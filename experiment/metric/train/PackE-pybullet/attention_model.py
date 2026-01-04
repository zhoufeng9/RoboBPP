import torch
from torch import nn
import math
from typing import NamedTuple
from graph_encoder import GraphAttentionEncoder
from distributions import FixedCategorical
from tools import observation_decode_leaf_node, init
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass

@dataclass
class AttentionModelFixed():
    node_embeddings: torch.Tensor
    context_node_projected: torch.Tensor
    glimpse_key: torch.Tensor
    glimpse_val: torch.Tensor
    logit_key: torch.Tensor

    def __getitem__(self, key):
        if torch.is_tensor(key) or isinstance(key, slice):
            return AttentionModelFixed(
                node_embeddings=self.node_embeddings[key],
                context_node_projected=self.context_node_projected[key],
                glimpse_key=self.glimpse_key[:, key],
                glimpse_val=self.glimpse_val[:, key],
                logit_key=self.logit_key[key]
            )
        # return super(AttentionModelFixed, self).__getitem__(key)
        return self[key]

class PCTModel(nn.Module):
    def __init__(self,
                 embedding_dim,
                 hidden_dim,
                 n_encode_layers=2,
                 tanh_clipping=10.,
                 mask_inner=False,
                 mask_logits=False,
                 n_heads=1,
                 internal_node_holder=None,
                 internal_node_length=None,
                 leaf_node_holder=None,
                 leaf_node_length = 8,
                 next_node_length = 6,
                 feature_aggregation = 'mean',
                 step_counter = 0,
                 args = None
                 ):
        super(PCTModel, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_encode_layers = n_encode_layers
        self.decode_type = None
        self.temp = 1.0
        self.step_counter = step_counter
        self.tanh_clipping = tanh_clipping
        self.mask_inner = mask_inner
        self.mask_logits = mask_logits

        self.n_heads = 1
        self.internal_node_holder = internal_node_holder
        self.internal_node_length = internal_node_length
        self.next_holder = 1

        self.leaf_node_holder = leaf_node_holder
        self.leaf_node_length = leaf_node_length
        self.next_node_length = next_node_length

        self.repeat_embed = args.repeat_embed
        self.gat_layer_num = args.gat_layer_num
        self.new_attention = args.new_attention
        self.draw_attention = args.draw_attention
        self.inner_leaf_attention_eliminate = args.inner_leaf_attention_eliminate
        self.policy_positional_encoding = args.policy_positional_encoding
        self.similarity = None
        self.no_next_item_input = args.no_next_item_input

        graph_size = internal_node_holder + leaf_node_holder + self.next_holder

        activate, ini = nn.LeakyReLU, 'leaky_relu'
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), nn.init.calculate_gain(ini))

        self.init_internal_node_embed = nn.Sequential(
            init_(nn.Linear(self.internal_node_length, 32)),
            activate(),
            init_(nn.Linear(32, embedding_dim)))

        self.init_leaf_node_embed = nn.Sequential(
            init_(nn.Linear(self.leaf_node_length, 32)),
            activate(),
            init_(nn.Linear(32, embedding_dim)))

        self.init_next_embed = nn.Sequential(
            init_(nn.Linear(self.next_node_length, 32)),
            activate(),
            init_(nn.Linear(32, embedding_dim)))

        d_model = embedding_dim
        max_len = 2000
        if self.policy_positional_encoding:
            position = torch.arange(max_len).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
            pe = torch.zeros(max_len, 1, d_model)
            pe[:, 0, 0::2] = torch.sin(position * div_term)
            pe[:, 0, 1::2] = torch.cos(position * div_term)  # [seq_len, batch_size, embedding_dim]
            self.pe = torch.transpose(pe, 0, 1)  # [batch_size, seq_len, embedding_dim]
        else:
            self.pe = torch.zeros(1, max_len, d_model)
        # assert False, 'some bug here'
        # Graph attention model
        self.embedder = nn.ModuleList([GraphAttentionEncoder(
            n_heads=n_heads,
            embed_dim=embedding_dim,
            n_layers=self.n_encode_layers,
            graph_size=graph_size,
            new_attention=self.new_attention,
            draw_attention=self.draw_attention,
            inner_leaf_attention_eliminate=self.inner_leaf_attention_eliminate,
            linear_net = args.linear_net,
            cate_attn = args.cate_attn
        ) for _ in range(self.gat_layer_num)])

        self.feature_aggregation = feature_aggregation
        self.project_node_embeddings = nn.Linear(embedding_dim, 3 * embedding_dim, bias=False) if feature_aggregation != 'rehearsal' else None
        if feature_aggregation == 'no_leaf' or feature_aggregation == 'item_attention':
            self.project_dim = embedding_dim * 2
        elif feature_aggregation == 'mean_norm' or feature_aggregation == 'mean_sep' or feature_aggregation == 'mean_running_norm' or feature_aggregation == 'rehearsal':
            self.project_dim = embedding_dim * 3
        else:
            self.project_dim = embedding_dim
        self.project_fixed_context = nn.Linear(self.project_dim, embedding_dim, bias=False)
        assert embedding_dim % n_heads == 0

        self.W_final_query = nn.Linear(embedding_dim, embedding_dim, bias=False) if feature_aggregation == 'item_attention' else None
        self.W_final_key = nn.Linear(embedding_dim, embedding_dim, bias=False) if feature_aggregation == 'item_attention' else None
        self.W_final_val = nn.Linear(embedding_dim, embedding_dim, bias=False) if feature_aggregation == 'item_attention' else None

        self.specified_mask_id = args.specified_mask_id
        self.specified_mask = torch.from_numpy(self.generate_eliminate_matrix(args.specified_mask_id)).to(args.device) if self.specified_mask_id != 0 else None

    def PositionalEncoding(self, x): # [batch_size, seq_len, embedding_dim]
        x = x + self.pe[:, :x.size(1)].to(x.device)
        # print(self.pe[:, x.size(1)].to(x.device))
        return x

    def set_step_counter(self, step_counter):
        self.step_counter = step_counter

    # 定义指数衰减函数
    def linear_decay(self, time_step, initial_value = 1, final_value = 1e-3, final_step = 100000):
        return final_value + (initial_value - final_value) * max(0, (1  - time_step / final_step))

    def batch_norm(self, x, epsilon=1e-5):
        mean = torch.mean(x, dim=0)
        variance = torch.var(x, dim=0)
        normalized = (x - mean) / torch.sqrt(variance + epsilon)
        return normalized

    def Min_Max_Scaling(self, x):
        # Replace X with your actual data
        # Min-Max Scaling
        min_val = torch.min(x, dim=0).values
        max_val = torch.max(x, dim=0).values
        x_scaled = (x - min_val) / (max_val - min_val)
        return x_scaled

    def Robust_Scaling(self, x):
        median = torch.median(x, dim=0).values
        q1 = torch.quantile(x, 0.25, dim=0)
        q3 = torch.quantile(x, 0.75, dim=0)
        iqr = q3 - q1
        iqr[iqr == 0.0] = 1.0  # Avoid division by zero for constant features
        x_robust_scaled = (x - median) / iqr
        return x_robust_scaled

    def layer_norm(self, x, epsilon=1e-5):
        mean = torch.mean(x, dim=-1).unsqueeze(-1)
        variance = torch.var(x, dim=-1).unsqueeze(-1)
        normalized = x - mean / torch.sqrt(variance + epsilon)
        return normalized

    def generate_eliminate_matrix(self, specified_mask_id):
        self.interval = [0, self.internal_node_holder, self.internal_node_holder + self.leaf_node_holder,
                         self.internal_node_holder + self.leaf_node_holder + self.next_holder]
        mask = np.zeros((self.internal_node_holder + self.leaf_node_holder + self.next_holder,
                            self.internal_node_holder + self.leaf_node_holder + self.next_holder))

        if specified_mask_id == 0: return None

        all_masks = []
        for i in range(3):
            for j in range(3):
                x_left = self.interval[i]
                x_right = self.interval[i + 1]
                y_left = self.interval[j]
                y_right = self.interval[j + 1]
                target_mask = mask.copy()
                target_mask[x_left:x_right, y_left:y_right] = 1
                all_masks.append(target_mask)
        all_masks = np.array(all_masks)

        binary_string = bin(specified_mask_id)[2:]
        binary_list = np.array([int(bit) for bit in binary_string])
        real_index = np.where(binary_list == 1)[0]
        matrices_list = all_masks[real_index]
        result_matrix = np.logical_or.reduce(matrices_list)

        return result_matrix

    def get_node_inputs(self, input, internal_node_holder, leaf_node_holder, normFactor=1):
        internal_nodes, leaf_nodes, next_item, invalid_leaf_nodes, full_mask = observation_decode_leaf_node(input,
                                                                                                            internal_node_holder,
                                                                                                            self.internal_node_length,
                                                                                                            leaf_node_holder)

        batch_size = input.size(0)
        internal_nodes_size = internal_nodes.size(1)
        leaf_node_size = leaf_nodes.size(1)
        next_size = next_item.size(1)

        internal_inputs = internal_nodes.contiguous().view(batch_size * internal_nodes_size,
                                                           self.internal_node_length) * normFactor
        leaf_inputs = leaf_nodes.contiguous().view(batch_size * leaf_node_size, self.leaf_node_length) * normFactor
        current_inputs = next_item.contiguous().view(batch_size * next_size, 6) * normFactor
        return internal_inputs, leaf_inputs, current_inputs



    def get_embedding(self, input, internal_node_holder, leaf_node_holder, normFactor=1, evaluate=False, decode_func = None):
        decode_func = observation_decode_leaf_node if decode_func is None else decode_func
        internal_nodes, leaf_nodes, next_item, invalid_leaf_nodes, full_mask = decode_func(input,
                                                                                           internal_node_holder,
                                                                                           self.internal_node_length,
                                                                                           leaf_node_holder)
        leaf_node_mask = 1 - invalid_leaf_nodes
        valid_length   = full_mask.sum(1)
        eliminate_mask = 1 - full_mask

        batch_size = input.size(0)
        graph_size = input.size(1)
        internal_nodes_size = internal_nodes.size(1)
        leaf_node_size = leaf_nodes.size(1)
        next_size = next_item.size(1)

        internal_inputs = internal_nodes.contiguous().view(batch_size * internal_nodes_size, self.internal_node_length) * normFactor
        leaf_inputs     = leaf_nodes.contiguous().view(batch_size * leaf_node_size, self.leaf_node_length) * normFactor
        current_inputs  = next_item.contiguous().view(batch_size * next_size, 6) * normFactor
        if self.no_next_item_input:
            current_inputs[:] = 0
        # We use three independent node-wise Multi-Layer Perceptron (MLP) blocks to project these raw space configuration nodes
        # presented by descriptors in different formats into the homogeneous node features.
        internal_embedded_inputs = self.init_internal_node_embed(internal_inputs).reshape(
            (batch_size, -1, self.embedding_dim))
        self.PositionalEncoding(internal_embedded_inputs)
        leaf_embedded_inputs = self.init_leaf_node_embed(leaf_inputs).reshape((batch_size, -1, self.embedding_dim))


        next_embedded_inputs = self.init_next_embed(current_inputs.squeeze()).reshape(batch_size, -1,
                                                                                      self.embedding_dim)


        # no activation function for the last layer
        embeddings = torch.cat((internal_embedded_inputs, leaf_embedded_inputs, next_embedded_inputs), dim=1).view(
            batch_size * graph_size, self.embedding_dim)

        # transform embeddings into high-level node features.
        for layer in self.embedder:
            for _ in range(self.repeat_embed):
                embeddings, _ = layer(embeddings,
                                          given_graph_size = internal_node_holder + leaf_node_holder + 1,
                                          mask = eliminate_mask,
                                          evaluate = evaluate,
                                          internal_node_holder = internal_node_holder,
                                          leaf_node_holder = leaf_node_holder,
                                          specified_mask = self.specified_mask
                                          )
        embedding_shape = (batch_size, graph_size, embeddings.shape[-1])
        return embeddings, embedding_shape, leaf_node_mask, eliminate_mask, valid_length

class AttentionModel(PCTModel):
    def __init__(self,
                 embedding_dim,
                 hidden_dim,
                 n_encode_layers=2,
                 tanh_clipping=10.,
                 mask_inner=False,
                 mask_logits=False,
                 n_heads=1,
                 internal_node_holder=None,
                 internal_node_length=None,
                 leaf_node_holder=None,
                 leaf_node_length=8,
                 next_node_length = 6,
                 feature_aggregation = 'mean',
                 args = None,
                 ):
        super(AttentionModel, self).__init__(embedding_dim = embedding_dim,
                                             hidden_dim = hidden_dim,
                                             n_encode_layers = n_encode_layers,
                                             tanh_clipping = tanh_clipping,
                                             mask_inner = mask_inner,
                                             mask_logits = mask_logits,
                                             n_heads = n_heads,
                                             internal_node_holder = internal_node_holder,
                                             internal_node_length = internal_node_length,
                                             leaf_node_holder = leaf_node_holder,
                                             leaf_node_length = leaf_node_length,
                                             feature_aggregation = feature_aggregation,
                                             next_node_length = next_node_length,
                                             args = args)

        self.pred_value_with_heightmap = args.pred_value_with_heightmap


    def forward(self, input, deterministic=False, evaluate_action=False, normFactor=1, evaluate=False,
                internal_node_holder=None, leaf_node_holder=None):
        internal_node_holder = internal_node_holder if internal_node_holder is not None else self.internal_node_holder
        leaf_node_holder = leaf_node_holder if leaf_node_holder is not None else self.leaf_node_holder
        embeddings, embedding_shape, leaf_node_mask, eliminate_mask, valid_length = self.get_embedding(input,
                                                                                                  internal_node_holder,
                                                                                                  leaf_node_holder,
                                                                                                  normFactor=normFactor,
                                                                                                  evaluate=evaluate)


        # Decide the leaf node indices for accommodating the current item
        log_p, action_log_prob, pointers, dist_entropy, dist, hidden = self._inner(embeddings,
                                                                                   deterministic=deterministic,
                                                                                   evaluate_action=evaluate_action,
                                                                                   shape=embedding_shape,
                                                                                   mask=leaf_node_mask,
                                                                                   eliminate_mask=eliminate_mask,
                                                                                   valid_length=valid_length,
                                                                                   internal_node_holder=internal_node_holder,
                                                                                   leaf_node_holder=leaf_node_holder
                                                                                   )
        return action_log_prob, pointers, dist_entropy, hidden, dist, embeddings, log_p

    def _inner(self, embeddings, mask=None, deterministic=False, evaluate_action=False,
               shape=None, eliminate_mask=None, valid_length=None, internal_node_holder=None, leaf_node_holder=None):
        # The aggregation of global feature
        fixed, internal_graph_embed = self._precompute(embeddings, shape=shape, eliminate_mask=eliminate_mask, valid_length=valid_length,
                                      internal_node_holder = internal_node_holder,
                                      leaf_node_holder = leaf_node_holder,
                                      evaluate_action = evaluate_action)
        # Calculate probabilities of selecting leaf nodes
        log_p, mask = self._get_log_p(fixed, mask, internal_node_holder=internal_node_holder,
                                      leaf_node_holder=leaf_node_holder)

        # The leaf node which is not feasible will be masked in a soft way.
        if deterministic:
            masked_outs = log_p * (1 - mask)
            masked_outs += 1e-20
        else:
            masked_outs = log_p * (1 - mask) + 1e-20

        log_p = torch.div(masked_outs, torch.sum(masked_outs, dim=1).unsqueeze(1))
        dist = FixedCategorical(probs=log_p)
        dist_entropy = dist.entropy()

        # Get maximum probabilities and indices
        if deterministic:
            # We take the argmax of the policy for the test.
            selected = dist.mode()
        else:
            # The action at is sampled from the distribution for training
            selected = dist.sample()

        if not evaluate_action:
            action_log_probs = dist.log_probs(selected)
        else:
            action_log_probs = None

        if self.pred_value_with_heightmap:
            return log_p, action_log_probs, selected, dist_entropy, dist, (fixed.context_node_projected, internal_graph_embed)
        else:
            return log_p, action_log_probs, selected, dist_entropy, dist, fixed.context_node_projected


    def _precompute(self, embeddings, num_steps=1, shape=None, eliminate_mask=None, valid_length=None,
                    internal_node_holder=None, leaf_node_holder=None, evaluate_action=False):

        # The aggregation of global feature, only happens on the eligible nodes.
        transEmbedding = embeddings.view(shape)
        valid_mask = 1 - eliminate_mask.int()
        origin_eliminate_mask = eliminate_mask.clone()
        eliminate_mask = eliminate_mask.view(shape[0], shape[1], 1).expand(shape).bool()
        transEmbedding[eliminate_mask] = 0
        transEmbedding = transEmbedding.view(embeddings.shape)
        internal_graph_embed = None
        if self.feature_aggregation == 'no_leaf':
            internal_graph_embed = transEmbedding.view(shape)[:, 0: internal_node_holder].sum(1)
            internal_graph_length = valid_mask[:, 0: internal_node_holder].sum(1)
            next_graph_embed = transEmbedding.view(shape)[:, -1]
            graph_embed = internal_graph_embed
            graph_embed = graph_embed / (internal_graph_length.reshape((-1, 1)) + 1)
            fixed_context = self.project_fixed_context(torch.cat((graph_embed, next_graph_embed), dim=1))
        elif self.feature_aggregation == 'mean':
            graph_embed = transEmbedding.view(shape).sum(1)
            graph_embed = graph_embed / valid_length.reshape((-1, 1))

            internal_graph_embed = transEmbedding.view(shape)[:, 0: internal_node_holder].sum(1)
            internal_graph_length = valid_mask[:, 0: internal_node_holder].sum(1)
            internal_graph_embed = internal_graph_embed / (internal_graph_length.reshape((-1, 1)) + 1e-6)

            fixed_context = self.project_fixed_context(graph_embed) # 这里可以改成仅仅和 internal nodes 和 item 相关

        elif self.feature_aggregation == 'item_attention': # 一个简单的 item attention 的实现，之后可能要参照 GCN 做标准实现

            norm_factor = 1 / math.sqrt(self.embedding_dim)
            next_graph_embed = transEmbedding.view(shape)[:, -1].reshape((-1, shape[-1]))

            internal_graph_embed = transEmbedding.view(shape)[:, 0: internal_node_holder].reshape((-1, shape[-1]))

            next_graph_embed_query   = self.W_final_query(next_graph_embed).reshape((shape[0], -1, shape[2]))
            internal_graph_embed_key = self.W_final_key(internal_graph_embed).reshape((shape[0], -1, shape[2]))
            internal_graph_embed_val = self.W_final_val(internal_graph_embed).reshape((shape[0], -1, shape[2]))

            internal_graph_mask = origin_eliminate_mask[:, 0: internal_node_holder]

            compatibility = norm_factor * torch.matmul(next_graph_embed_query, internal_graph_embed_key.transpose(1, 2))

            internal_graph_mask = internal_graph_mask.unsqueeze(1).bool()

            if evaluate_action:
                compatibility[internal_graph_mask] = -math.inf
            else:
                compatibility[internal_graph_mask] = -30

            attn = torch.softmax(compatibility, dim=-1)  #
            internal_graph_embed = torch.matmul(attn, internal_graph_embed_val).squeeze(1)
            fixed_context = self.project_fixed_context(torch.cat((internal_graph_embed, next_graph_embed_query.squeeze(1)), dim=1))

        elif self.feature_aggregation == 'mean_norm':

            internal_graph_embed = transEmbedding.view(shape)[:, 0: internal_node_holder].sum(1)
            internal_graph_length = valid_mask[:, 0: internal_node_holder].sum(1)
            internal_graph_embed = internal_graph_embed / torch.maximum(internal_graph_length.reshape((-1, 1)), torch.ones_like(internal_graph_length.reshape((-1, 1))) )
            internal_graph_embed = self.Robust_Scaling(internal_graph_embed)

            leaf_graph_embed = transEmbedding.view(shape)[:, internal_node_holder: internal_node_holder + leaf_node_holder].sum(1)
            leaf_graph_length = valid_mask[:, 0: internal_node_holder].sum(1)
            leaf_graph_embed = leaf_graph_embed / torch.maximum(leaf_graph_length.reshape((-1, 1)), torch.ones_like(leaf_graph_length.reshape((-1, 1))) )
            leaf_graph_embed = self.Robust_Scaling(leaf_graph_embed)

            next_graph_embed = transEmbedding.view(shape)[:, -1]
            next_graph_embed = self.Robust_Scaling(next_graph_embed)
            graph_embed = torch.cat((internal_graph_embed, leaf_graph_embed, next_graph_embed), dim=1)
            fixed_context = self.project_fixed_context(graph_embed)


        elif self.feature_aggregation == 'mean_sep':

            internal_graph_embed = transEmbedding.view(shape)[:, 0: internal_node_holder].sum(1)
            internal_graph_length = valid_mask[:, 0: internal_node_holder].sum(1)
            internal_graph_embed = internal_graph_embed / torch.maximum(internal_graph_length.reshape((-1, 1)), torch.ones_like(internal_graph_length.reshape((-1, 1))) )

            leaf_graph_embed = transEmbedding.view(shape)[:, internal_node_holder: internal_node_holder + leaf_node_holder].sum(1)
            leaf_graph_length = valid_mask[:, 0: internal_node_holder].sum(1)
            leaf_graph_embed = leaf_graph_embed / torch.maximum(leaf_graph_length.reshape((-1, 1)), torch.ones_like(leaf_graph_length.reshape((-1, 1))) )

            next_graph_embed = transEmbedding.view(shape)[:, -1]

            graph_embed = torch.cat((internal_graph_embed, leaf_graph_embed, next_graph_embed), dim=1)
            fixed_context = self.project_fixed_context(graph_embed)

        else:
            assert self.feature_aggregation == 'item'
            graph_embed = transEmbedding.view(shape)[:, -1]
            fixed_context = self.project_fixed_context(graph_embed) # 仅仅和 target item 相关

        if self.project_dim == self.embedding_dim:
            with torch.no_grad():
                internal_graph_embed = transEmbedding.view(shape)[:, 0: internal_node_holder].sum(1)
                internal_graph_length = valid_mask[:, 0: internal_node_holder].sum(1)
                internal_graph_embed = internal_graph_embed / torch.maximum(internal_graph_length.reshape((-1, 1)), torch.ones_like(internal_graph_length.reshape((-1, 1))) )

                leaf_graph_embed = transEmbedding.view(shape)[:, internal_node_holder: internal_node_holder + leaf_node_holder].sum(1)
                leaf_graph_length = valid_mask[:, 0: internal_node_holder].sum(1)
                leaf_graph_embed = leaf_graph_embed / torch.maximum(leaf_graph_length.reshape((-1, 1)), torch.ones_like(leaf_graph_length.reshape((-1, 1))) )

                next_graph_embed = transEmbedding.view(shape)[:, -1]

                cosine_sim_internal = F.cosine_similarity(graph_embed, internal_graph_embed, dim=1).mean().item()
                cosine_sim_leaf     = F.cosine_similarity(graph_embed, leaf_graph_embed, dim=1).mean().item()
                cosine_sim_next     = F.cosine_similarity(graph_embed, next_graph_embed, dim=1).mean().item()
                self.similarity =  (cosine_sim_internal, cosine_sim_leaf, cosine_sim_next)
        else:
            self.similarity = (0, 0, 0)

        glimpse_key_fixed, glimpse_val_fixed, logit_key_fixed = \
            self.project_node_embeddings(transEmbedding).view((shape[0], 1, shape[1], -1)).chunk(3, dim=-1)
        fixed_attention_node_data = (
            self._make_heads(glimpse_key_fixed, num_steps),
            self._make_heads(glimpse_val_fixed, num_steps),
            logit_key_fixed.contiguous()
        )

        return AttentionModelFixed(transEmbedding, fixed_context, *fixed_attention_node_data), internal_graph_embed

    def _get_log_p(self, fixed, mask=None, normalize=True, internal_node_holder=None, leaf_node_holder=None):
        # Compute query = context node embedding
        query = fixed.context_node_projected[:, None, :]

        # Compute keys and values for the nodes
        glimpse_K, glimpse_V, logit_K = self._get_attention_node_data(fixed)

        # Compute logits (unnormalized log_p)
        log_p, glimpse = self._one_to_many_logits(query, glimpse_K, glimpse_V, logit_K, mask,
                                                  internal_node_holder=internal_node_holder,
                                                  leaf_node_holder=leaf_node_holder)
        if normalize:
            log_p = torch.log_softmax(log_p / self.temp, dim=-1)
        assert not torch.isnan(log_p).any()
        return log_p.exp(), mask

    def _one_to_many_logits(self, query, glimpse_K, glimpse_V, logit_K, mask, internal_node_holder=None,
                            leaf_node_holder=None):
        batch_size, num_steps, embed_dim = query.size()
        key_size = val_size = embed_dim // self.n_heads
        # Compute the glimpse, rearrange dimensions so the dimensions are (n_heads, batch_size, num_steps, 1, key_size)
        glimpse_Q = query.view(batch_size, num_steps, self.n_heads, 1, key_size).permute(2, 0, 1, 3, 4)

        # Batch matrix multiplication to compute compatibilities (n_heads, batch_size, num_steps, graph_size)
        compatibility = torch.matmul(glimpse_Q, glimpse_K.transpose(-2, -1)) / math.sqrt(glimpse_Q.size(-1))
        logits = compatibility.reshape([-1, 1, compatibility.shape[-1]])

        # From the logits compute the probabilities by clipping, masking and softmax
        if self.tanh_clipping > 0:
            logits = torch.tanh(logits) * self.tanh_clipping

        logits = logits[:, 0, internal_node_holder: internal_node_holder + leaf_node_holder]
        if self.mask_logits:
            logits[mask.bool()] = -math.inf

        return logits, None

    def _one_to_many_logits_n_head(self, query, glimpse_K, glimpse_V, logit_K, mask, internal_node_holder=None,
                            leaf_node_holder=None):

        batch_size, num_steps, embed_dim = query.size()
        key_size = val_size = embed_dim // self.n_heads

        # Compute the glimpse, rearrange dimensions so the dimensions are (n_heads, batch_size, num_steps, 1, key_size)
        glimpse_Q = query.view(batch_size, num_steps, self.n_heads, 1, key_size).permute(2, 0, 1, 3, 4)

        # Batch matrix multiplication to compute compatibilities (n_heads, batch_size, num_steps, graph_size)
        compatibility = torch.matmul(glimpse_Q, glimpse_K.transpose(-2, -1)) / math.sqrt(glimpse_Q.size(-1))
        logits = compatibility.reshape([-1, 1, compatibility.shape[-1]])

        # From the logits compute the probabilities by clipping, masking and softmax
        if self.tanh_clipping > 0:
            logits = torch.tanh(logits) * self.tanh_clipping

        logits = logits[:, 0, internal_node_holder: internal_node_holder + leaf_node_holder]
        if self.mask_logits:
            logits[mask.bool()] = -math.inf

        return logits, None

    def _get_attention_node_data(self, fixed):
        return fixed.glimpse_key, fixed.glimpse_val, fixed.logit_key

    def _make_heads(self, v, num_steps=None):
        assert num_steps is None or v.size(1) == 1 or v.size(1) == num_steps

        return (
            v.contiguous().view(v.size(0), v.size(1), v.size(2), self.n_heads, -1)
            .expand(v.size(0), v.size(1) if num_steps is None else num_steps, v.size(2), self.n_heads, -1)
            .permute(3, 0, 1, 2, 4)
        )