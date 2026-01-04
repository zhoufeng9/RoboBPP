import torch
from torch import nn
import math
import numpy as np
import matplotlib.pyplot as plt


def draw_attention(attention_matrix, valid_mask, internal_node_holder, leaf_node_holder):

    internal_node_length = int(np.sum(valid_mask[0:internal_node_holder]))
    leaf_node_length     = int(np.sum(valid_mask[internal_node_holder:internal_node_holder + leaf_node_holder]))

    selected_rows = np.concatenate((attention_matrix[0:internal_node_length, :],
                                    attention_matrix[internal_node_holder:internal_node_holder + leaf_node_length, :],
                                    attention_matrix[-1:, :]), axis=0)
    attention_matrix = np.concatenate((selected_rows[:, 0:internal_node_length],
                                    selected_rows[:, internal_node_holder:internal_node_holder + leaf_node_length],
                                    selected_rows[:, -1:]), axis=1)

    labels = ['internal node', 'leaf node', 'next item']
    coords = [internal_node_length / 2, internal_node_length + leaf_node_length / 2, internal_node_length + leaf_node_length + 0.5]
    coords = np.array(coords) - 0.5
    plt.xticks(coords, labels, rotation=0, va='center')
    plt.yticks(coords, labels, rotation=90, va='center')

    plt.imshow(attention_matrix, cmap='viridis')
    plt.title('Attention Matrix')
    plt.xlabel('Item Index')
    plt.ylabel('Query Index')
    plt.colorbar()  # 显示颜色条，表示数值与颜色的对应关系
    plt.show()

class SkipConnection(nn.Module):
    def __init__(self, module):
        super(SkipConnection, self).__init__()
        self.module = module

    # GNN 里面的 encode 不是很必要
    def forward(self, input):
        return {'data':input['data'] + self.module(input), 'mask': input['mask'],
        # return {'data':input['data'], 'mask': input['mask'],
        # return {'data': self.module(input), 'mask': input['mask'],
                'graph_size':input['graph_size'],
                'internal_node_holder': input['internal_node_holder'],
                'leaf_node_holder': input['leaf_node_holder'],
                'specified_mask': input['specified_mask']
                }

class SkipConnection_Linear(nn.Module):
    def __init__(self, module):
        super(SkipConnection_Linear, self).__init__()
        self.module = module

    def forward(self, input):
        return {'data':input['data'] + self.module(input['data']), 'mask': input['mask'],
        # return {'data':input['data'], 'mask': input['mask'],
        # return {'data':self.module(input['data']), 'mask': input['mask'],
                'graph_size': input['graph_size'],
                'internal_node_holder':  input['internal_node_holder'],
                'leaf_node_holder':  input['leaf_node_holder'],
                'specified_mask': input['specified_mask'] }

class MultiHeadAttention(nn.Module):
    def __init__(
            self,
            n_heads,
            input_dim,
            embed_dim=None,
            val_dim=None,
            key_dim=None,
            draw_attention=False,
            inner_leaf_attention_eliminate = False,
            cate_attn = False
        ):
        super(MultiHeadAttention, self).__init__()

        if val_dim is None:
            assert embed_dim is not None, "Provide either embed_dim or val_dim"
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim
        self.draw_attention = draw_attention
        self.inner_leaf_attention_eliminate = inner_leaf_attention_eliminate
        self.cate_attn = cate_attn

        self.norm_factor = 1 / math.sqrt(key_dim)  # See Attention is all you need

        self.W_query = nn.Linear(input_dim, key_dim, bias=False)
        self.W_key = nn.Linear(input_dim, key_dim, bias=False)
        self.W_val = nn.Linear(input_dim, val_dim, bias=False)

        if embed_dim is not None:
            # self.W_out = nn.Parameter(torch.Tensor(n_heads, key_dim, embed_dim))
            self.W_out = nn.Linear(key_dim, embed_dim)

        self.init_parameters()

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, data, h=None):
        """
        :param q: queries (batch_size, n_query, input_dim)
        :param h: data (batch_size, graph_size, input_dim)
        :param mask: mask (batch_size, n_query, graph_size) or viewable as that (i.e. can be 2 dim if n_query == 1)
        Mask should contain 1 if attention is not possible (i.e. mask is negative adjacency)
        :return:
        """
        q = data['data']
        mask = data['mask']
        graph_size = data['graph_size']
        internal_node_holder = data['internal_node_holder']
        leaf_node_holder = data['leaf_node_holder']
        specified_mask = data['specified_mask']

        if h is None:
            h = q
        assert len(q.size()) == 2
        batch_size = int(q.size()[0] / graph_size)

        input_dim = h.size()[-1]
        n_query = graph_size
        assert input_dim == self.input_dim, "Wrong embedding dimension of input"

        hflat = h.contiguous().view(-1, input_dim)
        qflat = q.contiguous().view(-1, input_dim)

        # last dimension can be different for keys and values
        shp = (self.n_heads, batch_size, graph_size, -1)
        shp_q = (self.n_heads, batch_size, n_query, -1)
        Q = self.W_query(qflat).view(shp_q) # 把数据分解成若干个头，每个头里面的数据小一些了，也就是 feature 小一点了
        K = self.W_key(hflat).view(shp)
        V = self.W_val(hflat).view(shp)

        # Calculate compatibility (n_heads, batch_size, n_query, graph_size)
        compatibility = self.norm_factor * torch.matmul(Q, K.transpose(2, 3)) # 每个数据头里面的内积

        # Optionally apply mask to prevent attention
        if mask is not None:
            eliminate = mask.unsqueeze(1).repeat((1, graph_size, 1)) # 把最后一个维度的 mask 重复一遍
            if self.inner_leaf_attention_eliminate:
                eliminate[:,internal_node_holder:internal_node_holder+leaf_node_holder, internal_node_holder:internal_node_holder + leaf_node_holder] *= torch.eye(leaf_node_holder).cuda().to(eliminate.device)
            eliminate = eliminate.bool()
            if specified_mask is not None:
                eliminate = eliminate | specified_mask.unsqueeze(0).repeat((batch_size, 1, 1))
            eliminate = eliminate.view(1, batch_size, n_query, graph_size).expand_as(compatibility) # 一个头里面的 mask 拓展到多个
            if data['evaluate']:
                compatibility[eliminate] = -math.inf
            else:
                compatibility[eliminate] = -30

        if not self.cate_attn:
            attn = torch.softmax(compatibility, dim=-1) #
        else:
            attn_internal = torch.softmax(compatibility[:, :, :, 0:internal_node_holder], dim=-1)
            attn_leaf = torch.softmax(compatibility[:, :, :, internal_node_holder:internal_node_holder + leaf_node_holder], dim=-1)
            attn_next = torch.softmax(compatibility[:, :, :, -1:], dim=-1)
            attn = torch.cat((attn_internal, attn_leaf, attn_next), dim=-1) / 3

        if self.draw_attention:
            draw_attention(attn[0][0].detach().cpu().numpy(), 1 - mask[0].detach().cpu().numpy(), internal_node_holder, leaf_node_holder)

        # If there are nodes with no neighbours then softmax returns nan so we fix them to 0
        # 目前这里没有很明显的作用，可以先忽视
        if mask is not None:
            attnc = attn.clone()
            attnc[eliminate] = 0
            attn = attnc

        heads = torch.matmul(attn, V)
        out = self.W_out(heads.permute(1, 2, 0, 3).contiguous().view(-1, self.n_heads * self.val_dim)).view(batch_size * n_query, self.embed_dim)
        # out[:] = 0
        return out

    # def forward(self, data, h=None):
    #     """
    #     :param q: queries (batch_size, n_query, input_dim)
    #     :param h: data (batch_size, graph_size, input_dim)
    #     :param mask: mask (batch_size, n_query, graph_size) or viewable as that (i.e. can be 2 dim if n_query == 1)
    #     Mask should contain 1 if attention is not possible (i.e. mask is negative adjacency)
    #     :return:
    #     """
    #     q = data['data']
    #     mask = data['mask']
    #     graph_size = data['graph_size']
    #     internal_node_holder = data['internal_node_holder']
    #     leaf_node_holder = data['leaf_node_holder']
    #     specified_mask = data['specified_mask']
    #
    #     if h is None:
    #         h = q
    #     assert len(q.size()) == 2
    #     batch_size = int(q.size()[0] / graph_size)
    #
    #     input_dim = h.size()[-1]
    #     n_query = graph_size
    #     assert input_dim == self.input_dim, "Wrong embedding dimension of input"
    #
    #     hflat = h.contiguous().view(-1, input_dim)
    #     qflat = q.contiguous().view(-1, input_dim)
    #
    #     # last dimension can be different for keys and values
    #     shp = (self.n_heads, batch_size, graph_size, -1)
    #     shp_q = (self.n_heads, batch_size, n_query, -1)
    #     Q = self.W_query(qflat).view(shp_q) # 把数据分解成若干个头，每个头里面的数据小一些了，也就是 feature 小一点了
    #     K = self.W_key(hflat).view(shp)
    #     V = self.W_val(hflat).view(shp)
    #
    #     # Calculate compatibility (n_heads, batch_size, n_query, graph_size)
    #     compatibility = self.norm_factor * torch.matmul(Q, K.transpose(2, 3)) # 每个数据头里面的内积
    #
    #     # Optionally apply mask to prevent attention
    #     if mask is not None:
    #         eliminate = mask.unsqueeze(1).repeat((1, graph_size, 1)) # 把最后一个维度的 mask 重复一遍
    #         if self.inner_leaf_attention_eliminate:
    #             eliminate[:,internal_node_holder:internal_node_holder+leaf_node_holder, internal_node_holder:internal_node_holder + leaf_node_holder] *= torch.eye(leaf_node_holder).cuda().to(eliminate.device)
    #         eliminate = eliminate.bool()
    #         if specified_mask is not None:
    #             eliminate = eliminate | specified_mask.unsqueeze(0).repeat((batch_size, 1, 1))
    #         eliminate = eliminate.view(1, batch_size, n_query, graph_size).expand_as(compatibility) # 一个头里面的 mask 拓展到多个
    #         if data['evaluate']:
    #             compatibility[eliminate] = -math.inf
    #         else:
    #             compatibility[eliminate] = -30
    #     attn = torch.softmax(compatibility, dim=-1) #
    #     # If there are nodes with no neighbours then softmax returns nan so we fix them to 0
    #     if mask is not None:
    #         attnc = attn.clone()
    #         attnc[eliminate] = 0
    #         attn = attnc
    #
    #     if self.draw_attention:
    #         draw_attention(attn[0][0].detach().cpu().numpy(), 1 - mask[0].detach().cpu().numpy(), internal_node_holder, leaf_node_holder)
    #
    #     heads = torch.matmul(attn, V)
    #     out_with_internal = self.W_out(heads.permute(1, 2, 0, 3).contiguous().view(-1, self.n_heads * self.val_dim)).view(batch_size * n_query, self.embed_dim)
    #
    #
    #     q = data['data']
    #     mask = data['mask']
    #     graph_size = data['graph_size']
    #     internal_node_holder = data['internal_node_holder']
    #     leaf_node_holder = data['leaf_node_holder']
    #     specified_mask = data['specified_mask']
    #
    #     mask[:,0:internal_node_holder] = -math.inf
    #
    #     if h is None:
    #         h = q
    #     assert len(q.size()) == 2
    #     batch_size = int(q.size()[0] / graph_size)
    #
    #     input_dim = h.size()[-1]
    #     n_query = graph_size
    #     assert input_dim == self.input_dim, "Wrong embedding dimension of input"
    #
    #     hflat = h.contiguous().view(-1, input_dim)
    #     qflat = q.contiguous().view(-1, input_dim)
    #
    #     # last dimension can be different for keys and values
    #     shp = (self.n_heads, batch_size, graph_size, -1)
    #     shp_q = (self.n_heads, batch_size, n_query, -1)
    #     Q = self.W_query(qflat).view(shp_q)  # 把数据分解成若干个头，每个头里面的数据小一些了，也就是 feature 小一点了
    #     K = self.W_key(hflat).view(shp)
    #     V = self.W_val(hflat).view(shp)
    #
    #     # Calculate compatibility (n_heads, batch_size, n_query, graph_size)
    #     compatibility = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))  # 每个数据头里面的内积
    #
    #     # Optionally apply mask to prevent attention
    #     if mask is not None:
    #         eliminate = mask.unsqueeze(1).repeat((1, graph_size, 1))  # 把最后一个维度的 mask 重复一遍
    #         if self.inner_leaf_attention_eliminate:
    #             eliminate[:, internal_node_holder:internal_node_holder + leaf_node_holder,
    #             internal_node_holder:internal_node_holder + leaf_node_holder] *= torch.eye(leaf_node_holder).cuda().to(
    #                 eliminate.device)
    #         eliminate = eliminate.bool()
    #         if specified_mask is not None:
    #             eliminate = eliminate | specified_mask.unsqueeze(0).repeat((batch_size, 1, 1))
    #         eliminate = eliminate.view(1, batch_size, n_query, graph_size).expand_as(compatibility)  # 一个头里面的 mask 拓展到多个
    #         if data['evaluate']:
    #             compatibility[eliminate] = -math.inf
    #         else:
    #             compatibility[eliminate] = -30
    #     attn = torch.softmax(compatibility, dim=-1)  #
    #     # If there are nodes with no neighbours then softmax returns nan so we fix them to 0
    #     if mask is not None:
    #         attnc = attn.clone()
    #         attnc[eliminate] = 0
    #         attn = attnc
    #
    #     if self.draw_attention:
    #         draw_attention(attn[0][0].detach().cpu().numpy(), 1 - mask[0].detach().cpu().numpy(), internal_node_holder,
    #                        leaf_node_holder)
    #
    #     heads = torch.matmul(attn, V)
    #     out = self.W_out(
    #         heads.permute(1, 2, 0, 3).contiguous().view(-1, self.n_heads * self.val_dim)).view(batch_size * n_query,
    #                                                                                            self.embed_dim)
    #     print(torch.cosine_similarity(out, out_with_internal, dim=1).mean())
    #     return out

class MultiHeadAttention_new(nn.Module):
    def __init__(
            self,
            n_heads,
            input_dim,
            embed_dim=None,
            val_dim=None,
            key_dim=None,
            dropout_rate= 0.0,
            draw_attention = False
    ):
        super(MultiHeadAttention_new, self).__init__()

        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim   = embed_dim
        self.key_dim   = embed_dim
        self.head_dim = embed_dim // n_heads
        self.draw_attention = draw_attention

        # self.norm_factor = 1 / math.sqrt(key_dim)  # See Attention is all you need
        self.W_query = nn.Linear(input_dim, self.key_dim, bias=False)
        self.W_key = nn.Linear(input_dim, self.key_dim, bias=False)
        self.W_val = nn.Linear(input_dim, self.val_dim, bias=False)

        if embed_dim is not None:
            # self.W_out = nn.Parameter(torch.Tensor(n_heads, key_dim, embed_dim))
            # self.W_out = nn.ModuleList(nn.Linear(key_dim, embed_dim) for _ in range(n_heads))
            self.W_out = nn.Linear(input_dim, embed_dim)

        self.init_parameters()
        self.dropout = nn.Dropout(dropout_rate)

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, data, h=None):
        """
        :param q: queries (batch_size, n_query, input_dim)
        :param h: data (batch_size, graph_size, input_dim)
        :param mask: mask (batch_size, n_query, graph_size) or viewable as that (i.e. can be 2 dim if n_query == 1)
        Mask should contain 1 if attention is not possible (i.e. mask is negative adjacency)
        :return:
        """

        q = data['data']
        mask = data['mask']
        graph_size = data['graph_size']
        if h is None:
            h = q

        batch_size = int(q.size()[0] / graph_size)
        graph_size = graph_size
        input_dim = h.size()[-1]
        n_query = graph_size # n_query 可以不等于 graph_size
        assert input_dim == self.input_dim, "Wrong embedding dimension of input"

        hflat = h.contiguous().view(-1, input_dim)
        qflat = q.contiguous().view(-1, input_dim)

        # (batch_size, graph_size, n_heads, feat_size)
        shp = (batch_size,   -1, self.n_heads,  self.head_dim)
        shp_q = (batch_size, -1, self.n_heads,  self.head_dim)

        # 这里是用同一个网络预测的不同分量算 attention，然后加和，之前的那个版本是用不同的参数预测小的分量，然后拼接，我觉得这边这个版本更对一些吧
        # (batch_size, n_heads, graph_size, feat_size)
        Q = self.W_query(qflat).view(shp_q).transpose(1, 2)
        K = self.W_key(hflat).view(shp).transpose(1, 2)
        V = self.W_val(hflat).view(shp).transpose(1, 2)

        # Calculate compatibility (batch_size, n_heads, n_query, graph_size)
        compatibility = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))

        # Optionally apply mask to prevent attention
        if mask is not None:
            # (batch_size, 1, n_query, graph_size)
            mask = mask.unsqueeze(1).repeat((1, graph_size, 1)).bool()
            # (batch_size, n_heads, n_query, graph_size)
            mask = mask.view(batch_size, 1, n_query, graph_size).expand_as(compatibility)
            if data['evaluate']:
                compatibility[mask] = -math.inf
            else:
                compatibility[mask] = -30
        attn = torch.softmax(compatibility, dim=-1) #
        #############################################
        # attn_internal = torch.softmax(compatibility)

        # # If there are nodes with no neighbours then softmax returns nan so we fix them to 0
        if mask is not None:
            attnc = attn.clone()
            attnc[mask] = 0
            attn = attnc


        # (batch_size, n_heads, n_query, graph_size) Q K 被拆分了，V 没有
        weighted_sum = torch.matmul(attn, V)
        # (batch_size, n_query, n_heads, graph_size)
        output = weighted_sum.transpose(1, 2).contiguous().view(-1, input_dim)
        output = self.W_out(output).reshape(batch_size * n_query, self.embed_dim)
        return output

class MultiHeadAttentionLayer(nn.Sequential):
    def __init__(
            self,
            n_heads,
            embed_dim,
            feed_forward_hidden=128,
            new_attention = False,
            draw_attention = False,
            inner_leaf_attention_eliminate = False,
            linear_net = False,
            cate_attn = False
        ):
        attention_func = MultiHeadAttention_new if new_attention else MultiHeadAttention
        super(MultiHeadAttentionLayer, self).__init__(
            SkipConnection(
                attention_func(
                    n_heads,
                    input_dim=embed_dim,
                    embed_dim=embed_dim,
                    draw_attention = draw_attention,
                    inner_leaf_attention_eliminate = inner_leaf_attention_eliminate,
                    cate_attn = cate_attn
                )
            ),
            SkipConnection_Linear(
                nn.Sequential(
                    nn.Linear(embed_dim, feed_forward_hidden),
                    nn.ReLU(),
                    nn.Linear(feed_forward_hidden, embed_dim)
                ) if feed_forward_hidden > 0 else nn.Linear(embed_dim, embed_dim)
            ) if not linear_net else SkipConnection_Linear(
                nn.Sequential(
                    nn.Linear(embed_dim, feed_forward_hidden, bias=False),
                    nn.Linear(feed_forward_hidden, embed_dim, bias=False)
                ) if feed_forward_hidden > 0 else nn.Linear(embed_dim, embed_dim)
            ),

        )

class GraphAttentionEncoder(nn.Module):
    def __init__(
            self,
            n_heads,
            embed_dim,
            n_layers,
            node_dim=None,
            feed_forward_hidden=128,
            graph_size=None,
            new_attention = False,
            draw_attention = False,
            inner_leaf_attention_eliminate = False,
            linear_net = False,
            cate_attn = False
    ):
        super(GraphAttentionEncoder, self).__init__()
        n_layers = 1
        # To map input to embedding space
        self.init_embed = nn.Linear(node_dim, embed_dim) if node_dim is not None else None
        self.graph_size = graph_size
        self.layers = nn.Sequential(*(
            MultiHeadAttentionLayer(n_heads, embed_dim, feed_forward_hidden, new_attention, draw_attention,
                                    inner_leaf_attention_eliminate, linear_net, cate_attn)
            for _ in range(n_layers)
        ))

    def forward(self, x, mask=None, limited=False, given_graph_size = None, evaluate = False, internal_node_holder  = None, leaf_node_holder = None, specified_mask = None):

        graph_size = given_graph_size if given_graph_size is not None else self.graph_size
        # Batch multiply to get initial embeddings of nodes
        h = self.init_embed(x.view(-1, x.size(-1))).view(*x.size()[:2], -1) if self.init_embed is not None else x

        data = {'data':h, 'mask': mask, 'graph_size': graph_size,
                'evaluate': evaluate, 'internal_node_holder': internal_node_holder,
                'leaf_node_holder': leaf_node_holder, 'specified_mask': specified_mask}
        h = self.layers(data)['data']
        return (h, h.view(int(h.size()[0] / graph_size), graph_size, -1).mean(dim=1),)