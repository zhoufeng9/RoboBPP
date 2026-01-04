import torch.nn as nn
from tools import init, AddBias
from numpy import sqrt
from attention_model import AttentionModel
# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
from torch import optim
from torch.nn.utils import clip_grad_norm_
from rainbow_model import DQNBPP
import math
from attention_model import GraphAttentionEncoder
from distributions import FixedNormal, FixedCategorical

class DRL_GAT(nn.Module):
    def __init__(self, args):
        super(DRL_GAT, self).__init__()

        self.actor = AttentionModel(
                                    args.embedding_size,
                                    args.hidden_size,
                                    n_encode_layers = args.gat_layer_num,
                                    n_heads = args.head_num,
                                    internal_node_holder = args.internal_node_holder,
                                    internal_node_length = args.internal_node_length,
                                    leaf_node_holder = args.leaf_node_holder,
                                    feature_aggregation = args.feature_aggregation,
                                    args = args
                                    )
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), sqrt(2))

        self.critic_moduel = init_(nn.Linear(args.embedding_size * (2 if args.actor_with_q else 1), 1))
        self.args = args
        self.pred_value_with_heightmap = args.pred_value_with_heightmap
        self.critic_distill = init_(nn.Linear(args.embedding_size, 1)) if self.pred_value_with_heightmap else None

    def critic(self, hidden, embeddings, log_p, p):
        if self.args.actor_with_q:
            batch_size = hidden.shape[0]
            embeddings = torch.cat([hidden.unsqueeze(dim=1).repeat((1, embeddings.shape[1], 1)), embeddings], dim=-1)
            embeddings = embeddings.reshape((-1, embeddings.shape[-1]))
            full_q = self.critic_moduel(embeddings).reshape((batch_size, -1))
            v = (full_q * log_p).sum(1, keepdim = True)
            q = full_q.gather(1, p)
            return q, full_q, v, None
        elif self.pred_value_with_heightmap:
            v = self.critic_moduel(hidden[0])
            v_distill = self.critic_distill(hidden[1])
            return None, None, v, v_distill
        else:
            v = self.critic_moduel(hidden)
            return None, None, v, None

    def forward(self, items, deterministic = False, normFactor = 1, evaluate = False,
                internal_node_holder = None, leaf_node_holder = None, return_q = False, return_v_distill = False):
        internal_node_holder = internal_node_holder if internal_node_holder is not None else self.args.internal_node_holder
        leaf_node_holder = leaf_node_holder if leaf_node_holder is not None else self.args.leaf_node_holder
        o, p, dist_entropy, hidden, _, embeddings, log_p = self.actor(items, deterministic, normFactor = normFactor, evaluate = evaluate, internal_node_holder = internal_node_holder, leaf_node_holder = leaf_node_holder)
        embeddings = embeddings.reshape((items.shape[0], -1, self.args.embedding_size))[:, internal_node_holder: internal_node_holder + leaf_node_holder, :]
        q_values, full_q_values, v_values, v_distill = self.critic(hidden, embeddings, log_p, p)

        if return_q:
            return o, p, dist_entropy, (q_values, full_q_values, v_values)
        elif return_v_distill:
            return o, p, dist_entropy, (v_distill, v_values)
        else:
            return o, p, dist_entropy, v_values

    def evaluate_actions(self, items, actions, normFactor = 1,
                         return_q = False, return_v_distill = False):

        _, p, dist_entropy, hidden, dist, embeddings, log_p = self.actor(items, evaluate_action = True, normFactor = normFactor)
        embeddings = embeddings.reshape((items.shape[0], -1, self.args.embedding_size))[:, self.args.internal_node_holder: self.args.internal_node_holder + self.args.leaf_node_holder, :]
        action_log_probs = dist.log_probs(actions)
        q_values, full_q_values, v_values, v_distill =  self.critic(hidden, embeddings, log_p, p)

        if return_q:
            return (q_values, full_q_values, v_values), action_log_probs, dist_entropy.mean()
        elif return_v_distill:
            return (v_distill, v_values), action_log_probs, dist_entropy.mean()
        else:
            return v_values, action_log_probs, dist_entropy.mean()

    def get_node_inputs(self, items, normFactor = 1):
        internal_inputs, leaf_inputs, current_inputs = self.actor.get_node_inputs(items, self.args.internal_node_holder, self.args.leaf_node_holder, normFactor)
        return internal_inputs, leaf_inputs, current_inputs

    def get_action_rank(self, items, deterministic = False, normFactor = 1, evaluate = False, internal_node_holder = None, leaf_node_holder = None, return_q = False):
        internal_node_holder = internal_node_holder if internal_node_holder is not None else self.args.internal_node_holder
        leaf_node_holder = leaf_node_holder if leaf_node_holder is not None else self.args.leaf_node_holder
        o, p, dist_entropy, hidden, _, embeddings, log_p = self.actor(items, deterministic, normFactor = normFactor, evaluate = evaluate, internal_node_holder = internal_node_holder, leaf_node_holder = leaf_node_holder)
        embeddings = embeddings.reshape((items.shape[0], -1, self.args.embedding_size))[:, internal_node_holder: internal_node_holder + leaf_node_holder, :]
        q_values, full_q_values, v_values, v_distill = self.critic(hidden, embeddings, log_p, p)
        if return_q:
            return log_p, p, dist_entropy, (q_values, full_q_values, v_values)
        else:
            return log_p, p, dist_entropy, v_values

class RainbowAgent():
    def __init__(self, args):
        self.action_space = args.leaf_node_holder
        self.atoms = args.atoms
        self.Vmin = args.V_min
        self.Vmax = args.V_max
        self.support = torch.linspace(args.V_min, args.V_max, self.atoms).to(device=args.device)  # Support (range) of z
        self.delta_z = (args.V_max - args.V_min) / (self.atoms - 1)
        self.batch_size = args.batch_size
        self.n = args.multi_step # For GAE implementation.
        self.discount = args.discount
        self.norm_clip = args.norm_clip
        args.model = args.model_path
        network = DQNBPP
        self.online_net = network(args.embedding_size,
                                        args.hidden_size,
                                        n_encode_layers = args.gat_layer_num,
                                        n_heads = 1,
                                        internal_node_holder = args.internal_node_holder,
                                        internal_node_length = args.internal_node_length,
                                        leaf_node_holder = args.leaf_node_holder,
                                        args = args).to(device=args.device).share_memory()

        if args.model and args.load_model:  # Load pretrained model if provided
            if os.path.isfile(args.model):
                state_dict = torch.load(args.model, map_location='cpu')  # Always load tensors onto CPU by default, will shift to GPU if necessary
                self.online_net.load_state_dict(state_dict)
                print("Loading pretrained model: " + args.model)
            else:  # Raise error if incorrect model path provided
                raise FileNotFoundError(args.model)

        self.online_net.train()
        self.target_net = network(args.embedding_size,
                                        args.hidden_size,
                                        n_encode_layers = args.gat_layer_num,
                                        n_heads = 1,
                                        internal_node_holder = args.internal_node_holder,
                                        internal_node_length = args.internal_node_length,
                                        leaf_node_holder = args.leaf_node_holder,
                                        args = args).to(device=args.device).share_memory()
        self.update_target_net()
        self.target_net.train()
        for param in self.target_net.parameters():
            param.requires_grad = False
        self.optimiser = optim.Adam(self.online_net.parameters(), lr=args.learning_rate, eps=args.adam_eps)

    # Resets noisy weights in all linear layers (of online net only)
    def reset_noise(self):
        self.online_net.reset_noise()

    # Acts based on single state (no batch)
    def act(self, state):
        with torch.no_grad():
            q_map= self.online_net(state)
            mask = self.online_net.get_leaf_node_mask(state)
            sum_q_map = q_map * self.support
            sum_q_map = sum_q_map.sum(2)
            if mask is not None:
                sum_q_map[(1 - mask).bool()] = -math.inf
        return sum_q_map.argmax(1)

    # Acts with an ε-greedy policy (used for evaluation only)
    def act_e_greedy(self, state, mask, epsilon=0.001):  # High ε can reduce evaluation scores drastically
        if np.random.random() < epsilon:
            return np.random.randint(0, self.action_space)
        else:
            return self.act(state, mask)

    # core part
    def learn(self, memory):
        segment_size = int(self.batch_size/len(memory))
        idxs, states, actions, returns, next_states, nonterminals, weights = [],[],[],[],[],[],[]

        for mem in memory:
            idx, state, action, ret, next_state, nonterminal, weight = mem.sample(segment_size)
            idxs.append(idx), states.append(state), actions.append(action), returns.append(ret)
            next_states.append(next_state), nonterminals.append(nonterminal), weights.append(weight)

        states = torch.cat(states, 0)
        actions = torch.cat(actions, 0)
        returns = torch.cat(returns, 0)
        next_states = torch.cat(next_states, 0)
        nonterminals = torch.cat(nonterminals, 0).reshape(self.batch_size, 1)
        weights = torch.cat(weights, 0)

        # Calculate current state probabilities (online network noise already sampled)
        log_ps= self.online_net(states, log=True)  # Log probabilities log p(s_t, ·; θonline)
        log_ps_a = log_ps[range(self.batch_size), actions]  # log p(s_t, a_t; θonline)

        with torch.no_grad():
            # Calculate nth next state probabilities
            pns = self.online_net(next_states)  # Probabilities p(s_t+n, ·; θonline)
            dns = self.support.expand_as(pns) * pns  # Distribution d_t+n = (z, p(s_t+n, ·; θonline))
            argmax_indices_ns = dns.sum(2).argmax(1)  # Perform argmax action selection using online network: argmax_a[(z, p(s_t+n, a; θonline))]

            self.target_net.reset_noise()  # Sample new target net noise

            pns = self.target_net(next_states)  # Probabilities p(s_t+n, ·; θtarget)
            pns_a = pns[range(self.batch_size), argmax_indices_ns]  # Double-Q probabilities p(s_t+n, argmax_a[(z, p(s_t+n, a; θonline))]; θtarget)

            # Compute Tz (Bellman operator T applied to z)
            Tz = returns.unsqueeze(1) + nonterminals * (self.discount ** self.n) * self.support.unsqueeze(0)  # Tz = R^n + (γ^n)z (accounting for terminal states)
            Tz = Tz.clamp(min=self.Vmin, max=self.Vmax)  # Clamp between supported values

            # Compute L2 projection of Tz onto fixed support z
            b = (Tz - self.Vmin) / self.delta_z  # b = (Tz - Vmin) / Δz
            l, u = b.floor().to(torch.int64), b.ceil().to(torch.int64)

            # Fix disappearing probability mass when l = b = u (b is int)
            l[(u > 0) * (l == u)] -= 1
            u[(l < (self.atoms - 1)) * (l == u)] += 1

            # Distribute probability of Tz
            m = states.new_zeros(self.batch_size, self.atoms)
            offset = torch.linspace(0, ((self.batch_size - 1) * self.atoms), self.batch_size).unsqueeze(1).expand(self.batch_size, self.atoms).to(actions)
            m.view(-1).index_add_(0, (l + offset).view(-1), (pns_a * (u.float() - b)).view(-1))  # m_l = m_l + p(s_t+n, a*)(u - b)
            m.view(-1).index_add_(0, (u + offset).view(-1), (pns_a * (b - l.float())).view(-1))  # m_u = m_u + p(s_t+n, a*)(b - l)

        loss = -torch.sum(m * log_ps_a, 1)
        self.online_net.zero_grad()
        (weights * loss).mean().backward()  # Backpropagate importance-weighted minibatch loss
        clip_grad_norm_(self.online_net.parameters(), self.norm_clip)  # Clip gradients by L2 norm
        self.optimiser.step()

        for i in range(len(memory)):
            memory[i].update_priorities(idxs[i], loss[i * segment_size : (i+1) * segment_size].detach().cpu())  # Update priorities of sampled transitions
        return loss

    def update_target_net(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    # Save model parameters on current device (don't move model between devices)
    def save(self, path, name):
        torch.save(self.online_net.state_dict(), os.path.join(path, name))

    # Evaluates Q-value based on single state (no batch)
    def evaluate_q(self, state):
        with torch.no_grad():
            return (self.online_net(state.unsqueeze(0)) * self.support).sum(2).max(1)[0].item()

    def train(self):
        self.online_net.train()

    def eval(self):
        self.online_net.eval()

class DiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(DiagGaussian, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))

        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
        self.logstd = AddBias(torch.zeros(num_outputs))

        self.right_bound = 0.5
        self.left_bound  = 0.1

        self.max_logstd = 0
        self.min_logstd = -10

    def forward(self, x):

        x = x.reshape(-1, x.shape[-1])
        action_mean = self.fc_mean(x)

        #  An ugly hack for my KFAC implementation.
        zeros = torch.zeros(action_mean.size())
        if x.is_cuda:
            zeros = zeros.cuda()

        action_logstd = self.logstd(zeros)
        action_mean   = torch.sigmoid(action_mean) * (self.right_bound - self.left_bound + 0.04) + self.left_bound - 0.02
        # action_logstd = soft_clamp(action_logstd, self.min_logstd, self.max_logstd)
        return FixedNormal(action_mean, action_logstd.exp()), action_mean, action_logstd

