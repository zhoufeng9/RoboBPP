#!/usr/bin/env python3

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Size notations:

class SeqAttention(nn.Module):
    """Sequential self-attention layer.
    """

    def __init__(self, hidden_size, enable_mem, attn_span):
        nn.Module.__init__(self)
        self.hidden_size = hidden_size  # size of a single head
        self.attn_span = attn_span
        self.enable_mem = enable_mem

    def forward(self, query, key, value):
        # query size = B x M x H
        # key, value sizes = B x (M+L) x H

        # compute attention from context
        # B x M (dest) x (M+L) (src)
        attn = torch.matmul(query, key.transpose(-1, -2))

        attn = attn / math.sqrt(self.hidden_size)  # B x M X (M+L)
        attn = F.softmax(attn, dim=-1)


        out = torch.matmul(attn, value)  # B x M x H

        return out

    def get_cache_size(self):
            return self.attn_span


class MultiHeadSeqAttention(nn.Module):
    def __init__(self, hidden_size, enable_mem, nb_heads, attn_span):
        nn.Module.__init__(self)
        assert hidden_size % nb_heads == 0
        self.nb_heads = nb_heads
        self.head_dim = hidden_size // nb_heads
        self.attn = SeqAttention(
            hidden_size=self.head_dim, enable_mem=enable_mem, attn_span=attn_span)
        self.proj_query = nn.Linear(hidden_size, hidden_size, bias=False)
        self.proj_out = nn.Linear(hidden_size, hidden_size, bias=False)
        self.proj_val = nn.Linear(hidden_size, hidden_size, bias=False)
        self.proj_key = nn.Linear(hidden_size, hidden_size, bias=False)

        # note that the linear layer initialization in current Pytorch is kaiming uniform init

    def head_reshape(self, x):
        K = self.nb_heads
        D = self.head_dim
        x = x.view(x.size()[:-1] + (K, D))  # B x (M+L) x K x D
        x = x.transpose(1, 2).contiguous()  # B x K x (M+L) x D
        x = x.view(-1, x.size(-2), x.size(-1))  # B_K x (M+L) x D
        return x

    def forward(self, query, key, value):
        B = query.size(0)
        K = self.nb_heads
        D = self.head_dim
        M = query.size(1)

        query = self.proj_query(query)
        query = self.head_reshape(query)
        value = self.proj_val(value)
        value = self.head_reshape(value)
        key = self.proj_key(key)
        key = self.head_reshape(key)

        out = self.attn(query, key, value)  # B_K x M x D
        out = out.view(B, K, M, D)  # B x K x M x D
        out = out.transpose(1, 2).contiguous()  # B x M x K x D
        out = out.view(B, M, -1)  # B x M x K_D
        out = self.proj_out(out)
        return out


class FeedForwardLayer(nn.Module):
    def __init__(self, hidden_size, inner_hidden_size):
        nn.Module.__init__(self)
        self.fc1 = nn.Linear(hidden_size, inner_hidden_size)
        self.fc2 = nn.Linear(inner_hidden_size, hidden_size)

    def forward(self, h):
        h1 = F.relu(self.fc1(h))
        h2 = self.fc2(h1)
        return h2

class TransformerSeqLayer(nn.Module):
    def __init__(self, hidden_size, enable_mem, nb_heads, attn_span, inner_hidden_size):
        nn.Module.__init__(self)
        self.attn = MultiHeadSeqAttention(
            hidden_size=hidden_size, enable_mem=enable_mem, nb_heads=nb_heads, attn_span=attn_span)
        self.ff = FeedForwardLayer(hidden_size=hidden_size, inner_hidden_size=inner_hidden_size)
        self.norm1=nn.BatchNorm1d(hidden_size)
        self.norm2=nn.BatchNorm1d(hidden_size)
        self.enable_mem = enable_mem

    def forward(self, h, h_cache):
        # h = B x M x H
        # h_cache = B x L x H
        if self.enable_mem:
            h_all = torch.cat([h_cache, h], dim=1)  # B x (M+L) x H
        else:
            h_all = h_cache                         # B x M x H
        attn_out = self.attn(h, h_all, h_all)
        h = self.norm1((h + attn_out).view(-1, h.size(-1))).view(*h.size()) # B x M x H
        ff_out = self.ff(h)
        out = self.norm2((h+ ff_out).view(-1, h.size(-1))).view(*h.size()) # B x M x H
        return out


class EncoderSeq(nn.Module):
    def __init__(self, state_size, hidden_size, nb_heads, encoder_nb_layers,
                 attn_span, inner_hidden_size):
        nn.Module.__init__(self)
        # init embeddings
        self.init_embed = nn.Linear(state_size, hidden_size)

        self.layers = nn.ModuleList()
        self.layers.extend(
            TransformerSeqLayer(
                hidden_size=hidden_size, enable_mem=True, nb_heads=nb_heads,
                attn_span=attn_span, inner_hidden_size=inner_hidden_size)
            for _ in range(encoder_nb_layers))

    def forward(self, x, h_cache):
        # x size = B x M
        block_size = x.size(1)
        h = self.init_embed(x)  # B x M x H
        h_cache_next = []
        for l, layer in enumerate(self.layers):
            cache_size = layer.attn.attn.get_cache_size()

            # B x L x H
            h_cache_next_l = torch.cat(
                [h_cache[l][:, -cache_size + 1:, :], h[:, 0:1, :]],
                dim=1).detach()

            h_cache_next.append(h_cache_next_l)

            h = layer(h, h_cache[l])  # B x M x H

        return h, h_cache_next


class QDecoder(nn.Module):
    def __init__(self, state_size, hidden_size, nb_heads, decoder_nb_layers,
                 attn_span, inner_hidden_size):
        nn.Module.__init__(self)
        # init embeddings
        self.init_embed = nn.Linear(state_size, hidden_size)

        self.layers = nn.ModuleList()
        self.layers.extend(
            TransformerSeqLayer(
                hidden_size=hidden_size, enable_mem=False, nb_heads=nb_heads,
                attn_span=attn_span, inner_hidden_size=inner_hidden_size)
            for _ in range(decoder_nb_layers))

    def forward(self, x, embedding):
        # x size = B x Q_M
        block_size = x.size(1)
        h = self.init_embed(x)  # B x Q_M x H
        h_cache_next = []
        for l, layer in enumerate(self.layers):

            h = layer(h, embedding)  # B x Q_M x H

        return h

class PackDecoder(nn.Module):
    def __init__(self, head_hidden_size, res_size, state_size, hidden_size,nb_heads, decoder_layers,attn_span, inner_hidden_size):
        nn.Module.__init__(self)

        self.att_decoder = QDecoder(state_size, hidden_size,nb_heads=nb_heads,decoder_nb_layers=decoder_layers,attn_span=attn_span, inner_hidden_size=inner_hidden_size)

        self.head  = nn.Sequential(
                            nn.Linear(hidden_size, head_hidden_size),
                            nn.ReLU(),
                            nn.Linear(head_hidden_size, res_size)
                            )
        self.critic = nn.Sequential(
                            nn.Linear(hidden_size, head_hidden_size),
                            nn.ReLU(),
                            nn.Linear(head_hidden_size, 1)
                            )

    def forward(self, x, embedding):
        h = self.att_decoder(x, embedding)
        out = self.head(h)
        value = self.critic(h)
        return out, value

def orientation_transform(l,w,h,num):
    if(num==0):
        return l,w,h
    elif(num==1):
        return l,h,w
    elif(num==2):
        return w,l,h
    elif(num==3):
        return w,h,l
    elif(num==4):
        return h,l,w
    else:
        return h,w,l

def observation_decode(observation, args = None):
    batch_size = observation.size(0)
    observation = observation.reshape(batch_size, -1, 9)
    packed_state = observation[:, 0: args.internal_node_holder, 0:6]
    next_box = observation[:,args.internal_node_holder:, 3:6]
    return packed_state, next_box

class RCQL(nn.Module):
    def __init__(self, state_size, hidden_size, nb_heads, encoder_nb_layers, attn_span, inner_hidden_size,src_head_hidden_size,pos_head_hidden_size, 
                 s_res_size, r_res_size,x_res_size,y_res_size,decoder_nb_layers,item_state_size):
        nn.Module.__init__(self)

        self.encoder = nn.Linear(state_size, hidden_size)


        self.encoder=EncoderSeq(state_size=state_size,hidden_size=hidden_size,nb_heads=nb_heads,encoder_nb_layers=encoder_nb_layers,attn_span=attn_span,
                                inner_hidden_size=inner_hidden_size)
        # self.s_decoder=PackDecoder(head_hidden_size=src_head_hidden_size,res_size=s_res_size,state_size=item_state_size,hidden_size=hidden_size,
        #                            decoder_layers=decoder_nb_layers,attn_span=attn_span,inner_hidden_size=inner_hidden_size,nb_heads=nb_heads)
        self.r_decoder=PackDecoder(head_hidden_size=src_head_hidden_size,res_size=r_res_size,state_size=item_state_size,hidden_size=hidden_size,
                                   decoder_layers=decoder_nb_layers,attn_span=attn_span,inner_hidden_size=inner_hidden_size,nb_heads=nb_heads)
        self.x_decoder=PackDecoder(head_hidden_size=pos_head_hidden_size,res_size=x_res_size,state_size=item_state_size,hidden_size=hidden_size,
                                   decoder_layers=decoder_nb_layers,attn_span=attn_span,inner_hidden_size=inner_hidden_size,nb_heads=nb_heads)
        self.y_decoder=PackDecoder(head_hidden_size=pos_head_hidden_size,res_size=y_res_size,state_size=item_state_size,hidden_size=hidden_size,
                                   decoder_layers=decoder_nb_layers,attn_span=attn_span,inner_hidden_size=inner_hidden_size,nb_heads=nb_heads)
        self.softmax=nn.Softmax(dim=1)
        self.alpha=torch.nn.Parameter(torch.tensor(1.0),requires_grad=True)

    def calc_ori_idx(self,actor_encoder_out,select_seq_idx,isGreedy):
        r_out, r_value = self.r_decoder(select_seq_idx, actor_encoder_out)
        batch_size=actor_encoder_out.size()[0]
        templist=[i for i in range(batch_size)]
        r_out=torch.squeeze(r_out,dim=1)
        r_out=self.softmax(r_out)
        if(isGreedy):
            oriidx=torch.max(r_out,1)[1]
            # return oriidx
        else:
            oriidx=torch.multinomial(r_out,1)
            oriidx=torch.squeeze(oriidx)
        oripro=r_out[templist,oriidx]
        return oriidx, oripro, r_out, r_value
        
    def calc_x_idx(self,actor_encoder_out,select_seq_idx_ori,isGreedy):
        x_out, x_value = self.x_decoder(select_seq_idx_ori, actor_encoder_out)
        batch_size=actor_encoder_out.size()[0]
        templist=[i for i in range(batch_size)]
        x_out=torch.squeeze(x_out, dim=1)
        
        # for i in range(batch_size):
        #     x_out[i,maskx[i]]=-np.inf

        x_out=self.softmax(x_out)
        if(isGreedy):
            xidx=torch.max(x_out,1)[1]
            # return xidx
        else:
            xidx=torch.multinomial(x_out,1)
            xidx=torch.squeeze(xidx)
        xpro=x_out[templist,xidx]
        return xidx, xpro, x_out, x_value
    
    def calc_y_idx(self,actor_encoder_out,select_seq_idx_ori,isGreedy):
        y_out, y_value = self.y_decoder(select_seq_idx_ori, actor_encoder_out)
        batch_size=actor_encoder_out.size()[0]
        templist=[i for i in range(batch_size)]
        y_out=torch.squeeze(y_out, dim=1)
        # for i in range(batch_size):
        #     y_out[i,masky[i]]=-np.inf   
        y_out=self.softmax(y_out)
        if(isGreedy):
            yidx=torch.max(y_out,1)[1]
            # return yidx
        else:
            yidx=torch.multinomial(y_out,1)
            yidx=torch.squeeze(yidx)
        ypro=y_out[templist,yidx]
        return yidx, ypro, y_out, y_value
    
    def set_device(self,device):
        self.encoder=self.encoder.to(device)
        # self.s_decoder=self.s_decoder.to(device)
        self.r_decoder=self.r_decoder.to(device)
        self.x_decoder=self.x_decoder.to(device)
        self.y_decoder=self.y_decoder.to(device)
        
    def set_eval(self):
        self.encoder=self.encoder.eval()
        # self.s_decoder=self.s_decoder.eval()
        self.r_decoder=self.r_decoder.eval()
        self.x_decoder=self.x_decoder.eval()
        self.y_decoder=self.y_decoder.eval()

    def forward(self, observation, h_caches, IsGreedy, args):
        batch_size = observation.size(0)
        packed_state, select_item = observation_decode(observation, args)

        actor_encoder_out, h_caches = self.encoder(packed_state, h_caches)

        oriidx, oripro, oriout, orivalue = self.calc_ori_idx(actor_encoder_out, select_item, IsGreedy)

        select_ori_item=torch.zeros(select_item.size()).to(device)
        for jk in range(batch_size):
            select_ori_item[jk,:,0],select_ori_item[jk,:,1],select_ori_item[jk,:,2]=orientation_transform(select_item[jk,:,0],select_item[jk,:,1],select_item[jk,:,2],oriidx[jk])

        xidx, xpro, xout, xvalue = self.calc_x_idx(actor_encoder_out, select_ori_item, IsGreedy)

        yidx, ypro, yout, yvalue = self.calc_y_idx(actor_encoder_out, select_ori_item, IsGreedy)
        return torch.concat([oriidx.unsqueeze(1), xidx.unsqueeze(1), yidx.unsqueeze(1)], dim = 1), \
               torch.concat([oripro.unsqueeze(1), xpro.unsqueeze(1), ypro.unsqueeze(1)], dim = 1), \
               torch.mean(torch.concat([orivalue.squeeze(1), xvalue.squeeze(1), yvalue.squeeze(1)], dim = 1), dim = 1)

    def evaluate_actions(self, observation, h_caches, action, args):
        batch_size = observation.size(0)
        packed_state, select_item = observation_decode(observation, args)

        actor_encoder_out, h_caches = self.encoder(packed_state, h_caches)

        oriidx = action[:, 0]
        _, _, oriout, orivalue = self.calc_ori_idx(actor_encoder_out, select_item, False)

        select_ori_item=torch.zeros(select_item.size()).to(device)
        for jk in range(batch_size):
            select_ori_item[jk,:,0],select_ori_item[jk,:,1],select_ori_item[jk,:,2]=orientation_transform(select_item[jk,:,0],select_item[jk,:,1],select_item[jk,:,2],oriidx[jk])

        xidx = action[:, 1]
        _, _, xout, xvalue = self.calc_x_idx(actor_encoder_out, select_ori_item, False)
        yidx = action[:, 2]
        _, _, yout, yvalue = self.calc_y_idx(actor_encoder_out, select_ori_item, False)
        dist_ori = torch.distributions.Categorical(probs=oriout)
        dist_x = torch.distributions.Categorical(probs=xout)
        dist_y = torch.distributions.Categorical(probs=yout)
        action_log_probs_ori = dist_ori.log_prob(oriidx)
        entropy_ori = dist_ori.entropy().mean()
        action_log_probs_x = dist_x.log_prob(xidx)
        entropy_x = dist_x.entropy().mean()
        action_log_probs_y = dist_y.log_prob(yidx)
        entropy_y = dist_y.entropy().mean()
        return torch.mean(torch.concat([orivalue.squeeze(1), xvalue.squeeze(1), yvalue.squeeze(1)], dim = 1), dim = 1), \
               torch.concat([action_log_probs_ori.unsqueeze(1), action_log_probs_x.unsqueeze(1), action_log_probs_y.unsqueeze(1)], dim = 1), \
               entropy_ori + entropy_x + entropy_y




class RolloutStorage(object):
    def __init__(self, num_steps, num_processes, obs_shape, action_space,
                 recurrent_hidden_state_size = 128):
        self.obs = torch.zeros(num_steps + 1, num_processes, *obs_shape)
        self.recurrent_hidden_states = torch.zeros(
            num_steps + 1, num_processes, recurrent_hidden_state_size)
        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_processes, 1)
        self.returns = torch.zeros(num_steps + 1, num_processes, 1)
        self.action_log_probs = torch.zeros(num_steps, num_processes, 3)
        self.actions = torch.zeros(num_steps, num_processes, 3)
        self.actions = self.actions.long()
        self.masks = torch.ones(num_steps + 1, num_processes, 1)
        self.num_steps = num_steps
        self.step = 0

    def to(self, device):
        self.obs = self.obs.to(device)
        self.recurrent_hidden_states = self.recurrent_hidden_states.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)
        # self.bad_masks = self.bad_masks.to(device)
        # self.location_masks = self.location_masks.to(device)

    def insert(self, obs, recurrent_hidden_states, actions, action_log_probs,
               value_preds, rewards, masks):
        self.obs[self.step + 1].copy_(obs)
        actions = actions.view(-1, 3)
        self.actions[self.step].copy_(actions)
        action_log_probs = action_log_probs.view(-1, 3)
        self.action_log_probs[self.step].copy_(action_log_probs)
        value_preds = value_preds.view(-1, 1)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)
        self.step = (self.step + 1) % self.num_steps


    def after_update(self):
        self.obs[0].copy_(self.obs[-1])
        # self.recurrent_hidden_states[0].copy_(self.recurrent_hidden_states[-1])
        self.masks[0].copy_(self.masks[-1])


    def compute_returns(self,
                        next_value,
                        use_gae,
                        gamma,
                        gae_lambda,
                        use_proper_time_limits=True):
            next_value = next_value.view(-1, 1)
            self.returns[-1] = next_value#
            for step in reversed(range(self.rewards.size(0))):
                self.returns[step] = self.returns[step + 1] * \
                    gamma * self.masks[step + 1] + self.rewards[step]

    def feed_forward_generator(self,
                               advantages,
                               num_mini_batch=None,
                               mini_batch_size=None):
        num_steps, num_processes = self.rewards.size()[0:2]
        batch_size = num_processes * num_steps

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(num_processes, num_steps, num_processes * num_steps,
                          num_mini_batch))
            mini_batch_size = batch_size // num_mini_batch
        sampler = BatchSampler(
            SubsetRandomSampler(range(batch_size)),
            mini_batch_size,
            drop_last=True)
        for indices in sampler:
            obs_batch = self.obs[:-1].view(-1, *self.obs.size()[2:])[indices]
            recurrent_hidden_states_batch = self.recurrent_hidden_states[:-1].view(
                -1, self.recurrent_hidden_states.size(-1))[indices]
            actions_batch = self.actions.view(-1,
                                              self.actions.size(-1))[indices]
            value_preds_batch = self.value_preds[:-1].view(-1, 1)[indices]
            return_batch = self.returns[:-1].view(-1, 1)[indices]
            masks_batch = self.masks[:-1].view(-1, 1)[indices]
            old_action_log_probs_batch = self.action_log_probs.view(-1,
                                                                    1)[indices]
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages.view(-1, 1)[indices]

            yield obs_batch, recurrent_hidden_states_batch, actions_batch, \
                value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ