import torch
import torch.nn as nn
import numpy as np
import math
from torchvision import transforms
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def observation_decode(observation, contianer_size):
    frontiers = observation[:, 0: contianer_size[0] * contianer_size[2] * 2]
    next_box  = observation[:, contianer_size[0] * contianer_size[2] * 2:contianer_size[0] * contianer_size[2] * 2+3]
    next_box  = torch.unsqueeze(next_box, 1)
    maskposition = observation[:, contianer_size[0] * contianer_size[2] * 2 + 3:]
    maskseq = []
    seqidx = torch.zeros(frontiers.size()[0], dtype=torch.long)
    return next_box, frontiers, maskposition, maskseq, seqidx

class Attend2Pack(nn.Module):
    def __init__(self,originaldim,embedingdim,num_head,FFNdim,AFFNnum,inchannel,outchannel,C,outdim,L,W, scale, target_size = (256, 256)):
        super().__init__()
        self.AFFNnum=AFFNnum
        self.AFFNlist=nn.ModuleList()
        self.embeding=nn.Linear(originaldim,embedingdim)
        for i in range(AFFNnum):
            self.AFFNlist.append(AttentionFFN(embedingdim,num_head,FFNdim).to(device))
        self.FrontierEmbedding=FrontierEmbedding(inchannel,outchannel,embedingdim,L,W).to(device)
        self.DecodingPosition=DecodingPosition(embedingdim,outdim).to(device)
        self.scale = scale
        self.resize_transform = transforms.Resize(target_size)

    def clacb(self,input,scale):
        x=self.embeding(input)
        for i in range(self.AFFNnum):
            x=self.AFFNlist[i](x,scale)
        return x

    def forward(self, observation, IsGreedy, contianer_size):
        inputb, inputf, maskposition, maskseq, seqidx = observation_decode(observation, contianer_size)
        inputb = self.clacb(inputb, self.scale)
        inputf = inputf.reshape(inputf.size()[0], 2, contianer_size[0], contianer_size[2])
        inputf = self.resize_transform(inputf)
        inputf = self.FrontierEmbedding(inputf)
        posidx, policypro, policy, value = self.DecodingPosition(seqidx, inputb, inputf, maskseq, maskposition,IsGreedy)
        return policypro, posidx, value

    def evaluate_actions(self, observation, action, contianer_size):
        inputb, inputf, maskposition, maskseq, seqidx = observation_decode(observation, contianer_size)
        inputb = self.clacb(inputb, self.scale)
        inputf = inputf.reshape(inputf.size()[0], 2, contianer_size[0], contianer_size[2])
        inputf = self.resize_transform(inputf)
        inputf = self.FrontierEmbedding(inputf)
        _, _, policy, value = self.DecodingPosition(seqidx, inputb, inputf, maskseq, maskposition,IsGreedy = False)
        dist = torch.distributions.Categorical(probs=policy)
        action_log_probs = dist.log_prob(action.squeeze(1))
        return value, action_log_probs, dist.entropy().mean()

class MHA(nn.Module):
    def __init__(self,embedingdim,num_head):
        super().__init__()
        self.embedingdim=embedingdim
        self.num_head=num_head
        self.edn=embedingdim//num_head
        self.Q=nn.Linear(embedingdim,embedingdim,bias=False)
        self.K=nn.Linear(embedingdim,embedingdim,bias=False)
        self.V=nn.Linear(embedingdim,embedingdim,bias=False)
        self.softmax=nn.Softmax(dim=-1)
        self.O=nn.Linear(embedingdim,embedingdim,bias=False)
    
    def forward(self,input,scale):
        batch,n,dim=input.shape
        nh=self.num_head
        edn=self.edn
        q=self.Q(input).reshape(batch,n,nh,edn).transpose(1,2)
        k=self.K(input).reshape(batch,n,nh,edn).transpose(1,2)
        v=self.V(input).reshape(batch,n,nh,edn).transpose(1,2)
        dist=torch.matmul(q,k.transpose(2,3))*scale
        dist=self.softmax(dist)
        att=torch.matmul(dist,v)
        att=att.transpose(1,2).reshape(batch,n,dim)
        att=self.O(att)
        return att
                  
class AttentionFFN(nn.Module):
    def __init__(self,embedingdim,num_head,lineardim):
        super().__init__()
        self.mha=MHA(embedingdim,num_head)
        self.norm=nn.LayerNorm(embedingdim)
        self.linear1=nn.Linear(embedingdim,lineardim)
        self.linear2=nn.Linear(lineardim,embedingdim)
        
    def forward(self,input,scale):
        x=self.norm(input)
        x=self.mha(x,scale)+x
        x=self.norm(x)
        x2=torch.relu(self.linear1(x))
        x=torch.relu(self.linear2(x2))+x
        return x
    
class FrontierEmbedding(nn.Module):
    def __init__(self,inchannel,outchannel,embedingdim,L,W):
        super().__init__()
        self.conv1=nn.Conv2d(inchannel,  outchannel,kernel_size=4, stride=2, padding=1)
        self.conv2=nn.Conv2d(outchannel, outchannel,kernel_size=4, stride=4)
        self.conv3=nn.Conv2d(outchannel, outchannel,kernel_size=4, stride=4)
        L = 8
        W = 8
        self.norm1          = nn.LayerNorm([outchannel,L,W])
        self.linear1        = nn.Linear(outchannel*L*W,embedingdim)
        self.LinearFrontier = nn.Linear(embedingdim,embedingdim,bias=False)

    def forward(self,input):
        x=self.conv1(input)
        x=self.conv2(x)
        x=self.conv3(x)
        x=self.norm1(x)
        x=torch.relu(x)
        x=torch.flatten(x,1)
        x=self.linear1(x)
        x=self.LinearFrontier(x)
        return x


class DecodingPosition(nn.Module):
    def __init__(self,embedingdim,out_dim):
        super().__init__()
        self.embedingdim=embedingdim
        self.LinearW_selected=nn.Linear(embedingdim,embedingdim,bias=False)
        self.LinearP_leftover=nn.Linear(embedingdim,embedingdim,bias=False)
        self.LinearOut=nn.Linear(embedingdim,out_dim,bias=False)
        self.softmax=nn.Softmax(dim=1)
        self.critic_moduel = nn.Linear(embedingdim, 1)

    def forward(self,selectb,inputb,inputf,maskseq,maskposition,IsGreedy):
        batch_size=inputb.size()[0]

        bst=self.LinearW_selected(inputb)
        bst=torch.squeeze(bst,1)
        inputq=(bst+inputf)/2
        ct2=self.LinearOut(inputq)

        ct2[maskposition==0] = -1e6

        policy=self.softmax(ct2)
        batchX = torch.arange(batch_size)
        value = self.critic_moduel(inputq)

        if(IsGreedy):
            posidx = torch.max(policy,1)[1]
        else:
            posidx=torch.multinomial(policy,1)
        posidx=torch.squeeze(posidx)
        policypro=policy[batchX,posidx]
        return posidx, policypro, policy, value

class RolloutStorage(object):
    def __init__(self, num_steps, num_processes, obs_shape, action_space,
                 recurrent_hidden_state_size = 128):
        self.obs = torch.zeros(num_steps + 1, num_processes, *obs_shape)
        self.recurrent_hidden_states = torch.zeros(
            num_steps + 1, num_processes, recurrent_hidden_state_size)
        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_processes, 1)
        self.returns = torch.zeros(num_steps + 1, num_processes, 1)
        self.action_log_probs = torch.zeros(num_steps, num_processes, 1)
        self.actions = torch.zeros(num_steps, num_processes, 1)
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
        actions = actions.view(-1, 1)
        self.actions[self.step].copy_(actions)
        action_log_probs = action_log_probs.view(-1, 1)
        self.action_log_probs[self.step].copy_(action_log_probs)
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