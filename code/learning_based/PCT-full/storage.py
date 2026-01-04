import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

# Storage for n-step training.
class PCTRolloutStorage(object):
    def __init__(self, num_steps, num_processes, obs_shape, gamma):
        self.obs = torch.zeros(num_steps + 1, num_processes, *obs_shape)
        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.returns = torch.zeros(num_steps + 1, num_processes, 1)
        self.action_log_probs = torch.zeros(num_steps, num_processes, 1)
        self.actions = torch.zeros(num_steps, num_processes, 1).long()
        self.masks = torch.ones(num_steps + 1, num_processes, 1)
        self.value_preds = torch.zeros(num_steps+1, num_processes, 1)

        self.num_steps = num_steps
        self.gamma = gamma
        self.step = 0

    def to(self, device):
        self.obs = self.obs.to(device)
        self.rewards = self.rewards.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)
        self.value_preds = self.value_preds.to(device)

    def cuda(self):
        self.obs = self.obs.cuda()
        self.rewards = self.rewards.cuda()
        self.returns = self.returns.cuda()
        self.action_log_probs = self.action_log_probs.cuda()
        self.actions = self.actions.cuda()
        self.masks = self.masks.cuda()

    def insert(self, obs, actions, action_log_probs, value_preds, rewards, masks):
        self.obs[self.step + 1].copy_(obs)
        self.actions[self.step].copy_(actions)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)
        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        self.obs[0].copy_(self.obs[-1])
        self.masks[0].copy_(self.masks[-1])

    def compute_returns(self, next_value):
        self.returns[-1] = next_value#
        for step in reversed(range(self.rewards.size(0))):
            self.returns[step] = self.returns[step + 1] * \
                self.gamma * self.masks[step + 1] + self.rewards[step]

    def feed_forward_generator(self, advantages, num_mini_batch=None, mini_batch_size=None):
        num_steps, num_processes = self.rewards.size()[0:2]
        batch_size = num_processes * num_steps

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes ({}) * number of steps ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(num_processes, num_steps, num_processes * num_steps, num_mini_batch))

            mini_batch_size = batch_size // num_mini_batch

        sampler = BatchSampler(SubsetRandomSampler(range(batch_size)),
                               mini_batch_size,
                               drop_last=True)

        for indices in sampler:
            obs_batch = self.obs[:-1].view(-1, *self.obs.size()[2:])[indices]
            actions_batch = self.actions.view(-1, self.actions.size(-1))[indices]
            value_preds_batch = self.value_preds[:-1].view(-1, 1)[indices]
            return_batch = self.returns[:-1].view(-1, 1)[indices]
            masks_batch = self.masks[:-1].view(-1, 1)[indices]
            old_action_log_probs_batch = self.action_log_probs.view(-1, 1)[indices]

            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages.view(-1, 1)[indices]

            yield obs_batch, actions_batch, value_preds_batch, return_batch, masks_batch, \
                  old_action_log_probs_batch, adv_targ

class RehearsalRolloutStorage(object):
    def __init__(self, num_steps, num_processes, obs_shape, gamma, add_obs_shape, action_shape):
        self.obs = torch.zeros(num_steps + 1, num_processes, *obs_shape)
        self.add_obs = torch.zeros(num_steps + 1, num_processes, *add_obs_shape)

        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.returns = torch.zeros(num_steps + 1, num_processes, 1)
        self.action_log_probs = torch.zeros(num_steps, num_processes, 1)
        self.actions = torch.zeros(num_steps, num_processes, action_shape)
        self.masks = torch.ones(num_steps + 1, num_processes, 1)
        self.value_preds = torch.zeros(num_steps+1, num_processes, 1)

        self.num_steps = num_steps
        self.gamma = gamma
        self.step = 0

    def to(self, device):
        self.obs = self.obs.to(device)
        self.add_obs = self.add_obs.to(device)
        self.rewards = self.rewards.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)
        self.value_preds = self.value_preds.to(device)

    def cuda(self):
        self.obs = self.obs.cuda()
        self.add_obs = self.add_obs.cuda()
        self.rewards = self.rewards.cuda()
        self.returns = self.returns.cuda()
        self.action_log_probs = self.action_log_probs.cuda()
        self.actions = self.actions.cuda()
        self.masks = self.masks.cuda()

    def insert(self, obs, actions, action_log_probs, value_preds, rewards, masks, add_obs):
        self.obs[self.step + 1].copy_(obs)
        self.add_obs[self.step + 1].copy_(add_obs)
        self.actions[self.step].copy_(actions)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)
        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        self.obs[0].copy_(self.obs[-1])
        self.add_obs[0].copy_(self.add_obs[-1])
        self.masks[0].copy_(self.masks[-1])

    def compute_returns(self, next_value):
        self.returns[-1] = next_value#
        for step in reversed(range(self.rewards.size(0))):
            self.returns[step] = self.returns[step + 1] * \
                self.gamma * self.masks[step + 1] + self.rewards[step]

    def feed_forward_generator(self, advantages, num_mini_batch=None, mini_batch_size=None):
        num_steps, num_processes = self.rewards.size()[0:2]
        batch_size = num_processes * num_steps

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes ({}) * number of steps ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(num_processes, num_steps, num_processes * num_steps, num_mini_batch))

            mini_batch_size = batch_size // num_mini_batch

        sampler = BatchSampler(SubsetRandomSampler(range(batch_size)),
                               mini_batch_size,
                               drop_last=True)

        for indices in sampler:
            obs_batch = self.obs[:-1].view(-1, *self.obs.size()[2:])[indices]
            add_obs_batch = self.add_obs[:-1].view(-1, *self.add_obs.size()[2:])[indices]

            actions_batch = self.actions.view(-1, self.actions.size(-1))[indices]
            value_preds_batch = self.value_preds[:-1].view(-1, 1)[indices]
            return_batch = self.returns[:-1].view(-1, 1)[indices]
            masks_batch = self.masks[:-1].view(-1, 1)[indices]
            old_action_log_probs_batch = self.action_log_probs.view(-1, 1)[indices]

            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages.view(-1, 1)[indices]

            yield obs_batch, actions_batch, value_preds_batch, return_batch, masks_batch, \
                  old_action_log_probs_batch, adv_targ, add_obs_batch
