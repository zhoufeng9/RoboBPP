from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

# Categorical
FixedCategorical = torch.distributions.Categorical

old_sample = FixedCategorical.sample
FixedCategorical.sample = lambda self: old_sample(self).unsqueeze(-1)

log_prob_cat = FixedCategorical.log_prob
FixedCategorical.log_probs = lambda self, actions: log_prob_cat(
    self, actions.squeeze(-1)).view(actions.size(0), -1).sum(-1).unsqueeze(-1)
FixedCategorical.mode = lambda self: self.probs.argmax(dim=-1, keepdim=True)

class Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs, container_size):
        super(Categorical, self).__init__()

        init_ = lambda m: init(
            m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            gain=0.01)

        activate = nn.LeakyReLU
        self.linear = init_(nn.Linear(num_inputs * 2, container_size[0] * container_size[1] * 2))

        self.activation = nn.LeakyReLU()
        self.softmax = nn.Softmax()
        self.log_softmax = nn.LogSoftmax()

        self.init_mask_value = 1
        self.target_mask_value = 0.001
        self.epoch = 0
        self.total_num_epochs = 10e4
        self.mask_value = 1

        # for debug
        self.prob = None
        self.prob_before = None
        self.hidden = None

    # # 1.24 norm one       here mask is right???
    def forward(self, feature, mask):  # 如果eval是true的话, 额，这里似乎变得复杂了
        latent = self.linear(feature)
        inver_mask = 1 - mask
        latent[torch.where(inver_mask == 1)] = -torch.inf
        prob = self.softmax(latent)
        return FixedCategorical(probs=prob)

    def get_policy_distribution(self,x):
        x =self.linear(x)
        mx = self.softmax(x)
        return mx


class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, container_size, base=None, base_kwargs=None):
        super(Policy, self).__init__()
        self.recoder = []
        self.recoder_counter = 0
        if base_kwargs is None:
            base_kwargs = {}
        if base is None:
            base = CNNBase # 10 * 10

        self.base = base(obs_shape[0], container_size = container_size, **base_kwargs)
        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.base.output_size, num_outputs, container_size)
        self.container_size = container_size

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, deterministic=False):
        value, actor_features, location_masks = self.base(inputs)
        dist= self.dist(actor_features, location_masks)

        if deterministic:  # 这里要有一个解耦动作的机制
            action = dist.mode()
        else:
            action = dist.sample()
        action = action.reshape(-1, 1)
        action_log_probs = dist.log_probs(action)
        return value, action, action_log_probs

    def get_value(self, inputs):
        value, _, _ = self.base(inputs)
        return value

    def evaluate_actions(self, inputs, action):
        value, actor_features, location_masks = self.base(inputs)
        dist = self.dist(actor_features, location_masks)
        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()
        return value, action_log_probs, dist_entropy


class NNBase(nn.Module):  # 如果是递归的话，就在这里创建递归模型
    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size

    @property
    def output_size(self):
        return self._hidden_size


def observation_decode(observation, contianer_size):
    frontiers = observation[:, 0: contianer_size[0] * contianer_size[1]]
    maskposition = observation[:, contianer_size[0] * contianer_size[1]: -6]
    next_box = observation[:, -6:]
    return frontiers, maskposition, next_box

class CNNBase(NNBase):
    def __init__(self, num_inputs,  container_size,  recurrent=False,  hidden_size=128):
        recurrent = False,
        super(CNNBase, self).__init__(recurrent, num_inputs, hidden_size)

        activate, ini = nn.LeakyReLU, 'leaky_relu'
        self.container_size = container_size
        self.max_axis = max(self.container_size[0], self.container_size[1])
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), nn.init.calculate_gain(ini))
        self.target_size = (256, 256)
        self.resize_transform = transforms.Resize(self.target_size)
        outchannel = 4

        self.image_encoder = nn.Sequential(
            init_(nn.Conv2d(1, outchannel, 4, stride=2, padding=1)),
            activate(),
            init_(nn.Conv2d(outchannel, outchannel, 4, stride=4)),
            activate(),
            init_(nn.Conv2d(outchannel, 4, 4, stride=4)),
            activate(),
            Flatten())
        self.image_encoder_linear = nn.Sequential(init_(nn.Linear(hidden_size, hidden_size)),
                                                  activate())


        self.next_box_encoder = nn.Sequential(
            init_(nn.Linear(6, hidden_size)),
            activate(),
            init_(nn.Linear(hidden_size, hidden_size)),
            activate())

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))
        self.critic_linear = init_(nn.Linear(hidden_size * 2, 1))
        self.train()

    def forward(self, inputs):
        frontiers, maskposition, next_box = observation_decode(inputs, self.container_size)
        frontiers = frontiers.reshape((-1, 1, self.container_size[0], self.container_size[1]))
        frontiers = self.resize_transform(frontiers)
        image_vector = self.image_encoder(frontiers)
        image_vector = self.image_encoder_linear(image_vector)
        next_box_vector = self.next_box_encoder(next_box)
        hidden_vector = torch.cat((image_vector, next_box_vector), 1)
        return self.critic_linear(hidden_vector), hidden_vector, maskposition


def _flatten_helper(T, N, _tensor):
    return _tensor.view(T * N, *_tensor.size()[2:])

# obs的形状是 batch * cpu * length
class RolloutStorage(object):
    def __init__(self, num_steps, num_processes, obs_shape, action_space):
        self.obs = torch.zeros(num_steps + 1, num_processes, *obs_shape)
        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_processes, 1)
        self.returns = torch.zeros(num_steps + 1, num_processes, 1)
        self.action_log_probs = torch.zeros(num_steps, num_processes, 1)
        if action_space.__class__.__name__ == 'Discrete':
            action_shape = 1
        else:
            action_shape = action_space.shape[0]
        self.actions = torch.zeros(num_steps, num_processes, 1)
        if action_space.__class__.__name__ == 'Discrete':
            self.actions = self.actions.long()
        self.masks = torch.ones(num_steps + 1, num_processes, 1)

        # Masks that indicate whether it's a true terminal state
        # or time limit end state
        self.bad_masks = torch.ones(num_steps + 1, num_processes, 1)#为啥这里要加一

        self.num_steps = num_steps
        self.step = 0

    def to(self, device):
        self.obs = self.obs.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)
        self.bad_masks = self.bad_masks.to(device)

    def insert(self, obs,  actions, action_log_probs,
               value_preds, rewards, masks, bad_masks):
        self.obs[self.step + 1].copy_(obs)#分别给6个进程的观测赋值,这个copy函数是说把外来数据传进来
        self.actions[self.step].copy_(actions)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)
        self.bad_masks[self.step + 1].copy_(bad_masks)
        self.step = (self.step + 1) % self.num_steps


    def after_update(self):
        self.obs[0].copy_(self.obs[-1])
        self.masks[0].copy_(self.masks[-1])
        self.bad_masks[0].copy_(self.bad_masks[-1])


    def compute_returns(self,
                        next_value,
                        use_gae,
                        gamma,
                        gae_lambda,
                        use_proper_time_limits=True):
        if use_proper_time_limits:
            if use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.size(0))):
                    delta = self.rewards[step] + gamma * self.value_preds[
                        step + 1] * self.masks[step +
                                               1] - self.value_preds[step]
                    gae = delta + gamma * gae_lambda * self.masks[step +
                                                                  1] * gae
                    gae = gae * self.bad_masks[step + 1]
                    self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.size(0))):
                    self.returns[step] = (self.returns[step + 1] * \
                        gamma * self.masks[step + 1] + self.rewards[step]) * self.bad_masks[step + 1] \
                        + (1 - self.bad_masks[step + 1]) * self.value_preds[step]
        else:
            if use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.size(0))):
                    delta = self.rewards[step] + gamma * self.value_preds[
                        step + 1] * self.masks[step +
                                               1] - self.value_preds[step]
                    gae = delta + gamma * gae_lambda * self.masks[step +
                                                                  1] * gae
                    self.returns[step] = gae + self.value_preds[step]
            else:
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
            # recurrent_hidden_states_batch = self.recurrent_hidden_states[:-1].view(
            #     -1, self.recurrent_hidden_states.size(-1))[indices]
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

            yield obs_batch, actions_batch, \
                value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ

class A2C_XY_MASK():
    def __init__(self,
                 actor_critic,
                 value_loss_coef,
                 entropy_coef,
                 lr = None,
                 eps = None,
                 alpha = None,
                 max_grad_norm = None,
                 acktr = False):

        self.actor_critic = actor_critic
        self.acktr = acktr

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm

        self.loss_func = nn.MSELoss(reduce=False, size_average=True)
