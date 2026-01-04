from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import numpy as np
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
# from kfac    import KFACOptimizer

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
    def __init__(self, num_inputs, num_outputs, container_size, enable_rotation = True):
        super(Categorical, self).__init__()

        init_ = lambda m: init(
            m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            gain=0.01)

        self.linear_o = init_(nn.Linear(num_inputs, 1 + enable_rotation))
        self.linear_x = init_(nn.Linear(num_inputs, container_size[0]))
        self.linear_y = init_(nn.Linear(num_inputs, container_size[1]))

        self.activation = nn.LeakyReLU()
        self.softmax = nn.Softmax()
        self.log_softmax = nn.LogSoftmax()

        self.init_mask_value = 1
        self.target_mask_value = 0.001
        self.epoch = 0
        self.total_num_epochs = 10e4
        self.mask_value = 1

        # for debug
        self.prob_o = None
        self.prob_x = None
        self.prob_y = None

        self.prob_o_before = None
        self.prob_x_before = None
        self.prob_y_before = None

        self.hidden_o = None
        self.hidden_x = None
        self.hidden_y = None

    # # 1.24 norm one
    def forward(self, feature, mask, container_size, actiono=None, actionx=None):  # 如果eval是true的话, 额，这里似乎变得复杂了
        compensation = 1e-20
        prob, latentV = None, None
        enable_rotation = True
        if actiono is None and actionx is None:  #
            latentV = self.linear_o(feature)

            prob = self.softmax(latentV)
            self.prob_o_before = prob
        elif actiono is not None and actionx is None:  #
            x_norm = (1 + enable_rotation)
            # latentV = self.linear_x(feature)
            latentV = self.linear_x(feature + actiono.float().detach().reshape(-1, 1).to(feature.device) / x_norm)
            prob = self.softmax(latentV)  # 看来这个值不能太小
            self.prob_x_before = prob
        elif actiono is not None and actionx is not None:  # action o 和 action x 都不能为空
            y_norm = (1 + enable_rotation) * container_size[0]
            # latentV = self.linear_y(feature)
            latentV = self.linear_y(feature + actionx.float().detach().reshape(-1, 1).to(feature.device) * actiono.float().detach().reshape(-1, 1).to(feature.device) / y_norm)
            prob = self.softmax(latentV)  # + 0.01 # 要分清楚这个函数里面的logits和prob!
            self.prob_y_before = prob

        ones = torch.ones_like(mask)
        inver_mask = ones - mask

        logx = torch.log(1 + F.softmax(latentV))  # 没有感觉这东西，先softmax，然后再拿出来有啥用
        inver_mask = inver_mask.to(logx.device)
        bad_prob = logx.mul(inver_mask)

        mask_value = 1e-3
        mask = mask.to(logx.device)
        prob = prob.mul(mask + mask_value * inver_mask) + compensation
        return FixedCategorical(probs=prob), bad_prob



    def get_policy_distribution(self,x):
        x =self.linear(x)
        mx = self.softmax(x)
        return mx

class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, channel=6, container_size = None, base=None, base_kwargs=None):
        super(Policy, self).__init__()
        self.recoder = []
        self.recoder_counter = 0
        if base_kwargs is None:
            base_kwargs = {}
        if base is None:
            base = CNNBase

        self.enable_rotation = True
        self.base = base(obs_shape[0], channel, container_size, **base_kwargs)
        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.base.output_size, num_outputs, container_size, self.enable_rotation)
        self.o_entropy_coef = np.log(container_size[0]) / np.log(2)
        self.o_prob_coef = container_size[0] / 2
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

    def get_o_mask(self, location_masks):
        oxy_mask = location_masks.reshape((-1, 1 + self.enable_rotation, self.container_size[0] * self.container_size[1]))
        o_mask = torch.sum(oxy_mask, 2)
        o_mask[o_mask > 0] = 1
        return o_mask

    def get_x_mask(self, location_masks, action_o):
        oxy_mask = location_masks.reshape((-1, 1 + self.enable_rotation, self.container_size[0], self.container_size[1]))
        batch_index = torch.arange(oxy_mask.size()[0])
        xy_masks = oxy_mask[batch_index, action_o.reshape(-1)]
        x_mask = torch.sum(xy_masks, 2)
        x_mask[x_mask > 0] = 1
        return x_mask, xy_masks

    def get_y_mask(self, xy_masks, action_x):
        oxy_mask = xy_masks.reshape((-1, self.container_size[0], self.container_size[1]))
        batch_index = torch.arange(oxy_mask.size()[0])
        y_mask = oxy_mask[batch_index,action_x.reshape(-1)]
        return y_mask

    def act(self, inputs, rnn_hxs, masks, location_masks, deterministic=False):

        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)

        o_mask = self.get_o_mask(location_masks)
        dist_o, bad_prob_o = self.dist(actor_features, o_mask, container_size = self.container_size)

        if deterministic:  # 这里要有一个解耦动作的机制
            action_o = dist_o.mode()
        else:
            action_o = dist_o.sample()

        x_mask, xy_masks = self.get_x_mask(location_masks, action_o.to("cpu"))
        dist_x, bad_prob_x = self.dist(actor_features, x_mask, actiono = action_o.to("cpu"), container_size = self.container_size)

        if deterministic:  # 这里要有一个解耦动作的机制
            action_x = dist_x.mode()
        else:
            action_x = dist_x.sample()

        y_mask = self.get_y_mask(xy_masks, action_x.to("cpu"))
        dist_y, bad_prob_y = self.dist(actor_features, y_mask, actiono = action_o.to("cpu"), actionx = action_x.to("cpu"), container_size = self.container_size)

        if deterministic:  # 这里要有一个解耦动作的机制
            action_y = dist_y.mode()
        else:
            action_y = dist_y.sample()

        action_log_probs_o = dist_o.log_probs(action_o)
        action_log_probs_x = dist_x.log_probs(action_x)
        action_log_probs_y = dist_y.log_probs(action_y)

        action_log_probs = torch.cat((action_log_probs_o, action_log_probs_x, action_log_probs_y), 1)
        action = torch.cat((action_o, action_x, action_y), 1)

        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks):
        value, _, _ = self.base(inputs, rnn_hxs, masks)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action, location_masks):

        action_o, action_x, action_y = action[:, 0], action[:, 1], action[:, 2]
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)

        o_mask = self.get_o_mask(location_masks)
        x_mask, xy_masks = self.get_x_mask(location_masks, action_o)
        y_mask = self.get_y_mask(xy_masks, action_x)

        dist_o, bad_prob_o = self.dist(actor_features, o_mask, container_size = self.container_size)
        dist_x, bad_prob_x = self.dist(actor_features, x_mask, actiono = action_o, container_size = self.container_size)
        dist_y, bad_prob_y = self.dist(actor_features, y_mask, actiono = action_o, actionx = action_x, container_size = self.container_size)

        action_log_probs_o = dist_o.log_probs(action_o)
        action_log_probs_x = dist_x.log_probs(action_x)
        action_log_probs_y = dist_y.log_probs(action_y)
        action_log_probs = torch.cat((action_log_probs_o, action_log_probs_x, action_log_probs_y), 1)

        dist_entropy_o = dist_o.entropy().mean()
        dist_entropy_x = dist_x.entropy().mean()
        dist_entropy_y = dist_y.entropy().mean()
        dist_entropy = dist_entropy_o + dist_entropy_x + dist_entropy_y # 我有必要

        bad_prob_o = bad_prob_o.mean()
        bad_prob_x = bad_prob_x.mean()
        bad_prob_y = bad_prob_y.mean()
        bad_prob = self.o_prob_coef * bad_prob_o + bad_prob_x + bad_prob_y


        return value, action_log_probs, dist_entropy, rnn_hxs, bad_prob, \
               [dist_entropy_o, dist_entropy_x, dist_entropy_y, bad_prob_o, bad_prob_x, bad_prob_y]


class NNBase(nn.Module):  # 如果是递归的话，就在这里创建递归模型
    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            self.gru = nn.GRU(recurrent_input_size, hidden_size)
            for name, param in self.gru.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)

    # 返回模型属性
    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size



class CNNBase(NNBase):
    def __init__(self, num_inputs, channel = 6, container_size = None, recurrent=False, hidden_size=128):
        super(CNNBase, self).__init__(recurrent, num_inputs, hidden_size)
        activate, ini = nn.LeakyReLU, 'leaky_relu'
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), nn.init.calculate_gain(ini))

        self.container_size = container_size
        self.channel = channel
        self.target_size = (256, 256)
        outchannel = 4

        self.image_encoder = nn.Sequential(
            init_(nn.Conv2d(channel, outchannel, 4, stride=2, padding=1)),
            activate(),
            init_(nn.Conv2d(outchannel, outchannel, 4, stride=4)),
            activate(),
            init_(nn.Conv2d(outchannel, 4, 4, stride=4)),
            activate(),
            Flatten())
        self.image_encoder_linear = nn.Sequential(init_(nn.Linear(hidden_size, hidden_size)),
                                                  activate())
        self.resize_transform = transforms.Resize(self.target_size)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))
        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs.reshape((-1, self.channel, self.container_size[0], self.container_size[1]))
        x = self.resize_transform(x)
        x = self.image_encoder(x)
        x = self.image_encoder_linear(x)
        assert not self.is_recurrent
        return self.critic_linear(x), x, rnn_hxs


def _flatten_helper(T, N, _tensor):
    return _tensor.view(T * N, *_tensor.size()[2:])

# obs的形状是 batch * cpu * length
class RolloutStorage(object):
    def __init__(self, num_steps, num_processes, obs_shape, action_space,
                 recurrent_hidden_state_size, can_give_up, enable_rotation, pallet_size):
        self.obs = torch.zeros(num_steps + 1, num_processes, *obs_shape)
        self.recurrent_hidden_states = torch.zeros(
            num_steps + 1, num_processes, recurrent_hidden_state_size)
        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_processes, 1)
        self.returns = torch.zeros(num_steps + 1, num_processes, 1)
        self.action_log_probs = torch.zeros(num_steps, num_processes, 3)
        if action_space.__class__.__name__ == 'Discrete':
            action_shape = 1
        else:
            action_shape = action_space.shape[0]
        self.actions = torch.zeros(num_steps, num_processes, 3)
        if action_space.__class__.__name__ == 'Discrete':
            self.actions = self.actions.long()
        self.masks = torch.ones(num_steps + 1, num_processes, 1)

        # Masks that indicate whether it's a true terminal state
        # or time limit end state
        self.bad_masks = torch.ones(num_steps + 1, num_processes, 1)#为啥这里要加一

        self.num_steps = num_steps
        self.step = 0
        self.location_masks = torch.zeros(num_steps+1, num_processes, int(2 * pallet_size[0] * pallet_size[1]))

    def to(self, device):
        self.obs = self.obs.to(device)
        self.recurrent_hidden_states = self.recurrent_hidden_states.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)
        self.bad_masks = self.bad_masks.to(device)
        self.location_masks = self.location_masks.to(device)



    def insert(self, obs, recurrent_hidden_states, actions, action_log_probs,
               value_preds, rewards, masks, bad_masks,location_masks):
        self.obs[self.step + 1].copy_(obs)#分别给6个进程的观测赋值,这个copy函数是说把外来数据传进来
        self.recurrent_hidden_states[self.step + 1].copy_(recurrent_hidden_states)#递归层的大小
        self.actions[self.step].copy_(actions)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)
        self.bad_masks[self.step + 1].copy_(bad_masks)
        self.location_masks[self.step + 1].copy_(location_masks)
        self.step = (self.step + 1) % self.num_steps


    def after_update(self):
        self.obs[0].copy_(self.obs[-1])
        self.recurrent_hidden_states[0].copy_(self.recurrent_hidden_states[-1])
        self.masks[0].copy_(self.masks[-1])
        self.bad_masks[0].copy_(self.bad_masks[-1])
        self.location_masks[0].copy_(self.location_masks[-1])


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

# class A2C_XY_MASK():
#     def __init__(self,
#                  actor_critic,
#                  value_loss_coef,
#                  entropy_coef,
#                  lr = None,
#                  eps = None,
#                  alpha = None,
#                  max_grad_norm = None,
#                  acktr = False):
#
#         self.actor_critic = actor_critic
#         self.acktr = acktr
#
#         self.value_loss_coef = value_loss_coef
#         self.entropy_coef = entropy_coef
#
#         self.max_grad_norm = max_grad_norm
#
#         self.loss_func = nn.MSELoss(reduce=False, size_average=True)
#         # self.loss_func = nn.NLLLoss(reduce=False, size_average=True) #1205
#         self.acktr_optimizer = KFACOptimizer(actor_critic)
#
#     def update(self, rollouts):
#         # check_nan(self.actor_critic, 1)
#         obs_shape = rollouts.obs.size()[2:]
#         action_shape = rollouts.actions.size()[-1]
#         num_steps, num_processes, _ = rollouts.rewards.size()
#         mask_size = rollouts.location_masks.size()[-1]
#
#         values, action_log_probs, dist_entropy, _, bad_prob, metrics = self.actor_critic.evaluate_actions(
#             rollouts.obs[:-1].view(-1, *obs_shape), # 这里调用的是前五个observation， 不包括最后一个
#             rollouts.recurrent_hidden_states[0].view(-1, self.actor_critic.recurrent_hidden_state_size),
#             rollouts.masks[:-1].view(-1, 1),
#             rollouts.actions.view(-1, action_shape),
#             rollouts.location_masks[:-1].view(-1, mask_size))
#         values = values.view(num_steps, num_processes, 1)
#         action_log_probs = action_log_probs.view(num_steps, num_processes, 3)
#
#         advantages = rollouts.returns[:-1] - values
#         value_loss = advantages.pow(2).mean()
#         action_loss = -(advantages.detach() * action_log_probs).mean()
#
#         prob_loss = bad_prob
#
#         if self.acktr and self.acktr_optimizer.steps % self.acktr_optimizer.Ts == 0:
#             # Sampled fisher, see Martens 2014d
#             self.actor_critic.zero_grad()
#             pg_fisher_loss = -action_log_probs.mean()
#
#             value_noise = torch.randn(values.size())
#             if values.is_cuda:
#                 value_noise = value_noise.cuda()
#
#             sample_values = values + value_noise
#             vf_fisher_loss = -(values - sample_values.detach()).pow(2).mean()
#
#             fisher_loss = pg_fisher_loss + vf_fisher_loss
#             self.acktr_optimizer.acc_stats = True
#             fisher_loss.backward(retain_graph=True)
#             self.acktr_optimizer.acc_stats = False
#
#         self.acktr_optimizer.zero_grad()
#         prob_loss_coef = 1
#         action_loss_coef = 1  #
#
#         # 10 is reasonable (cause totalreward is in [0,10])
#         (
#             prob_loss * prob_loss_coef+
#             value_loss * self.value_loss_coef
#             + action_loss * action_loss_coef
#             - dist_entropy * self.entropy_coef
#         ).backward()
#         if self.acktr == False:
#             nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
#                                      self.max_grad_norm)
#         try:
#             self.acktr_optimizer.step()
#         except RuntimeError as e:
#             print('Optimizer wrong!')
#
#         for name, parameters in self.actor_critic.named_parameters():
#             if np.isnan(torch.sum(parameters.grad).item()):
#                 print('Wrong and NAN appears!')
#
#         return value_loss.item(), action_loss.item(), dist_entropy.item(), metrics