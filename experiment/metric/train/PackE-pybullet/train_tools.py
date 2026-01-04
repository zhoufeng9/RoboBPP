import copy
import os

import torch
import torch.optim as optim
import numpy as np
import tools
import time
from collections import deque
from storage import PCTRolloutStorage
import random
import gym
np.set_printoptions(threshold=np.inf)
from tqdm import trange
from memory import ReplayMemory
from CDRL_model import *
from models.model_Attend2Pack import RolloutStorage as Attend2PackRolloutStorage
from models.model_RCQL        import RolloutStorage as RCQLRolloutStorage

class train_tools(object):
    def __init__(self, writer, timeStr, packing_policy, args):
        self.writer = writer
        self.timeStr = timeStr
        self.step_counter = 0
        self.packing_policy = packing_policy
        self.use_acktr = args.drl_method == 'acktr'
        seed = args.seed

        if args.drl_method != 'rainbow' and args.model_architecture != 'CDRL':
            if self.use_acktr:
                self.policy_optim = KFACOptimizer(self.packing_policy) # For ACKTR method. （https://proceedings.neurips.cc/paper/2017/hash/361440528766bbaaaa1901845cf4152b-Abstract.html）
            else:
                self.policy_optim = optim.Adam(self.packing_policy.parameters(), lr=args.learning_rate) # For naive A2C method.

        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            random.seed(seed)

    def policy_update(self, pct_rollout, args):
        obs_shape = pct_rollout.obs.size()[2:]
        action_shape = pct_rollout.actions.size()[-1]
        leaf_node_value, selectedlogProb, dist_entropy = self.packing_policy.evaluate_actions(
            pct_rollout.obs[:-1].view(-1, *obs_shape),
            pct_rollout.actions.view(-1, action_shape),
            normFactor=args.normFactor)
        leaf_node_value = leaf_node_value.view(args.num_steps, args.num_processes, 1)
        selectedlogProb = selectedlogProb.view(args.num_steps, args.num_processes, 1)

        advantages = pct_rollout.returns[:-1] - leaf_node_value
        critic_loss = advantages.pow(2).mean()
        actor_loss = -(advantages.detach() * selectedlogProb).mean()

        if self.use_acktr and self.policy_optim.steps % self.policy_optim.Ts == 0:
            # Sampled fisher, see Martens 2014d
            self.packing_policy.zero_grad()
            pg_fisher_loss = - selectedlogProb.mean()

            value_noise = torch.randn(leaf_node_value.size())
            if leaf_node_value.is_cuda:
                value_noise = value_noise.to(args.device)

            sample_values = leaf_node_value + value_noise
            vf_fisher_loss = -(leaf_node_value - sample_values.detach()).pow(2).mean()

            fisher_loss = pg_fisher_loss + vf_fisher_loss
            self.policy_optim.acc_stats = True
            fisher_loss.backward(retain_graph=True)
            self.policy_optim.acc_stats = False

        self.policy_optim.zero_grad()
        (args.actor_loss_coef * actor_loss
         + args.critic_loss_coef * critic_loss).backward()
        torch.nn.utils.clip_grad_norm_(self.packing_policy.parameters(), args.max_grad_norm)
        self.policy_optim.step()

        return actor_loss, critic_loss, dist_entropy

    def policy_update_attend_reinforce(self, pct_rollout, args):
        obs_shape = pct_rollout.obs.size()[2:]
        action_shape = pct_rollout.actions.size()[-1]

        values, selectedlogProb, dist_entropy = self.packing_policy.evaluate_actions(
            pct_rollout.obs[:-1].view(-1, *obs_shape),
            pct_rollout.actions.view(-1, action_shape),
            args.container_size)

        values = values.view(args.num_steps, args.num_processes, 1)
        selectedlogProb = selectedlogProb.view(args.num_steps, args.num_processes, 1)

        advantages = pct_rollout.returns[:-1] - values
        critic_loss = advantages.pow(2).mean()
        returns = pct_rollout.returns[:-1]  # 返回值计算方式取决于具体实现
        actor_loss = -(returns * selectedlogProb).mean()  # 不需要 leaf_node_value
        # actor_loss = -(advantages * selectedlogProb).mean()  # 不需要 leaf_node_value
        
        self.policy_optim.zero_grad()
        (args.actor_loss_coef * actor_loss
         + args.critic_loss_coef * critic_loss).backward()
        torch.nn.utils.clip_grad_norm_(self.packing_policy.parameters(), args.max_grad_norm)
        self.policy_optim.step()

        return actor_loss, critic_loss, dist_entropy


    def policy_update_RCQL_actor_critic(self, pct_rollout, args):
        obs_shape = pct_rollout.obs.size()[2:]
        action_shape = pct_rollout.actions.size()[-1]
        batch_size = pct_rollout.obs[:-1].view(-1, *obs_shape).size(0)
        attn_span = 20
        hidden_size = 128
        encoder_nb_layers = 3
        h_cache = [torch.zeros(batch_size, attn_span, hidden_size).to(args.device) for i in range(encoder_nb_layers)]
        values, selectedlogProb, dist_entropy = self.packing_policy.evaluate_actions(
            pct_rollout.obs[:-1].view(-1, *obs_shape),
            h_cache,
            pct_rollout.actions.view(-1, action_shape),
            args)
        values = values.view(args.num_steps, args.num_processes, 1)
        selectedlogProb = selectedlogProb.view(args.num_steps, args.num_processes, 3)
        advantages = pct_rollout.returns[:-1] - values
        critic_loss = advantages.pow(2).mean()
        actor_loss = -(advantages.detach()  * selectedlogProb).mean()  # 不需要 leaf_node_value
        self.policy_optim.zero_grad()
        (args.actor_loss_coef * actor_loss
         + args.critic_loss_coef * critic_loss).backward()
        torch.nn.utils.clip_grad_norm_(self.packing_policy.parameters(), args.max_grad_norm)
        self.policy_optim.step()

        return actor_loss, critic_loss, dist_entropy

    def policy_update_with_q(self, pct_rollout, args):
        obs_shape = pct_rollout.obs.size()[2:]
        action_shape = pct_rollout.actions.size()[-1]
        leaf_node_value, selectedlogProb, dist_entropy = self.packing_policy.evaluate_actions(
            pct_rollout.obs[:-1].view(-1, *obs_shape),
            pct_rollout.actions.view(-1, action_shape),
            normFactor=args.normFactor,
            return_q = args.actor_with_q)
        leaf_node_q_value, _, leaf_node_v_value = leaf_node_value

        leaf_node_q_value = leaf_node_q_value.view(args.num_steps, args.num_processes, 1)
        leaf_node_v_value = leaf_node_v_value.view(args.num_steps, args.num_processes, 1)
        selectedlogProb = selectedlogProb.view(args.num_steps, args.num_processes, 1)

        advantages = pct_rollout.returns[:-1] - leaf_node_v_value
        critic_loss = (pct_rollout.returns[:-1] - leaf_node_q_value).pow(2).mean()
        actor_loss = -(advantages.detach() * selectedlogProb).mean()

        if self.use_acktr and self.policy_optim.steps % self.policy_optim.Ts == 0:
            # Sampled fisher, see Martens 2014d
            self.packing_policy.zero_grad()
            pg_fisher_loss = - selectedlogProb.mean()

            value_noise = torch.randn(leaf_node_v_value.size())
            if leaf_node_v_value.is_cuda:
                value_noise = value_noise.to(args.device)

            sample_values = leaf_node_v_value + value_noise
            vf_fisher_loss = -(leaf_node_v_value - sample_values.detach()).pow(2).mean()

            fisher_loss = pg_fisher_loss + vf_fisher_loss
            self.policy_optim.acc_stats = True
            fisher_loss.backward(retain_graph=True)
            self.policy_optim.acc_stats = False

        self.policy_optim.zero_grad()
        (args.actor_loss_coef * actor_loss
         + args.critic_loss_coef * critic_loss).backward()
        torch.nn.utils.clip_grad_norm_(self.packing_policy.parameters(), args.max_grad_norm)

        self.policy_optim.step()

        return actor_loss, critic_loss, dist_entropy

    def policy_update_with_ppo(self,
               rollouts,
               args):

        obs_shape    = rollouts.obs.size()[2:]
        action_shape = rollouts.actions.size()[-1]

        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        # todo to be update later
        self.ppo_epoch = args.ppo_epoch
        self.num_mini_batch = args.num_mini_batch # 跑几个小 batch 的意思, 其实这个值对我越小越好
        self.clip_param = 0.2                     # default 0.2
        self.use_clipped_value_loss = True
        self.max_grad_norm = 0.5
        self.value_loss_coef = 1.
        # self.entropy_coef = 0.01

        for e in range(self.ppo_epoch):
            data_generator = rollouts.feed_forward_generator(
                    advantages, self.num_mini_batch)
            for sample in data_generator:
                obs_batch, actions_batch, value_preds_batch, return_batch, \
                masks_batch, old_action_log_probs_batch, adv_targ = sample

                # with torch.no_grad():
                    # values_exp, action_exp, action_log_probs_exp = self.packing_policy.act(
                    #     obs_batch.view(-1, *obs_shape))
                    # values, action_log_probs, dist_entropy = self.packing_policy.evaluate_actions(
                    #     obs_batch.view(-1, *obs_shape),
                    #     action_exp.view(-1, action_shape))
                    # exit()

                values, action_log_probs, dist_entropy = self.packing_policy.evaluate_actions(
                    obs_batch.view(-1, *obs_shape),
                    actions_batch.view(-1, action_shape))

                ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()

                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + \
                        (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (value_pred_clipped - return_batch).pow(2)
                    value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()
                self.policy_optim.zero_grad()
                (value_loss * self.value_loss_coef
                 + action_loss
                 # - dist_entropy * self.entropy_coef
                 ).backward()
                nn.utils.clip_grad_norm_(self.packing_policy.parameters(), self.max_grad_norm)
                self.policy_optim.step()

                value_loss_epoch += value_loss
                action_loss_epoch += action_loss
                dist_entropy_epoch += dist_entropy

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        return action_loss_epoch, value_loss_epoch,  dist_entropy_epoch

    def eval_record(self, label, eval_episode_ratio, episode_length):
        self.writer.add_scalar('{}/Mean ratio'.format(label), np.mean(eval_episode_ratio), self.step_counter)
        self.writer.add_scalar("{}/Max  ratio".format(label), np.max(eval_episode_ratio), self.step_counter)
        self.writer.add_scalar("{}/Min  ratio".format(label), np.min(eval_episode_ratio), self.step_counter)

        self.writer.add_scalar('{}/Mean length'.format(label), np.mean(episode_length), self.step_counter)
        self.writer.add_scalar("{}/Max  length".format(label), np.max(episode_length), self.step_counter)
        self.writer.add_scalar("{}/Min  length".format(label), np.min(episode_length), self.step_counter)

    def evaluation(self, args, label, sub_time_str, large_scale,
                   distribution,  sample_from_distribution, sample_left_bound, sample_right_bound,
                   internal_node_holder, leaf_node_holder, no_internal_node_input = False, normal_mean = None, normal_std = None, setting = 2):
        eval_args = copy.deepcopy(args)
        eval_args.num_processes = 1

        if distribution == 'normal': assert normal_mean is not None and normal_std is not None

        eval_args.distribution = distribution
        eval_args.large_scale = large_scale
        eval_args.sample_from_distribution = sample_from_distribution
        eval_args.sample_left_bound = sample_left_bound
        eval_args.sample_right_bound = sample_right_bound
        eval_args.internal_node_holder = internal_node_holder
        eval_args.leaf_node_holder = leaf_node_holder
        eval_args.no_internal_node_input = no_internal_node_input
        eval_args.setting = setting

        from evaluation_tools import evaluate_PCT
        eval_envs = gym.make(eval_args.id, args=eval_args)
        eval_func = evaluate_PCT
        eval_episode_ratio, eval_episode_length = eval_func(self.packing_policy, eval_envs, sub_time_str + '_' + label,
                                                                       eval_args, eval_args.device, eval_freq=eval_args.evaluation_episodes, factor=eval_args.normFactor)
        self.eval_record(label, eval_episode_ratio, eval_episode_length)


    def set_meta_next_box(self, envs, world_obs, action, term, args):
        raise NotImplementedError

    def train_PCT(self, envs, args, train_episodes = None, given_distribution_label = None):
        device = args.device
        model_save_path = os.path.join(args.model_save_path, self.timeStr)
        sub_time_str    = time.strftime('%Y.%m.%d-%H-%M-%S', time.localtime(time.time()))
        self.packing_policy.train()
        factor = args.normFactor # NormFactor controlls the maximum value of the original input of the network to less than 1.0, which helps the training of the network

        obs = envs.reset()
        all_nodes, leaf_nodes = tools.get_leaf_nodes(obs, args.internal_node_holder, args.leaf_node_holder)
        all_nodes, leaf_nodes = all_nodes.to(device), leaf_nodes.to(device)

        num_steps, num_processes = args.num_steps, args.num_processes
        if args.drl_method == 'ppo': num_steps = 30

        pct_rollout = PCTRolloutStorage(num_steps,
                                        num_processes,
                                        obs_shape=all_nodes.shape[1:],
                                        gamma = args.gamma)
        pct_rollout.to(device)

        start = time.time()
        ratio_recorder = 0
        episode_rewards = deque(maxlen=10)
        episode_constraint_rewards = deque(maxlen=10)
        episode_ratio = deque(maxlen=10)
        episode_final_length = deque(maxlen=10)
        batchX = torch.arange(args.num_processes)
        inside_counter = self.step_counter

        pct_rollout.obs[0].copy_(all_nodes)
        if train_episodes is None: train_episodes = 1e20

        for _ in range(int(train_episodes)):
            ##############################################
            ####### Collect n-step training sample #######
            ##############################################
            self.step_counter += 1
            distribution_label_storage = []
            for step in range(num_steps):
                with torch.no_grad():
                    selectedlogProb, selectedIdx, dist_entropy, value = self.packing_policy(all_nodes, normFactor=factor)
                selected_leaf_node = leaf_nodes[batchX, selectedIdx.squeeze()]
                obs, reward, done, infos = envs.step(selected_leaf_node.cpu().numpy())

                all_nodes, leaf_nodes = tools.get_leaf_nodes(obs, args.internal_node_holder, args.leaf_node_holder)
                all_nodes, leaf_nodes = all_nodes.to(device), leaf_nodes.to(device)
                pct_rollout.insert(all_nodes, selectedIdx.squeeze(1), selectedlogProb, value, reward, torch.tensor(1-done).unsqueeze(1))

                env_distribution_label = []
                for _ in range(len(infos)):
                    if done[_]:
                        if 'reward' in infos[_].keys():
                            episode_rewards.append(infos[_]['reward'])
                        else:
                            episode_rewards.append(infos[_]['episode']['r'])
                        if 'ratio' in infos[_].keys():
                            episode_ratio.append(infos[_]['ratio'])
                        if 'final_length' in infos[_].keys():
                            episode_final_length.append(infos[_]['final_length'])
                        if 'constraint_reward' in infos[_].keys():
                            episode_constraint_rewards.append(infos[_]['constraint_reward'])
                    if 'distribution_label' in infos[_].keys():
                        env_distribution_label.append(infos[_]['distribution_label'])

                if given_distribution_label is not None:
                    distribution_label_storage.append(given_distribution_label)
                else:
                    distribution_label_storage.append(env_distribution_label)

            with torch.no_grad():
                _, _, _, next_value = self.packing_policy(pct_rollout.obs[-1], normFactor = factor)
            next_value = next_value[1] if isinstance(next_value, tuple) else next_value
            pct_rollout.compute_returns(next_value)

            ##############################################
            ########### PCT policy evaluation ############
            ##############################################
            # Save the trained policy model
            if (self.step_counter % args.model_save_interval == 0) or (self.step_counter == 1):
                if self.step_counter % args.model_update_interval == 0 or self.step_counter == 1:
                    sub_time_str = args.custom + '_' + time.strftime('%Y.%m.%d-%H-%M-%S', time.localtime(time.time()))
                    # also evaluate the model
                    self.packing_policy.eval()

                    if not args.training_without_evaluate:
                        self.evaluation(args, 'Evaluate', sub_time_str, large_scale = False,
                                       distribution = args.distribution,
                                       sample_from_distribution = args.sample_from_distribution,
                                       sample_left_bound = args.sample_left_bound,
                                       sample_right_bound = args.sample_right_bound,
                                       internal_node_holder = args.internal_node_holder,
                                       leaf_node_holder = args.leaf_node_holder,
                                       normal_mean = args.normal_mean,
                                       normal_std = args.normal_std,
                                       setting=args.setting)

                    # if args.evaluate_with_large_data:
                    if not args.training_without_evaluate:
                        self.evaluation(args, 'Evaluate_large_scale_simple', sub_time_str, large_scale = False,
                                       distribution = 'uniform',
                                       sample_from_distribution = True,
                                       sample_left_bound = 0.05,
                                       sample_right_bound = 0.25,
                                       internal_node_holder = 500,
                                       leaf_node_holder = 500)

                    if not args.training_without_evaluate:
                        self.evaluation(args, 'Evaluate_with_no_interval_node', sub_time_str, large_scale=False,
                                        distribution='uniform',
                                        sample_from_distribution=args.sample_from_distribution,
                                        sample_left_bound=args.sample_left_bound,
                                        sample_right_bound=args.sample_right_bound,
                                        internal_node_holder=args.internal_node_holder,
                                        leaf_node_holder=args.leaf_node_holder,
                                        no_internal_node_input=True,
                                        normal_mean=args.normal_mean,
                                        normal_std=args.normal_std,
                                        setting=args.setting)

                        self.evaluation(args, 'Evaluate_with_no_interval_node_large_scale', sub_time_str, large_scale = False,
                                       distribution = 'uniform',
                                       sample_from_distribution = True,
                                       sample_left_bound = 0.05,
                                       sample_right_bound = 0.25,
                                       internal_node_holder = 500,
                                       leaf_node_holder = 500,
                                       no_internal_node_input = True)

                    # if args.evaluate_with_given_data:
                    if not args.training_without_evaluate:
                        self.evaluation(args, 'Evaluate_U_01_05', sub_time_str, large_scale = False,
                                       distribution = 'uniform',
                                       sample_from_distribution = True,
                                       sample_left_bound = 0.1,
                                       sample_right_bound = 0.5,
                                       internal_node_holder = 80,
                                       leaf_node_holder = 150)

                        self.evaluation(args, 'Evaluate_U_00_10', sub_time_str, large_scale = False,
                                       distribution = 'uniform',
                                       sample_from_distribution = True,
                                       sample_left_bound = 0.001,
                                       sample_right_bound = 1,
                                       internal_node_holder = 80,
                                       leaf_node_holder =     150)

                        self.evaluation(args, 'Evaluate_N_02_02', sub_time_str, large_scale = False,
                                       distribution = 'normal',
                                       sample_from_distribution = True,
                                       sample_left_bound = 0.001,
                                       sample_right_bound = 1,
                                       internal_node_holder = 150,
                                       leaf_node_holder =     150,
                                       normal_mean=0.2,
                                       normal_std =0.2)

                        self.evaluation(args, 'Evaluate_N_05_02', sub_time_str, large_scale = False,
                                       distribution = 'normal',
                                       sample_from_distribution = True,
                                       sample_left_bound = 0.001,
                                       sample_right_bound = 1,
                                       internal_node_holder = 80,
                                       leaf_node_holder =     150,
                                       normal_mean=0.5,
                                       normal_std =0.2)

                        self.evaluation(args, 'Evaluate_N_08_02', sub_time_str, large_scale = False,
                                       distribution = 'normal',
                                       sample_from_distribution = True,
                                       sample_left_bound = 0.001,
                                       sample_right_bound = 1,
                                       internal_node_holder = 80,
                                       leaf_node_holder =     150,
                                       normal_mean=0.8,
                                       normal_std =0.2)


                    self.packing_policy.train()

                if not os.path.exists(model_save_path):
                    os.makedirs(model_save_path)
                if not os.path.exists(args.regular_model_save_path):
                    os.makedirs(args.regular_model_save_path)

                torch.save(self.packing_policy.state_dict(), os.path.join(model_save_path, sub_time_str + ".pt"))
                torch.save(self.packing_policy.state_dict(), os.path.join(args.regular_model_save_path, args.custom + ".pt")) # save the last model to a regular position

            ##############################################
            ########### PCT policy optimzation ###########
            ##############################################
            distribution_loss, class_accuracy = 0, 0
            self.packing_policy.actor.set_step_counter(self.step_counter)
            if args.actor_with_q:
                actor_loss, critic_loss, dist_entropy = self.policy_update_with_q(pct_rollout, args)
            else:
                s = time.time()
                if args.drl_method == 'ppo':
                    actor_loss, critic_loss, dist_entropy = self.policy_update_with_ppo(pct_rollout, args)
                else:
                    actor_loss, critic_loss, dist_entropy = self.policy_update(pct_rollout, args)
                e = time.time()
                # print('policy update time', e - s)
            ##############################################
            ############ After optimzation ###############
            ##############################################
            pct_rollout.after_update()

            # Write tensorboard logs.
            if self.step_counter % args.print_log_interval == 0 and len(episode_rewards)>1:
                total_num_steps = (self.step_counter + 1 - inside_counter) * num_processes * num_steps
                end = time.time()
                if len(episode_ratio) != 0:
                    ratio_recorder = max(ratio_recorder, np.max(episode_ratio))

                print("Time version: {} is training\n"
                      "Updates {}, num timesteps {}, FPS {}\n"
                      "Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                      "The dist entropy {:.5f}, the value loss {:.5f}, the action loss {:.5f}\n"
                      "The mean space ratio is {:.5f}, the ratio threshold is {:.5f}\n"
                        .format(self.timeStr,
                                self.step_counter, total_num_steps, int(total_num_steps / (end - start)),
                                len(episode_rewards), np.mean(episode_rewards), np.median(episode_rewards), np.min(episode_rewards), np.max(episode_rewards),
                                dist_entropy.item(), critic_loss.item(), actor_loss.item(),
                                np.mean(episode_ratio), ratio_recorder))
                if len(episode_rewards) != 0:
                    self.writer.add_scalar('Rewards/Mean rewards', np.mean(episode_rewards), self.step_counter)
                    self.writer.add_scalar("Rewards/Max rewards", np.max(episode_rewards), self.step_counter)
                    self.writer.add_scalar('Rewards/Min rewards', np.min(episode_rewards), self.step_counter)
                if len(episode_ratio) != 0:
                    self.writer.add_scalar("Ratio/The max ratio", np.max(episode_ratio), self.step_counter)
                    self.writer.add_scalar("Ratio/The mean ratio", np.mean(episode_ratio), self.step_counter)
                    self.writer.add_scalar("Ratio/The min ratio", np.min(episode_ratio), self.step_counter)
                if len(episode_final_length) != 0:
                    self.writer.add_scalar("Length/The max length", np.max(episode_final_length), self.step_counter)
                    self.writer.add_scalar("Length/The mean length", np.mean(episode_final_length), self.step_counter)
                    self.writer.add_scalar("Length/The min length", np.min(episode_final_length), self.step_counter)
                if len(episode_constraint_rewards) != 0:
                    self.writer.add_scalar("Constraint reward/Mean constraint reward", np.mean(episode_constraint_rewards), self.step_counter)
                    self.writer.add_scalar("Constraint reward/Max constraint reward", np.max(episode_constraint_rewards), self.step_counter)
                    self.writer.add_scalar("Constraint reward/Min constraint reward", np.min(episode_constraint_rewards), self.step_counter)
                self.writer.add_scalar("Ratio/The max ratio in history", ratio_recorder, self.step_counter)
                self.writer.add_scalar("Training/Value loss", critic_loss.item(), self.step_counter)
                self.writer.add_scalar("Training/Action loss", actor_loss.item(), self.step_counter)
                self.writer.add_scalar('Training/Distribution entropy', dist_entropy.item(), self.step_counter)
                self.writer.add_scalar('Similarity/Internal node', self.packing_policy.actor.similarity[0], self.step_counter)
                self.writer.add_scalar('Similarity/Leaf node', self.packing_policy.actor.similarity[1], self.step_counter)
                self.writer.add_scalar('Similarity/Next node', self.packing_policy.actor.similarity[2], self.step_counter)

    def train_CDRL(self, envs, args):
        import CDRL_model
        model_save_path = os.path.join(args.model_save_path, self.timeStr)
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
        sub_time_str = args.custom + '_' + time.strftime('%Y.%m.%d-%H-%M-%S', time.localtime(time.time()))

        value_loss_coef = 0.5
        entropy_coef = 1e-2
        pallet_size = args.container_size
        actor_critic = self.packing_policy
        enable_rotation = True
        use_mask = True
        device = args.device

        if args.setting == 2:
            action_space = gym.spaces.Discrete(600)
        else:
            action_space = gym.spaces.Discrete(200)

        agent = CDRL_model.A2C_XY_MASK(
                actor_critic, value_loss_coef, entropy_coef, acktr=True)
        obs = envs.reset()

        rollouts = CDRL_model.RolloutStorage(args.num_steps,  # number of forward steps in A2C (default: 5)
                                  args.num_processes,  # how many training CPU processes to use (default: 16)
                                  (obs.shape[-1],),
                                  action_space,
                                  actor_critic.recurrent_hidden_state_size,
                                  can_give_up=False,
                                  enable_rotation=enable_rotation,
                                  pallet_size=pallet_size)

        plainum = args.container_size[0] * args.container_size[1]
        location_masks = (obs[:, plainum:3 * plainum] > 0.0).float()

        rollouts.obs[0].copy_(obs)
        rollouts.location_masks[0].copy_(location_masks)
        rollouts.to(device)

        episode_rewards = deque(maxlen=10)
        episode_ratio = deque(maxlen=10)
        episode_constraint_rewards = deque(maxlen=10)

        start = time.time()

        j = 0
        ratio_recorder = 0


        while True:
            j += 1
            self.step_counter += 1
            for step in range(args.num_steps):
                with torch.no_grad():
                    value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                        rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                        rollouts.masks[step], rollouts.location_masks[step])
                action = action.squeeze(-1)
                obs, reward, done, infos = envs.step(action.cpu().numpy())

                if use_mask:
                    location_masks = obs[:, plainum: (2 + enable_rotation) * plainum]
                else:
                    location_masks = torch.ones((args.num_processes, (1 + args.enable_rotation) * plainum)).to(
                        device)

                for i in range(len(infos)):
                    if 'episode' in infos[i].keys():
                        episode_rewards.append(infos[i]['episode']['r'])
                        if 'ratio' in infos[i].keys():
                            episode_ratio.append(infos[i]['ratio'])
                        if 'constraint_reward' in infos[i].keys():
                            episode_constraint_rewards.append(infos[i]['constraint_reward'])

                if episode_ratio.__len__() != 0:
                    ratio_recorder = max(ratio_recorder, np.max(episode_ratio))

                masks = torch.FloatTensor(1 - done).reshape((-1, 1))
                bad_masks = torch.FloatTensor(
                    [[0.0] if 'bad_transition' in info.keys() else [1.0]
                     for info in infos])
                rollouts.insert(obs, recurrent_hidden_states, action,
                                action_log_prob, value, reward, masks, bad_masks, location_masks)

            with torch.no_grad():
                next_value = actor_critic.get_value(
                    rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                    rollouts.masks[-1]).detach()

            rollouts.compute_returns(next_value, False, args.gamma, 0.95, False)
            value_loss, action_loss, dist_entropy, metrics = agent.update(rollouts)
            # value_loss, action_loss, dist_entropy, metrics = 0,0,0,[0,0,0,0,0,0,]
            rollouts.after_update()

            if (self.step_counter % args.model_save_interval == 0) or (self.step_counter == 1):
                if self.step_counter % args.model_update_interval == 0 or self.step_counter == 1:
                    sub_time_str = args.custom + '_' + time.strftime('%Y.%m.%d-%H-%M-%S', time.localtime(time.time()))

                torch.save(self.packing_policy.state_dict(), os.path.join(model_save_path, sub_time_str + ".pt"))

            # Write tensorboard logs.
            if self.step_counter % args.print_log_interval == 0 and len(episode_rewards) > 1:
                total_num_steps = self.step_counter * args.num_processes * args.num_steps
                end = time.time()
                if len(episode_ratio) != 0:
                    ratio_recorder = max(ratio_recorder, np.max(episode_ratio))

                print("Time version: {} is training\n"
                      "Updates {}, num timesteps {}, FPS {}\n"
                      "Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                      "The dist entropy {:.5f}, the value loss {:.5f}, the action loss {:.5f}\n"
                      "The mean space ratio is {:.5f}, the ratio threshold is {:.5f}\n"
                      .format(self.timeStr,
                              self.step_counter, total_num_steps, int(total_num_steps / (end - start)),
                              len(episode_rewards), np.mean(episode_rewards), np.median(episode_rewards),
                              np.min(episode_rewards), np.max(episode_rewards),
                              dist_entropy, value_loss, action_loss,

                              np.mean(episode_ratio), ratio_recorder))
                if len(episode_rewards) != 0:
                    self.writer.add_scalar('Rewards/Mean rewards', np.mean(episode_rewards), self.step_counter)
                    self.writer.add_scalar("Rewards/Max rewards", np.max(episode_rewards), self.step_counter)
                    self.writer.add_scalar('Rewards/Min rewards', np.min(episode_rewards), self.step_counter)
                if len(episode_ratio) != 0:
                    self.writer.add_scalar("Ratio/The max ratio", np.max(episode_ratio), self.step_counter)
                    self.writer.add_scalar("Ratio/The mean ratio", np.mean(episode_ratio), self.step_counter)
                    self.writer.add_scalar("Ratio/The min ratio", np.min(episode_ratio), self.step_counter)

                if len(episode_constraint_rewards) != 0:
                    self.writer.add_scalar("Constraint reward/Mean constraint reward", np.mean(episode_constraint_rewards), self.step_counter)
                    self.writer.add_scalar("Constraint reward/Max constraint reward", np.max(episode_constraint_rewards), self.step_counter)
                    self.writer.add_scalar("Constraint reward/Min constraint reward", np.min(episode_constraint_rewards), self.step_counter)

                self.writer.add_scalar("Training/Value loss", value_loss, self.step_counter)
                self.writer.add_scalar("Training/Action loss", action_loss, self.step_counter)
                self.writer.add_scalar('Training/Distribution entropy', dist_entropy, self.step_counter)

    def train_Attend2Pack(self, envs, args, train_episodes = None, given_distribution_label = None):
        device = args.device
        model_save_path = os.path.join(args.model_save_path, self.timeStr)
        sub_time_str    = time.strftime('%Y.%m.%d-%H-%M-%S', time.localtime(time.time()))
        self.packing_policy.train()

        obs = envs.reset()
        num_steps, num_processes = args.num_steps, args.num_processes
        if args.drl_method == 'ppo': num_steps = 30

        Attend2Pack_rollout = Attend2PackRolloutStorage(num_steps,
                                        num_processes,
                                        obs_shape=obs.shape[1:],
                                        action_space = args.container_size[0] * 2)
        Attend2Pack_rollout.to(device)

        start = time.time()
        ratio_recorder = 0
        episode_rewards = deque(maxlen=10)
        episode_constraint_rewards = deque(maxlen=10)
        episode_ratio = deque(maxlen=10)
        episode_final_length = deque(maxlen=10)
        inside_counter = self.step_counter

        Attend2Pack_rollout.obs[0].copy_(obs)
        if train_episodes is None: train_episodes = 1e20

        for _ in range(int(train_episodes)):
            ##############################################
            ####### Collect n-step training sample #######
            ##############################################
            self.step_counter += 1
            for step in range(num_steps):
                with torch.no_grad():
                    action_log_prob, action, value = self.packing_policy(obs, False, args.container_size)
                obs, reward, done, infos = envs.step(action.cpu().numpy())
                Attend2Pack_rollout.insert(obs, None, action, action_log_prob,
                                           value, reward, torch.tensor(1-done).unsqueeze(1))
                for _ in range(len(infos)):
                    if done[_]:
                        if 'reward' in infos[_].keys():
                            episode_rewards.append(infos[_]['reward'])
                        else:
                            episode_rewards.append(infos[_]['episode']['r'])
                        if 'ratio' in infos[_].keys():
                            episode_ratio.append(infos[_]['ratio'])
                        if 'final_length' in infos[_].keys():
                            episode_final_length.append(infos[_]['final_length'])
                        if 'constraint_reward' in infos[_].keys():
                            episode_constraint_rewards.append(infos[_]['constraint_reward'])

            with torch.no_grad():
                _, _, next_value = self.packing_policy(Attend2Pack_rollout.obs[-1], False, args.container_size)

            next_value = next_value[1] if isinstance(next_value, tuple) else next_value
            Attend2Pack_rollout.compute_returns(next_value, False, args.gamma, 0.95, False)

            ##############################################
            ########### PCT policy evaluation ############
            ##############################################
            # Save the trained policy model
            if (self.step_counter % args.model_save_interval == 0) or (self.step_counter == 1):
                if not os.path.exists(model_save_path):
                    os.makedirs(model_save_path)
                if not os.path.exists(args.regular_model_save_path):
                    os.makedirs(args.regular_model_save_path)

                torch.save(self.packing_policy.state_dict(), os.path.join(model_save_path, sub_time_str + ".pt"))
                torch.save(self.packing_policy.state_dict(), os.path.join(args.regular_model_save_path, args.custom + ".pt")) # save the last model to a regular position

            ##############################################
            ########### PCT policy optimzation ###########
            ##############################################
            actor_loss, critic_loss, dist_entropy = self.policy_update_attend_reinforce(Attend2Pack_rollout, args)

            ##############################################
            ############ After optimzation ###############
            ##############################################
            Attend2Pack_rollout.after_update()

            # Write tensorboard logs.
            if self.step_counter % args.print_log_interval == 0 and len(episode_rewards)>1:
                total_num_steps = (self.step_counter + 1 - inside_counter) * num_processes * num_steps
                end = time.time()
                if len(episode_ratio) != 0:
                    ratio_recorder = max(ratio_recorder, np.max(episode_ratio))

                print("Time version: {} is training\n"
                      "Updates {}, num timesteps {}, FPS {}\n"
                      "Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                      "The dist entropy {:.5f}, the value loss {:.5f}, the action loss {:.5f}\n"
                      "The mean space ratio is {:.5f}, the ratio threshold is {:.5f}\n"
                        .format(self.timeStr,
                                self.step_counter, total_num_steps, int(total_num_steps / (end - start)),
                                len(episode_rewards), np.mean(episode_rewards), np.median(episode_rewards), np.min(episode_rewards), np.max(episode_rewards),
                                dist_entropy.item(), critic_loss.item(), actor_loss.item(),
                                np.mean(episode_ratio), ratio_recorder))
                if len(episode_rewards) != 0:
                    self.writer.add_scalar('Rewards/Mean rewards', np.mean(episode_rewards), self.step_counter)
                    self.writer.add_scalar("Rewards/Max rewards", np.max(episode_rewards), self.step_counter)
                    self.writer.add_scalar('Rewards/Min rewards', np.min(episode_rewards), self.step_counter)
                if len(episode_ratio) != 0:
                    self.writer.add_scalar("Ratio/The max ratio", np.max(episode_ratio), self.step_counter)
                    self.writer.add_scalar("Ratio/The mean ratio", np.mean(episode_ratio), self.step_counter)
                    self.writer.add_scalar("Ratio/The min ratio", np.min(episode_ratio), self.step_counter)
                if len(episode_final_length) != 0:
                    self.writer.add_scalar("Length/The max length", np.max(episode_final_length), self.step_counter)
                    self.writer.add_scalar("Length/The mean length", np.mean(episode_final_length), self.step_counter)
                    self.writer.add_scalar("Length/The min length", np.min(episode_final_length), self.step_counter)
                if len(episode_constraint_rewards) != 0:
                    self.writer.add_scalar("Constraint reward/Mean constraint reward", np.mean(episode_constraint_rewards), self.step_counter)
                    self.writer.add_scalar("Constraint reward/Max constraint reward", np.max(episode_constraint_rewards), self.step_counter)
                    self.writer.add_scalar("Constraint reward/Min constraint reward", np.min(episode_constraint_rewards), self.step_counter)
                self.writer.add_scalar("Ratio/The max ratio in history", ratio_recorder, self.step_counter)
                self.writer.add_scalar("Training/Value loss", critic_loss.item(), self.step_counter)
                self.writer.add_scalar("Training/Action loss", actor_loss.item(), self.step_counter)
                self.writer.add_scalar('Training/Distribution entropy', dist_entropy.item(), self.step_counter)


    def train_q_value(self, envs, args):
        priority_weight_increase = (1 - args.priority_weight) / (args.T_max - args.learn_start)
        sub_time_str = time.strftime('%Y.%m.%d-%H-%M-%S', time.localtime(time.time()))

        model_save_path = os.path.join(args.model_save_path, self.timeStr)
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)

        if args.save_memory_path is not None:
            memory_save_path = os.path.join(model_save_path, args.save_memory_path)
            if not os.path.exists(memory_save_path):
                os.makedirs(memory_save_path)

        episode_rewards = deque(maxlen=10)
        episode_ratio = deque(maxlen=10)
        episode_counter = deque(maxlen=10)
        state = envs.reset()
        _, leaf_nodes = tools.get_leaf_nodes(state, args.internal_node_holder, args.leaf_node_holder)

        memNum = args.num_processes
        memory_capacity = int(args.memory_capacity / memNum)
        self.mem = [ReplayMemory(args, memory_capacity, len(state[0].reshape(-1))) for _ in range(memNum)]

        reward_clip = torch.ones((args.num_processes, 1)) * args.reward_clip
        R, loss = 0, 0
        batchX = torch.arange(args.num_processes)

        # Training loop
        self.packing_policy.train()
        for T in trange(1, args.T_max + 1):

            if T % args.replay_frequency == 0:
                self.packing_policy.reset_noise()  # Draw a new set of noisy weights

            _, leaf_nodes = tools.get_leaf_nodes(state, args.internal_node_holder, args.leaf_node_holder)
            leaf_nodes = leaf_nodes.to(args.device)

            action = self.packing_policy.act(state)  # Choose an action greedily (with noisy weights)
            # print(action)
            selected_leaf_node = leaf_nodes[batchX, action.squeeze()]
            next_state, reward, done, infos = envs.step(selected_leaf_node.cpu().numpy())  # Step
            next_state = next_state

            for _ in range(len(infos)):
                if done[_]:
                    if 'reward' in infos[_].keys():
                        episode_rewards.append(infos[_]['reward'])
                    else:
                        episode_rewards.append(infos[_]['episode']['r'])
                    if 'ratio' in infos[_].keys():
                        episode_ratio.append(infos[_]['ratio'])
                    if 'counter' in infos[_].keys():
                        episode_counter.append(infos[_]['counter'])

            if args.reward_clip > 0:
                reward = torch.maximum(torch.minimum(reward, reward_clip), -reward_clip)  # Clip rewards

            for i in range(len(state)):
                self.mem[i].append(state[i].reshape(-1), action[i], reward[i], done[i])  # Append transition to memory

            if T >= args.learn_start:
                for i in range(len(self.mem)):
                    self.mem[i].priority_weight = min(self.mem[i].priority_weight + priority_weight_increase, 1)  # Anneal importance sampling weight β to 1

                if T % args.replay_frequency == 0:
                    loss = self.packing_policy.learn(self.mem)  # Train with n-step distributional double-Q learning
                # Update target network
                if T % args.target_update == 0:
                    self.packing_policy.update_target_net()

                # Checkpoint the network #
                if (args.checkpoint_interval != 0) and (T % args.save_interval == 0):
                    if T % args.checkpoint_interval == 0:
                        sub_time_str = time.strftime('%Y.%m.%d-%H-%M-%S', time.localtime(time.time()))
                    self.packing_policy.save(model_save_path, 'checkpoint{}.pt'.format(sub_time_str))

                if T % args.print_log_interval == 0:
                    self.writer.add_scalar("Training/Value loss",  loss.mean().item(), T)

            state = next_state

            if len(episode_rewards)!= 0:
                self.writer.add_scalar('Metric/Reward mean', np.mean(episode_rewards), T)
                self.writer.add_scalar('Metric/Reward max', np.max(episode_rewards), T)
                self.writer.add_scalar('Metric/Reward min', np.min(episode_rewards), T)

            if len(episode_ratio) != 0:
                self.writer.add_scalar('Metric/Ratio', np.mean(episode_ratio), T)

            if len(episode_counter) != 0:
                self.writer.add_scalar('Metric/Length', np.mean(episode_counter), T)

    def train_RCQL(self, envs, args, train_episodes = None, given_distribution_label = None):
        device = args.device
        model_save_path = os.path.join(args.model_save_path, self.timeStr)
        sub_time_str    = time.strftime('%Y.%m.%d-%H-%M-%S', time.localtime(time.time()))
        self.packing_policy.train()

        obs = envs.reset()
        num_steps, num_processes = args.num_steps, args.num_processes
        if args.drl_method == 'ppo': num_steps = 30

        RCQL_rollout = RCQLRolloutStorage(num_steps,
                                        num_processes,
                                        obs_shape=obs.shape[1:],
                                        action_space = args.container_size[0] * 2)
        RCQL_rollout.to(device)

        start = time.time()
        ratio_recorder = 0
        episode_rewards = deque(maxlen=10)
        episode_constraint_rewards = deque(maxlen=10)
        episode_ratio = deque(maxlen=10)
        episode_final_length = deque(maxlen=10)
        inside_counter = self.step_counter

        RCQL_rollout.obs[0].copy_(obs)
        if train_episodes is None: train_episodes = 1e20

        attn_span = 20
        hidden_size = 128
        encoder_nb_layers = 3
        h_cache = [torch.zeros(num_processes, attn_span, hidden_size).to(device) for i in range(encoder_nb_layers)]

        for _ in range(int(train_episodes)):
            ##############################################
            ####### Collect n-step training sample #######
            ##############################################
            self.step_counter += 1
            for step in range(num_steps):
                with torch.no_grad():
                    # pro, [oriidx, xidx, yidx], [oripro, xpro, ypro], [oriout, xout, yout], [orivalue, xvalue, yvalue]
                    action, action_log_prob, value = self.packing_policy(obs, h_cache, False, args)
                obs, reward, done, infos = envs.step(action.cpu().numpy())
                RCQL_rollout.insert(obs, None, action, action_log_prob,
                                           value, reward, torch.tensor(1-done).unsqueeze(1))
                for _ in range(len(infos)):
                    if done[_]:
                        if 'reward' in infos[_].keys():
                            episode_rewards.append(infos[_]['reward'])
                        else:
                            episode_rewards.append(infos[_]['episode']['r'])
                        if 'ratio' in infos[_].keys():
                            episode_ratio.append(infos[_]['ratio'])
                        if 'final_length' in infos[_].keys():
                            episode_final_length.append(infos[_]['final_length'])
                        if 'constraint_reward' in infos[_].keys():
                            episode_constraint_rewards.append(infos[_]['constraint_reward'])

            with torch.no_grad():
                _, _, next_value = self.packing_policy(RCQL_rollout.obs[-1], h_cache, False, args)

            next_value = next_value[1] if isinstance(next_value, tuple) else next_value
            RCQL_rollout.compute_returns(next_value, False, args.gamma, 0.95, False)

            ##############################################
            ########### PCT policy evaluation ############
            ##############################################
            # Save the trained policy model
            if (self.step_counter % args.model_save_interval == 0) or (self.step_counter == 1):
                if not os.path.exists(model_save_path):
                    os.makedirs(model_save_path)
                if not os.path.exists(args.regular_model_save_path):
                    os.makedirs(args.regular_model_save_path)

                torch.save(self.packing_policy.state_dict(), os.path.join(model_save_path, sub_time_str + ".pt"))
                torch.save(self.packing_policy.state_dict(), os.path.join(args.regular_model_save_path, args.custom + ".pt")) # save the last model to a regular position

            ##############################################
            ########### PCT policy optimzation ###########
            ##############################################
            actor_loss, critic_loss, dist_entropy = self.policy_update_RCQL_actor_critic(RCQL_rollout, args)

            ##############################################
            ############ After optimzation ###############
            ##############################################
            RCQL_rollout.after_update()

            # Write tensorboard logs.
            if self.step_counter % args.print_log_interval == 0 and len(episode_rewards)>1:
                total_num_steps = (self.step_counter + 1 - inside_counter) * num_processes * num_steps
                end = time.time()
                if len(episode_ratio) != 0:
                    ratio_recorder = max(ratio_recorder, np.max(episode_ratio))

                print("Time version: {} is training\n"
                      "Updates {}, num timesteps {}, FPS {}\n"
                      "Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                      "The dist entropy {:.5f}, the value loss {:.5f}, the action loss {:.5f}\n"
                      "The mean space ratio is {:.5f}, the ratio threshold is {:.5f}\n"
                        .format(self.timeStr,
                                self.step_counter, total_num_steps, int(total_num_steps / (end - start)),
                                len(episode_rewards), np.mean(episode_rewards), np.median(episode_rewards), np.min(episode_rewards), np.max(episode_rewards),
                                dist_entropy.item(), critic_loss.item(), actor_loss.item(),
                                np.mean(episode_ratio), ratio_recorder))
                if len(episode_rewards) != 0:
                    self.writer.add_scalar('Rewards/Mean rewards', np.mean(episode_rewards), self.step_counter)
                    self.writer.add_scalar("Rewards/Max rewards", np.max(episode_rewards), self.step_counter)
                    self.writer.add_scalar('Rewards/Min rewards', np.min(episode_rewards), self.step_counter)
                if len(episode_ratio) != 0:
                    self.writer.add_scalar("Ratio/The max ratio", np.max(episode_ratio), self.step_counter)
                    self.writer.add_scalar("Ratio/The mean ratio", np.mean(episode_ratio), self.step_counter)
                    self.writer.add_scalar("Ratio/The min ratio", np.min(episode_ratio), self.step_counter)
                if len(episode_final_length) != 0:
                    self.writer.add_scalar("Length/The max length", np.max(episode_final_length), self.step_counter)
                    self.writer.add_scalar("Length/The mean length", np.mean(episode_final_length), self.step_counter)
                    self.writer.add_scalar("Length/The min length", np.min(episode_final_length), self.step_counter)
                if len(episode_constraint_rewards) != 0:
                    self.writer.add_scalar("Constraint reward/Mean constraint reward", np.mean(episode_constraint_rewards), self.step_counter)
                    self.writer.add_scalar("Constraint reward/Max constraint reward", np.max(episode_constraint_rewards), self.step_counter)
                    self.writer.add_scalar("Constraint reward/Min constraint reward", np.min(episode_constraint_rewards), self.step_counter)
                self.writer.add_scalar("Ratio/The max ratio in history", ratio_recorder, self.step_counter)
                self.writer.add_scalar("Training/Value loss", critic_loss.item(), self.step_counter)
                self.writer.add_scalar("Training/Action loss", actor_loss.item(), self.step_counter)
                self.writer.add_scalar('Training/Distribution entropy', dist_entropy.item(), self.step_counter)

    def train_PackE(self, envs, args):
        import PackE_model


        model_save_path = os.path.join(args.model_save_path, self.timeStr)

        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
        sub_time_str = args.custom + '_' + time.strftime('%Y.%m.%d-%H-%M-%S', time.localtime(time.time()))

        value_loss_coef = 0.5
        entropy_coef = 1e-2
        pallet_size = args.container_size
        actor_critic = self.packing_policy
        enable_rotation = True
        use_mask = True
        device = args.device

        action_space = gym.spaces.Discrete(2 * args.container_size[0] * args.container_size[1])
        obs = envs.reset()
            
        rollouts = PackE_model.RolloutStorage(args.num_steps,  # number of forward steps in A2C (default: 5)
                                             args.num_processes,  # how many training CPU processes to use (default: 16)
                                             (obs.shape[-1],),
                                             action_space)


        rollouts.obs[0].copy_(obs)
        rollouts.to(device)

        episode_rewards = deque(maxlen=10)
        episode_ratio = deque(maxlen=10)
        episode_constraint_rewards = deque(maxlen=10)

        start = time.time()

        j = 0
        ratio_recorder = 0

        while True:
            j += 1
            self.step_counter += 1
            for step in range(args.num_steps):
                with torch.no_grad():
                    value, action, action_log_prob = actor_critic.act(
                        rollouts.obs[step])

                obs, reward, done, infos = envs.step(action.cpu().numpy())

                for i in range(len(infos)):
                    if 'episode' in infos[i].keys():
                        episode_rewards.append(infos[i]['episode']['r'])
                        if 'ratio' in infos[i].keys():
                            episode_ratio.append(infos[i]['ratio'])
                        if 'constraint_reward' in infos[i].keys():
                            episode_constraint_rewards.append(infos[i]['constraint_reward'])

                if episode_ratio.__len__() != 0:
                    ratio_recorder = max(ratio_recorder, np.max(episode_ratio))

                masks = torch.FloatTensor(1 - done).reshape((-1, 1))
                bad_masks = torch.FloatTensor(
                    [[0.0] if 'bad_transition' in info.keys() else [1.0]
                     for info in infos])
                rollouts.insert(obs, action, action_log_prob, value, reward, masks, bad_masks)

            with torch.no_grad():
                next_value = actor_critic.get_value(rollouts.obs[-1]).detach()

            rollouts.compute_returns(next_value, False, args.gamma, 0.95, False)
            # value_loss, action_loss, dist_entropy, metrics = agent.update(rollouts)
            value_loss, action_loss, dist_entropy = self.policy_update_with_ppo(rollouts, args)
            rollouts.after_update()

            if (self.step_counter % args.model_save_interval == 0) or (self.step_counter == 1):
                if self.step_counter % args.model_update_interval == 0 or self.step_counter == 1:
                    sub_time_str = args.custom + '_' + time.strftime('%Y.%m.%d-%H-%M-%S', time.localtime(time.time()))

                torch.save(self.packing_policy.state_dict(), os.path.join(model_save_path, sub_time_str + ".pt"))

            # Write tensorboard logs.
            if self.step_counter % args.print_log_interval == 0 and len(episode_rewards) > 1:
                total_num_steps = self.step_counter * args.num_processes * args.num_steps
                end = time.time()
                if len(episode_ratio) != 0:
                    ratio_recorder = max(ratio_recorder, np.max(episode_ratio))

                print("Time version: {} is training\n"
                      "Updates {}, num timesteps {}, FPS {}\n"
                      "Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                      "The dist entropy {:.5f}, the value loss {:.5f}, the action loss {:.5f}\n"
                      "The mean space ratio is {:.5f}, the ratio threshold is {:.5f}\n"
                      .format(self.timeStr,
                              self.step_counter, total_num_steps, int(total_num_steps / (end - start)),
                              len(episode_rewards), np.mean(episode_rewards), np.median(episode_rewards),
                              np.min(episode_rewards), np.max(episode_rewards),
                              dist_entropy, value_loss, action_loss,

                              np.mean(episode_ratio), ratio_recorder))
                if len(episode_rewards) != 0:
                    self.writer.add_scalar('Rewards/Mean rewards', np.mean(episode_rewards), self.step_counter)
                    self.writer.add_scalar("Rewards/Max rewards", np.max(episode_rewards), self.step_counter)
                    self.writer.add_scalar('Rewards/Min rewards', np.min(episode_rewards), self.step_counter)
                if len(episode_ratio) != 0:
                    self.writer.add_scalar("Ratio/The max ratio", np.max(episode_ratio), self.step_counter)
                    self.writer.add_scalar("Ratio/The mean ratio", np.mean(episode_ratio), self.step_counter)
                    self.writer.add_scalar("Ratio/The min ratio", np.min(episode_ratio), self.step_counter)

                if len(episode_constraint_rewards) != 0:
                    self.writer.add_scalar("Constraint reward/Mean constraint reward",
                                           np.mean(episode_constraint_rewards), self.step_counter)
                    self.writer.add_scalar("Constraint reward/Max constraint reward",
                                           np.max(episode_constraint_rewards), self.step_counter)
                    self.writer.add_scalar("Constraint reward/Min constraint reward",
                                           np.min(episode_constraint_rewards), self.step_counter)

                self.writer.add_scalar("Training/Value loss", value_loss, self.step_counter)
                self.writer.add_scalar("Training/Action loss", action_loss, self.step_counter)
                self.writer.add_scalar('Training/Distribution entropy', dist_entropy, self.step_counter)