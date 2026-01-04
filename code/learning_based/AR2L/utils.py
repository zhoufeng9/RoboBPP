import os, sys
from shutil import copyfile, copytree
import argparse
import numpy as np
from gym.envs.registration import register
import gym
from time import time
import re
import random

import torch
import torch.nn as nn

import matplotlib
if sys.platform != 'linux': matplotlib.use('TkAgg')
from matplotlib import pyplot as plt


def get_args():
    parser = argparse.ArgumentParser(description='3D Bin Packing Arguments')
    parser.add_argument('--setting', type=int, default=1,
                        help='Experiemnt setting: 1 | 2 | 3')
    # parser.add_argument('--container-size', type=float, default=10,
    #                     help='The width, length and height of the container')
    parser.add_argument('--max-item-size', type=int, default=5,
                        help='the maximum size of box')
    parser.add_argument('--min-item-size', type=int, default=1,
                        help='the minimum size of box')
    parser.add_argument('--continuous', action='store_true', default=False,
                        help='Use continuous environment or discrete environment')
    parser.add_argument('--num-box', type=int, default=80,
                        help='The maximum number of nodes to represent the bin state')
    parser.add_argument('--num-next-box', type=int, default=5,
                        help='The maximum number of next box, default is 5')
    parser.add_argument('--num-candidate-action', type=int, default=120,
                        help='The maximum number of particles to represent the feasible actions')
    parser.add_argument('--node-dim', type=int, default=9,
                        help='The vector size to represent one node')
    parser.add_argument('--sparse-reward', type=int, default=1,
                        help='The reward from env can be dense (0) or sparse (1)')

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Donot use cuda')
    parser.add_argument('--device', type=int, default=0,
                        help='Choose GPUs to train model')
    parser.add_argument('--seed', type=int, default=0,
                        help='Set the random seed')

    parser.add_argument('--training-algorithm', type=str, default='ppo',
                        help='Choose one training algorithm: ppo')
    parser.add_argument('--num-processes', type=int, default=16,
                        help='The number of parallel processes used for training')
    parser.add_argument('--num-steps', type=int, default=30,
                        help='The rollout length for n-step training')
    # parser.add_argument('--num-processes', type=int, default=1,
    #                     help='The number of parallel processes used for training')
    # parser.add_argument('--num-steps', type=int, default=4,
    #                     help='The rollout length for n-step training')
    parser.add_argument('--actor-loss-coef', type=float, default=1.0,
                        help='The coefficient of actor loss')
    parser.add_argument('--critic-loss-coef', type=float, default=1.0,
                        help='The coefficient of critic loss')
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                        help='The maximum norm of gradients')
    parser.add_argument('--gamma', type=float, default=1.0,
                        help='Discount factor of Return')

    parser.add_argument('--embedding-size', type=int, default=64,
                        help='Dimension of the input embedding')
    parser.add_argument('--hidden-size', type=int, default=128,
                        help='Dimension of hidden layers')
    parser.add_argument('--gat-layer-num', type=int, default=1,
                        help='The number of GAT layers')

    parser.add_argument('--print-log-interval', type=int, default=10,
                        help='The frequency of printing the training logs')
    parser.add_argument('--sample-from-distribution', action='store_true', default=False,
                        help='Sample continuous item size from a Uniform distribution')
    parser.add_argument('--sample-left-bound', type=float, default=1.,
                        help='The left bound of the uniform distribution')
    parser.add_argument('--sample-right-bound', type=float, default=5.,
                        help='the right bound of the uniform distribution')
    parser.add_argument('--unit-interval', type=float, default=1.,
                        help='the unit interval for height samples')

    parser.add_argument('--learning-rate', type=float, default=3e-4,
                        help='Set learning rate for a2c (default: 1e-6) or ppo (default: 3e-4)')
    parser.add_argument('--num-env-steps', type=int, default=10e6,
                        help='number of environment steps to train (default: 10e6)')

    parser.add_argument('--use-gae', action='store_true', default=True,
                        help='choose whether to use GAE for advantage approximation or not')
    parser.add_argument('--gae-lambda', type=float, default=0.95,
                        help='gae lambda parameter (default: 0.95)')
    parser.add_argument('--clip-param', type=float, default=0.1,
                        help='ppo clip parameter (default: 0.1)')
    parser.add_argument('--num-mini-batch', type=int, default=32,
                        help='number of batches for ppo (default: 32)')
    parser.add_argument('--ppo-epoch', type=int, default=1,
                        help='number of ppo epochs (default: 10)')
    parser.add_argument('--use-proper-time-limits', action='store_true', default=False,
                        help='compute returns taking into account time limits')
    parser.add_argument('--value-loss-coef', type=float, default=1.,
                        help='The coefficient of value loss of PPO')
    parser.add_argument('--entropy-coef', type=float, default=0.01,
                        help='The coefficient of entropy of PPO')
    parser.add_argument('--eps', type=float, default=1e-5,
                        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument('--alpha', type=float, default=0.5)

    # add by wzf
    parser.add_argument('--model-save-interval', type=int, default=100, help='How often to save the model')
    parser.add_argument('--model-update-interval', type=int, default=10e3, help='How often to create a new model')
    # parser.add_argument('--model-save-interval', type=int, default=20, help='How often to save the model')
    # parser.add_argument('--model-update-interval', type=int, default=60, help='How often to create a new model')
    parser.add_argument('--model-save-path', type=str, default='./logs/experiment',
                        help='The path to save the trained model')
    parser.add_argument('--custom', type=str, help='Test name', default='time_series')
    parser.add_argument('--container-size', type=int, help='Container size', nargs="*", default=[120, 120, 100])
    parser.add_argument('--dataset-path', type=str, help='The path to load dataset')

    args = parser.parse_args()
    args.no_cuda = args.no_cuda | (not torch.cuda.is_available())

    if args.no_cuda:
        args.device = 'cpu'

    # args.container_size = int(args.container_size) if not args.continuous else args.container_size
    # args.container_size = 3 * (args.container_size,)
    args.env_id = '3dBP-Discrete-v0' if not args.continuous else '3dBP-Continuous-v0'
    # args.item_size_set = DiscreteBoxData(lower=args.min_item_size, higher=args.max_item_size)
    args.item_size_set = [[43, 20,  7],[45, 30,  6]]
    args.normFactor = 1. / max(args.container_size)

    if args.custom == 'time_series':
        args.container_size = [134, 125, 100]
        args.threshold = 0.02
    elif args.custom == 'occupancy':
        args.container_size = [120, 100, 170]
        args.threshold = 0.02
    elif args.custom == 'flat_long':
        args.container_size = [250, 120, 100]
        args.threshold = 0.04

    return args


def DecodeObs4Place(observation, bin_node_len, box_node_len, candidata_node_len, node_dim):
    bin_node = observation[:, 0:bin_node_len, 0:node_dim-2]
    box_node = observation[:, bin_node_len:bin_node_len+box_node_len, 3:node_dim-2]
    candidata_node = observation[:, bin_node_len+box_node_len:, 0:node_dim-1]

    assert candidata_node.size(1) == candidata_node_len
    assert observation[:, bin_node_len:bin_node_len+box_node_len, 0:3].sum() == 0

    full_mask = observation[:, :, -1]
    valid_mask = observation[:, bin_node_len+box_node_len:, node_dim-2]

    return bin_node, box_node, candidata_node, valid_mask, full_mask

def DecodeObs4Adv(observation, bin_node_len, box_node_len, node_dim):
    bin_node = observation[:, 0:bin_node_len, 0:node_dim-2]
    box_node = observation[:, bin_node_len:bin_node_len+box_node_len, 3:node_dim-2]

    assert observation[:, bin_node_len:bin_node_len+box_node_len, 0:3].sum() == 0

    full_mask = observation[:, :, -1]
    valid_mask = observation[:, bin_node_len:, -1]

    return bin_node, box_node, valid_mask, full_mask

def DecodeObs4Critic(observation, bin_node_len, box_node_len, node_dim):
    bin_node = observation[:, 0:bin_node_len, 0:node_dim-2]
    box_node = observation[:, bin_node_len:bin_node_len+box_node_len, 3:node_dim-2]

    assert observation[:, bin_node_len:bin_node_len+box_node_len, 0:3].sum() == 0

    full_mask = observation[:, 0:bin_node_len+box_node_len, -1]
    return bin_node, box_node, full_mask



def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr, minimum_lr=1e-4):
    """Decreases the learning rate linearly"""
    if epoch - 12000 >= 0:
        decay_ratio = epoch / float(total_num_epochs) if epoch < total_num_epochs else \
            float(total_num_epochs-1)/float(total_num_epochs)
        lr = initial_lr - (initial_lr * decay_ratio)
        lr = minimum_lr if lr < minimum_lr else lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def DiscreteBoxData(lower=1, higher=5, resolution=1):
    item_size_set = []
    for i in range(lower, higher + 1):
        for j in range(lower, higher + 1):
            for k in range(lower, higher + 1):
                item_size_set.append((i * resolution, j * resolution, k * resolution))
    return item_size_set

def registration_envs():
    register(
        id='3dBP-Discrete-v0',
        entry_point='3dBP_envs.3dBP_Discrete0:PackingDiscrete',
    )
    register(
        id='3dBP-Continuous-v0',
        entry_point='3dBP_envs.3dBP_Continuous0:PackingContinuous',
    )

