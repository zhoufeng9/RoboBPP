import os
import sys
import torch
import time as mytime
from time import strftime, localtime, time
import numpy as np
import random
import gym

from models.graph_attention import DRL_GAT
from utils import *
from envs import make_vec_envs
from ppo import PPO_Training
from utils import get_args, registration_envs

def main(args):
    custom = args.custom
    timeStr = custom + '-' + mytime.strftime('%Y.%m.%d-%H-%M-%S', mytime.localtime(mytime.time()))

    if args.no_cuda:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda', args.device)
        torch.cuda.set_device(args.device)

    torch.set_num_threads(1)
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    envs = make_vec_envs(args, None, True)

    bppObs_size = (args.num_box + args.num_next_box + args.num_candidate_action) * args.node_dim
    BPP_policy = DRL_GAT(gym.spaces.Box(low=0., high=args.container_size[-1], shape=(bppObs_size, )),
                         gym.spaces.Discrete(n=args.num_candidate_action),
                         args.embedding_size,
                         args.hidden_size,
                         args.gat_layer_num,
                         args.num_box,
                         args.num_next_box,
                         args.num_candidate_action,
                         args.node_dim,
                         policy_ctg='place',
                         )

    advObs_size = (args.num_box + args.num_next_box) * args.node_dim
    adv_policy = DRL_GAT(gym.spaces.Box(low=0., high=args.container_size[-1], shape=(advObs_size, )),
                         gym.spaces.Discrete(n=args.num_next_box),
                         args.embedding_size,
                         args.hidden_size,
                         args.gat_layer_num,
                         args.num_box,
                         args.num_next_box,
                         0,
                         args.node_dim,
                         policy_ctg='permutation',
                         )

    balObs_size = (args.num_box + args.num_next_box) * args.node_dim
    bal_policy = DRL_GAT(gym.spaces.Box(low=0., high=args.container_size[-1], shape=(balObs_size,)),
                         gym.spaces.Discrete(n=args.num_next_box),
                         args.embedding_size,
                         args.hidden_size,
                         args.gat_layer_num,
                         args.num_box,
                         args.num_next_box,
                         0,
                         args.node_dim,
                         policy_ctg='permutation',
                         )

    BPP_policy = BPP_policy.to(device)
    adv_policy = adv_policy.to(device)
    bal_policy = bal_policy.to(device)


    train_model = PPO_Training(BPP_policy,
                               adv_policy,
                               bal_policy,
                               args,)

    train_model.train_n_steps(envs, args, device, timeStr)


if __name__ == '__main__':
    registration_envs()
    args = get_args()
    main(args)


