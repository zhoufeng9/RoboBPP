import IPython
import gymnasium as gym
import numpy as np
import tianshou as ts
import os

# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from tqdm import tqdm
import argparse
import torch
from torch.distributions import Categorical
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter

from tianshou.utils import TensorboardLogger
from tianshou.data import Collector, VectorReplayBuffer, ReplayBuffer
from tianshou.data import Batch
from packer.tapnet.envs.env import *


def get_tap_policy(args):
    
    box_dim = 3
    # ems_dim = 7 if args.use_bridge else 6    
    ems_dim = 6 + (args.container_type == 'multi')
    # ems_dim = 7 + (args.container_type == 'multi')

    args.ems_dim = ems_dim

    rot_num = 2
    device = args.device
    
    if args.require_box_num is not None:
        box_state_num = rot_num * len(args.rotate_axes) * args.require_box_num
    else:
        box_state_num = rot_num * len(args.rotate_axes) * args.box_num
        
    ems_per_num = 10
    max_ems_num = args.box_num * ems_per_num
    args.ems_per_num = ems_per_num

    args.fact_data_folder = None
    prec_dim = 2

    args.stable_predict = args.stable_predict == 1


    from packer.tapnet.models.network import Net, Critic
    actor = Net(box_dim, ems_dim, args.hidden_dim, prec_dim, args.prec_type, args.stable_predict, args.corner_num, device).to(device)
    critic = Critic( box_dim, ems_dim, box_state_num, max_ems_num, args.hidden_dim, prec_dim, args.prec_type, device=device).to(device)
    args.action_type = 'box-ems'
    optim = torch.optim.Adam( list(actor.parameters()) + list(critic.parameters()) , lr=float(args.lr))

    lr_scheduler = None
    if args.lr_decay:
        # decay learning rate to 0 linearly
        max_update_num = np.ceil(
            args.step_per_epoch / args.step_per_collect
        ) * args.max_epoch

        lr_scheduler = LambdaLR(
            optim, lr_lambda=lambda epoch: 1 - epoch / max_update_num
        )

    def dist_fn(*logits):
        return Categorical(*logits)

    policy = ts.policy.PPOPolicy(actor, critic, optim, dist_fn, \
            discount_factor=args.gamma,
            max_grad_norm=args.max_grad_norm,
            action_bound_method=args.action_bound_method,
            lr_scheduler=lr_scheduler,
            gae_lambda=args.gae_lambda,
            vf_coef=args.vf_coef,
            ent_coef=args.ent_coef,
            reward_normalization=args.rew_norm,
            action_scaling=True,
            eps_clip=args.eps_clip,
            value_clip=args.value_clip,
            dual_clip=args.dual_clip,
            advantage_normalization=args.norm_adv,
            recompute_advantage=args.recompute_adv,
    )
    
    if args.resume_path is not None and args.model != 'greedy':
        # policy_pth = os.path.join(args.resume_path, "policy.pth")
        policy_pth = args.resume_path
        print(f"loading {policy_pth}")
        state_dict = torch.load(policy_pth, map_location=torch.device('cuda:0'))

        if args.train == 0:
            # tianshou save the params as: actor.xxxx. , we need xxx.
            actor_dict = { k[6:] :v for k,v in state_dict.items() if ( '_actor_critic' not in k and 'critic' not in k ) }
            # actor_dict = { k[8:] :v for k,v in state_dict.items() if ( '_actor_critic' not in k and 'critic' not in k ) }

            policy.actor.load_state_dict(actor_dict,strict=True)
        else:
            policy.load_state_dict(state_dict)

    return policy


def get_tap_env(args):
    if args.train == 0:
        test_num = 1
    else:
        test_num = 1
    use_bridge = args.use_bridge == 1
    
    test_envs = TAP(box_num=args.box_num, ems_dim=args.ems_dim,container_size=args.container_size, box_range=args.box_range, threshold=args.threshold, check_stability=args.check_stability,
                stable_rule=args.stable_rule,allow_unstable=args.allow_unstable,use_bridge=use_bridge,same_height_threshold = args.same_height_threshold, 
                min_ems_width = args.min_ems_width, min_height_diff = args.min_height_diff,fact_type=args.fact_type,
                data_type=args.data_type,ems_type=args.ems_type,rotate_axes=args.rotate_axes,action_type=args.action_type,require_box_num=args.require_box_num,
                world_type=args.world_type,container_type=args.container_type,pack_type=args.pack_type,ems_per_num = args.ems_per_num,
                init_ctn_num = args.init_ctn_num,stable_predict=args.stable_predict,gripper_size=args.gripper_size,corner_num=args.corner_num,reward_type='H')

    test_envs.seed(args.seed)

    return test_envs