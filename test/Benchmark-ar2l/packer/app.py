import numpy as np
from . import ems_tools as ET
from .container  import Container
from packer.pct_model.model import DRL_GAT
from packer.ar2l_models.graph_attention import DRL_GAT as ar2l_DRL_GAT
from packer.pct_model.tools import *

from packer.irrgular_tools import *
import functools
# from packer.pct_model.tools import get_args as get_pct_args
from ar2l_tools.utils import execute_permute_policy

from tools import load_config


import time
import gym

def get_policy(args, method):
    # LSAH   MACS   RANDOM  OnlineBPH   DBL  BR  MTPE IR_HM BLBF FF
    infos = dict()
    infos['args'] = None
    
    if method in ['LeftBottom', 'HeightmapMin', 'LSAH', 'MACS', 'RANDOM', 'OnlineBPH', 'DBL', 'BR', 'SDFPack', 'SDF_Pack', 'MTPE', 'IR_HM', 'BLBF', 'FF']:
        policy = 'heuristic'
        
    elif method == 'PCT' or method == 'PackE':
        # args = get_pct_args()
        # infos['args'] = args
        args_method = load_config(args.config_learning_method)
        
        if args_method.no_cuda: args.device = 'cpu'
        if args_method.setting == 1:
            args_method.internal_node_length = 6
        elif args_method.setting == 2:
            args_method.internal_node_length = 6
        elif args_method.setting == 3:
            args_method.internal_node_length = 7
        if args_method.evaluate:
            args_method.num_processes = 1

        if args.data == 'random' or args.data == 'time_series':
            if args.setting == 1:
                args_method.model_path = args_method.model_path_random_time_series
            else:
                args_method.model_path = args_method.model_path_random_time_series_pybullet
        elif args.data == 'occupancy':
            if args.setting == 1:
                args_method.model_path = args_method.model_path_occupancy
            else:
                args_method.model_path = args_method.model_path_occupancy_pybullet
        elif args.data == 'flat_long':
            if args.setting == 1:
                args_method.model_path = args_method.model_path_flat_long
            else:
                args_method.model_path = args_method.model_path_flat_long_pybullet

        infos['args'] = args_method
        policy = DRL_GAT()
        # Load the trained model
        model_path = args_method.model_path
        policy = load_policy(model_path, policy)
        print('Pre-train model loaded!', model_path)
        policy.eval()
        
    elif method == 'AR2L':
        args_method = load_config(args.config_learning_method)

        if args_method.no_cuda: args_method.device = 'cpu'
        args_method.env_id = '3dBP-Discrete-v0' if not args_method.continuous else '3dBP-Continuous-v0'

        if args.data == 'random' or args.data == 'time_series':
            if args.setting == 1:
                args_method.model_path = args_method.model_path_random_time_series
            else:
                args_method.model_path = args_method.model_path_random_time_series_pybullet
            args_method.container_size = [134, 125, 100]
            args_method.threshold = 0.07
        elif args.data == 'occupancy':
            if args.setting == 1:
                args_method.model_path = args_method.model_path_occupancy
            else:
                args_method.model_path = args_method.model_path_occupancy_pybullet
            args_method.container_size = [120, 100, 170]
            args_method.threshold = 0.04
        elif args.data == 'flat_long':
            if args.setting == 1:
                args_method.model_path = args_method.model_path_flat_long
            else:
                args_method.model_path = args_method.model_path_flat_long_pybullet
            args_method.container_size = [250, 120, 100]
            args_method.threshold = 0.07

        args_method.item_size_set = [[43, 20, 7], [45, 30, 6]]
        args_method.normFactor = 1. / max(args_method.container_size)

        infos['args'] = args_method
        bppObs_size = (args_method.num_box + args_method.num_next_box + args_method.num_candidate_action) * args_method.node_dim
        policy = ar2l_DRL_GAT(gym.spaces.Box(low=0., high=args_method.container_size[-1], shape=(bppObs_size, )),
                         gym.spaces.Discrete(n=args_method.num_candidate_action),
                         args_method.embedding_size,
                         args_method.hidden_size,
                         args_method.gat_layer_num,
                         args_method.num_box,
                         args_method.num_next_box,
                         args_method.num_candidate_action,
                         args_method.node_dim,
                         policy_ctg='place',
                         )
        # Load the trained model
        model_path = args_method.model_path
        # policy = load_policy(model_path, policy)
        pretrained_state_dict = torch.load(model_path, map_location='cpu')
        policy.load_state_dict(pretrained_state_dict)
        print('Pre-train model loaded!', model_path)
        policy.eval()

        
    return policy, infos




def pct(PCT_policy, env, obs, args, eval_freq = 100, factor = 1):

    # obs = env.cur_observation()
    # obs = torch.FloatTensor(obs).unsqueeze(dim=0)

    all_nodes, leaf_nodes = get_leaf_nodes_with_factor(obs, 1, args.internal_node_holder, args.leaf_node_holder)
    batchX = torch.arange(1)

    with torch.no_grad():
        selectedlogProb, selectedIdx, policy_dist_entropy, value = PCT_policy(all_nodes, True, normFactor=factor)
    selected_leaf_node = leaf_nodes[batchX, selectedIdx.squeeze()]      # tensor([[ 8.,  0.,  0., 10.,  5., 10.,  0.,  0.,  1.]], device='cuda:0')
    # if len(env.packed) == 2:
    #     print("here")

    action = selected_leaf_node.cpu().numpy()[0][0:6]
    now_action, box_size = env.LeafNode2Action(action)

    # check rot
    init_box_size = env.next_box
    if box_size[0] == init_box_size[0] and box_size[1] == init_box_size[1]:
        rot = 0
    else:
        rot = 1

    rec = env.space.plain[now_action[1]:now_action[1] + env.next_box[0], now_action[2]:now_action[2] + env.next_box[1]]
    lz = np.max(rec)

    obs, reward, done, infos = env.step(action)

    new_action = (rot,) + now_action[1:]
    # now_action[0] = rot

    if done:
        # obs = env.reset()
        pass

    return done, new_action, lz, obs



def ar2l(policy, env, ar2l_env, bpp_obs, block_index, args):
    num_box, num_next_box, num_candidate_action, node_dim = args.num_box, args.num_next_box, args.num_candidate_action, args.node_dim
    bpp_obs_tensor = torch.from_numpy(bpp_obs)

    # device = torch.device('cuda:0')
    # 调整形状为 (1, 220, 9)
    bpp_obs_tensor = bpp_obs_tensor.unsqueeze(0).to(torch.float32)

    with torch.no_grad():
        action_log_probs, action, entropy = policy.forward_actor(
            bpp_obs_tensor, deterministic=True, normFactor=args.normFactor
        )
        value = policy.forward_critic(
            bpp_obs_tensor, deterministic=True, normFactor=args.normFactor
        )

    location = bpp_obs[num_box+num_next_box:, :][action][:7]
    # location = bpp_obs.split([num_box, num_next_box, num_candidate_action], dim=1)[-1][batchX, action.squeeze(1)][:, :7]
    zero_padding = 0
    # zero_padding = torch.zeros((location.size(0), 1))
    box_idx = 0
    execution = np.append(np.append(location, box_idx), zero_padding)
    # execution = torch.cat((location, box_idx, zero_padding), dim=-1)

    pmt_obs, reward, done, infos = ar2l_env.step(execution)
    pmt_obs = pmt_obs.reshape(num_box + num_next_box, node_dim)
    # pmt_obs = pmt_obs.reshape(pmt_obs.shape[0], num_box + num_next_box, node_dim)
    next_bpp_obs, box_idx, done = execute_permute_policy(ar2l_env, pmt_obs, args)

    if done is True:
        return done, None, None, None

    # lx, ly, lz = int(execution[0]), int(execution[1]), int(execution[2])
    lx, ly, lz = int(execution[0]), int(execution[1]), execution[2]
    if int(env.next_box[0]) == int(execution[3]) and int(env.next_box[1]) == int(execution[4]):
        rotation_flag = 0
    else:
        rotation_flag = 1
    bestAction = [rotation_flag, lx, ly]
    obs, reward, done_env, infos = env.step(bestAction)

    return done, bestAction, lz, next_bpp_obs


class Packer():
    def __init__(self, container_size) -> None:
        self.container = Container(container_size)
    
    def pack_box(self, env, infos, obs, method, policy):
        # LSAH   MACS   RANDOM  OnlineBPH   DBL  BR
           
        if method == 'PCT' or method == 'PackE':
            args = infos['args']
            normFactor = 1.0 / np.max(env.bin_size)
            start_time = time.time()
            done, action, lz, next_obs = pct(policy, env, obs, args, eval_freq=100, factor=normFactor)    # 0.008333333333333333  0.05
            end_time = time.time()
            planning_time = end_time - start_time

            placeable = not done
            if placeable == False:
                return placeable, [], None, infos, planning_time
            rotation_flag, lx, ly = action[0], action[1], action[2]
            pos = np.array([lx, ly, lz], dtype=np.float32)
            
            infos['next_obs'] = next_obs

        elif method == 'AR2L':
            args = infos['args']
            normFactor = 1.0 / np.max(env.bin_size)
            start_time = time.time()
            done, action, lz, next_obs = pct(policy, env, obs, args, eval_freq=100,
                                             factor=normFactor)  # 0.008333333333333333  0.05
            end_time = time.time()
            planning_time = end_time - start_time

            placeable = not done
            if placeable == False:
                return placeable, [], None, infos, planning_time
            rotation_flag, lx, ly = action[0], action[1], action[2]
            pos = np.array([lx, ly, lz], dtype=np.float32)

            infos['next_obs'] = next_obs
    
        return placeable, pos, rotation_flag, infos, planning_time



    def pack_box_ar2l(self, env, ar2l_env, infos, bpp_obs, method, policy, args):
        if method == 'PCT' or method == 'PackE':
            pass

        elif method == 'AR2L':
            block_index = infos['block_index']
            start_time = time.time()
            done, action, lz, next_obs = ar2l(policy, env, ar2l_env, bpp_obs, block_index, args)
            end_time = time.time()
            planning_time = end_time - start_time

            placeable = not done
            if placeable == False:
                return placeable, [], None, infos, planning_time
            rotation_flag, lx, ly = action[0], action[1], action[2]
            pos = np.array([lx, ly, lz], dtype=np.float32)

            infos['next_obs'] = next_obs

        return placeable, pos, rotation_flag, infos, planning_time

    def add_box(self, box, pos):
        pos = self.container.add_new_box(box, pos)
        add_success = True
        if pos is None:
            add_success = False
        
        return add_success


