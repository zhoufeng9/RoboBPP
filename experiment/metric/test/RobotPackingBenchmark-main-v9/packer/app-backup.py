import numpy as np
from . import ems_tools as ET
from .container  import Container
from packer.pct_model.model import DRL_GAT
from packer.pct_model.tools import *
from packer.sdf_pack import *
from packer.irrgular_tools import *
import functools
# from packer.pct_model.tools import get_args as get_pct_args
from packer.irbpp_model.arguments import get_args as get_irbpp_args
from tools import load_config
from packer.irbpp_model.agent import Agent
from packer.irbpp_model.tools import get_mask_from_state, load_shape_dict, shotInfoPre, shapeProcessing

from packer.pack2d_model.pack.packer2d import *
from packer.pack2d_model.pack.tool_2d import *

import time

def get_policy(args, method):
    # LSAH   MACS   RANDOM  OnlineBPH   DBL  BR  MTPE IR_HM BLBF FF
    infos = dict()
    infos['args'] = None
    
    if method in ['LeftBottom', 'HeightmapMin', 'LSAH', 'MACS', 'RANDOM', 'OnlineBPH', 'DBL', 'BR', 'SDFPack', 'SDF_Pack', 'MTPE', 'IR_HM', 'BLBF', 'FF']:
        PCT_policy = 'heuristic'
        
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
            args_method.model_path = args_method.model_path_random_time_series
        elif args.data == 'occupancy':
            args_method.model_path = args_method.model_path_occupancy
        elif args.data == 'flat_long':
            args_method.model_path = args_method.model_path_flat_long

        infos['args'] = args_method
        PCT_policy = DRL_GAT()
        # Load the trained model
        model_path = args_method.model_path
        PCT_policy = load_policy(model_path, PCT_policy)
        print('Pre-train model loaded!', model_path)
        PCT_policy.eval()
        
    elif method == 'IR_BPP':
        # args = get_irbpp_args()
        args_method = load_config(args.config_learning_method)
        
        args_method.action_space = args_method.selectedAction
        args_method.objPath = './dataset/{}/shape_vhacd'.format(args_method.dataset)
        args_method.pointCloud = './dataset/{}/pointCloud'.format(args_method.dataset)
        args_method.dicPath = './dataset/{}/id2shape.pt'.format(args_method.dataset)
        if  args_method.dataset == 'kitchen':
            args_method.dataSample = 'category'
        else:
            args_method.dataSample = 'instance'
        args_method.categories = len(torch.load(args_method.dicPath))
        args_method.bin_dimension = np.round(args_method.bin_dimension, decimals=6)
        args_method.ZRotNum = args_method.resolutionRot  # Max: 4/8
        args_method.heightResolution = args_method.resolutionZ
        args_method.shapeDict, args_method.infoDict = load_shape_dict(args_method, True, scale=args_method.meshScale)
        args_method.globalView = True if args_method.evaluate else False
        args_method.shotInfo = shotInfoPre(args_method, args_method.meshScale)
        args_method.test_name = './dataset/{}/test_sequence.pt'.format(args_method.dataset)
        args_method.shapeArray = shapeProcessing(args_method.shapeDict, args_method)

        dqn = Agent(args_method)
        dqn.online_net.eval()
        infos['args'] = args_method
        PCT_policy = dqn
    
    elif method == 'pack_2d':
        packer2d = Packer_2d("packer/pack2d_model/config.yml")
        infos['packer2d'] = packer2d
        PCT_policy = 'heuristic'
        
    return PCT_policy, infos



def left_bottom(env):
    bin_size = env.bin_size
    bestAction = []
    bestScore = 1e10
    rotation_flag = 0
    bestZpose = None

    next_box = env.next_box
    next_den = env.next_den

    for rot in range(env.orientation):
        # Find the most suitable placement within the allowed orientation.
        if rot == 0:
            x, y, z = next_box
        elif rot == 1:
            y, x, z = next_box
        for lx in range(bin_size[0] - x + 1):
            for ly in range(bin_size[1] - y + 1):
                # Check the feasibility of this placement
                feasible = env.space.drop_box_virtual([x, y, z], (lx, ly), False, next_den, env.setting, False, False)
                if not feasible:
                    continue
                rec = env.space.plain[lx:lx + x, ly:ly + y]
                lz = np.max(rec)
                score = lz * 100 + lx * 10 + ly * 1

                if score < bestScore:
                    bestScore = score
                    env.next_box = [x, y, z]
                    rotation_flag = rot
                    bestAction = [0, lx, ly]
                    bestZpose = lz

    if len(bestAction) != 0:
        lz = bestZpose
        # Place this item in the environment with the best action.
        env.step(bestAction)
        bestAction[0] = rotation_flag
        done = False
    else:
        # No feasible placement, this episode is done.
        lz = None
        done = True
        env.box_creator.drop_box()  # remove current box from the list
        env.box_creator.generate_box_size()  # add a new box to the list
        env.cur_observation()

    return done, bestAction, lz


def heightmap_min(env):
    bin_size = env.bin_size
    bestScore = 1e10
    bestAction = []
    rotation_flag = 0

    next_box = env.next_box
    next_den = env.next_den

    for lx in range(bin_size[0] - next_box[0] + 1):
        for ly in range(bin_size[1] - next_box[1] + 1):
            # Find the most suitable placement within the allowed orientation.
            for rot in range(env.orientation):
                if rot == 0:
                    x, y, z = next_box
                elif rot == 1:
                    y, x, z = next_box
                elif rot == 2:
                    z, x, y = next_box
                elif rot == 3:
                    z, y, x = next_box
                elif rot == 4:
                    x, z, y = next_box
                elif rot == 5:
                    y, z, x = next_box

                # Check the feasibility of this placement
                feasible, heightMap = env.space.drop_box_virtual([x, y, z], (lx, ly), False,
                                                                    next_den, env.setting, False, True)
                if not feasible:
                    continue

                # Score the given placement.
                score = lx + ly + 100 * np.sum(heightMap)
                if score < bestScore:
                    bestScore = score
                    env.next_box = [x, y, z]
                    rotation_flag = rot
                    bestAction = [0, lx, ly]


    if len(bestAction) != 0:
        rec = env.space.plain[bestAction[1]:bestAction[1] + env.next_box[0], bestAction[2]:bestAction[2] + env.next_box[1]]
        lz = np.max(rec)
        # Place this item in the environment with the best action.
        env.step(bestAction)
        bestAction[0] = rotation_flag
        done = False
    else:
        # No feasible placement, this episode is done.
        lz = None
        done = True
        env.box_creator.drop_box()  # remove current box from the list
        env.box_creator.generate_box_size()  # add a new box to the list
        env.cur_observation()

    return  done, bestAction, lz


def LASH(env, maxXY, minXY):
    '''
    Solving a new 3D bin packing problem with deep reinforcement learning method.
    https://arxiv.org/abs/1708.05930 
    '''
    bin_size = env.bin_size
    rotation_flag = 0

    # maxXY = [0,0]
    # minXY = [bin_size[0], bin_size[1]]


    bestScore = bin_size[0] * bin_size[1] + bin_size[1] * bin_size[2] + bin_size[2] * bin_size[0]
    EMS = env.space.EMS

    bestAction = None
    next_box = env.next_box
    next_den = env.next_den

    for ems in EMS:
        # Find the most suitable placement within the allowed orientation.
        if np.sum(np.abs(ems)) == 0:
            continue
        for rot in range(env.orientation):
            if rot == 0:
                x, y, z = next_box
            elif rot == 1:
                y, x, z = next_box
            elif rot == 2:
                z, x, y = next_box
            elif rot == 3:
                z, y, x = next_box
            elif rot == 4:
                x, z, y = next_box
            elif rot == 5:
                y, z, x = next_box

            if ems[3] - ems[0] >= x and ems[4] - ems[1] >= y and ems[5] - ems[2] >= z:
                lx, ly = ems[0], ems[1]
                # Check the feasibility of this placement
                feasible, height = env.space.drop_box_virtual([x, y, z], (lx, ly), False,
                                                            next_den, env.setting, returnH=True)

                if feasible:
                    score = (max(lx + x, maxXY[0]) - min(lx, minXY[0])) * (
                                max(ly + y, maxXY[1]) - min(ly, minXY[1])) \
                            + (height + z) * (max(ly + y, maxXY[1]) - min(ly, minXY[1])) \
                            + (height + z) * (max(lx + x, maxXY[0]) - min(lx, minXY[0]))

                    # The placement which keeps pack items with less surface area is better.
                    if score < bestScore:
                        bestScore = score
                        env.next_box = [x, y, z]
                        rotation_flag = rot
                        bestAction = [0, lx, ly, height, ems[3] - ems[0], ems[4] - ems[1], ems[5] - ems[2]]

                    elif score == bestScore and bestAction is not None:
                        if min(ems[3] - ems[0] - x, ems[4] - ems[1] - y, ems[5] - ems[2] - z) < \
                                min(bestAction[4] - x, bestAction[5] - y, bestAction[6] - z):
                            env.next_box = [x, y, z]
                            rotation_flag = rot
                            bestAction = [0, lx, ly, height, ems[3] - ems[0], ems[4] - ems[1], ems[5] - ems[2]]
    
    if bestAction is not None:
        x, y, _ = env.next_box
        _, lx, ly, _, _, _, _ = bestAction

        if lx + x > maxXY[0]: maxXY[0] = lx + x
        if ly + y > maxXY[1]: maxXY[1] = ly + y
        if lx < minXY[0]: minXY[0] = lx
        if ly < minXY[1]: minXY[1] = ly
        
        rec = env.space.plain[bestAction[1]:bestAction[1] + env.next_box[0], bestAction[2]:bestAction[2] + env.next_box[1]]
        lz = np.max(rec)
        # Place this item in the environment with the best action.
        
        _, _, done, _ = env.step(bestAction[0:3])
        bestAction[0] = rotation_flag
    else:
        # No feasible placement, this episode is done.
        lz = None
        done = True
        env.box_creator.drop_box()  # remove current box from the list
        env.box_creator.generate_box_size()  # add a new box to the list
        env.cur_observation()
    
    return done, bestAction, lz, maxXY, minXY


def MACS(env, block_index, container):
    '''
    Tap-net: transportand-pack using reinforcement learning.
    https://dl.acm.org/doi/abs/10.1145/3414685.3417796
    '''
    def calc_maximal_usable_spaces(ctn, H):
        '''
        Score the given placement.
        This score function comes from https://github.com/Juzhan/TAP-Net/blob/master/tools.py
        '''
        score = 0
        for h in range(H):
            level_max_empty = 0
            # build the histogram map
            hotmap = (ctn[:, :, h] == 0).astype(int)
            histmap = np.zeros_like(hotmap).astype(int)
            for i in reversed(range(container_size[0])):
                for j in range(container_size[1]):
                    if i==container_size[0]-1: histmap[i, j] = hotmap[i, j]
                    elif hotmap[i, j] == 0: histmap[i, j] = 0
                    else: histmap[i, j] = histmap[i+1, j] + hotmap[i, j]

            # scan the histogram map
            for i in range(container_size[0]):
                for j in range(container_size[1]):
                    if histmap[i, j] == 0: continue
                    if j>0 and histmap[i, j] == histmap[i, j-1]: continue
                    # look right
                    for j2 in range(j, container_size[1]):
                        if j2 == container_size[1] - 1: break
                        if histmap[i, j2+1] < histmap[i, j]: break
                    # look left
                    for j1 in reversed(range(0, j+1)):
                        if j1 == 0: break
                        if histmap[i, j1-1] < histmap[i, j]: break
                    area = histmap[i, j] * (j2 - j1 + 1)
                    if area > level_max_empty: level_max_empty = area
            score += level_max_empty
        return score

    def update_container(ctn, pos, boxSize):
        _x, _y, _z = pos
        block_x, block_y, block_z = boxSize
        ctn[_x:_x+block_x, _y:_y+block_y, _z:_z+block_z] = block_index + 1
        under_space = ctn[_x:_x+block_x, _y:_y+block_y, 0:_z]
        ctn[_x:_x+block_x, _y:_y+block_y, 0:_z][ under_space==0 ] = -1

    
    container_size = env.bin_size
    rotation_flag = 0
    bestScore = -1e10
    EMS = env.space.EMS

    bestAction = None
    next_box = env.next_box
    next_den = env.next_den

    for ems in EMS:
        # Find the most suitable placement within the allowed orientation.
        for rot in range(env.orientation):
            if rot == 0:
                x, y, z = next_box
            elif rot == 1:
                y, x, z = next_box
            elif rot == 2:
                z, x, y = next_box
            elif rot == 3:
                z, y, x = next_box
            elif rot == 4:
                x, z, y = next_box
            elif rot == 5:
                y, z, x = next_box

            if ems[3] - ems[0] >= x and ems[4] - ems[1] >= y and ems[5] - ems[2] >= z:
                for corner in range(4):
                    if corner == 0:
                        lx, ly = ems[0], ems[1]
                    elif corner == 1:
                        lx, ly = ems[3] - x, ems[1]
                    elif corner == 2:
                        lx, ly = ems[0], ems[4] - y
                    elif corner == 3:
                        lx, ly = ems[3] - x, ems[4] - y

                    # Check the feasibility of this placement
                    feasible, height = env.space.drop_box_virtual([x, y, z], (lx, ly), False,
                                                                        next_den, env.setting, returnH=True)
                    if feasible:
                        updated_containers = container.copy()
                        update_container(updated_containers, np.array([lx, ly, height]), np.array([x, y, z]))
                        score = calc_maximal_usable_spaces(updated_containers, height)

                        if score > bestScore:
                            bestScore = score
                            env.next_box = [x, y, z]
                            rotation_flag = rot
                            bestAction = [0, lx, ly, height]

    if bestAction is not None:
        # Place this item in the environment with the best action.
        update_container(container, bestAction[1:4], env.next_box)
        rec = env.space.plain[bestAction[1]:bestAction[1] + env.next_box[0], bestAction[2]:bestAction[2] + env.next_box[1]]
        lz = np.max(rec)
        
        _, _, done, _ = env.step(bestAction[0:3])
        bestAction[0] = rotation_flag
    else:
        # No feasible placement, this episode is done.
        lz = None
        done = True
        env.box_creator.drop_box()  # remove current box from the list
        env.box_creator.generate_box_size()  # add a new box to the list
        env.cur_observation()

    return done, bestAction, lz, container


def RANDOM(env, times = 2000):
    '''
    Randomly pick placements from full coordinates.
    '''
    bin_size = env.bin_size
    rotation_flag = 0
    bestAction = None

    next_box = env.next_box
    next_den = env.next_den

    # Check the feasibility of all placements.
    candidates = []
    rotation_flags = []
    for lx in range(bin_size[0] - next_box[0] + 1):
        for ly in range(bin_size[1] - next_box[1] + 1):
            for rot in range(env.orientation):
                if rot == 0:
                    x, y, z = next_box
                elif rot == 1:
                    y, x, z = next_box
                elif rot == 2:
                    z, x, y = next_box
                elif rot == 3:
                    z, y, x = next_box
                elif rot == 4:
                    x, z, y = next_box
                elif rot == 5:
                    y, z, x = next_box

                feasible, heightMap = env.space.drop_box_virtual([x, y, z], (lx, ly), False,
                                                                    next_den, env.setting, False, True)
                if not feasible:
                    continue
                
                rotation_flags.append(rot)
                candidates.append([[x, y, z], [0, lx, ly]])

    if len(candidates) != 0:
        # Pick one placement randomly from all possible placements
        idx = np.random.randint(0, len(candidates))
        rotation_flag = rotation_flags[idx]
        env.next_box = candidates[idx][0]
        bestAction = candidates[idx][1]
        rec = env.space.plain[bestAction[1]:bestAction[1] + env.next_box[0], bestAction[2]:bestAction[2] + env.next_box[1]]
        lz = np.max(rec)
        env.step(bestAction)
        bestAction[0] = rotation_flag
        done = False
    else:
        # No feasible placement, this episode is done.
        lz = None
        done = True
        env.box_creator.drop_box()  # remove current box from the list
        env.box_creator.generate_box_size()  # add a new box to the list
        env.cur_observation()

    return  done, bestAction, lz


def OnlineBPH(env, times = 2000):
    '''
    An Online Packing Heuristic for the Three-Dimensional Container Loading
    Problem in Dynamic Environments and the Physical Internet
    https://doi.org/10.1007/978-3-319-55792-2\_10
    '''

    # Sort the ems placement with deep-bottom-left order.
    EMS = env.space.EMS
    EMS = sorted(EMS, key=lambda ems: (ems[2], ems[1], ems[0]), reverse=False)

    rotation_flag = 0
    bestAction = None
    next_box = env.next_box
    next_den = env.next_den
    stop = False


    for ems in EMS:
        # Find the first suitable placement within the allowed orientation.
        if np.sum(np.abs(ems)) == 0:
            continue
        for rot in range(env.orientation):
            if rot == 0:
                x, y, z = next_box
            elif rot == 1:
                y, x, z = next_box
            elif rot == 2:
                z, x, y = next_box
            elif rot == 3:
                z, y, x = next_box
            elif rot == 4:
                x, z, y = next_box
            elif rot == 5:
                y, z, x = next_box

            if ems[0] + x > env.space.plain.shape[0] or ems[1] + y > env.space.plain.shape[1]:
                continue

            # Check the feasibility of this placement
            if env.space.drop_box_virtual([x, y, z], (ems[0], ems[1]), False, next_den, env.setting):
                env.next_box = [x, y, z]
                rotation_flag = rot
                bestAction = [0, ems[0], ems[1]]
                stop = True
                break
        if stop: break

    if bestAction is not None:
        # Place this item in the environment with the best action.
        rec = env.space.plain[bestAction[1]:bestAction[1] + env.next_box[0], bestAction[2]:bestAction[2] + env.next_box[1]]
        lz = np.max(rec)
        _, _, done, _ = env.step(bestAction)
        bestAction[0] = rotation_flag
    else:
        # No feasible placement, this episode is done.
        lz = None
        done = True
        env.box_creator.drop_box()  # remove current box from the list
        env.box_creator.generate_box_size()  # add a new box to the list
        env.cur_observation()

    return done, bestAction, lz


def DBL(env, times = 2000):
    '''
    A Hybrid Genetic Algorithm for Packing in 3D with Deepest Bottom Left with Fill Method
    https://doi.org/10.1007/978-3-540-30198-1\_45
    '''
    bin_size = env.bin_size
    rotation_flag = 0

    bestScore = 1e10
    bestAction = []

    next_box = env.next_box
    next_den = env.next_den

    for lx in range(bin_size[0] - next_box[0] + 1):
        for ly in range(bin_size[1] - next_box[1] + 1):
            # Find the most suitable placement within the allowed orientation.
            for rot in range(env.orientation):
                if rot == 0:
                    x, y, z = next_box
                elif rot == 1:
                    y, x, z = next_box
                elif rot == 2:
                    z, x, y = next_box
                elif rot == 3:
                    z, y, x = next_box
                elif rot == 4:
                    x, z, y = next_box
                elif rot == 5:
                    y, z, x = next_box

                # Check the feasibility of this placement
                feasible, height = env.space.drop_box_virtual([x, y, z], (lx, ly), False,
                                                                    next_den, env.setting, True, False)
                if not feasible:
                    continue

                # Score the given placement.
                score = lx + ly + 100 * height
                if score < bestScore:
                    bestScore = score
                    env.next_box = [x, y, z]
                    rotation_flag = rot
                    bestAction = [0, lx, ly]

    if len(bestAction) != 0:
        # Place this item in the environment with the best action.
        rec = env.space.plain[bestAction[1]:bestAction[1] + env.next_box[0], bestAction[2]:bestAction[2] + env.next_box[1]]
        lz = np.max(rec)
        env.step(bestAction)
        bestAction[0] = rotation_flag
        done = False
    else:
        # No feasible placement, this episode is done.
        lz = None
        done = True
        env.box_creator.drop_box()  # remove current box from the list
        env.box_creator.generate_box_size()  # add a new box to the list
        env.cur_observation()

    return done, bestAction, lz


def BR(env, times = 2000):
    '''
    Online 3D Bin Packing with Constrained Deep Reinforcement Learning
    https://ojs.aaai.org/index.php/AAAI/article/view/16155
    '''
    def eval_ems(ems):
        # Score the given placement.
        s = 0
        valid = []
        for bs in env.item_set:
            bx, by, bz = bs
            if ems[3] - ems[0] >= bx and ems[4] - ems[1] >= by and ems[5] - ems[2] >= bz:
                valid.append(1)
        s += (ems[3] - ems[0]) * (ems[4] - ems[1]) * (ems[5] - ems[2])
        s += len(valid)
        if len(valid) == len(env.item_set):
            s += 10
        return s
            

    bestScore = -1e10
    rotation_flag = 0
    EMS = env.space.EMS

    bestAction = None
    next_box = env.next_box
    next_den = env.next_den

    for ems in EMS:
        # Find the most suitable placement within the allowed orientation.
        for rot in range(env.orientation):
            if rot == 0:
                x, y, z = next_box
            elif rot == 1:
                y, x, z = next_box
            elif rot == 2:
                z, x, y = next_box
            elif rot == 3:
                z, y, x = next_box
            elif rot == 4:
                x, z, y = next_box
            elif rot == 5:
                y, z, x = next_box

            if ems[3] - ems[0] >= x and ems[4] - ems[1] >= y and ems[5] - ems[2] >= z:
                lx, ly = ems[0], ems[1]
                # Check the feasibility of this placement
                feasible, height = env.space.drop_box_virtual([x, y, z], (lx, ly), False,
                                                                    next_den, env.setting, returnH=True)
                if feasible:
                    score = eval_ems(ems)
                    if score > bestScore:
                        bestScore = score
                        env.next_box = [x, y, z]
                        rotation_flag = rot
                        bestAction = [0, lx, ly, height]


    if bestAction is not None:
        # Place this item in the environment with the best action.
        rec = env.space.plain[bestAction[1]:bestAction[1] + env.next_box[0], bestAction[2]:bestAction[2] + env.next_box[1]]
        lz = np.max(rec)
        _, _, done, _ = env.step(bestAction[0:3])
        bestAction[0] = rotation_flag
    else:
        # No feasible placement, this episode is done.
        lz = None
        done = True
        env.box_creator.drop_box()  # remove current box from the list
        env.box_creator.generate_box_size()  # add a new box to the list
        env.cur_observation()

    return  done, bestAction, lz



def pct(PCT_policy, env, obs, args, eval_freq = 100, factor = 1):
    
    # obs = env.cur_observation()
    # obs = torch.FloatTensor(obs).unsqueeze(dim=0)

    all_nodes, leaf_nodes = get_leaf_nodes_with_factor(obs, 1, args.internal_node_holder, args.leaf_node_holder)
    batchX = torch.arange(1)

    with torch.no_grad():
        selectedlogProb, selectedIdx, policy_dist_entropy, value = PCT_policy(all_nodes, True, normFactor=factor)
    selected_leaf_node = leaf_nodes[batchX, selectedIdx.squeeze()]      # tensor([[ 8.,  0.,  0., 10.,  5., 10.,  0.,  0.,  1.]], device='cuda:0')
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


def sdf_pack_regular(env):
    bin_size = env.bin_size
    bestScore = 1e10
    bestAction = []
    rotation_flag = 0

    next_box = env.next_box
    next_den = env.next_den


    for rot in range(env.orientation):
        # Find the most suitable placement within the allowed orientation.
        if rot == 0:
            x, y, z = next_box
        elif rot == 1:
            y, x, z = next_box
        now_size = x, y, z

        step = 2
        for lx in range(0, bin_size[0] - x + 1, step):
            for ly in range(0, bin_size[1] - y + 1, step):
                # Check the feasibility of this placement
                feasible = env.space.drop_box_virtual([x, y, z], (lx, ly), False, next_den, env.setting, False, False)
                if not feasible:
                    continue

                # get real Z
                rec = env.space.plain[lx: lx + x, ly: ly + y]
                posZ = np.max(rec)

                position = [lx, ly, posZ]
                z_score = posZ * env.block_unit
                # get tsdf
                tsdf = get_TSDF_v2(env, position, now_size)

                if tsdf == None:
                    continue

                # get rot_score
                # rot_score = 1 - ((np.sum(contains)) / (x * y * z)) ** (1 / 3)
                rot_score = 1 - ((x * y * z) / (x * y * z)) ** (1 / 3)

                # Score the given placement.
                score = tsdf + rot_score + z_score
                if score < bestScore:
                    bestScore = score
                    env.next_box = [x, y, z]
                    rotation_flag = rot
                    bestAction = [0, lx, ly]
                    bestZpose = posZ

    if len(bestAction) != 0:
        lz = bestZpose
        # Place this item in the environment with the best action.
        env.step(bestAction)
        bestAction[0] = rotation_flag
        done = False
    else:
        # No feasible placement, this episode is done.
        lz = None
        done = True
        env.box_creator.drop_box()  # remove current box from the list
        env.box_creator.generate_box_size()  # add a new box to the list
        env.cur_observation()

    return done, bestAction, lz


def sdf_pack(env, scene, heightmapB, maskB, contains):
    bin_size = env.bin_size
    bestScore = 1e10
    bestAction = []
    rotation_flag = 0

    next_box = env.next_box
    next_den = env.next_den

    # 如果contains值均为0，则说明该物体为极小或极细物品，将contains全赋值为1
    if np.all(contains == 0):
        contains[:] = 1

    for rot in range(env.orientation):  
        # Find the most suitable placement within the allowed orientation.
        if rot == 0:
            x, y, z = next_box
        elif rot == 1:
            y, x, z = next_box
            # 旋转 heightmapB
            heightmapB = np.transpose(heightmapB)
            # 旋转 maskB
            maskB = np.transpose(maskB)
            # 旋转 contains
            # contains = np.transpose(contains, (1, 2, 0))
            contains = np.transpose(contains, (1, 0, 2))
        for lx in range(bin_size[0] - x + 1):
            for ly in range(bin_size[1] - y + 1):
                # Check the feasibility of this placement
                feasible = env.space.drop_box_virtual([x, y, z], (lx, ly), False, next_den, env.setting, False, False)
                if not feasible:
                    continue
                
                # get real Z
                posZ = np.max((env.space.plain[lx: lx + x, ly: ly + y] - heightmapB) * maskB)
                position = [lx, ly, posZ]
                z_score = posZ * scene.block_unit
                # get tsdf
                tsdf = get_TSDF_v2(env, position, contains, scene)
                if tsdf == None:
                    continue

                # get rot_score
                rot_score = 1 - ( (np.sum(contains)) / (x * y * z) ) ** (1/3)

                # Score the given placement.
                score = tsdf + rot_score + z_score
                if score < bestScore:
                    bestScore = score
                    env.next_box = [x, y, z]
                    rotation_flag = rot
                    bestAction = [0, lx, ly]
                    bestZpose = posZ


    if len(bestAction) != 0:
        lz = bestZpose
        # Place this item in the environment with the best action.
        env.step(bestAction)
        bestAction[0] = rotation_flag
        done = False
    else:
        # No feasible placement, this episode is done.
        lz = None
        done = True
        env.box_creator.drop_box()  # remove current box from the list
        env.box_creator.generate_box_size()  # add a new box to the list
        env.cur_observation()

    return  done, bestAction, lz



def MTPE(env, scene, heightmapB, maskB, contains):
    bin_size = env.bin_size
    bestScore = 1e10
    bestAction = []
    rotation_flag = 0

    next_box = env.next_box
    next_den = env.next_den

    for rot in range(env.orientation):  
        # Find the most suitable placement within the allowed orientation.
        if rot == 0:
            x, y, z = next_box
        elif rot == 1:
            y, x, z = next_box
            # 旋转 heightmapB
            heightmapB = np.transpose(heightmapB)
            # 旋转 maskB
            maskB = np.transpose(maskB)
            # 旋转 contains
            contains = np.transpose(contains, (1, 2, 0))
        for lx in range(bin_size[0] - x + 1):
            for ly in range(bin_size[1] - y + 1):
                # Check the feasibility of this placement
                feasible = env.space.drop_box_virtual([x, y, z], (lx, ly), False, next_den, env.setting, False, False)
                if not feasible:
                    continue
                
                # get real Z
                posZ = np.max((env.space.plain[lx: lx + x, ly: ly + y] - heightmapB) * maskB)

                # Score the given placement.
                score = posZ
                if score < bestScore:
                    bestScore = score
                    env.next_box = [x, y, z]
                    rotation_flag = rot
                    bestAction = [0, lx, ly]
                    bestZpose = posZ


    if len(bestAction) != 0:
        lz = bestZpose
        # Place this item in the environment with the best action.
        env.step(bestAction)
        bestAction[0] = rotation_flag
        done = False
    else:
        # No feasible placement, this episode is done.
        lz = None
        done = True
        env.box_creator.drop_box()  # remove current box from the list
        env.box_creator.generate_box_size()  # add a new box to the list
        env.cur_observation()

    return  done, bestAction, lz



def BLBF(env, scene, heightmapB, maskB, contains):
    bin_size = env.bin_size
    bestAction = []
    bestScore = 1e10
    rotation_flag = 0

    next_box = env.next_box
    next_den = env.next_den

    for rot in range(env.orientation):  
        # Find the most suitable placement within the allowed orientation.
        if rot == 0:
            x, y, z = next_box
        elif rot == 1:
            y, x, z = next_box
            # 旋转 heightmapB
            heightmapB = np.transpose(heightmapB)
            # 旋转 maskB
            maskB = np.transpose(maskB)
            # 旋转 contains
            contains = np.transpose(contains, (1, 2, 0))
        for lx in range(bin_size[0] - x + 1):
            for ly in range(bin_size[1] - y + 1):
                # Check the feasibility of this placement
                feasible = env.space.drop_box_virtual([x, y, z], (lx, ly), False, next_den, env.setting, False, False)
                if not feasible:
                    continue
                # get real Z
                posZ = np.max((env.space.plain[lx: lx + x, ly: ly + y] - heightmapB) * maskB)
                score = posZ * 100 + lx * 10 + ly * 1
                
                if score < bestScore:
                    bestScore = score
                    env.next_box = [x, y, z]
                    rotation_flag = rot
                    bestAction = [0, lx, ly]
                    bestZpose = posZ


    if len(bestAction) != 0:
        lz = bestZpose
        # Place this item in the environment with the best action.
        env.step(bestAction)
        bestAction[0] = rotation_flag
        done = False
    else:
        # No feasible placement, this episode is done.
        lz = None
        done = True
        env.box_creator.drop_box()  # remove current box from the list
        env.box_creator.generate_box_size()  # add a new box to the list
        env.cur_observation()

    return  done, bestAction, lz



def FF(env, scene, heightmapB, maskB, contains):
    bin_size = env.bin_size
    bestAction = []
    rotation_flag = 0

    next_box = env.next_box
    next_den = env.next_den

    for rot in range(env.orientation):  
        # Find the most suitable placement within the allowed orientation.
        if rot == 0:
            x, y, z = next_box
        elif rot == 1:
            y, x, z = next_box
            # 旋转 heightmapB
            heightmapB = np.transpose(heightmapB)
            # 旋转 maskB
            maskB = np.transpose(maskB)
            # 旋转 contains
            contains = np.transpose(contains, (1, 2, 0))
        for lx in range(bin_size[0] - x + 1):
            for ly in range(bin_size[1] - y + 1):
                # Check the feasibility of this placement
                feasible = env.space.drop_box_virtual([x, y, z], (lx, ly), False, next_den, env.setting, False, False)
                # get real Z
                posZ = np.max((env.space.plain[lx: lx + x, ly: ly + y] - heightmapB) * maskB)

                if feasible:
                    env.next_box = [x, y, z]
                    rotation_flag = rot
                    bestAction = [0, lx, ly]
                    bestZpose = posZ
                    break
            if feasible:
                break
        if feasible:
            break

    if len(bestAction) != 0:
        lz = bestZpose
        # Place this item in the environment with the best action.
        env.step(bestAction)
        bestAction[0] = rotation_flag
        done = False
    else:
        # No feasible placement, this episode is done.
        lz = None
        done = True
        env.box_creator.drop_box()  # remove current box from the list
        env.box_creator.generate_box_size()  # add a new box to the list
        env.cur_observation()

    return  done, bestAction, lz



class Packer():
    def __init__(self, container_size) -> None:
        self.container = Container(container_size)
    
    def pack_box(self, env, infos, obs, method, policy):
        # LSAH   MACS   RANDOM  OnlineBPH   DBL  BR
        
        if method == 'LeftBottom':
            start_time = time.time()
            done, action, lz = left_bottom(env)
            end_time = time.time()
            planning_time = end_time - start_time

            placeable = not done
            if placeable == False:
                return placeable, [], None, infos, planning_time
            rotation_flag, lx, ly = action[0], action[1], action[2]
            pos = np.array([lx, ly, lz], dtype=np.float32)
            infos['next_obs'] = None
            placeable = True
            
        elif method == 'HeightmapMin':
            start_time = time.time()
            done, action, lz = heightmap_min(env)
            end_time = time.time()
            planning_time = end_time - start_time

            placeable = not done
            if placeable == False:
                return placeable, [], None, infos, planning_time
            rotation_flag, lx, ly = action[0], action[1], action[2]
            pos = np.array([lx, ly, lz], dtype=np.float32)
            infos['next_obs'] = None
            placeable = True
        
        elif method == 'LSAH':
            if len(env.packed) == 0:
                # 初始化
                maxXY = [0,0]
                minXY = [env.bin_size[0], env.bin_size[1]]
            else:
                maxXY = infos['maxXY']
                minXY = infos['minXY']
            start_time = time.time()
            done, action, lz, new_maxXY, new_minXY = LASH(env, maxXY, minXY)
            end_time = time.time()
            planning_time = end_time - start_time
            
            infos['maxXY'] = new_maxXY
            infos['minXY'] = new_minXY
            infos['next_obs'] = None
            
            placeable = not done
            if placeable == False:
                return placeable, [], None, infos, planning_time
            rotation_flag, lx, ly = action[0], action[1], action[2]
            pos = np.array([lx, ly, lz], dtype=np.float32)

            placeable = True
        
        elif method == 'MACS':
            block_index = infos['block_index']
            if len(env.packed) == 0:
                # 初始化
                container = np.zeros(env.bin_size)
            else:
                container = infos['container']
            start_time = time.time()
            done, action, lz, new_container = MACS(env, block_index, container)
            end_time = time.time()
            planning_time = end_time - start_time
            
            infos['container'] = new_container
            infos['next_obs'] = None
            
            placeable = not done
            if placeable == False:
                return placeable, [], None, infos, planning_time
            rotation_flag, lx, ly = action[0], action[1], action[2]
            pos = np.array([lx, ly, lz], dtype=np.float32)

            placeable = True
            
        elif method == 'RANDOM':
            start_time = time.time()
            done, action, lz = RANDOM(env)
            end_time = time.time()
            planning_time = end_time - start_time

            infos['next_obs'] = None
            
            placeable = not done
            if placeable == False:
                return placeable, [], None, infos, planning_time
            rotation_flag, lx, ly = action[0], action[1], action[2]
            pos = np.array([lx, ly, lz], dtype=np.float32)

            placeable = True
            
        elif method == 'OnlineBPH':
            start_time = time.time()
            done, action, lz = OnlineBPH(env)
            end_time = time.time()
            planning_time = end_time - start_time

            infos['next_obs'] = None
            
            placeable = not done
            if placeable == False:
                return placeable, [], None, infos, planning_time
            rotation_flag, lx, ly = action[0], action[1], action[2]
            pos = np.array([lx, ly, lz], dtype=np.float32)

            placeable = True
            
        elif method == 'DBL':
            start_time = time.time()
            done, action, lz = DBL(env)
            end_time = time.time()
            planning_time = end_time - start_time

            infos['next_obs'] = None
            
            placeable = not done
            if placeable == False:
                return placeable, [], None, infos, planning_time
            rotation_flag, lx, ly = action[0], action[1], action[2]
            pos = np.array([lx, ly, lz], dtype=np.float32)

            placeable = True
            
        elif method == 'BR':
            start_time = time.time()
            done, action, lz = BR(env)
            end_time = time.time()
            planning_time = end_time - start_time

            infos['next_obs'] = None
            
            placeable = not done
            if placeable == False:
                return placeable, [], None, infos, planning_time
            rotation_flag, lx, ly = action[0], action[1], action[2]
            pos = np.array([lx, ly, lz], dtype=np.float32)

            placeable = True


        elif method == 'SDFPack':
            # heightmapB, maskB, contains = get_heightmapB(num, env, scene, pack_init_sizes)
            start_time = time.time()
            done, action, lz = sdf_pack_regular(env)
            end_time = time.time()
            planning_time = end_time - start_time

            infos['next_obs'] = None

            placeable = not done
            if placeable == False:
                return placeable, [], None, infos, planning_time
            rotation_flag, lx, ly = action[0], action[1], action[2]
            pos = np.array([lx, ly, lz], dtype=np.float32)

            placeable = True

           
        elif method == 'PCT' or method == 'PackE':
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
    
        return placeable, pos, rotation_flag, infos, planning_time

    
    def pack_irregular_box(self, num, env, scene, pack_init_sizes, infos, obs, method, policy):
        if method == 'IR_HM':
            start_time = time.time()
            done, action, lz = heightmap_min(env)
            end_time = time.time()
            planning_time = end_time - start_time

            placeable = not done
            if placeable == False:
                return placeable, [], [], infos
            rotation_flag, lx, ly = action[0], action[1], action[2]
            pos = np.array([lx, ly, lz], dtype=np.float32)
            infos['next_obs'] = None
            placeable = True
            
        elif method == 'SDF_Pack':
            heightmapB, maskB, contains = get_heightmapB(num, env, scene, pack_init_sizes)
            start_time = time.time()
            done, action, lz = sdf_pack(env, scene, heightmapB, maskB, contains)
            end_time = time.time()
            planning_time = end_time - start_time

            placeable = not done
            if placeable == False:
                return placeable, [], [], infos
            rotation_flag, lx, ly = action[0], action[1], action[2]
            pos = np.array([lx, ly, lz], dtype=np.float32)
            infos['next_obs'] = None
            placeable = True
        
        elif method == 'MTPE':
            heightmapB, maskB, contains = get_heightmapB(num, env, scene, pack_init_sizes)
            start_time = time.time()
            done, action, lz = MTPE(env, scene, heightmapB, maskB, contains)
            end_time = time.time()
            planning_time = end_time - start_time

            placeable = not done
            if placeable == False:
                return placeable, [], [], infos
            rotation_flag, lx, ly = action[0], action[1], action[2]
            pos = np.array([lx, ly, lz], dtype=np.float32)
            infos['next_obs'] = None
            placeable = True
        
        elif method == 'BLBF':
            heightmapB, maskB, contains = get_heightmapB(num, env, scene, pack_init_sizes)
            start_time = time.time()
            done, action, lz = BLBF(env, scene, heightmapB, maskB, contains)
            end_time = time.time()
            planning_time = end_time - start_time

            placeable = not done
            if placeable == False:
                return placeable, [], [], infos
            rotation_flag, lx, ly = action[0], action[1], action[2]
            pos = np.array([lx, ly, lz], dtype=np.float32)
            infos['next_obs'] = None
            placeable = True
        
        elif method == 'FF':
            heightmapB, maskB, contains = get_heightmapB(num, env, scene, pack_init_sizes)
            start_time = time.time()
            done, action, lz = FF(env, scene, heightmapB, maskB, contains)
            end_time = time.time()
            planning_time = end_time - start_time

            placeable = not done
            if placeable == False:
                return placeable, [], [], infos
            rotation_flag, lx, ly = action[0], action[1], action[2]
            pos = np.array([lx, ly, lz], dtype=np.float32)
            infos['next_obs'] = None
            placeable = True
              
        elif method == 'IR_BPP':
            dqn = policy
            args = infos['args']
            state = obs
            state = torch.FloatTensor(state).reshape((1, -1)).to(args.device)
            mask = get_mask_from_state(state, args, args.bufferSize)

            start_time = time.time()
            action = dqn.act_e_greedy(state, mask, -1)
            end_time = time.time()
            planning_time = end_time - start_time

            rotIdx, lx, ly = env.candidates[action.item()][0:3].astype(np.int)
            rotation = env.transformation[int(rotIdx)]
            if np.array_equal(rotation, np.array([0., 0., 0., 1.])):
                rotation_flag = 0
            else:
                rotation_flag = 1

            bestAction = [0, lx, ly]

            env.step(bestAction)  # Step
            bestAction[0] = rotation_flag

            heightmapB, maskB, contains = get_heightmapB(num, env, scene, pack_init_sizes)
            size = pack_init_sizes[num]
            posZ = np.max((env.space.plain[lx: lx + size[0], ly: ly + size[1]] - heightmapB) * maskB)
            pos = np.array([lx, ly, posZ], dtype=np.float32)
            infos['next_obs'] = env.ir_cur_observation(scene)
            placeable = True

        elif method == 'pack_2d':
            
            packer2d = infos['packer2d']
            name = scene.box_names[num]
            # img_path = '/home/wzf/Workspace/rl/RobotPackingBenchmark-main-v2/dataset/pack2d/test_fit/' + name + '.png'
            # img_path = scene.img_path_2d + '/' + name + '.png'
            # img = cv2.imread(img_path)
            obj_path = scene.objPath_2d + '/' + name + '.obj'
            poly = get_poly(obj_path)

            start_time = time.time()
            ret, obj, rot = packer2d.calculate_location(container_ids = [[1]], poly = poly,
                    reach_area = [[0, 0], [750, 750]], obj_gap = 20, obj_shape = 2, check_collision = 1, rotate_angle = None, balance = 0, search_next = 0, img_path="./images")
            end_time = time.time()
            planning_time = end_time - start_time

            infos['packer2d'] = packer2d
            infos['next_obs'] = None
            
            if ret['code'] == 200:
                placeable = True
            else:
                placeable = False
                return placeable, [], [], infos
            
            scale = scene.scale_2d[0]
            lx, ly, lz = obj.obj_center[0] * scale, obj.obj_center[1] * scale, 0
            pos = np.array([lx, ly, lz], dtype=np.float32)
            if rot == 0:
                rotation_flag = 0
            elif rot == 90:
                rotation_flag = 1
            elif rot == 180:
                rotation_flag = 2
            elif rot == 270:
                rotation_flag = 3

        return placeable, pos, rotation_flag, infos, planning_time

    
    def add_box(self, box, pos):
        pos = self.container.add_new_box(box, pos)
        add_success = True
        if pos is None:
            add_success = False
        
        return add_success


