# code from https://github.com/Juzhan/TAP-Net.git


import os
import numpy as np

import random
import itertools
from scipy.spatial import ConvexHull

import time
from matplotlib.path import Path
def is_stable_2d(support, obj_left, obj_width):
    '''
    check if the obj is stable
    ---
    params:
    ---
        support: obj_width x 1 array / list / tnesor, the container data under obj
        obj_left: a float number, left index of obj's position (x of obj.position)
        obj_width: a float number, width of obj
    return:
    ---
        is_stable: bool, stable or not
    '''
    object_center = obj_left + obj_width/2
    left_index = obj_left
    right_index = obj_left + obj_width
    for left in support:
        if left == 0:
            left_index += 1
        else:
            break
    for right in reversed(support):
        if right == 0:
            right_index -= 1
        else:
            break
    # 如果物体中心线不在范围内，不稳定
    if object_center <= left_index or object_center >= right_index:
        return False
    return True

def is_stable(block, position, container):
    '''
    check for 3D packing
    ----
    '''
    if (position[2]==0):
        return True
    x_1 = position[0]
    x_2 = x_1 + block[0] - 1
    y_1 = position[1]
    y_2 = y_1 + block[1] - 1
    z = position[2] - 1
    obj_center = ( (x_1+x_2)/2, (y_1+y_2)/2 )

    # valid points right under this object
    points = []
    for x in range(x_1, x_2+1):
        for y in range(y_1, y_2+1):
            if (container[x][y][z] > 0):
                points.append([x, y])
    if(len(points) > block[0]*block[1]/2):
        return True
    if(len(points)==0 or len(points)==1):
        return False
    elif(len(points)==2):
        # whether the center lies on the line of the two points
        a = obj_center[0] - points[0][0]
        b = obj_center[1] - points[0][1]
        c = obj_center[0] - points[1][0]
        d = obj_center[1] - points[1][1]
        # same ratio and opposite signs
        if (b==0 or d==0):
            if (b!=d): return False
            else: return (a<0)!=(c<0) 
        return ( a/b == c/d and (a<0)!=(c<0) and (b<0)!=(d<0) )
    else:
        # calculate the convex hull of the points
        points = np.array(points)
        try:
            convex_hull = ConvexHull(points)
        except:
            # error means co-lines
            min_point = points[np.argmin( points[:,0] )]
            max_point = points[np.argmax( points[:,0] )]
            points = np.array( (min_point, max_point) )
            a = obj_center[0] - points[0][0]
            b = obj_center[1] - points[0][1]
            c = obj_center[0] - points[1][0]
            d = obj_center[1] - points[1][1]
            if (b==0 or d==0):
                if (b!=d): return False
                else: return (a<0)!=(c<0)
            return ( a/b == c/d and (a<0)!=(c<0) and (b<0)!=(d<0) )

        hull_path = Path(points[convex_hull.vertices])
        return hull_path.contains_point((obj_center))

# NOTE choose from Left-Bottom corners by greedy strategy

def calc_one_position_lb_greedy_2d(block, block_index, container_size, reward_type, 
                                container, positions, stable, heightmap, valid_size, empty_size):
    """
    calculate the latest block's position in the container by lb-greedy in 2D cases
    ---
    params:
    ---
        static params:
            block: int * 2 array, size of the block to pack
            block_index: int, index of the block to pack, previous were already packed
            container_size: 1 x 2 array, size of the container
            reward_type: string, options:
                'C+P-lb-soft'
                'C+P-lb-hard'
                'C+P+S-lb-soft'
                'C+P+S-lb-hard'
        dynamic params:
            container: width * height array, the container state
            positions: int * 2 array, coordinates of the blocks, [0, 0] for blocks after block_index
            stable: n * 1 bool list, the blocks' stability state
            heightmap: width * 1 array, heightmap of the container
            valid_size: int, sum of the packed blocks' size
            empty_size: int, size of the empty space under packed blocks
    return:
    ---
        container: width * height array, updated container
        positions: int * 2 array, updated positions
        stable: n * 1 bool list, updated stable
        heightmap: width * 1 array, updated heightmap
        valid_size: updated valid_size
        empty_size: updated empty_size
    """
    block_dim = len(block)
    block_x, block_z = block
    valid_size += block_x * block_z

    # get empty-maximal-spaces list from heightmap
    # each ems represented as a left-bottom corner
    ems_list = []
    # hm_diff: height differences of neightbor columns, padding 0 in the front
    hm_diff = heightmap.copy()
    hm_diff = np.insert(hm_diff, 0, hm_diff[0])
    hm_diff = np.delete(hm_diff, len(hm_diff)-1)
    hm_diff = heightmap - hm_diff
    # get the x coordinates of all left-bottom corners
    ems_x_list = np.nonzero(hm_diff)
    ems_x_list = np.insert(ems_x_list, 0, 0)
    # get ems_list
    for x in ems_x_list:
        if x+block_x > container_size[0]: break
        z = np.max( heightmap[x:x+block_x] )
        ems_list.append( [x, z] )
    # firt consider the most bottom, then left
    def bottom_first(pos): return pos[1]
    ems_list.sort(key=bottom_first, reverse=False)

    # if no ems found
    if len(ems_list) == 0:
        valid_size -= block_x * block_z
        stable[block_index] = False
        return container, positions, stable, heightmap, valid_size, empty_size

    # varients to store results of searching ems corners
    ems_num = len(ems_list)
    pos_ems = np.zeros((ems_num, block_dim)).astype(int)
    is_settle_ems  = [False] * ems_num
    is_stable_ems  = [False] * ems_num
    compactness_ems  = [0.0] * ems_num
    pyramidality_ems = [0.0] * ems_num
    stability_ems    = [0.0] * ems_num
    empty_ems = [empty_size] * ems_num
    under_space_mask  = [[]] * ems_num
    heightmap_ems = [np.zeros(container_size[:-1]).astype(int)] * ems_num
    visited = []

    # check if a position suitable
    def check_position(index, _x, _z):
        # check if the pos visited
        if [_x, _z] in visited: return
        if _z>0 and (container[_x:_x+block_x, _z-1]==0).all(): return
        visited.append([_x, _z])
        # if (container[_x:_x+block_x, _z:_z+block_z] == 0).all():
        if (container[_x:_x+block_x, _z] == 0).all():
            if _z > 0:
                support = container[_x:_x+block_x, _z-1]
                if not is_stable_2d(support, _x, block_x):
                    if reward_type.endswith('hard'):
                        return
                else:
                    is_stable_ems[index] = True
            else:
                is_stable_ems[index] = True
            pos_ems[index] = np.array([_x, _z])
            heightmap_ems[index][_x:_x+block_x] = _z + block_z
            is_settle_ems[index] = True

    # calculate socres
    def calc_C_P_S(index):
        _x, _z = pos_ems[index]
        # compactness
        height = np.max(heightmap_ems[index])
        # if _z+block_x > height: height = _z+block_z
        bbox_size = height * container_size[0]
        compactness_ems[index] = valid_size / bbox_size
        # pyramidality
        under_space = container[_x:_x+block_x, 0:_z]
        under_space_mask[index] = under_space==0
        empty_ems[index] += np.sum(under_space_mask[index])
        if 'P' in reward_type:
            pyramidality_ems[index] = valid_size / (empty_ems[index] + valid_size)
        # stability
        if 'S' in reward_type:
            stable_num = np.sum(stable[:block_index]) + np.sum(is_stable_ems[index])
            stability_ems[index] = stable_num / (block_index + 1)

    # search positions in each ems
    X = int(container_size[0] - block_x + 1)
    for ems_index, ems in enumerate(ems_list):
        # using buttom-left strategy in each ems
        heightmap_ems[ems_index] = heightmap.copy()
        _z = int(ems[-1])
        for _x  in range( int(ems[0]), X ):
            if is_settle_ems[ems_index]: break
            check_position(ems_index, _x, _z)
        if is_settle_ems[ems_index]: 
            calc_C_P_S(ems_index)

    # if the block has not been settled
    if np.sum(is_settle_ems) == 0:
        valid_size -= block_x * block_z
        stable[block_index] = False
        return container, positions, stable, heightmap, valid_size, empty_size

    # get the best ems
    ratio_ems = [c+p+s for c, p, s in zip(compactness_ems, pyramidality_ems, stability_ems)]
    best_ems_index = np.argmax(ratio_ems)
    while not is_settle_ems[best_ems_index]:
        ratio_ems.remove(ratio_ems[best_ems_index])
        best_ems_index = np.argmax(ratio_ems)

    # update the dynamic parameters
    _x, _z = pos_ems[best_ems_index]
    container[_x:_x+block_x, _z:_z+block_z] = block_index + 1
    container[_x:_x+block_x, 0:_z][ under_space_mask[best_ems_index] ] = -1
    positions[block_index] = pos_ems[best_ems_index]
    stable[block_index] = is_stable_ems[best_ems_index]
    heightmap = heightmap_ems[best_ems_index]
    empty_size = empty_ems[best_ems_index]

    return container, positions, stable, heightmap, valid_size, empty_size

def calc_one_position_lb_greedy_3d(block, block_index, container_size, reward_type, 
                                container, positions, stable, heightmap, valid_size, empty_size):
    """
    calculate the latest block's position in the container by lb-greedy in 2D cases
    ---
    params:
    ---
        static params:
            block: int * 3 array, size of the block to pack
            block_index: int, index of the block to pack, previous were already packed
            container_size: 1 x 3 array, size of the container
            reward_type: string, options:
                'C+P-lb-soft'
                'C+P-lb-hard'
                'C+P+S-lb-soft'
                'C+P+S-lb-hard'
        dynamic params:
            container: width * length * height array, the container state
            positions: int * 3 array, coordinates of the blocks, [0, 0] for blocks after block_index
            stable: n * 1 bool list, the blocks' stability state
            heightmap: width * length array, heightmap of the container
            valid_size: int, sum of the packed blocks' size
            empty_size: int, size of the empty space under packed blocks
    return:
    ---
        container: width * length * height array, updated container
        positions: int * 3 array, updated positions
        stable: n * 1 bool list, updated stable
        heightmap: width * length array, updated heightmap
        valid_size: int, updated valid_size
        empty_size: int, updated empty_size
    """
    block_dim = len(block)
    block_x, block_y, block_z = block
    valid_size += block_x * block_y * block_z

    # get empty-maximal-spaces list from heightmap
    # each ems represented as a left-bottom corner
    ems_list = []
    # hm_diff: height differences of neightbor columns, padding 0 in the front
    # x coordinate
    hm_diff_x = np.insert(heightmap, 0, heightmap[0, :], axis=0)
    hm_diff_x = np.delete(hm_diff_x, len(hm_diff_x)-1, axis=0)
    hm_diff_x = heightmap - hm_diff_x
    # y coordinate
    hm_diff_y = np.insert(heightmap, 0, heightmap[:, 0], axis=1)
    hm_diff_y = np.delete(hm_diff_y, len(hm_diff_y.T)-1, axis=1)
    hm_diff_y = heightmap - hm_diff_y

    # get the xy coordinates of all left-deep-bottom corners
    ems_x_list = np.array(np.nonzero(hm_diff_x)).T.tolist()
    ems_y_list = np.array(np.nonzero(hm_diff_y)).T.tolist()
    ems_xy_list = []
    ems_xy_list.append([0,0])
    for xy in ems_x_list:
        x, y = xy
        if y!=0 and [x, y-1] in ems_x_list:
            if heightmap[x, y] == heightmap[x, y-1] and \
                hm_diff_x[x, y] == hm_diff_x[x, y-1]:
                continue
        ems_xy_list.append(xy)
    for xy in ems_y_list:
        x, y = xy
        if x!=0 and [x-1, y] in ems_y_list:
            if heightmap[x, y] == heightmap[x-1, y] and \
                hm_diff_x[x, y] == hm_diff_x[x-1, y]:
                continue
        if xy not in ems_xy_list:
            ems_xy_list.append(xy)

    # sort by y coordinate, then x
    def y_first(pos): return pos[1]
    ems_xy_list.sort(key=y_first, reverse=False)

    # get ems_list
    for xy in ems_xy_list:
        x, y = xy
        if x+block_x > container_size[0] or \
            y+block_y > container_size[1]: continue
        z = np.max( heightmap[x:x+block_x, y:y+block_y] )
        ems_list.append( [ x, y, z ] )
    
    # firt consider the most bottom, sort by z coordinate, then y last x
    def z_first(pos): return pos[2]
    ems_list.sort(key=z_first, reverse=False)

    # if no ems found
    if len(ems_list) == 0:
        valid_size -= block_x * block_y * block_z
        stable[block_index] = False
        return container, positions, stable, heightmap, valid_size, empty_size

    # varients to store results of searching ems corners
    ems_num = len(ems_list)
    pos_ems = np.zeros((ems_num, block_dim)).astype(int)
    is_settle_ems  = [False] * ems_num
    is_stable_ems  = [False] * ems_num
    compactness_ems  = [0.0] * ems_num
    pyramidality_ems = [0.0] * ems_num
    stability_ems    = [0.0] * ems_num
    empty_ems = [empty_size] * ems_num
    under_space_mask  = [[]] * ems_num
    heightmap_ems = [np.zeros(container_size[:-1]).astype(int)] * ems_num
    visited = []

    # check if a position suitable
    def check_position(index, _x, _y, _z):
        # check if the pos visited
        if [_x, _y, _z] in visited: return
        if _z>0 and (container[_x:_x+block_x, _y:_y+block_y, _z-1]==0).all(): return
        visited.append([_x, _y, _z])
        if (container[_x:_x+block_x, _y:_y+block_y, _z] == 0).all():
            if not is_stable(block, np.array([_x, _y, _z]), container):
                if reward_type.endswith('hard'):
                    return
            else:
                is_stable_ems[index] = True
            pos_ems[index] = np.array([_x, _y, _z])
            heightmap_ems[index][_x:_x+block_x, _y:_y+block_y] = _z + block_z
            is_settle_ems[index] = True

    # calculate socres
    def calc_C_P_S(index):
        _x, _y, _z = pos_ems[index]
        # compactness
        height = np.max(heightmap_ems[index])
        bbox_size = height * container_size[0] *container_size[1]
        compactness_ems[index] = valid_size / bbox_size
        # pyramidality
        under_space = container[_x:_x+block_x, _y:_y+block_y, 0:_z]
        under_space_mask[index] = under_space==0
        empty_ems[index] += np.sum(under_space_mask[index])
        if 'P' in reward_type:
            pyramidality_ems[index] = valid_size / (empty_ems[index] + valid_size)
        # stability
        if 'S' in reward_type:
            stable_num = np.sum(stable[:block_index]) + np.sum(is_stable_ems[index])
            stability_ems[index] = stable_num / (block_index + 1)

    # search positions in each ems
    X = int(container_size[0] - block_x + 1)
    Y = int(container_size[1] - block_y + 1)
    for ems_index, ems in enumerate(ems_list):
        # using buttom-left strategy in each ems
        heightmap_ems[ems_index] = heightmap.copy()
        X0, Y0, _z = ems
        for _x, _y  in itertools.product( range(X0, X), range(Y0, Y) ):
            if is_settle_ems[ems_index]: break
            check_position(ems_index, _x, _y, _z)
        if is_settle_ems[ems_index]: calc_C_P_S(ems_index)

    # if the block has not been settled
    if np.sum(is_settle_ems) == 0:
        valid_size -= block_x * block_y * block_z
        stable[block_index] = False
        return container, positions, stable, heightmap, valid_size, empty_size

    # get the best ems
    ratio_ems = [c+p+s for c, p, s in zip(compactness_ems, pyramidality_ems, stability_ems)]
    best_ems_index = np.argmax(ratio_ems)
    while not is_settle_ems[best_ems_index]:
        ratio_ems.remove(ratio_ems[best_ems_index])
        best_ems_index = np.argmax(ratio_ems)

    # update the dynamic parameters
    _x, _y, _z = pos_ems[best_ems_index]
    container[_x:_x+block_x, _y:_y+block_y, _z:_z+block_z] = block_index + 1
    container[_x:_x+block_x, _y:_y+block_y, 0:_z][ under_space_mask[best_ems_index] ] = -1
    positions[block_index] = pos_ems[best_ems_index]
    stable[block_index] = is_stable_ems[best_ems_index]
    heightmap = heightmap_ems[best_ems_index]
    empty_size = empty_ems[best_ems_index]

    return container, positions, stable, heightmap, valid_size, empty_size

def calc_one_position_lb_greedy(block, block_index, container_size, reward_type, 
                                container, positions, stable, heightmap, valid_size, empty_size):
    """
    calculate the latest block's position in the container by lb-greedy
    ---
    params:
    ---
        static params:
            block: int * 2/3 array, size of the block to pack
            block_index: int, index of the block to pack, previous were already packed
            container_size: 1 x 2/3 array, size of the container
            reward_type: string, options:
                'C+P-lb-soft'
                'C+P-lb-hard'
                'C+P+S-lb-soft'
                'C+P+S-lb-hard'
        dynamic params:
            container: width (* depth) * height array, the container state
            positions: int * 2/3 array, coordinates of the blocks, [0, 0]/[0, 0, 0] for blocks after block_index
            stable: n * 1 bool list, the blocks' stability state
            heightmap: width (* depth) * 1 array, heightmap of the container
            valid_size: int, sum of the packed blocks' volume
            empty_size: int, size of the empty space under packed blocks
    return:
    ---
        container: width (* depth) * height array, updated container
        positions: n * 2/3 array, updated positions
        stable: n * 1 bool list, updated stable
        heightmap: width (* depth) * 1 array, updated heightmap
        valid_size: updated valid_size
        empty_size: updated empty_size
    """
    block_dim = len(container_size)
    if block_dim == 2:
        return calc_one_position_lb_greedy_2d(block, block_index, container_size, reward_type, 
                                            container, positions, stable, heightmap, valid_size, empty_size)
    elif block_dim == 3:
        return calc_one_position_lb_greedy_3d(block, block_index, container_size, reward_type, 
                                            container, positions, stable, heightmap, valid_size, empty_size)

def calc_positions_lb_greedy(blocks, container_size, reward_type="C+P+S-lb-hard"):
    '''
    calculate the positions to pack a group of blocks into a container by lb-greedy
    ---
    params:
    ---
        blocks: n x 2/3 array, blocks with an order
        container_size: 1 x 2/3 array, size of the container
        reward_type: string
            'C+P-lb-soft'
            'C+P-lb-hard'
            'C+P+S-lb-soft'
            'C+P+S-lb-hard'
    return:
    ---
        positions: int x 2/3 array, packing positions of the blocks
        container: width (* depth) * height array, the final state of the container
        stable: n x 1 bool list, each element indicates whether a block is placed(hard)/stable(soft) or not
        ratio: float, C / C*S / C+P / (C+P)*S / C+P+S, calculated by the following scores
        scores: 5 integer numbers: valid_size, box_size, empty_size, stable_num and packing_height
    '''
    # Initialize
    blocks = blocks.astype('int')
    blocks_num = len(blocks)
    block_dim = len(blocks[0])
    positions = np.zeros((blocks_num, block_dim)).astype(int)
    container = np.zeros(list(container_size)).astype(int)
    stable = [False] * blocks_num
    heightmap = np.zeros(container_size[:-1]).astype(int)
    valid_size = 0
    empty_size = 0

    # try to use cuda to accelerate but fail
    # container_tensor = torch.from_numpy(container)
    # container_tensor = container_tensor.cuda().detach()

    for block_index in range(blocks_num):
        container, positions, stable, heightmap, valid_size, empty_size = \
            calc_one_position_lb_greedy(blocks[block_index], block_index, container_size, reward_type, 
                                        container, positions, stable, heightmap, valid_size, empty_size)

    stable_num = np.sum(stable)
    if block_dim == 2:      box_size = np.max(heightmap) * container_size[0]
    elif block_dim == 3:    box_size = np.max(heightmap) * container_size[0] * container_size[1]

    C = valid_size / box_size
    P = valid_size / (empty_size + valid_size)
    S = stable_num / blocks_num

    if   reward_type == 'C+P-lb-soft':      ratio = C + P + S
    elif reward_type == 'C+P-lb-hard':      ratio = C + P + S
    elif reward_type == 'C+P+S-lb-soft':    ratio = C + P + S
    elif reward_type == 'C+P+S-lb-hard':    ratio = C + P + S
    elif reward_type == 'C-lb-hard' or reward_type == 'C-lb-soft':	ratio = C
    else:
        ratio = 0
        num = 0
        if 'C' in reward_type:
            ratio += C
            num += 1
        if 'P' in reward_type:
            ratio += P
            num += 1
        if 'S' in reward_type:
            ratio += S
            num += 1

        if num == 0:
            ratio = None
            print('Calc positions lb greedy ................')
            print("Unknown reward type for lb_greedy packing")

    scores = [valid_size, box_size, empty_size, stable_num, np.max(heightmap)]
    return positions, container, stable, ratio, scores

