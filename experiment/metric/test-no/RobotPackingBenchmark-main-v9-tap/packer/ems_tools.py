import numpy as np
import copy
import itertools


from scipy.spatial import ConvexHull
from matplotlib.path import Path

def check_stable(box, pos, layer_under_box=None, mask_under_box=None):
    '''
    check for 3D packing
    '''
    if layer_under_box is None and mask_under_box is None: return True
    
    if (pos[2]==0):
        if mask_under_box is None:
            return True
        # else:
        #     return False

    if mask_under_box is not None:
        if (mask_under_box == 0).any(): return False


    # obj_center = ( pos[0] + box[0]/2, pos[1] + box[1]/2  )
    obj_center = ( box[0]/2, box[1]/2  )
    # valid points right under this object
    
    if layer_under_box is None:
        layer_under_box = np.ones([box[0], box[1]]) * pos[2]
    if mask_under_box is None:
        mask_under_box = np.ones_like(layer_under_box)
    max_height = layer_under_box.max()
    x, y = np.where( (layer_under_box == max_height) * (mask_under_box == True) )

    points = np.concatenate([[x],[y]]).transpose()

    if(len(points) > box[0]*box[1]/2):
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

def compute_corners(heightmap, get_xy_corners=False):
    # NOTE find corners by heightmap
    
    hm_shape = heightmap.shape
    extend_hm = np.ones( (hm_shape[0]+2, hm_shape[1]+2) ) * 10000
    extend_hm[1:-1, 1:-1] = heightmap

    x_diff_hm_1 = extend_hm[:-1] - extend_hm[1:]
    x_diff_hm_1 = x_diff_hm_1[:-1, 1:-1]

    x_diff_hm_2 = extend_hm[1:] - extend_hm[:-1]
    x_diff_hm_2 = x_diff_hm_2[1:, 1:-1]

    y_diff_hm_1 = extend_hm[:, :-1] - extend_hm[:, 1:]
    y_diff_hm_1 = y_diff_hm_1[1:-1, :-1]

    y_diff_hm_2 = extend_hm[:, 1:] - extend_hm[:, :-1]
    y_diff_hm_2 = y_diff_hm_2[1:-1, 1:]
    
    x_diff_hms = [x_diff_hm_1 != 0, x_diff_hm_2 != 0]
    y_diff_hms = [y_diff_hm_1 != 0, y_diff_hm_2 != 0]

    corner_hm = np.zeros_like(heightmap)
    for xhm in x_diff_hms:
        for yhm in y_diff_hms:
            corner_hm += xhm * yhm

    left_bottom_hm = (x_diff_hm_1 != 0) * (y_diff_hm_1 != 0)

    left_bottom_corners = np.where(left_bottom_hm > 0)
    left_bottom_corners = np.array(left_bottom_corners).transpose()

    corners = np.where(corner_hm > 0)
    corners = np.array(corners).transpose()

    x_borders = list(np.where(x_diff_hm_1.sum(axis=1))[0])
    y_borders = list(np.where(y_diff_hm_1.sum(axis=0))[0])
    
    cross_corners = []
    if get_xy_corners:
        xy_corners = list(itertools.product(x_borders, y_borders))
        xy_corners = np.array(xy_corners)
        
        if len(left_bottom_corners) >= 3:
            try:
                convex_hull = ConvexHull(left_bottom_corners)
                hull_path = Path(left_bottom_corners[convex_hull.vertices])
                xy_inside_hull = hull_path.contains_points(xy_corners)
                # lb_inside_hull = hull_path.contains_points(left_bottom_corner)

                for c in left_bottom_corners:
                    c = list(c)
                    cross_corners.append(c)
                
                for i, c in enumerate(xy_corners):
                    c = list(c)
                    if c[0] == 0 or c[1] == 0:
                        cross_corners.append(c)

                    if xy_inside_hull[i] and c not in cross_corners:
                        cross_corners.append(c)
            except:
                pass

    x_borders.append(hm_shape[0])
    y_borders.append(hm_shape[1])

    return corners, left_bottom_corners, x_borders, y_borders, cross_corners

def compute_stair_corners(heightmap, corners):


    corners, _, _, _, _ = compute_corners(heightmap)

    stair_hm = np.zeros_like(heightmap)
    corner_heights = heightmap[corners[:,0], corners[:,1]]
    sort_ids = np.argsort(corner_heights)
    sort_corners = corners[sort_ids]

    for c in sort_corners:
        cx, cy = c
        h = heightmap[cx, cy]
        stair_hm[:cx+1, :cy+1] = h
    
    # bin_x, bin_y = heightmap.shape
    # stair_hm = np.zeros([bin_x, bin_y])
    # for xx in reversed(range(bin_x-1)):
    #     for yy in reversed(range(bin_y-1)):
    #         stair_hm[xx, yy] = max(stair_hm[xx+1, yy], stair_hm[xx, yy+1], heightmap[xx, yy])

    _, slb_corner, _, _, _ = compute_corners(stair_hm)
    return slb_corner


def compute_empty_space(container_size, corners, x_borders, y_borders, heightmap, empty_space_list, x_side='left-right', y_side='left-right', min_ems_width=0):
    
    # NOTE find ems from corners
    def check_valid_height_layer(height_layer):
        return (height_layer <= 0).all()

    for corner in corners:
        x,y = corner
        # h = int(heightmap[x, y])
        h = heightmap[x, y]
        if h == container_size[2]: continue

        h_layer = heightmap - h

        for axes in itertools.permutations(range(2), 2):
            x_small = x
            x_large = x+1
            
            y_small = y
            y_large = y+1

            for axis in axes:
                if axis == 0:
                    if 'left' in x_side:
                        for xb in x_borders:
                            if x_small > xb:
                                # if (h_layer[xb:x, y_small:y_large] <= 0).all():
                                if check_valid_height_layer(h_layer[xb:x, y_small:y_large]):
                                    x_small = xb
                            else: break

                    if 'right' in x_side:
                        for xb in x_borders[::-1]:
                            if x_large < xb:
                                if check_valid_height_layer(h_layer[x:xb, y_small:y_large]):
                                # if (h_layer[x:xb, y_small:y_large] <= 0).all():
                                    x_large = xb
                            else: break
                
                elif axis == 1:
                    if 'left' in y_side:
                        for yb in y_borders:
                            if y_small > yb:
                                if check_valid_height_layer(h_layer[ x_small:x_large, yb:y]):
                                # if (h_layer[ x_small:x_large, yb:y] <= 0).all():
                                    y_small = yb
                            else: break

                    if 'right' in y_side:
                        for yb in y_borders[::-1]:
                            if y_large < yb:
                                if check_valid_height_layer(h_layer[ x_small:x_large, y:yb]):
                                # if (h_layer[ x_small:x_large, y:yb] <= 0).all():
                                    y_large = yb
                            else: break

            # if (h_layer[ x_small:x_large, y_small:y_large] <= 0).all():
            if check_valid_height_layer(h_layer[ x_small:x_large, y_small:y_large]):

                new_ems = [[x_small, y_small, h], [x_large, y_large, container_size[2]],[1]*3 ]

                # NOTE remove small ems
                if min_ems_width > 0:
                    if x_large - x_small < min_ems_width or y_large - y_small < min_ems_width:
                        new_ems = None

                if new_ems is not None and new_ems not in empty_space_list:
                    empty_space_list.append( new_ems )

def compute_ems(id_map, heightmap, container_size, min_ems_width=0, ems_type='ems-id-stair', valid_container_mask=None):
    empty_max_spaces = []

    corners = None
    # NOTE normal_corners
    if 'id' in ems_type:
        corners, id_left_bottom_corners, x_borders, y_borders, xy_corners = compute_corners(id_map)
        compute_empty_space(container_size, id_left_bottom_corners, x_borders, y_borders, heightmap, empty_max_spaces, 'right', 'right', min_ems_width=min_ems_width)

    # NOTE stair corners
    if 'stair' in ems_type:
        if corners is None:
            corners, id_left_bottom_corners, x_borders, y_borders, xy_corners = compute_corners(id_map)
        stair_corners = compute_stair_corners(heightmap, corners)
        compute_empty_space(container_size, stair_corners, x_borders, y_borders, heightmap, empty_max_spaces, 'right', 'right', min_ems_width=min_ems_width)

    if 'ems' in ems_type:
        if corners is None:
            corners, hei_left_bottom_corners, x_borders, y_borders, xy_corners = compute_corners(id_map)
        compute_empty_space(container_size, corners, x_borders, y_borders, heightmap, empty_max_spaces, 'left-right', 'left-right', min_ems_width=min_ems_width)
        
    return empty_max_spaces

def compute_box_ems_mask(box_states, ems, check_z):
    # can place or not

    # box_states: state_num * 3
    # ems: ems_num * 6
    
    # pos | size, ems_len x 6
    ems_x = ems[:, 3:4]
    ems_y = ems[:, 4:5]
    ems_z = ems[:, 5:6]
    
    box_x = box_states[:, 0][None, :]
    box_y = box_states[:, 1][None, :]
    box_z = box_states[:, 2][None, :]

    ems_to_box_x = (ems_x - box_x) >= 0
    ems_to_box_y = (ems_y - box_y) >= 0
    ems_to_box_z = (ems_z - box_z) >= 0

    if check_z:
        ems_to_box_mask = ems_to_box_x * ems_to_box_y * ems_to_box_z
    else:
        ems_to_box_mask = ems_to_box_x * ems_to_box_y

    # ems_num x box_num 01
    return ems_to_box_mask
