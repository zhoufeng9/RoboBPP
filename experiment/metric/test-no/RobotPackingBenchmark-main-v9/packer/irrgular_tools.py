import math

import numpy as np
from packer.pct_model.tools import *
from packer.sdf_pack import *
from scipy.spatial import KDTree
from numba import jit


# irrgular
def rotate_180(matrix):
    # 转换为 NumPy 数组以方便操作
    matrix = np.array(matrix)
    
    # 上下翻转
    matrix = np.flipud(matrix)
    
    # 左右翻转
    matrix = np.fliplr(matrix)
    
    return matrix


def get_heightmapB(num, env, scene, sizes):
    name = scene.box_names[num]
    size = sizes[num]
    # objPath = 'dataset/blockout/shape_vhacd/'+name+'_0.obj'
    objPath = os.path.join(scene.objPath, name) + '_0.obj'
    mesh = trimesh.load(objPath)
    scale = scene.scale
    mesh.apply_scale(scale)
    # 目标四元数
    quaternion = [0, 0, 0, 1]
    # 创建旋转矩阵
    rotation_matrix = trimesh.transformations.quaternion_matrix(quaternion)
    # 应用旋转矩阵
    mesh.apply_transform(rotation_matrix)
    
    origin, direction = gen_ray_origin_direction(env.space.width, env.space.length, env.space.block_unit)
    
    heightMapH, heightMapB, maskH, maskB = shot_item(mesh, origin, direction, size[0], size[1], env.space.block_unit)
    contains = get_contains(mesh, size[0], size[1], size[2], env.space.block_unit)
    
    
    return heightMapB, maskB, contains


def gen_ray_origin_direction(xRange, yRange, resolution_h, boxPack = False, shift = 0.001):

    bottom = np.arange(0, xRange * yRange)      
    bottom = bottom.reshape((xRange, yRange))   

    origin = np.zeros((xRange, yRange, 3))
    origin[:, :, 0] = bottom // yRange * resolution_h + shift
    origin[:, :, 1] = bottom %  yRange * resolution_h + shift
    origin[:, :, 2] = -10e2

    origin = origin.reshape((xRange, yRange, 3))

    direction = np.zeros_like(origin)
    direction[:,:,2] = 1

    return origin, direction


def shot_item(mesh, ray_origins_ini, ray_directions_ini, xRange = 20, yRange = 20, block_unit = 0.03, start = [0,0,0]): # xRange, yRange the grid range.
    mesh = mesh.copy()
    mesh.apply_translation(- mesh.bounding_box.vertices[0]) # 将box沿着空间中的一个方向平移，使得box的包围盒的一个顶点移动到原点 (0, 0, 0) 处。

    heightMapB = np.zeros(xRange * yRange)
    heightMapH = np.zeros(xRange * yRange)
    maskB = np.zeros(xRange * yRange).astype(np.int32)
    maskH = np.zeros(xRange * yRange).astype(np.int32)

    ray_origins = ray_origins_ini[start[0] : start[0] + xRange, start[1] : start[1] + yRange].copy().reshape((-1,3))
    ray_directions = ray_directions_ini[start[0] : start[0] + xRange, start[1] : start[1] + yRange].copy().reshape(-1,3)
    index_triB, index_rayB, locationsB = mesh.ray.intersects_id( ray_origins=ray_origins, ray_directions=ray_directions,
                                                                 return_locations=True,   multiple_hits=False)

    if len(index_rayB) != 0:
        heightMapB[index_rayB] = locationsB[:, 2]
        maskB[index_rayB] = 1
    else:
        heightMapB[:] = 0
        maskB[:] = 1
    heightMapB = heightMapB.reshape((xRange, yRange))
    maskB = maskB.reshape((xRange, yRange))
    
    heightMapB_real = (np.array(heightMapB) / block_unit).astype(float)
    heightMapB_real = np.round(heightMapB_real).astype(np.int32)
    

    ray_origins[:, 2] *= -1
    ray_directions[:, 2] *= -1
    # print(np.concatenate((ray_origins, ray_origins + ray_directions), axis=1))
    index_triH, index_rayH, locationsH = mesh.ray.intersects_id( ray_origins=ray_origins, ray_directions=ray_directions,
                                                                 return_locations=True,   multiple_hits=False)
    if len(index_rayH) != 0:
        heightMapH[index_rayH] = locationsH[:, 2]
        maskH[index_rayH] = 1
    else:
        heightMapH[:] = mesh.extents[2]
        maskH[:] = 1
    heightMapH = heightMapH.reshape((xRange, yRange))
    maskH = maskH.reshape((xRange, yRange))

    heightMapH_real = (np.array(heightMapH) / block_unit).astype(float)
    heightMapH_real = np.round(heightMapH_real).astype(np.int32)



    return heightMapH_real, heightMapB_real, maskH, maskB



def get_contains(mesh, xRange, yRange, zRange, block_unit):
    contains = np.zeros((xRange, yRange, zRange)).astype(np.int32)
    min_bound, max_bound = mesh.bounds
    xRange_bound = max_bound[0] - min_bound[0]
    yRange_bound = max_bound[1] - min_bound[1]
    zRange_bound = max_bound[2] - min_bound[2]
    
    x_unit = xRange_bound / xRange
    y_unit = yRange_bound / yRange
    z_unit = zRange_bound / zRange
    
    
    for x in range(xRange):
        for y in range(yRange):
            for z in range(zRange):
                position = [max_bound[0] - (x + 0.5) * x_unit, max_bound[1] - (y + 0.5) * y_unit, max_bound[2] - (z + 0.5) * z_unit]
                position_array = np.array(position).reshape(1, 3)
                result = mesh.contains(position_array)
                if result:
                    contains[x][y][z] = 1
    
    return contains



def get_real_position(env, now_position):
    x, y, z = now_position

    now_real_x = env.space.real_position_x[x, y, z]
    now_real_y = env.space.real_position_y[x, y, z]
    now_real_z = env.space.real_position_z[x, y, z]
    real_position = [now_real_x, now_real_y, now_real_z]

    return real_position



def get_real_position_backup(env, now_position):
    real_position_x = env.space.real_position_x
    real_position_y = env.space.real_position_y
    real_position_z = env.space.real_position_z
    now_real_x = real_position_x[now_position[0]][now_position[1]][now_position[2]] # IndexError: index 30 is out of bounds for axis 0 with size 30
    now_real_y = real_position_y[now_position[0]][now_position[1]][now_position[2]]
    now_real_z = real_position_z[now_position[0]][now_position[1]][now_position[2]]
    real_position = [now_real_x, now_real_y, now_real_z]

    return real_position


def get_heightmap_points(env):
    heightmap_points = []
    plain = env.space.plain
    xRange, yRange = plain.shape[0], plain.shape[1]
    step = 5
    for x in range(0, xRange, step):
        for y in range(0, yRange, step):
            # real_position = get_real_position(env, [x, y, plain[x][y]-1])
            real_position = get_real_position(env, [x, y, math.ceil(plain[x][y])-1])
            heightmap_points.append(real_position)

    return heightmap_points


def get_heightmap_points_2(env):
    heightmap_points = []
    plain = env.space.plain
    xRange, yRange = plain.shape[0], plain.shape[1]
    real_position_x = env.space.real_position_x
    real_position_y = env.space.real_position_y
    real_position_z = env.space.real_position_z
    start_time = time.time()
    for x in range(xRange):
        for y in range(yRange):
            # real_position = get_real_position(env, [x, y, plain[x][y]-1])
            now_position = [x, y, plain[x][y] - 1]
            now_real_x = real_position_x[now_position[0]][now_position[1]][now_position[2]]  # IndexError: index 30 is out of bounds for axis 0 with size 30
            now_real_y = real_position_y[now_position[0]][now_position[1]][now_position[2]]
            now_real_z = real_position_z[now_position[0]][now_position[1]][now_position[2]]
            real_position = [now_real_x, now_real_y, now_real_z]

            heightmap_points.append(real_position)
    start_time2 = time.time()
    print(start_time2 - start_time)
    return heightmap_points


def get_heightmap_points_backup(env):
    heightmap_points = []
    plain = env.space.plain
    xRange, yRange = plain.shape[0], plain.shape[1]
    for x in range(xRange):
        for y in range(yRange):
            real_position = get_real_position(env, [x, y, plain[x][y]-1])
            heightmap_points.append(real_position)

    return heightmap_points


def get_container_points(env, now_position):
    container_points = []
    xRange, yRange = env.space.plain.shape[0], env.space.plain.shape[1]
    x, y, z = now_position
    container_point_left = get_real_position(env, [x, 0, z])
    container_point_right = get_real_position(env, [x, yRange-1, z])
    container_point_up = get_real_position(env, [0, y, z])
    container_point_down = get_real_position(env, [xRange-1, y, z])
    container_point_bottom = get_real_position(env, [x, y, 0])
    container_points.extend([container_point_left, container_point_right, container_point_up, container_point_down, container_point_bottom])
    
    return container_points
    



# 用于计算每个位置点的函数，使用 Numba 加速
@jit(nopython=True)
def compute_target_points(lx, ly, lz, xRange, yRange, zRange, step):
    points = []
    step = 5
    for x in range(0, xRange, step):
        for y in range(0, yRange, step):
            for z in range(0, zRange, step):
                points.append([lx + x, ly + y, lz + z])
    return points

# 优化和简化的 get_TSDF_v2 函数
def get_TSDF_v2(env, position, now_size):
    lx, ly, lz = position
    xRange, yRange, zRange = now_size
    heightmap_points = get_heightmap_points(env)  # 假设此函数已经优化

    # 使用 Numba 生成目标点
    target_points = compute_target_points(lx, ly, lz, xRange, yRange, zRange, 5)
    container_points = []

    for point in target_points:
        real_position = get_real_position(env, point)
        if real_position is not None:
            container_points.extend(get_container_points(env, point))

    test_points = heightmap_points + container_points
    if not test_points:
        return None

    # 使用 KDTree 查找最近点
    tree = KDTree(test_points)
    tsdfs, _ = tree.query(target_points)
    total_tsdf = np.sum(tsdfs)

    return total_tsdf





def get_TSDF_v2_backup(env, position, now_size):
    lx, ly, lz = position
    xRange, yRange, zRange = now_size[0], now_size[1], now_size[2]
    heightmap_points = get_heightmap_points(env)
    container_points = []
    target_points = []

    step = 5
    for x in range(0, xRange, step):
        for y in range(0, yRange, step):
            for z in range(0, zRange, step):
                now_position = [lx + x, ly + y, lz + z]
                # if lx + x == 30 or ly + y == 30 or lz + z == 30:
                #     print('here')
                #     pass
                real_position = get_real_position(env, now_position)
                target_points.append(real_position)
                
                # get container points
                container_point = get_container_points(env, now_position)
                container_points.extend(container_point)

    test_points = heightmap_points + container_points

    # 使用KDTree来查找最近点
    tree = KDTree(test_points)

    if len(target_points) == 0:
        return None
    # 计算每个点到最近mesh顶点的距离
    tsdfs, _ = tree.query(target_points)
    total_tsdf = np.sum(tsdfs)

    return total_tsdf  


def get_TSDF(env, position, contains, scene):
    lx, ly, lz = position
    xRange, yRange, zRange = contains.shape[0], contains.shape[1], contains.shape[2]
    tsdf = np.zeros((xRange, yRange, zRange))
    real_position_x = env.space.real_position_x
    real_position_y = env.space.real_position_y
    real_position_z = env.space.real_position_z
    
    for x in range(xRange):
        for y in range(yRange):
            for z in range(zRange):
                if contains[x][y][z] == 0:  # 此处无效，不在box内部
                    continue
                now_position = [lx + x, ly + y, lz + z]
                # env.space.ray_origins[0]
                now_real_x = real_position_x[now_position[0]][now_position[1]][now_position[2]]
                now_real_y = real_position_y[now_position[0]][now_position[1]][now_position[2]]
                now_real_z = real_position_z[now_position[0]][now_position[1]][now_position[2]]
                
                real_position = [now_real_x, now_real_y, now_real_z]
                distance = []
                
                # 剔除其他的obstacles
                other_obstacles = [0, 3, 4]
                obstacles = [x for x in scene.obstacles if x not in other_obstacles]
                
                for packed_id in obstacles:
                    # 使用一个虚拟对象（球体）来模拟点位置
                    virtual_sphere_radius = 0.001  # 虚拟球体的半径，设为很小的值来模拟点
                    virtual_sphere_collision_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=virtual_sphere_radius)
                    virtual_sphere_id = p.createMultiBody(baseCollisionShapeIndex=virtual_sphere_collision_shape, basePosition=real_position)

                    # 计算最近点
                    closest_points = p.getClosestPoints(virtual_sphere_id, packed_id, distance=0.2)

                    # 提取最短距离
                    if closest_points:
                        shortest_distance = closest_points[0][8]  # 结果中的第9个元素是距离
                        contact_point_on_object = closest_points[0][5]  # 结果中的第6个元素是物体表面的接触点
                        print(f"Shortest distance: {shortest_distance}")
                        print(f"Contact point on object: {contact_point_on_object}")
                        distance.append(shortest_distance)
                    else:
                        print("No contact points found.")

                    # 移除虚拟球体
                    p.removeBody(virtual_sphere_id)
                # 找到最小值
                min_distance = min(distance)
                tsdf[x][y][z] = min_distance
    total_tsdf = np.sum(tsdf)
    return total_tsdf           
                

