import numpy as np
import copy
import open3d as o3d
# import pymesh
# import random
# import pcl
from detector.transform_tool import *

border_indices = np.array([
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 0],
    
    [4, 5],
    [5, 6],
    [6, 7],
    [7, 4],
    
    # [4, 0],
    # [5, 1],
    # [6, 2],
    # [7, 3],
    
    [2, 1],
    [1, 5],
    [5, 6],
    [6, 2],
    
    [3, 0],
    [0, 4],
    [4, 7],
    [7, 3],
    
    [0, 1],
    [1, 5],
    [5, 4],
    [4, 0],
    
    [3, 2],
    [2, 6],
    [6, 7],
    [7, 3],
])


x_ = np.linspace(-1., 1., 20)
y_ = np.linspace(-1., 1., 20)
z_ = np.linspace(-1., 1., 20)
x, y, z = np.meshgrid(x_, y_, z_, indexing='ij')
box_inside_points = np.concatenate([[x,y,z]]).reshape(3, -1).transpose(1,0)


pbox_origin = o3d.io.read_triangle_mesh("F:\\workspace\\TAP\\mask_rcnn\\app\\assets\\cube.obj")


def pcl_fit(points, thresh=0.001, ksearch=50, max_iter=100):
    import pcl
    cloud = pcl.PointCloud()
    cloud.from_array(points.astype('float32'))
    seg = cloud.make_segmenter_normals(ksearch=ksearch)
    seg.set_optimize_coefficients(True)
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_normal_distance_weight(0.05)
    seg.set_method_type(pcl.SAC_RANSAC)
    seg.set_max_iterations(max_iter)
    seg.set_distance_threshold(thresh)
    inliers, eqs = seg.segment()
    return eqs, inliers
    

def get_box_inside( block_size, block_pose=np.eye(4), max_width=4, unit_width=0.03, max_insert=20 ):
    
    block_max_length = max(block_size)
    insert_num = int( max_insert * block_max_length / (max_width * unit_width) )

    x_ = np.linspace(-1., 1., insert_num)
    y_ = np.linspace(-1., 1., insert_num)
    z_ = np.linspace(-1., 1., insert_num)
    x, y, z = np.meshgrid(x_, y_, z_, indexing='ij')
    box_inside_points = np.concatenate([[x,y,z]]).reshape(3, -1).transpose(1,0)

    rot = block_pose[:3, :3]
    trans = block_pose[:3, 3]
    inside_points = box_inside_points * block_size / 2.0
    inside_points = np.einsum( 'ij,aj->ai', rot, inside_points ) + trans
    
    return inside_points


def check_point_cross_block(corners, cam_pos):
    c_to_cam = cam_pos - corners

    to_cam_dist = np.linalg.norm(c_to_cam, axis=1)

    sort_order = np.argsort(to_cam_dist)
    
    # c_to_far = corners[far_id] - corners

    # cos = np.einsum('ij,ij->i', c_to_cam, c_to_far)

    return sort_order

def get_block_line(block_size, cam_to_world, cam_intrinsic, pose):
    
    p = np.zeros_like(cam_to_world)
    p[:3,:] = pose[:3,:]
    p[3,3] = 1
    final_pose = p
    
    x, y, z = block_size / 2.0
    block_corners = np.array([
        [ x, y, z ],
        [ -x, y, z ],
        [ -x, -y, z ],
        [ x, -y, z ],

        [ x, y, -z ],
        [ -x, y, -z ],
        [ -x, -y, -z ],
        [ x, -y, -z ],
    ])

    corners = transform_by_matrix( block_corners, final_pose )
    cam_corners = transform_by_matrix( corners, np.linalg.inv(cam_to_world) )

    fx = cam_intrinsic[0, 0]
    fy = cam_intrinsic[1, 1]
    cx = cam_intrinsic[0, 2]
    cy = cam_intrinsic[1, 2]

    image_corners = cam_corners * (fx, fy, 1) / cam_corners[:,2:] + (cx, cy, 0)
    image_corners = image_corners[:,:2].astype("int")
    
    sort_order = check_point_cross_block(corners, cam_to_world[:3,3])
    far_id = sort_order[-1]
    close_id = sort_order[0]

    close_dist = np.linalg.norm(cam_to_world[:3,3] - corners[close_id] )

    global border_indices

    final_border_indices = []
    f_border_indices = border_indices.reshape(6, -1, 2)
    for ids in f_border_indices:
        if far_id not in ids:
            final_border_indices.append(ids)
    face_num = len(final_border_indices)
    final_border_indices = np.array(final_border_indices).reshape(-1, 2)

    main_lines = image_corners[ final_border_indices ]

    main_lines = main_lines.reshape(face_num, -1, 2, 2)
    final_lines = []
    for lines in main_lines:
        if far_id not in lines:
            final_lines.append(lines)
    
    final_lines = np.array(final_lines).reshape(-1, 2, 2)

    return final_lines, image_corners, corners[ final_border_indices ], close_dist

def create_pc(points, colors=None) -> o3d.geometry.PointCloud:
    """
    Args:

        points: np.array [N, 3]
            points
        colors: np.array [3] | np.array [N, 3]
            Just color values between [0,1]
    
    Returns:
        point_cloud: open3d.geometry.PointCloud
        
    """
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector( points )

    if colors is not None:
        if len(colors) == 3 and len(colors.shape) == 1:
            pc.paint_uniform_color(colors)
        else:
            pc.colors = o3d.utility.Vector3dVector( colors )
    return pc

def create_line_pc(start, end, step_num, colors=None):
    step = (end - start) / step_num
    points = []
    for i in range(step_num):
        points.append( start + step * i )
    points = np.array(points)
    return create_pc(points, colors)

def create_cube_line(lines, colors=None):
    lines_pc = []
    for line in lines:
        a, b = line
        lines_pc.append( create_line_pc(a, b, 100, colors ) )
    return lines_pc


def get_box(block_size, pose=None, unit_height=0.03):
    """
    Args:
        block_sizes: Dict{'x', 'y', 'z'}
        pose: Array [4, 4] / [3, 4]

    Returns:
        pbox: pymesh.Mesh
    """
    if pose is None:
        pose = np.identity(4)

    block_size = np.array( [ block_size['x'], block_size['y'], block_size['z'] ] )
    x, y, z = block_size / 2.0
    
    v = np.array(pbox_origin.vertices)
    
    v[:,0] = x * v[:,0]
    v[:,1] = y * v[:,1]
    v[:,2] = z * v[:,2]

    rot = pose[:3,:3]
    trans = pose[:3,3]
    v = np.einsum( 'ij,aj->ai', rot, v ) + trans

    # inside_points = get_box_inside( block_size, pose, unit_width=unit_height )
    inside_points = box_inside_points * block_size / 2
    inside_points = np.einsum( 'ij,aj->ai', rot, inside_points ) + trans

    # pbox = pymesh.form_mesh(v, f)
    # return pbox, inside_points
    return inside_points

class Block(object):
    def __init__( self, size, pose, basic_unit = 0.02 ):
        length, width, height = size

        self.length = length
        self.width = width
        self.height = height

        self.size = {}
        self.size['x'] = length
        self.size['y'] = width
        self.size['z'] = height
        
        self.pose = pose
        self.center = pose[:3,3]

        # generate side block, but side block would not generate
        # self.mesh, self.inside_points = get_box( self.size, self.pose, basic_unit )
        self.inside_points = get_box( self.size, self.pose, basic_unit )
        # self.corners = self.mesh.vertices
        self.basic_unit = basic_unit
    
    def point_inside_num( self, points, cube_size ):
        abs_points = abs(points)
        sub = cube_size/2.0 - abs_points
        return (( sub > 0).sum(axis=1) == 3).sum()

    def get_points_num(self):
        return len(box_inside_points)


    def is_intersect(self, block):
        """
        Args:
            block: Block
        Returns:
            ret: bool
        """

        a_size = np.array( [ self.size['x'], self.size['y'], self.size['z'] ] )
        b_size = np.array( [ block.size['x'], block.size['y'], block.size['z'] ] )

        # a_pose = self.pose
        b_pose = block.pose
        
        a_points = self.inside_points
        # b_points = block.inside_points

        b_inv_pose = np.linalg.inv(b_pose)
        trans = b_inv_pose[:3, 3]
        rot = b_inv_pose[:3,:3]

        a_points_at_b = np.einsum( 'ij,aj->ai', rot, a_points ) + trans
        
        return self.point_inside_num(a_points_at_b, b_size) > 0

        # a_mesh = self.mesh
        # b_mesh = block.mesh
        # intersect = pymesh.boolean(a_mesh, b_mesh, "intersection")
        # a_volume = abs(a_mesh.volume)
        # b_volume = abs(b_mesh.volume)
        # min_volume = min(a_volume, b_volume)
        # intersect_rate = intersect.volume / min_volume
        # # return intersect.volume > 0
        # return intersect_rate > 1e-3
        
    def intersect_rate(self, block):
        """
        Args:
            block: Block
        Returns:
            ret: float
        """
        # a_mesh = self.mesh
        # b_mesh = block.mesh
        # intersect = pymesh.boolean(a_mesh, b_mesh, "intersection")
        # a_volume = abs(a_mesh.volume)
        # b_volume = abs(b_mesh.volume)
        # intersect_rate = intersect.volume / a_volume
        # return intersect_rate

        a_size = np.array( [ self.size['x'], self.size['y'], self.size['z'] ] )
        b_size = np.array( [ block.size['x'], block.size['y'], block.size['z'] ] )

        # a_pose = self.pose
        b_pose = block.pose
        
        a_points = self.inside_points
        # b_points = block.inside_points

        b_inv_pose = np.linalg.inv(b_pose)
        trans = b_inv_pose[:3, 3]
        rot = b_inv_pose[:3,:3]

        a_points_at_b = np.einsum( 'ij,aj->ai', rot, a_points ) + trans
        return self.point_inside_num(a_points_at_b, b_size * 0.999)


class Cube:
    '''
    A class of the detected cube
    
    Args:
        x: x
    
    Functions:
        x -> x
    
    '''
    def __init__(self, points, max_block_size, block_unit, fit_thresh=0.001, match_close_one=False):

        # np.random.seed(666)
        # random.seed(666)

        self.fit_thresh = fit_thresh
        # self.fit_thresh = 0.002
        self.match_close_one = match_close_one

        pc = create_pc(points)

        self.pc = pc.voxel_down_sample(voxel_size=self.fit_thresh / 2.0)

        self.labels = None

        self.plane_labels = []
        self.plane_norms = []
        self.plane_pcs = []
        self.plane_models = []

        self.rotation_z_axis = None
        self.euler_to_z = None
        self.back_to_z_rot = None

        self.rotation_y_axis = None
        self.euler_to_y = None
        self.back_to_y_rot = None
        
        self.size = None
        self.center = None
        self.pose = None
        
        self.max_block_size = max_block_size
        self.block_unit = block_unit
        self.is_single_surface = False


    def fit_planes(self, pc, thresh=0.001):

        ksearch=10
        max_iter = 100
        # max_iter = 200

        points_num = len(pc.points)
        
        planes = []
        plane_pcs = []
        plane_normals = []

        obj_center = pc.get_center()

        colors = [
            [1,0,0],
            [0,1,0],
            [0,0,1],
        ]
        

        # print(points_num)
        if points_num < 1000:
            return plane_pcs, plane_normals, planes

        while True:

            if not self.match_close_one: 
                if points_num > 22000:
                    break

            if points_num < 50:
                break
            if len(planes) >= 3:
                break
            
            eqs, lins = pc.segment_plane( distance_threshold=thresh, ransac_n=4, num_iterations=1000 )

            p_num = len(lins)

            fit_pc = pc.select_by_index(lins)

            # print(len(fit_pc.points))
            # fit_pc, _ = fit_pc.remove_radius_outlier(nb_points=5, radius=0.003)
            # fit_pc = fit_pc.voxel_down_sample(voxel_size=0.002)
            # print(len(fit_pc.points))
            # print('******')

            fit_pc.paint_uniform_color(colors[len(planes)])

            # import IPython
            # IPython.embed()

            # o3d.visualization.draw_geometries(
            #     [fit_pc, pc]
            # )

            pc = pc.select_by_index(lins, invert=True)
            points_num -= p_num

            a, b, c, d = eqs
            plane_norm = np.array([a,b,c])
            plane_norm = plane_norm / np.linalg.norm( plane_norm )
            # print(p_num, '----', plane_norm)

            # compare with pre_norm, each plane must 90
            is_same_plane = False
            for pi, pre_norm in enumerate(plane_normals):
                # print(abs(vec_cos_theta(plane_norm, plane_normals[0])))

                abs_cos = abs(vec_cos_theta(plane_norm, pre_norm))
                # if abs_cos > 0.1:
                if abs_cos > 0.6:

                    is_same_plane = True
                    
                    break
                

            if is_same_plane: continue
            
            fit_pc, _ = fit_pc.remove_statistical_outlier(nb_neighbors=5, std_ratio=0.005)

            cluster_center = fit_pc.get_center()
            if vec_cos_theta( plane_norm, obj_center-cluster_center ) > 0:
                plane_norm = -plane_norm

            plane_normals.append(plane_norm)
            plane_pcs.append(fit_pc)
            planes.append(eqs)

        return plane_pcs, plane_normals, planes


    def match_planes(self, top_K=3):
        # start matching
        # cluster for plane of cuboid
        plane_pcs, plane_norms, plane_models = self.fit_planes( self.pc, thresh=self.fit_thresh )
        
        self.plane_norms = plane_norms
        self.plane_pcs = plane_pcs
        self.plane_models = plane_models

        self.is_single_surface = len( self.plane_norms ) == 1


    def compute_rotation_matrix(self):
        '''
        compute two rotation matrix

        1. back_to_z_axis

        2. back_to_y_axis
            
        '''
        plane_num = len(self.plane_models)
        z_scores = []
        # find the norm close to z-axis of the world
        for i in range(plane_num):
            p_norm = self.plane_norms[i]
            z_scores.append( vec_cos_theta( p_norm, np.array((0,0,1)), ) )

        min_z_ind = np.argmax(z_scores)
        z_norm = self.plane_norms[min_z_ind]
        z_axis = np.array((0,0,1))

        if (z_norm == [0,0,1]).all():
            # compute the rotation axis
            rotation_z_axis = np.array([0,0,1])
            # compute the rotation degree
            euler_to_z = 0
            
            # get the rotation matrix to match z axis
            back_to_z_rot = vec_euler_to_matrix( rotation_z_axis, euler_to_z )

        else:
            # compute the rotation axis
            rotation_z_axis = np.cross( z_norm, z_axis )
            rotation_z_axis = rotation_z_axis / np.linalg.norm(rotation_z_axis)
            # compute the rotation degree
            cost = vec_cos_theta( z_norm, z_axis )
            euler_to_z = np.math.acos( cost )
            euler_to_z = np.math.pi - euler_to_z
            
            # get the rotation matrix to match z axis
            back_to_z_rot = vec_euler_to_matrix( rotation_z_axis, euler_to_z )

        # transform norm, then z_norm and z_axis are collinear
        trans_norms = transform_by_matrix( self.plane_norms, back_to_z_rot, True )
        # normalize the normal vectors
        for i in range(plane_num):
            trans_norms[i] = trans_norms[i] / np.linalg.norm(trans_norms[i])

        # now compute the rotation degree along z-axis to make y-axis-collinear
        if not self.is_single_surface:
            max_y_score = -1000
            min_y_norm = None
            for i in range(len(trans_norms)):
                if min_z_ind == i: continue
                
                p_norm = trans_norms[i]
                y_score = vec_cos_theta( p_norm, np.array((0,1,0)) )
                if y_score > max_y_score:
                    min_y_norm = p_norm
                    max_y_score = y_score

            rotation_y_axis = np.cross( min_y_norm, [0,1,0] )
            rotation_y_axis = rotation_y_axis / np.linalg.norm(rotation_y_axis)
            cost = vec_cos_theta( min_y_norm, np.array((0,1,0)) )

            euler_to_y = np.math.acos( cost )
            
            euler_to_y = np.math.pi - euler_to_y

            if rotation_y_axis[-1] < 0:
                rotation_y_axis = [0, 0, -1]
            else:
                rotation_y_axis = [0, 0, 1]
        else:
            rotation_y_axis = [0,0,1]
            euler_to_y = 0
            # TODO
            # project top face to plane, and match cube ...

        back_to_y_rot = vec_euler_to_matrix( rotation_y_axis, euler_to_y )
        
        self.rotation_z_axis = rotation_z_axis
        self.euler_to_z = euler_to_z
        self.back_to_z_rot = back_to_z_rot

        self.rotation_y_axis = rotation_y_axis
        self.euler_to_y = euler_to_y
        self.back_to_y_rot = back_to_y_rot
    
    def compute_pose(self):
        """
        compute the cube pose based on pointcloud and rotation matrix
        
        """
        self.back_to_z_rot = vec_euler_to_matrix( self.rotation_z_axis, self.euler_to_z )
        self.back_to_y_rot = vec_euler_to_matrix( self.rotation_y_axis, self.euler_to_y )

        back_to_origin = self.back_to_y_rot @ self.back_to_z_rot
        origin_to_pose = np.linalg.inv(back_to_origin)

        cube_pc = copy.deepcopy(self.plane_pcs[0])
        for i in range(len(self.plane_pcs)):
            if i == 0: continue
            cube_pc += self.plane_pcs[i]
        cube_points = np.array(cube_pc.points)

        not_rot_cube_points = transform_by_matrix( cube_points, back_to_origin )

        if self.is_single_surface == 1:
            # not_rot_cube_points
            z_offset_points = not_rot_cube_points + [0, 0, self.block_unit] # ros
            
            # z_offset_points = not_rot_cube_points - [0, 0, self.block_unit]

            not_rot_cube_points = np.concatenate( (not_rot_cube_points, z_offset_points), axis=0 )

        origin_max_corner = not_rot_cube_points.max(axis=0)
        origin_min_corner = not_rot_cube_points.min(axis=0)

        cube_size =  origin_max_corner - origin_min_corner
        
        cube_center = ( origin_max_corner + origin_min_corner ) / 2
        cube_center = origin_to_pose @ cube_center

        # project to image
        cube_pose = np.identity(4)
        cube_pose[:3, :3] = origin_to_pose
        cube_pose[:3, 3] = cube_center

        self.center = cube_center
        self.size = cube_size
        self.pose = cube_pose

    def is_oversize(self):
        if self.size is not None:
            for s in self.size:
                # print('oversize? ', s, self.max_block_size)
                if s > (self.max_block_size + 0.5) * self.block_unit:
                # if s > (self.max_block_size + 0.7) * self.block_unit:
                    return True
                if s < (0.5) * self.block_unit:
                # if s < 3 * self.block_unit:
                    return True
        return False

def block_match(mask, depth, cam_intrinsic, cam_to_world, max_block_size, block_unit, match_close_one=False, fit_thresh=0.001):
    block_lines = []
    block_poses = []
    block_sizes = []
    block_remove = []
    block_ids = []

    cubes = []

    debug = False

    ids = np.unique(mask)[1:]   # [8, 9, ..., 17] 即箱子的bodyUniqueId

    block_num = len(ids)

    skip_num = 0    # 跳过背景（通常ID为0）
    
    
    for obj_id in range(block_num):

        # print("Matching block %d" % obj_id)

        # get obj_data  将深度图转换为点云（point cloud），得到三维空间中每个像素的坐标信息
        points = depth_to_pc_fast(depth, cam_intrinsic, cam_to_world)
        
        obj_mask = (mask == ids[obj_id])    # 来选择当前物体的点

        p_num = obj_mask.sum()      # 如果点数太少，则跳过该物体并记录到 block_remove 列表
        if p_num < 200:
            skip_num += 1
            block_remove.append(obj_id)
            # cubes.append(None)
            if debug:
                print('here == ')
            continue

        # obj_points = points[ obj_mask > 0 ]   # 得到该id对应的箱子对应的3d点(点云)
        obj_points = points[ obj_mask ]

        cube = Cube(obj_points, max_block_size, block_unit, fit_thresh=fit_thresh, match_close_one=match_close_one)

        cube.match_planes()     # 拟合物体的平面

        if len(cube.plane_models) == 0:     # 平面模型数量为0
            skip_num += 1
            block_remove.append(obj_id)     # 跳过该物体并记录到 block_remove 列表
            if debug:
                print("   ] len(cube.plane_models) == 0")
            continue

        if cube.is_single_surface:      # 物体是单表面

            if debug:
                print("   ] is_single_surface")
            block_remove.append(obj_id)
            skip_num += 1

            continue

        cube.compute_rotation_matrix()      # 计算物体的旋转矩阵
        cube.compute_pose()                 # 计算物体的位姿（位置和方向）

        if cube.is_oversize():      # 检查物体是否超尺寸
            if debug:
                print('oversize ', cube.size)
            skip_num += 1
            block_remove.append(obj_id)
            continue
        
        block_ids.append(ids[obj_id])   # 记录id与cube
        cubes.append(cube)

    opt_remove_list = []
    
    block_num -= skip_num
    # block_num -= len(opt_remove_list)
    
    all_dist = []
    for obj_id in range(block_num):

        # if obj_id in opt_remove_list: continue      # 此处代码有误，opt_remove_list为空，应该用block_remove
        if obj_id in block_remove: continue

        block_poses.append( cubes[obj_id].pose )
        block_sizes.append( cubes[obj_id].size )

        cube_lines, _, _, close_dist = get_block_line( cubes[obj_id].size, cam_to_world, cam_intrinsic, cubes[obj_id].pose)
        # cube_lines: 一个包含立方体边线的数组，每条边由两个端点定义，每个端点是一个3D坐标，可以用于在图像或3D视图中可视化立方体的边缘
        # close_dist: 相机与立方体最近顶点的距离，可以用于确定立方体相对于相机的深度
        block_lines.append(cube_lines)

        all_dist.append(close_dist)

    sort_order = np.argsort(all_dist)[::-1]     # 根据与摄像头之间的距离对物体进行排序

    return block_poses, block_sizes, block_lines, block_remove, ids, block_ids, sort_order
    
