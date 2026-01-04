import math
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import pybullet as p
import trimesh
import torch

from arm_pybullet_env.arm_envs.envs import Env, colors
from arm_pybullet_env.arm_envs import cameras
from arm_pybullet_env.arm_envs import models


from arm_pybullet_env.utils import pybullet_utils
from arm_pybullet_env.utils import utils

from block import Block
import random
import copy
import time
from scipy.spatial.transform import Rotation as R, Slerp

mpl.use('Agg')


def transmit(box_id, prepare_final_pose, target_final_pose):
    # 总运动时间（秒）
    total_time = 2.0
    # 仿真帧率（每秒执行的仿真步数）
    frame_rate = 240
    # 总帧数
    total_frames = int(total_time * frame_rate)

    # 分别提取初始和目标的位置信息和四元数姿态
    start_pos = np.array(prepare_final_pose[0])
    end_pos = np.array(target_final_pose[0])
    start_orn = np.array(prepare_final_pose[1])
    end_orn = np.array(target_final_pose[1])

    # 使用 scipy 的 Rotation 类来处理四元数
    start_rot = R.from_quat(start_orn)
    end_rot = R.from_quat(end_orn)

    # 创建时间插值数组
    times = [0, 1]

    # 创建 Slerp 对象用于插值
    slerp = Slerp(times, R.from_quat([start_orn, end_orn]))

    # 生成时间进度数组
    interp_times = np.linspace(0, 1, total_frames)

    # 开始仿真移动过程
    for frame in range(total_frames):
        # 计算当前时间的进度（0 到 1 之间）
        alpha = frame / total_frames

        # 插值位置
        current_pos = (1 - alpha) * start_pos + alpha * end_pos

        # 获取当前插值的四元数
        current_rot = slerp([alpha])[0]
        current_orn = current_rot.as_quat()  # 转换为四元数形式

        # 更新box的位置和姿态
        p.resetBasePositionAndOrientation(box_id, current_pos, current_orn)

        # 步进仿真
        p.stepSimulation()

        # 控制仿真步进速度，使得移动时间为2秒
        time.sleep(1 / frame_rate)




class Pack_Env(Env):
    def __init__(self, args_base, assets_root=models.get_data_path(), disp=False, shared_memory=False, hz=240, use_egl=False) -> None:
        super().__init__(assets_root, disp=disp, shared_memory=shared_memory, hz=hz, use_egl=use_egl)

        # planner
        self.ompl_plan_allowed_time = args_base.Scene.ompl_plan_allowed_time
        self.interpolate_num = args_base.Scene.interpolate_num
        self.step_size = args_base.Scene.step_size

        # scene
        self.arm_urdf_path = args_base.Scene.arm_urdf_path
        self.homej = args_base.Scene.homej
        
        self.support_pose = args_base.Scene.support_pose
        self.conveyor_pose = args_base.Scene.conveyor_pose
        self.blocks_size = []
        self.sizes = []
        self.packed_positions = []
        self.plane_offset = args_base.Scene.plane_offset
    
        self.block_unit = args_base.Scene.block_unit
        self.blocks_num = args_base.Scene.blocks_num
        self.block_gap = args_base.Scene.block_gap

        self.threshold = args_base.Scene.threshold
        
        self.init_box_offset = np.array(args_base.Scene.init_box_offset) # 初始盒子的偏移量

        self.block_min_size = args_base.Scene.block_min_size
        self.block_max_size = args_base.Scene.block_max_size

        
        # 摆放容器左上角
        self.target_origin = np.array(args_base.Scene.target_origin)   # 目标容器的左上角位置
        self.target_container_size = np.array(args_base.Scene.target_container_size)  # 目标容器的整体尺寸
        # self.target_width = args_base.Scene.target_container_size[0]    # numpy目标容器的宽度
        self.target_base_size = np.array([self.target_container_size[0] * self.block_unit, self.target_container_size[1] * self.block_unit, 0.02])   # 目标容器的基础尺寸
        # self.target_container_size = np.array(args_base.Scene.target_container_size)    # 目标容器的整体尺寸
        
        
        self.wall_width = args_base.Scene.wall_width
        # self.wall_height = args_base.Scene.wall_height
        self.wall_height = self.target_container_size[-1] * self.block_unit + 0.01

        self.remove_position = np.array(args_base.Scene.remove_position) # 无法摆放物体的丢弃位置
        
        self.height_pose_offset_up = args_base.Scene.height_pose_offset_up
        self.height_pose_offset_down = args_base.Scene.height_pose_offset_down
        self.height_pose_offset_up_2d = args_base.Scene.height_pose_offset_up_2d
        self.height_pose_offset_down_2d = args_base.Scene.height_pose_offset_down_2d

        self.is_3d_packing = 1

        # irregular
        self.objPath = args_base.Scene.objPath
        self.scale = args_base.Scene.scale

        # 2d packing data
        self.scale_2d = args_base.Scene.scale_2d
        self.objPath_2d = args_base.Scene.objPath_2d
        # self.img_path_2d = args_base.Scene.img_path_2d


    
    def reset(self, setting=1, show_gui=True):
        super().reset(setting=setting, show_gui=show_gui)
        self.blocks_size = []
        self.sizes = []
        self.packed_positions = []



    def get_2d_data(self):
        scale = [0.0006, 0.0006, 0.0006]
        rotation = [0, 0, 0, 1]
        
        def load_unique_obj_files(obj_path):
            # obj_files = sorted(os.listdir(obj_path))
            obj_files = [f'panel_furniture_{i}.obj' for i in range(0, 1000)]
            object_sizes = []
            names = []

            for file in obj_files:
                objPath = os.path.join(obj_path, file)

                # 进行相应处理
                mesh = trimesh.load(objPath)
                mesh.apply_scale(scale)

                visual_shape_id = p.createVisualShape(shapeType=p.GEOM_MESH, fileName=objPath, meshScale=scale)
                collision_shape_id = p.createCollisionShape(shapeType=p.GEOM_MESH, fileName=objPath, collisionFramePosition=[0.0, 0.0, 0.0], meshScale=scale)
                mass = mesh.volume
                # 0.39056174, -0.32140087,  0.08197798    0.13197798
                object_id = p.createMultiBody(baseMass=mass,basePosition=[0.53056174, -0.20140087,  0.08197798],
                                        baseOrientation=rotation,
                                        baseCollisionShapeIndex=collision_shape_id,
                                        baseVisualShapeIndex=visual_shape_id,
                                        useMaximalCoordinates=True)
                # p.changeDynamics(id, -1, linearDamping = linearDamping, angularDamping = angularDamping, contactProcessingThreshold = 0,)
                aabb_min, aabb_max = p.getAABB(object_id)
                object_size = [aabb_max[0] - aabb_min[0], aabb_max[1] - aabb_min[1], aabb_max[2] - aabb_min[2]]
                if object_size[0] > 0.09 or object_size[1] > 0.09 or object_size[0] < 0.03 or object_size[1] < 0.03:
                    continue
                object_sizes.append(object_size)
                real_name = file.split('.')[0]
                names.append(real_name)
            
            object_sizes = np.array(object_sizes) 
            np.save('object_sizes_2d.npy', object_sizes)
            names = np.array(names) 
            np.save('names_2d.npy', names)
            
            return object_sizes
        
        sizes = load_unique_obj_files('/media/wzf/goodluck/Workspace/rl/panel_furniture/panel_furniture_obj')
        
        print(sizes.shape)
        print("here")


    def get_irrgular_data(self):
        scale = [0.5, 0.5, 0.5]
        
        rotation = [0, 0, 0, 1]
        
        def load_unique_obj_files(obj_path):
            obj_files = sorted(os.listdir(obj_path))
            object_sizes = []
            unique_files = []

            for file in obj_files:
                if file.endswith('.obj'):
                    # 获取文件名的主要部分（去掉最后的 '_数字.obj'）
                    main_part = '_'.join(file.split('_')[:-1])
                    if main_part not in unique_files:
                        unique_files.append(main_part)
                        objPath = os.path.join(obj_path, file)
                        # 进行相应处理
                        mesh = trimesh.load(objPath)
                        mesh.apply_scale(scale)
                        
                        visual_shape_id = p.createVisualShape(shapeType=p.GEOM_MESH, fileName=objPath, meshScale=scale)
                        collision_shape_id = p.createCollisionShape(shapeType=p.GEOM_MESH, fileName=objPath, collisionFramePosition=[0.0, 0.0, 0.0], meshScale=scale)
                        mass = mesh.volume
                        object_id = p.createMultiBody(baseMass=mass,basePosition=[0.53056174, -0.20140087,  0.13197798],
                                                baseOrientation=rotation,
                                                baseCollisionShapeIndex=collision_shape_id,
                                                baseVisualShapeIndex=visual_shape_id,
                                                useMaximalCoordinates=True)
                        # p.changeDynamics(id, -1, linearDamping = linearDamping, angularDamping = angularDamping, contactProcessingThreshold = 0,)
                        aabb_min, aabb_max = p.getAABB(object_id)
                        object_size = [aabb_max[0] - aabb_min[0], aabb_max[1] - aabb_min[1], aabb_max[2] - aabb_min[2]]
                        object_sizes.append(object_size)
            
            object_sizes = np.array(object_sizes) 
            np.save('object_sizes.npy', object_sizes)
            
            return object_sizes, unique_files
        
        sizes, unique_files = load_unique_obj_files('/media/wzf/goodluck/Workspace/rl/abc/shape_vhacd')
        
        print(sizes.shape)


    def load_2d_irregular_data(self, data, args):    
        tex_path = args.Data.tex_path
        self.textures = None
        if os.path.exists(tex_path):
            self.textures = np.load(tex_path, allow_pickle=True)
        
        init_position = np.array(args.Data.init_position)
        rotation = np.array(args.Data.init_rotation)
        poses_mat = []
        self.box_ids = []
        self.box_urdf = []
        choice_names = None
        
        if data == 'steel_plate':
            sizes = np.load(args.Data.Steel_plate.data_path)
            names = np.load(args.Data.Steel_plate.data_names)

            choice_sizes = sizes.tolist()
            choice_names = names.tolist()
            num_samples = len(choice_names)
            
            for size in choice_sizes:
                position = init_position + size
                pose = [position, rotation]
                pose_mat = pybullet_utils.pose_to_mat(pose)
                poses_mat.append(pose_mat)

            info = choice_names
        elif data == 'panel_furniture':
            sizes = np.load(args.Data.Panel_furniture.data_path)
            names = np.load(args.Data.Panel_furniture.data_names)

            choice_sizes = sizes.tolist()
            choice_names = names.tolist()
            num_samples = len(choice_names)

            for size in choice_sizes:
                position = init_position + size
                pose = [position, rotation]
                pose_mat = pybullet_utils.pose_to_mat(pose)
                poses_mat.append(pose_mat)

            info = choice_names
        
        poses_mat = np.array(poses_mat) 
        self.box_names = choice_names
        
        return num_samples, choice_sizes, poses_mat, info


    def load_irregular_data(self, data, args):    
        tex_path = args.Data.tex_path
        self.textures = None
        if os.path.exists(tex_path):
            self.textures = np.load(tex_path, allow_pickle=True)
        
        init_position = np.array(args.Data.init_position)
        rotation = np.array(args.Data.init_rotation)
        poses_mat = []
        self.box_ids = []
        self.box_urdf = []
        
        if data == 'blockout':
            # file_path = 'dataset/10_[7]_[1_4]_003_mess_real/train/0/poses.npy'
            # poses_mat = np.load( file_path, allow_pickle=True )[0]
            # mat = pybullet_utils.mat_to_pose(poses_mat)
            sizes = np.load(args.Data.Blockout.data_path)
            names = args.Data.Blockout.data_names
            for size in sizes:
                position = init_position + size
                pose = [position, rotation]
                pose_mat = pybullet_utils.pose_to_mat(pose)
                poses_mat.append(pose_mat)
        

            # poses_mat = np.tile(pose_mat, (sizes.shape[0], 1, 1))
            info = names
        elif data == 'armbench':
            sizes = np.load(args.Data.Armbench.data_path)
            names = args.Data.Armbench.data_names
            for size in sizes:
                position = init_position + size
                pose = [position, rotation]
                pose_mat = pybullet_utils.pose_to_mat(pose)
                poses_mat.append(pose_mat)
            info = names
        elif data == 'abc':
            sizes = np.load(args.Data.Abc.data_path)
            names = args.Data.Abc.data_names
            for size in sizes:
                position = init_position + size
                pose = [position, rotation]
                pose_mat = pybullet_utils.pose_to_mat(pose)
                poses_mat.append(pose_mat)
            info = names
        
        poses_mat = np.array(poses_mat) 
        self.box_names = names
        
        return len(sizes), sizes, poses_mat, info


    def load_data(self, data, data_config, args):
        tex_path = args.Data.tex_path
        self.textures = None
        if os.path.exists(tex_path):
            self.textures = np.load(tex_path, allow_pickle=True)

        position = np.array(args.Data.init_position)
        rotation = np.array(args.Data.init_rotation)
        pose = [position, rotation]
        pose_mat = pybullet_utils.pose_to_mat(pose)

        self.box_ids = []
        self.box_urdf = []

        if data == 'random':
            sizes = np.load(args.Data.Random.data_path, allow_pickle=True )
            poses_mat = np.tile(pose_mat, (sizes.shape[0], 1, 1))
            info = None

        elif data == 'time_series':
            file_path = args.Data.Time_series.data_path + '/data_time_series_' + str(data_config) + '.pt'
            data = torch.load(file_path)
            sizes = np.array(data['data'])
            poses_mat = np.tile(pose_mat, (sizes.shape[0], 1, 1))
            info = None
            # info = df[['Material', 'Case_pick_start_time']].to_numpy()

        elif data == 'occupancy':
            file_path = args.Data.Occupancy.data_path + '/data_occupancy_' + str(data_config) + '.pt'
            data = torch.load(file_path)
            sizes = np.array(data['data'])
            poses_mat = np.tile(pose_mat, (sizes.shape[0], 1, 1))
            info = None
            # info = df[['商品code', '名称']].to_numpy()

        elif data == 'flat_long':
            file_path = args.Data.Flat_long.data_path + '/data_flat_long_' + str(data_config) + '.pt'
            data = torch.load(file_path)
            sizes = np.array(data['data'])

            poses_mat = np.tile(pose_mat, (sizes.shape[0], 1, 1))
            info = None
            # 创建info数组，包含从1开始的连续整数
            # info = np.arange(1, sizes.shape[0] + 1).reshape(-1, 1)

        return len(sizes), sizes, poses_mat, info

    def load_data_old(self, data, test_data_config, args):
        tex_path = args.Data.tex_path
        self.textures = None
        if os.path.exists(tex_path):
            self.textures = np.load(tex_path, allow_pickle=True)

        position = np.array(args.Data.init_position)
        rotation = np.array(args.Data.init_rotation)
        pose = [position, rotation]
        pose_mat = pybullet_utils.pose_to_mat(pose)

        self.box_ids = []
        self.box_urdf = []

        if data == 'random':
            sizes = np.load(args.Data.Random.data_path, allow_pickle=True)
            poses_mat = np.tile(pose_mat, (sizes.shape[0], 1, 1))
            info = None

        elif data == 'time_series':
            file_path = 'packing_dataset/宝洁-Simulation+data.xlsx'
            df = pd.read_excel(file_path, engine='openpyxl', index_col=None)
            # Extract the relevant columns
            sizes = df[['Length_cm', 'Width_cm', 'Height_cm']].to_numpy() / 100  # 将单位由cm转换为m
            # sizes = sizes / 2   # 此处缩小四倍方便packing验证
            poses_mat = np.tile(pose_mat, (sizes.shape[0], 1, 1))
            info = df[['Material', 'Case_pick_start_time']].to_numpy()

        elif data == 'occupancy':
            file_path = 'packing_dataset/得力选品desk.xlsx'
            df = pd.read_excel(file_path, sheet_name='道口物料清单', engine='openpyxl', index_col=None)
            # Extract the relevant columns
            sizes = df[['尺寸（L）', '尺寸（W）', '尺寸（H）']].to_numpy() / 100  # 将单位由cm转换为m
            # sizes = sizes / 3   # 此处缩小四倍方便packing验证
            poses_mat = np.tile(pose_mat, (sizes.shape[0], 1, 1))
            info = df[['商品code', '名称']].to_numpy()

        elif data == 'flat_long':
            folder_path = 'packing_dataset/欧派家居opai'
            # 初始化一个集合来存储去重后的数据
            unique_data = set()
            # 读取文件夹内的每个txt文件
            for file_name in sorted(os.listdir(folder_path)):
                if file_name.endswith('.txt'):
                    file_path = os.path.join(folder_path, file_name)
                    with open(file_path, 'r') as file:
                        for line in file:
                            # 将每一行数据转换为元组，并加入集合进行去重
                            data_tuple = tuple(map(float, line.strip().split()))
                            if data_tuple[0] > 200 or data_tuple[1] > 200:
                                continue
                            unique_data.add(data_tuple)
            # 将集合转换为numpy数组
            sizes = np.array(list(unique_data)) / 100  # 将单位由cm转换为m
            sizes = sizes / 2  # 此处缩小两倍方便packing验证
            poses_mat = np.tile(pose_mat, (sizes.shape[0], 1, 1))
            # 创建info数组，包含从1开始的连续整数
            info = np.arange(1, sizes.shape[0] + 1).reshape(-1, 1)

        return len(sizes), sizes, poses_mat, info




    def load_irregular_scene(self, num, rotation_flag, poses_mat, sizes):
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)

        # for i in range(len(poses_mat)):
        mat = poses_mat[num]
        size = sizes[num]
        name = self.box_names[num]
        
        if rotation_flag == 0:
            pass
        else:  # 将box旋转
            size[0], size[1] = size[1], size[0]

        pose = pybullet_utils.mat_to_pose(mat)  # mat (4, 4) -> pose 3),4)
        pose[0] += self.init_box_offset
        

        color = colors[np.random.randint(len(colors))]

        color = pybullet_utils.color_random(color)
        if self.is_3d_packing:
            # blockout
            objPath = self.objPath + '/' + name + '_0.obj'
        else:
            objPath = self.objPath_2d + '/' + name + '.obj'
        box_id = self.addObject_irregular(rotation_flag, objPath, pose, color)

        self.box_ids.append(box_id)
        # self.box_urdf.append(obj_urdf)

        if self.textures is not None:  # 纹理
            p.changeVisualShape(box_id, -1, textureUniqueId=self.tex_uids[self.textures[np.random.randint(len(self.textures))]])

        self.refresh_boxes()
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

        pybullet_utils.simulate_step(1000)
        
        pose_real = p.getBasePositionAndOrientation(box_id)[0]
        pose[0] = np.array(pose_real)
        mat = pybullet_utils.pose_to_mat(pose)  # pose 3),4) -> mat (4, 4)

        return mat, size



    def load_scene(self, num, rotation_flag, poses_mat, sizes):
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)

        mat = poses_mat[num]
        size = sizes[num]
        if rotation_flag == 0:
            pass
        else:   # 将box旋转
            size[0], size[1] = size[1], size[0]
        
        pose = pybullet_utils.mat_to_pose(mat)      # mat (4, 4) -> pose 3),4)
        pose[0] += self.init_box_offset
        # mat = pybullet_utils.pose_to_mat(pose)      # pose 3),4) -> mat (4, 4)
        target_pose = copy.deepcopy(pose)
        prepare_pose = copy.deepcopy(pose)
        prepare_pose[0][0] = prepare_pose[0][0] + 2
        
        if len(poses_mat) <= len(colors):
            color = colors[num]
        else:
            color = colors[np.random.randint(len(colors))]
        
        # color = pybullet_utils.color_random(color)
        box_id, obj_urdf = self.add_box(prepare_pose, size, color, use_tex=True )   # here
        
        self.box_ids.append(box_id)
        self.box_urdf.append(obj_urdf)

        if self.target_container_size[1] == 250:
            # 获取物体的当前位置和方向
            position, orientation = p.getBasePositionAndOrientation(box_id)

            p.resetBasePositionAndOrientation(box_id, position, (0,0,1,1))

        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        pybullet_utils.simulate_step(1000)
        prepare_final_pose = p.getBasePositionAndOrientation(box_id)
        target_pose[0][-1] = prepare_final_pose[0][-1]
        target_final_pose = (tuple(target_pose[0]), tuple(target_pose[1]))

        if self.target_container_size[1] == 250:
            target_final_pose = (tuple(target_pose[0]), tuple(prepare_final_pose[1]))

        # 匀速移动2s到target位置
        transmit(box_id, prepare_final_pose, target_final_pose)

        final_pose = p.getBasePositionAndOrientation(box_id)
        mat = pybullet_utils.pose_to_mat(final_pose)  # pose 3),4) -> mat (4, 4)
        
        return mat, size



    def refresh_boxes(self):
        if self.textures is not None:
            for i, box_id in enumerate(self.box_ids):
                p.changeVisualShape(box_id, -1, textureUniqueId=self.tex_uids[ self.textures[i] ])

    def add_wall_v2(self, target_origin, target_container_size, block_gap, block_unit, wall_width, wall_height, only_base=False):
        # block_width = target_width * (block_unit + block_gap * 2 + block_gap) + wall_width * 2
        l, w, h = target_container_size
        block_length = l * (block_unit + block_gap * 2) + wall_width * 2      # here
        block_width = w * (block_unit + block_gap * 2) + wall_width * 2
        # block_length = l * (block_unit + block_gap) + wall_width * 2
        # block_width = w * (block_unit + block_gap) + wall_width * 2
        half_length = block_length / 2.0
        half_width = block_width / 2.0
        half_height = wall_height / 2.0

        tx = target_origin[0]
        ty = target_origin[1]
        target_center = np.array([tx + l * block_unit / 2.0, ty + w * block_unit / 2.0, 0])
        target_center += [wall_width, wall_width, 0]

        wall_color = (0.79, 0.196, 0.198, 1)

        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        container_offset = np.array([0, 0, 0.025])
        wall_1 = None
        if not only_base:
            wall_1, wall_1_urdf = self.add_box(
                (target_center + container_offset + [0, half_width, half_height], (0, 0, 0, 1)),
                (block_length + wall_width, wall_width, wall_height), color=wall_color, category='fixed', mass=0)
            wall_2, wall_2_urdf = self.add_box(
                (target_center + container_offset - [0, half_width, -half_height], (0, 0, 0, 1)),
                (block_length + wall_width, wall_width, wall_height), color=wall_color, category='fixed', mass=0)
            wall_3, wall_3_urdf = self.add_box(
                (target_center + container_offset + [half_length, 0, half_height], (0, 0, 0, 1)),
                (wall_width, block_width + wall_width, wall_height), color=wall_color, category='fixed', mass=0)
            wall_4, wall_4_urdf = self.add_box(
                (target_center + container_offset - [half_length, 0, -half_height], (0, 0, 0, 1)),
                (wall_width, block_width + wall_width, wall_height), color=wall_color, category='fixed', mass=0)

        wall_5, wall_5_urdf = self.add_box((target_center + container_offset, (0, 0, 0, 1)),
                                           (block_length, block_width, wall_width), color=(0.9, 0.2, 0.2, 1),
                                           category='fixed', mass=0)

        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)


        if wall_1 is not None:
            return [wall_1, wall_2, wall_3, wall_4, wall_5]
        else:
            return [wall_5]
        # return [ container_id ]

    def add_wall(self, target_origin, target_width, block_gap, block_unit, wall_width, wall_height, only_base=False):
        block_width = target_width * (block_unit + block_gap * 2) + wall_width * 2
        half_width = block_width / 2.0
        half_height = wall_height / 2.0
        
        tx = target_origin[0]
        ty = target_origin[1]
        target_center = np.array( [  tx + target_width * block_unit / 2.0, ty + target_width * block_unit / 2.0, 0 ] )
        target_center += [ wall_width, wall_width, 0 ]

        wall_color = (0.79, 0.196, 0.198, 1)
        
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        container_offset = np.array([0, 0, 0.025])
        wall_1 = None
        if not only_base:
            wall_1, wall_1_urdf = self.add_box((  target_center + container_offset + [ 0, half_width, half_height ] , (0, 0, 0, 1)),  (block_width + wall_width, wall_width, wall_height), color=wall_color, category='fixed', mass=0 )
            wall_2, wall_2_urdf = self.add_box((  target_center + container_offset - [ 0, half_width, -half_height ] , (0, 0, 0, 1)),  (block_width + wall_width, wall_width, wall_height), color=wall_color, category='fixed', mass=0 )
            wall_3, wall_3_urdf = self.add_box((  target_center + container_offset + [ half_width, 0,  half_height ] , (0, 0, 0, 1)),  (wall_width, block_width + wall_width, wall_height), color=wall_color, category='fixed', mass=0 )
            wall_4, wall_4_urdf = self.add_box((  target_center + container_offset - [ half_width, 0, -half_height ] , (0, 0, 0, 1)),  (wall_width, block_width + wall_width, wall_height), color=wall_color, category='fixed', mass=0 )
        wall_5, wall_5_urdf = self.add_box((  target_center + container_offset, (0, 0, 0, 1)),  (block_width, block_width, wall_width), color=(0.9, 0.2, 0.2, 1), category='fixed', mass=0 )
        
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        
        if self.recorder is not None:
            if wall_1 is not None:
                self.recorder.register_object(wall_1, wall_1_urdf)
                self.recorder.register_object(wall_2, wall_2_urdf)
                self.recorder.register_object(wall_3, wall_3_urdf)
                self.recorder.register_object(wall_4, wall_4_urdf)
            self.recorder.register_object(wall_5, wall_5_urdf)
            
        if wall_1 is not None:
            return [wall_1, wall_2, wall_3, wall_4, wall_5]
        else:
            return [wall_5]
        # return [ container_id ]


    def pick_block(self, block:Block):

        pick_pose = block.grasp_poses['down']

        pick_pose = pybullet_utils.mat_to_pose(pick_pose)

        pick_pose = utils.multiply( pick_pose, ( (0,0,0), utils.eulerXYZ_to_quatXYZW((np.pi, 0, 0)) ) )

        new_pose_z = pick_pose[0][2] + block.size['z']  # 在pose的z坐标再加上height
        new_rotation = np.array([pick_pose[1][3], pick_pose[1][2], pick_pose[1][1], pick_pose[1][0]])   # 调整顺序
        # new_rotation = [9.59142239e-01, 2.82924257e-01, 1.15512163e-04, 1.29994100e-04]
        new_pose = ((pick_pose[0][0], pick_pose[0][1], new_pose_z), new_rotation)    

        max_count = 1
        while True:
            max_count -= 1
            pick_success = self.pick(new_pose)
            # pick_success = self.pick(((0.3838593363761902, -0.2102079540491104, 0.1123405247926712), (-0.00012999412138015032, -0.00011551217176020145, 0.2829242944717407, 0.95914226770401)))
            if pick_success:
                break
            if max_count <= 0:
                break

        # while not self.robot.is_static:
        #     pybullet_utils.simulate_step(1)

        return pick_success, new_pose



    def place_at_irregular_2d(self, bodyUniqueId, position, rotation_flag, size, offset, pos_offset=None, rot_offset=None):

        if rotation_flag == 0:
            rotation = [0, 0, 0, 1]
        elif rotation_flag == 1:      # 90
            rotation = [0, 0, 1, 1]
        elif rotation_flag == 2:    # 180
            rotation = [0, 0, 1, 0]
        elif rotation_flag == 3:    # 270
            rotation = [0, 0, 1, 1]
        rotation = [0, 0, 0, 1]
        
        position = position * 1.0
        position[-1] += size[-1] / 2      # 此处做了修改，之后check这个高度对不对

        real_position = [
            offset[0] + position[0],
            offset[1] + position[1],
            offset[2] + position[2]
        ]

        pose_mat = np.identity(4)
        # pose[:3, :3] = rotation
        pose_mat[:3, 3] = real_position

        init_rotation = np.identity(3)
        if pos_offset is not None:
            rot_pos_offset = init_rotation @ pos_offset

        if rot_offset is not None:
            init_rotation = init_rotation @ rot_offset
        
        rot = np.identity(4)
        rot[:3, :3] = init_rotation

        pose_mat = pose_mat @ rot
        pose_mat = pose_mat[:3]

        if pos_offset is not None:
            pose_mat[:3, 3] += rot_pos_offset

        pose = pybullet_utils.mat_to_pose(pose_mat)
        max_count = 1
        
        while True:
            max_count -= 1
            height_pose = [ list(i) for i in pose ]
            height_pose[0][-1] += self.height_pose_offset_up_2d
            
            move_success, _ = self.move_ee_to(height_pose)         # 移动到pose处   [3),4)]
            
            if not move_success:
                break
            
            self.robot.ee.release()
            self.robot.update_robot_ompl()

            # height_pose[0][-1] -= 0.253         # 当为irrgular时注释掉
            p.resetBasePositionAndOrientation(bodyUniqueId, height_pose[0], rotation)   
            

            height_pose[0][-1] -= self.height_pose_offset_down_2d   # 0.117
            p.resetBasePositionAndOrientation(bodyUniqueId, height_pose[0], rotation)
            p.changeDynamics(bodyUniqueId, -1, mass=0)
            self.obstacles.append(bodyUniqueId)
            pybullet_utils.simulate_step(1000)

            if max_count <= 0:
                break

        
        while not self.robot.is_static:
            pybullet_utils.simulate_step(1)
            if self.recorder is not None:
                self.recorder.add_keyframe()
        
        # pybullet_utils.simulate_step(20)
        for _ in range(20):
            pybullet_utils.simulate_step(1)
            if self.recorder is not None:
                self.recorder.add_keyframe()

        return move_success, pose_mat


    def place_at_irregular(self, setting, bodyUniqueId, position, rotation, size, offset, pos_offset=None, rot_offset=None):
        stability_reward = None
        position = position * 1.0
        position[-1] += size[-1] / 2      # 此处做了修改，之后check这个高度对不对
        position[0] = self.target_container_size[0] - (position[0] + size[0])
        position[1] = self.target_container_size[1] - (position[1] + size[1])
        
        position  = np.array(position) * (self.block_unit + self.block_gap)

        size = np.array(size) * ( self.block_unit + self.block_gap )
        
        real_position = [
            offset[0] + position[0] + size[0]/2,
            offset[1] + position[1] + size[1]/2,
            offset[2] + position[2] + size[2]/2
        ]

        pose_mat = np.identity(4)
        # pose[:3, :3] = rotation
        pose_mat[:3, 3] = real_position

        init_rotation = np.identity(3)
        if pos_offset is not None:
            rot_pos_offset = init_rotation @ pos_offset

        if rot_offset is not None:
            init_rotation = init_rotation @ rot_offset
        
        rot = np.identity(4)
        rot[:3, :3] = init_rotation

        pose_mat = pose_mat @ rot
        pose_mat = pose_mat[:3]

        if pos_offset is not None:
            pose_mat[:3, 3] += rot_pos_offset

        pose = pybullet_utils.mat_to_pose(pose_mat)
        max_count = 1
        
        while True:
            stop_flag = True
            max_count -= 1
            height_pose = [ list(i) for i in pose ]
            height_pose[0][-1] += self.height_pose_offset_up    # 0.2

            if setting == 3:
                move_success, _ = self.move_ee_to(height_pose)         # 移动到pose处   [3),4)]
            
                if not move_success:
                    break
            
                self.robot.ee.release()
                self.robot.update_robot_ompl()
            else:
                move_success = True

            # height_pose[0][-1] -= 0.253         # 当为irrgular时注释掉
            p.resetBasePositionAndOrientation(bodyUniqueId, height_pose[0], rotation)   
            
            # 循环直到box与其他物体接触
            while True:
                height_pose[0][-1] -= 0.001
                p.resetBasePositionAndOrientation(bodyUniqueId, height_pose[0], rotation)   
                pybullet_utils.simulate_step(1)
                contact_points = []
                # 检测当前物体是否与任何其他物品接触
                for packed_id in self.obstacles:
                    contact_points = p.getContactPoints(bodyUniqueId, packed_id)
                    if contact_points:  # 如果有接触点
                        print(f"contact_points: {contact_points}")
                        break  # 找到接触点，退出循环
                    else:
                        continue
                if contact_points:  # 如果有接触点
                    break  # 找到接触点，退出循环
                # break  # 退出循环

            if setting == 1:
                # 将盒子的线速度和角速度设置为零
                p.resetBaseVelocity(bodyUniqueId, linearVelocity=[0, 0, 0], angularVelocity=[0, 0, 0])
                # 将盒子的质量设置为零，使其成为静态物体
                p.changeDynamics(bodyUniqueId, -1, mass=0)

            if setting == 1:
                stability_reward, stop_flag, now_position_offset = None, None, None
            else:
                # init
                num_steps = 30
                velocities = {}
                positions_init = {}
                for i in range(bodyUniqueId - self.containers[0][-1]):
                    id = self.containers[0][-1] + i + 1
                    velocities[str(id)] = []
                    positions_init[str(id)] = p.getBasePositionAndOrientation(id)[0]

                # 运行10个仿真步长并测量速度
                for step in range(num_steps):
                    p.stepSimulation()
                    for i in range(bodyUniqueId - self.containers[0][-1]):
                        id = self.containers[0][-1] + i + 1
                        linear_vel, angular_vel = p.getBaseVelocity(id)
                        velocities[str(id)].append([linear_vel, angular_vel])

                # For each object in the bin we select the maximum measured velocity
                max_linear_vel = {}
                max_angular_vel = {}
                for key, values in velocities.items():
                    p_linear, p_angular = [], []
                    for value in values:
                        now_linear = math.sqrt(sum(x ** 2 for x in value[0]))
                        now_angular = math.sqrt(sum(x ** 2 for x in value[1]))
                        p_linear.append(now_linear)
                        p_angular.append(now_angular)
                    p_linear_max = max(p_linear)
                    p_angular_max = max(p_angular)
                    max_linear_vel[key] = p_linear_max
                    max_angular_vel[key] = p_angular_max

                # Take the mean of all objects in the bin for linear and angular velocities respectively.
                id_final = []
                for i in range(bodyUniqueId - self.containers[0][-1]):
                    id = self.containers[0][-1] + i + 1
                    positions_final = p.getBasePositionAndOrientation(id)[0]
                    positions_offset = tuple(a - b for a, b in zip(positions_final, positions_init[str(id)]))
                    print(positions_offset)
                    # 使用any()函数检查列表中是否包含True
                    if any([abs(o) > 1e-05 for o in positions_offset]):
                        id_final.append(id)
                    if id == bodyUniqueId:
                        now_position_offset = positions_offset[0] + positions_offset[1]
                        stop_flag = any([positions_offset[0] > 1e-04, positions_offset[1] > 1e-04])

                selected_linear_values = [max_linear_vel[str(id)] for id in id_final]
                average_linear_value = sum(selected_linear_values) / len(selected_linear_values)
                selected_angular_values = [max_angular_vel[str(id)] for id in id_final]
                average_angular_value = sum(selected_angular_values) / len(selected_angular_values)

                # 计算linear_stability_reward
                linear_stability_reward = max(0, min(1, -(average_linear_value ** 0.4) + 1))
                angular_stability_reward = max(0, min(1, -(average_angular_value ** 0.3) + 1))
                stability_reward = linear_stability_reward * 0.5 + angular_stability_reward * 0.5

            pybullet_utils.simulate_step(1000)
            self.obstacles.append(bodyUniqueId)

            if max_count <= 0:
                break
        if setting == 3:
            while not self.robot.is_static:
                pybullet_utils.simulate_step(1)
        
        # pybullet_utils.simulate_step(20)
        for _ in range(20):
            pybullet_utils.simulate_step(1)

        return move_success, stability_reward, stop_flag, now_position_offset

    def place_at(self, setting, bodyUniqueId, position, rotation, size, offset, pos_offset=None, rot_offset=None):
        total_distance = 0
        stability_reward = None
        position = position * 1.0
        position[-1] += size[-1] / 2  # 此处做了修改，之后check这个高度对不对
        position[0] = self.target_container_size[0] - (position[0] + size[0])       # here
        position[1] = self.target_container_size[1] - (position[1] + size[1])

        position = np.array(position) * (self.block_unit + self.block_gap)

        size = np.array(size) * (self.block_unit + self.block_gap)

        real_position = [
            offset[0] + position[0] + size[0]/2,
            offset[1] + position[1] + size[1]/2,
            offset[2] + position[2] + size[2]/2
        ]

        pose_mat = np.identity(4)
        # pose[:3, :3] = rotation
        pose_mat[:3, 3] = real_position

        init_rotation = np.identity(3)
        if pos_offset is not None:
            rot_pos_offset = init_rotation @ pos_offset

        if rot_offset is not None:
            init_rotation = init_rotation @ rot_offset
        
        rot = np.identity(4)
        rot[:3, :3] = init_rotation

        pose_mat = pose_mat @ rot
        pose_mat = pose_mat[:3]

        if pos_offset is not None:
            pose_mat[:3, 3] += rot_pos_offset

        pose = pybullet_utils.mat_to_pose(pose_mat)
        max_count = 1
        target_pose = list(pose[0])

        while True:
            max_count -= 1
            height_pose = [ list(i) for i in pose ]
            height_pose[0][-1] += self.height_pose_offset_up    # 0.2

            if setting == 3:
                move_success, dangerous_flag, total_distance = self.move_ee_to(height_pose, allowed_time=self.ompl_plan_allowed_time)      # 移动到pose处   [3),4)]
                if dangerous_flag is True:
                    print("This packing operation is dangerous! Id: ", bodyUniqueId)
                if not move_success:
                    break

                self.robot.ee.release()
                self.robot.update_robot_ompl()
            else:
                move_success = True
                dangerous_flag = None

            height_pose[0][-1] -= self.height_pose_offset_down  # 0.253
            p.resetBasePositionAndOrientation(bodyUniqueId, height_pose[0], rotation)   
            
            # 循环直到box与其他物体接触
            while True:
                height_pose[0][-1] -= 0.001
                p.resetBasePositionAndOrientation(bodyUniqueId, height_pose[0], rotation)   
                pybullet_utils.simulate_step(1)
                contact_points = []
                # 检测当前物体是否与任何其他物品接触
                for packed_id in self.obstacles:
                    contact_points = p.getContactPoints(bodyUniqueId, packed_id)
                    if contact_points:  # 如果有接触点
                        print(f"contact_points: {contact_points}")
                        break  # 找到接触点，退出循环
                    else:
                        continue
                if contact_points:  # 如果有接触点
                    break  # 找到接触点，退出循环

            if setting == 1:
                # 将盒子的线速度和角速度设置为零
                p.resetBaseVelocity(bodyUniqueId, linearVelocity=[0, 0, 0], angularVelocity=[0, 0, 0])
                # 将盒子的质量设置为零，使其成为静态物体
                p.changeDynamics(bodyUniqueId, -1, mass=0)

            if setting == 1:
                stability_reward = 0
            else:
                # init
                num_steps = 200
                velocities = {}
                for i in range(bodyUniqueId - self.containers[0][-1]):
                    id = self.containers[0][-1] + i + 1
                    velocities[str(id)] = []
                # 运行200个仿真步长并测量速度
                for step in range(num_steps):
                    p.stepSimulation()
                    for i in range(bodyUniqueId - self.containers[0][-1]):
                        id = self.containers[0][-1] + i + 1
                        linear_vel, angular_vel = p.getBaseVelocity(id)
                        velocities[str(id)].append([linear_vel, angular_vel])

                # For each object in the bin we select the maximum measured velocity
                max_linear_vel = {}
                max_angular_vel = {}
                for key, values in velocities.items():
                    p_linear, p_angular = [], []
                    for value in values:
                        now_linear = math.sqrt(sum(x ** 2 for x in value[0]))
                        now_angular = math.sqrt(sum(x ** 2 for x in value[1]))
                        p_linear.append(now_linear)
                        p_angular.append(now_angular)
                    p_linear_max = max(p_linear)
                    p_angular_max = max(p_angular)
                    max_linear_vel[key] = p_linear_max
                    max_angular_vel[key] = p_angular_max

                # Take the max of all objects in the bin for linear and angular velocities respectively.
                max_linear_vel_key = max(max_linear_vel, key=max_linear_vel.get)
                max_linear_vel_value = max_linear_vel[max_linear_vel_key]
                max_angular_vel_key = max(max_angular_vel, key=max_angular_vel.get)
                max_angular_vel_value = max_angular_vel[max_angular_vel_key]

                # 计算linear_stability_reward
                linear_stability_reward = max(0, min(1, -(max_linear_vel_value ** 0.4) + 1))
                angular_stability_reward = max(0, min(1, -(max_angular_vel_value ** 0.3) + 1))
                stability_reward = linear_stability_reward * 0.5 + angular_stability_reward * 0.5

            pybullet_utils.simulate_step(1000)
            self.obstacles.append(bodyUniqueId)

            if max_count <= 0:
                break

        # if setting == 3:
        #     for obj_id in self.robot.obj_ids['rigid']:
        #         num_joints = p.getNumJoints(obj_id)
        #         for j in range(num_joints):
        #             # 设置关节速度为0，并使用PD控制模式
        #             p.setJointMotorControl2(bodyUniqueId=obj_id, jointIndex=j, controlMode=p.VELOCITY_CONTROL, targetVelocity=0, force=0)
        #     while not self.robot.is_static:
        #         for obj_id in self.robot.obj_ids['rigid']:
        #             num_joints = p.getNumJoints(obj_id)
        #             for j in range(num_joints):
        #                 # 设置关节速度为0，并使用PD控制模式
        #                 p.setJointMotorControl2(bodyUniqueId=obj_id, jointIndex=j, controlMode=p.VELOCITY_CONTROL,
        #                                         targetVelocity=0, force=0)
        #         print("robot is not static")
        #         pybullet_utils.simulate_step(1)
        
        # pybullet_utils.simulate_step(20)
        for _ in range(20):
            pybullet_utils.simulate_step(1)

        return move_success, stability_reward, target_pose, dangerous_flag, total_distance


    def remove_current_block(self, setting, bodyUniqueId, block_size, height_offset):
                            
        offset = self.remove_position.copy()
        offset[2] += height_offset

        self.place_at(1, bodyUniqueId, np.array([0, 0, 0], dtype=np.float32), [0, 0, 0, 1], block_size, offset)

        remove_success = True
        pybullet_utils.p.resetBasePositionAndOrientation( bodyUniqueId, [-10, -10, 5], [0,0,0,1] )
            
        return remove_success
    
    def new_container(self, only_base=False):
        for c in self.containers:
            for obj in c:
                pose = p.getBasePositionAndOrientation(obj)
                pose = [ list(i) for i in pose ]
                p.resetBasePositionAndOrientation( obj, pose[0] + self.plane_offset, pose[1] )
        walls = self.add_wall_v2(self.target_origin, self.target_container_size, self.block_gap, self.block_unit, self.wall_width, self.wall_height, only_base)
        # walls = self.add_wall_v2(self.target_origin, [120, 250, 100], self.block_gap, self.block_unit, self.wall_width, self.wall_height, only_base)

        self.containers.append([])
        self.containers[-1] += walls
    
    def addObject_irregular(self, rotation_flag, objPath, pose, color):
        mesh = trimesh.load(objPath)
        if self.is_3d_packing:
            scale = self.scale
        else:
            scale = self.scale_2d
        mesh.apply_scale(scale)
        
        position = pose[0]
        
        angle = np.deg2rad(0)  # 0度转换为弧度
        if rotation_flag == 1:      # 90
            angle = np.deg2rad(90)  # 90度转换为弧度
            # rotation = (0, 0, 1, 1)
            
        elif rotation_flag == 2:    # 180
            angle = np.deg2rad(180)  # 180度转换为弧度
        elif rotation_flag == 3:    # 270
            angle = np.deg2rad(90)  # 90度转换为弧度
        
        # # 目标四元数
        # quaternion = [0, 0, 0, 1]
        # 创建旋转矩阵
        quaternion = trimesh.transformations.quaternion_from_euler(0, 0, angle)
        # rotation_matrix = trimesh.transformations.quaternion_matrix(list(rotation))
        rotation_matrix = trimesh.transformations.quaternion_matrix(list(quaternion))
        mesh.apply_transform(rotation_matrix)
        
        # 获取网格的中心
        center = mesh.centroid
        # 创建将网格平移到原点的矩阵
        translation_to_origin = trimesh.transformations.translation_matrix(-center)
        mesh.apply_transform(translation_to_origin)
        
        objPath_resize = objPath.replace('.obj', '_resize.obj')
        mesh.export(objPath_resize)


        visual_shape_id = p.createVisualShape(shapeType=p.GEOM_MESH, fileName=objPath_resize, meshScale=[1,1,1])
        collision_shape_id = p.createCollisionShape(shapeType=p.GEOM_MESH, fileName=objPath_resize,
                                                    collisionFramePosition=[0.0, 0.0, 0.0], meshScale=[1,1,1])
        mass = mesh.volume
        object_id = p.createMultiBody(baseMass=mass, basePosition=position,
                                      baseOrientation=(0,0,0,1),
                                      baseCollisionShapeIndex=collision_shape_id,
                                      baseVisualShapeIndex=visual_shape_id,
                                      useMaximalCoordinates=True)
        color = pybullet_utils.color_random(utils.COLORS['brown'])
        p.changeVisualShape(object_id, -1, rgbaColor=color)
        p.changeDynamics(object_id, -1,
                         mass=0.5,       # 0.5
                         linearDamping = 0.1,
                         angularDamping = 0.1,
                         contactProcessingThreshold = 0,
                         )
        p.setGravity(0, 0, -9.8)  # 设置重力
        category = 'rigid'
        self.robot.add_object_to_list(category, object_id)
        self.obstacles.append(object_id)

        return object_id
    
    def addObject(self, rotation_flag, objPath, pose, color):
        mesh = trimesh.load(objPath)
        if self.is_3d_packing:
            scale = self.scale
        else:
            scale = self.scale_2d
        mesh.apply_scale(scale)

        position = pose[0]
        rotation = pose[1]
        
        if rotation_flag == 1:      # 90
            rotation = (0,0,1,1)
        elif rotation_flag == 2:    # 180
            rotation = (0,0,1,0)
        elif rotation_flag == 3:    # 270
            rotation = (0, 0, 1, 1)


        visual_shape_id = p.createVisualShape(shapeType=p.GEOM_MESH, fileName=objPath, meshScale=scale)
        collision_shape_id = p.createCollisionShape(shapeType=p.GEOM_MESH, fileName=objPath,
                                                    collisionFramePosition=[0.0, 0.0, 0.0], meshScale=scale)
        mass = mesh.volume
        object_id = p.createMultiBody(baseMass=mass, basePosition=position,
                                      baseOrientation=rotation,
                                      baseCollisionShapeIndex=collision_shape_id,
                                      baseVisualShapeIndex=visual_shape_id,
                                      useMaximalCoordinates=True)
        color = pybullet_utils.color_random(utils.COLORS['brown'])
        p.changeVisualShape(object_id, -1, rgbaColor=color)
        p.changeDynamics(object_id, -1,
                         mass=0.5,       # 0.5
                         linearDamping = 0.1,
                         angularDamping = 0.1,
                         contactProcessingThreshold = 0,
                         )
        p.setGravity(0, 0, -9.8)  # 设置重力
        category = 'rigid'
        self.robot.add_object_to_list(category, object_id)
        self.obstacles.append(object_id)

        return object_id

    def is_regular(self, method):
        if method in ['LeftBottom', 'HeightmapMin', 'PCT', 'PackE', 'LSAH', 'MACS', 'RANDOM', 'OnlineBPH', 'DBL', 'BR', 'SDFPack']:
            is_regular_data = True
        elif method in ['SDF_Pack', 'IR_BPP', 'MTPE', 'IR_HM', 'BLBF', 'FF', 'pack_2d']:
            is_regular_data = False
        else:
            print("error!")
            is_regular_data = None
        
        return is_regular_data
