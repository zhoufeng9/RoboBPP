import os
import sys
import copy
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import time
from tqdm import tqdm
import pybullet as p

import time
import cv2
import numpy as np
from irb6640_pybullet_env.irb6640_envs.envs import Env, colors
from irb6640_pybullet_env.irb6640_envs import cameras
from irb6640_pybullet_env.irb6640_envs import models

from irb6640_pybullet_env.utils import pybullet_utils
from irb6640_pybullet_env.utils import utils
from irb6640_pybullet_env.utils import pack

from scipy.spatial.transform import Rotation as R

mpl.use('Agg')

scale_pos = pybullet_utils.scale_pos

def fast_pack(blocks, container_size):
    cx, cy, cz = container_size
    heightmap = np.zeros([cx,cy])
    positions = []

    rand_num = 10
    for b in blocks:
        bx, by, bz = b
        rand_px = np.random.random_integers(0, cx-bx-1, (rand_num))
        rand_py = np.random.random_integers(0, cy-by-1, (rand_num))

        min_z = 100000
        min_one = 0
        for k in range(rand_num):
            px = rand_px[k]
            py = rand_py[k]
            pz = np.max(heightmap[px:px+bx, py:py+by])
            if pz < min_z:
                min_z = pz
                min_one = k
        
        px = rand_px[min_one]
        py = rand_py[min_one]
        pz = np.max(heightmap[px:px+bx, py:py+by])
    
        heightmap[px:px+bx, py:py+by] = pz + bz
        
        positions.append([px, py, pz])
        
    return np.array(positions)    

def init_cam():
    width = 640
    height = 480
    
    ratio = width / height

    # vision sensor > object properties > common
    focal_dist = 2.0
    
    if (ratio > 1):
        fov_x = np.deg2rad(60.0)
        fov_y = 2 * np.math.atan( np.math.tan(fov_x / 2) / ratio )
    else:
        fov_y = np.deg2rad(60.0)
        fov_x = 2 * np.math.atan( np.math.tan(fov_y / 2) / ratio )

    # fov = np.deg2rad(60.0)

    focal_x = width / ( focal_dist * np.math.tan( fov_x / 2 ))
    focal_y = height / ( focal_dist * np.math.tan( fov_y / 2 ))
    
    ins = np.identity(3)
    ins[0,0] = focal_x
    ins[1,1] = focal_y
    ins[0,2] = width / 2
    ins[1,2] = height / 2

    return list(np.reshape(ins, (-1)))

class TAP_Env(Env):
    def __init__(self, assets_root=models.get_data_path(), disp=False, shared_memory=False, hz=240, use_egl=False, globalScaling=1) -> None:
        super().__init__(assets_root, disp=disp, shared_memory=shared_memory, hz=hz, use_egl=use_egl)

        self.globalScaling = 1
        
        image_size = (480, 640)
        # intrinsics = (450., 0, 320., 0, 450., 240., 0, 0, 1)
        intrinsics = init_cam()
        
        # init_position = tuple(np.array((0.20327461, 0.05055285, 0.40699884)) * (self.globalScaling) )
        # init_position = tuple(np.array((0.20327461, 0.15055285, 0.50699884)) * (self.globalScaling) )
        init_position = tuple(np.array((0.20327461, 0.15055285, 0.50699884)) * (self.globalScaling) )
        init_rotation = (np.pi / 4, 4 * np.pi / 4, 1 * np.pi / 4)
        init_rotation = p.getQuaternionFromEuler(init_rotation)

        init_config = {
            'image_size': image_size,
            'intrinsics': intrinsics,
            'position': init_position,
            'rotation': init_rotation,
            'zrange': (0.01, 10.),
            'noise': False
        }
        
        tmp_position = [0.37944994, 0.08588685, 0.23093843]
        tmp_rotation = (np.pi / 4, 4 * np.pi / 4, 3 * np.pi / 4)
        tmp_rotation = p.getQuaternionFromEuler(tmp_rotation)

        tmp_config = {
            'image_size': image_size,
            'intrinsics': intrinsics,
            'position': tmp_position,
            'rotation': tmp_rotation,
            'zrange': (0.01, 10.),
            'noise': False
        }

        self.blocks_size = []

        self.cams['init'] = cameras.Camera(init_config, self._random)
        # TODO
        self.cams['tmp'] = cameras.Camera(tmp_config, self._random)

        self.test_cams = []

        self.add_cam(( init_position, init_rotation))
        self.add_cam( ( scale_pos([0.20327461, 0.15055285, 0.45699884], self.globalScaling), init_rotation))
        self.add_cam( ( scale_pos([0.64524717, 0.06026357, 0.23334967], self.globalScaling),[ 0.12379133,  0.80005916, -0.57380898, -0.12379133]) )
        self.add_cam( ( scale_pos([0.20327461, 0.15055285, 0.55699884], self.globalScaling), init_rotation))

        # self.add_cam( ([0.45885849, 0.1093576 , 0.21334855],[-0.02718741,  0.80400643, -0.59337623,  0.02718741]) )
        self.add_cam( ( scale_pos([ 0.92799311, -0.68947782,  0.25822451], self.globalScaling), [ 0.82474008,  0.16867879, -0.16867879, -0.51273652]) )
        self.add_cam( ( scale_pos([ 0.39724696, -0.63417996,  0.38054319], self.globalScaling), [ 0.86221021, -0.02407117,  0.02407117, -0.5054055 ]) )
        self.add_cam( ( scale_pos([0.3604007 , 0.47290363, 0.58034994], self.globalScaling), [-0.1931027 ,  0.83019314, -0.48600622,  0.1931027 ]) )
        self.add_cam( ( scale_pos([ 0.74692205, -0.58308107,  0.36238718], self.globalScaling), [ 0.8693235 ,  0.1441227 , -0.1441227 , -0.45025986]) )

        self.add_cam( ( scale_pos([ 0.92799311, -1.08947782,  0.65822451], self.globalScaling),[ 0.82474008,  0.16867879, -0.16867879, -0.51273652]) )
        self.add_cam( ( scale_pos([ 0.94692205, -0.78308107,  0.56238718], self.globalScaling),[ 0.8693235 ,  0.1441227 , -0.1441227 , -0.45025986]) )

        self.add_cam( ( scale_pos([ 0.1, -0.78947782,  0.75822451], self.globalScaling), p.getQuaternionFromEuler( [np.pi / 4, 4 * np.pi / 4, 3.5 * np.pi / 4] ) ), (640, 640) )
        self.add_cam( ( scale_pos([ 0.1, -0.58947782,  0.65822451], self.globalScaling), p.getQuaternionFromEuler( [np.pi / 4, 4 * np.pi / 4, 3 * np.pi / 4] ) ), (640, 640) )
        # self.add_cam( ([ 0.05, -0.38947782,  0.65822451], p.getQuaternionFromEuler( [np.pi / 4, 4 * np.pi / 4, 2.4 * np.pi / 4] ) ), (640, 640) )
        self.add_cam( ( scale_pos([ 0.1, -0.44947782,  0.65822451], self.globalScaling), p.getQuaternionFromEuler( [np.pi / 4, 4 * np.pi / 4, 3 * np.pi / 4] ) ), (640, 640) )
        self.add_cam( ( scale_pos((0.10327461, 0.15055285, 0.55699884), self.globalScaling), init_rotation))
        self.texture_ids = []

    def add_cam(self, pose, image_size=(480, 640)):
        
        # intrinsics = (450., 0, 320., 0, 450., 240., 0, 0, 1)
        intrinsics = init_cam()
        
        # rot = ((0,0,0), utils.eulerXYZ_to_quatXYZW((0,0,np.pi)))
        # pose = utils.multiply(pose, rot)
        position = pose[0]
        rotation = pose[1]

        config = {
            'image_size': image_size,
            'intrinsics': intrinsics,
            'position': position,
            'rotation': rotation,
            'zrange': (0.01, 10.),
            'noise': False
        }
        self.test_cams.append( cameras.Camera(config, self._random) )

    def init_scene(self, \
        block_tasks, \
        # init_container_origin=[0.35, -0.35, 0], \
        init_container_origin=[0.35, -0.35, 0], \
        init_container_size=[7,7,20],\
        init_space_width=0.5, init_space_height=0.6, use_mean=True, unit_length=1):

        # init_container_origin=[0.4, -0.35, 0], \
        init_container_origin = scale_pos(init_container_origin, self.globalScaling)
        # init_container_size = scale_pos(init_container_size, self.globalScaling)

        init_wall_width = 0.1

        wall1 = self.add_box(( scale_pos( (init_space_width,  init_wall_width * 0.55, init_space_height/2 + 0.05), self.globalScaling), (0, 0, 0, 1)),  
                             scale_pos((init_space_width, init_wall_width, init_space_height), self.globalScaling), color=(0,0,0, 1), category='fixed' )
        wall2 = self.add_box(( scale_pos( (0.25 - init_wall_width * 0.55, -0.25, init_space_height/2 + 0.05), self.globalScaling), (0, 0, 0, 1)),  
                             scale_pos((init_wall_width, init_space_width, init_space_height), self.globalScaling), color=(0,0,0, 1), category='fixed' )
        wall3 = self.add_box((  scale_pos((0.25 + init_space_width + init_wall_width * 0.55, -0.25, init_space_height/2 + 0.05), self.globalScaling), (0, 0, 0, 1)),  
                             scale_pos((init_wall_width, init_space_width, init_space_height), self.globalScaling), color=(0,0,0, 1), category='fixed' )
        wall4 = self.add_box(( scale_pos((init_space_width, -init_space_width - init_wall_width * 0.55, init_space_height/2 + 0.05), self.globalScaling), (0, 0, 0, 1)),  
                             scale_pos((init_space_width, init_wall_width, init_space_height), self.globalScaling), color=(0,0,0, 1), category='fixed' )

        wall1 = wall1[0]
        wall2 = wall2[0]
        wall3 = wall3[0]
        wall4 = wall4[0]

        p.changeDynamics(wall1, -1, mass=100)
        p.changeDynamics(wall2, -1, mass=100)
        p.changeDynamics(wall3, -1, mass=100)
        p.changeDynamics(wall4, -1, mass=100)
        

        self.texture_ids = []
        p.setGravity(0, 0, -9.8)

        for task in block_tasks:
            block_unit = task['unit']
            block_num = task['num']
            block_min_size = task['min_size']
            block_max_size = task['max_size']
            is_mess = task['is_mess']
            self.generate_boxes( block_num, block_min_size, block_max_size, init_container_size, init_container_origin, \
                block_unit, is_mess, use_mean=use_mean, unit_length=unit_length )
            pybullet_utils.simulate_step(20)
            # pybullet_utils.simulate_step(100)

        pybullet_utils.simulate_step(2000)
        # pybullet_utils.simulate_step(500)

        # p.setGravity(0, 0, -9.8 / self.globalScaling)

        p.removeBody(wall1)
        p.removeBody(wall2)
        p.removeBody(wall3)
        p.removeBody(wall4)
        
        pybullet_utils.simulate_step(2000)
        

    def add_boxes(self, zone_pose):
        # Add stack of boxes on pallet.
        margin = 0.01
        object_ids = []

        stack_size = (0.19, 0.19, 0.19)

        stack_dim = np.int32([2, 3, 3])
        # stack_dim = np.random.randint(low=2, high=4, size=3)
        box_size = (stack_size - (stack_dim - 1) * margin) / stack_dim
        for z in range(stack_dim[2]):

            # Transpose every layer.
            stack_dim[0], stack_dim[1] = stack_dim[1], stack_dim[0]
            box_size[0], box_size[1] = box_size[1], box_size[0]

            for y in range(stack_dim[1]):
                for x in range(stack_dim[0]):
                    position = list((x + 0.5, y + 0.5, z + 0.5) * box_size)
                    position[0] += x * margin - stack_size[0] / 2
                    position[1] += y * margin - stack_size[1] / 2
                    position[2] += z * margin + 0.03
                    pose = (position, (0, 0, 0, 1))
                    pose = utils.multiply(zone_pose, pose)
                    
                    box_id = self.add_box(pose, box_size)
                    box_id = box_id[0]

                    object_ids.append((box_id, (0, None)))
                    pybullet_utils.color_random_brown(box_id)

    def generate_boxes(self, blocks_num, block_min_size, block_max_size, \
        container_size, container_origin, block_unit, is_mess=False, loop_num=1, use_mean=True, unit_length=1):
        ''' Generate blocks in simulator

        Args:
            - `blocks_num`: int
            - `block_min_size`: int
            - `block_max_size`: int
            - `container_size`: list(int) [3]
            - `container_origin`: list(float) [3]
                * left down corner of container
            - `block_unit`: float
                * how many meters each unit is
            - `is_mess`: boolean
                * True if we want to generate random placement
            - `loop_num`: int
                * how many time you want to generate blocks data (usually 1)
        
        Returns:
            - `positions`: np.array [n, 3]
            - `blocks`: np.array [n, 3]
        '''

        if len(self.blocks_size) == 0:
            self.blocks_size = []
        
        # generate blocks
        for _ in range(loop_num):
            while True:
                # blocks = np.random.randint( block_min_size, block_max_size+1, (blocks_num, 3) )
                
                size_list = [ i for i in range(block_min_size, block_max_size+1, unit_length) ]
                if use_mean:
                    slen = len(size_list)
                    prob_blocks = np.ones(slen) / (slen * 1.0)
                    
                else:
                    # print('generate ', size_list)

                    # Gaussian distribution
                    mu = 0.5
                    sigma = 0.16
                    prob_x = np.linspace(mu - 3*sigma, mu + 3*sigma, len(size_list))
                    prob_blocks = np.exp( - (prob_x-mu)**2 / (2*sigma**2) ) / (np.sqrt(2*np.pi) * sigma)
                    prob_blocks = prob_blocks / np.sum(prob_blocks)

                blocks = np.random.choice(size_list, (blocks_num, 3), p=prob_blocks )
                
                # positions, container, stable, ratio, scores = pack.calc_positions_lb_greedy(blocks, container_size)
                # if np.sum(stable) == blocks_num:
                #     break

                positions = fast_pack(blocks, container_size)
                break

        # add block into scene
        for i in range(blocks_num):
            quat = (0, 0, 0, 1)
            
            texture_id = np.random.randint(0, 10)
            self.texture_ids.append(texture_id)

            if is_mess:
                # quat = utils.eulerXYZ_to_quatXYZW( np.random.rand(3) * np.math.pi / 6 )
                quat = utils.eulerXYZ_to_quatXYZW( [0, np.random.rand() * np.math.pi / 7, np.random.rand() * np.math.pi / 7 ])
            
            size = blocks[i] * block_unit 
            self.blocks_size.append(size)
            
            pos = positions[i]
            # pos[0] = container_size[0] - pos[0]
            # pos[1] = container_size[1] - pos[1]
            pos = pos * block_unit + size/2
            pos += container_origin
            # 加点高度掉落
            # pos += [0, 0, 0.15]
            # pos += scale_pos([0, 0, 0.25 + 0.0005 * i], self.globalScaling)
            pos += scale_pos([0, 0, 0.05 + 0.01 * i], self.globalScaling)

            # 这里 size 减去一点点大小是为了生成间距的数据，这样可能避免碰撞，方便抓取规划？
            # margin = 0.0002
            margin = 0
            color = colors[np.random.randint(len(colors))]
            color = pybullet_utils.color_random(color)

            box_id = self.add_box( (pos, quat), size - margin, color, use_tex=True )
            box_id = box_id[0]
            
            p.changeVisualShape(box_id, -1, textureUniqueId=self.tex_uids[texture_id])


        return blocks
    
    def reset(self, show_gui=True):
        super().reset(show_gui=show_gui)
        self.blocks_size = []

    
    def take_images(self, cam_name):
        if type(cam_name) == int:
            rgb, dep, seg = self.test_cams[cam_name].take_images()
        else:
            rgb, dep, seg = self.cams[cam_name].take_images()
        return rgb, dep, seg


    def save_scene(self, save_path, cam_id):

        os.makedirs(save_path, exist_ok=True)


        poses = []
        sizes = []
        ids = []
        
        for i in self.robot.obj_ids['rigid']:
            pose = p.getBasePositionAndOrientation(i)
            pose = pybullet_utils.pose_to_mat(pose)
            ids.append(i)
            poses.append(pose)

            # aabb = p.getAABB(i)
            # aabb = np.array(aabb)
            # size = aabb[1] - aabb[0]
            
            # sizes.append(size)

        np.save( os.path.join( save_path, "id" ), ids )
        np.save( os.path.join( save_path, "poses" ), poses )
        np.save( os.path.join( save_path, "sizes" ), self.blocks_size )
        np.save( os.path.join( save_path, "textures" ), self.texture_ids )
    
        # intrinsics = self.cams['init'].get_intrinsics()
        # cam_pose = self.cams['init'].get_pose()

        if cam_id >= 0:
            cam_path = os.path.join(save_path, str(cam_id))
            os.makedirs(cam_path, exist_ok=True)
            
            intrinsics = self.test_cams[cam_id].get_intrinsics()
            cam_pose = self.test_cams[cam_id].get_pose()

            cam = {
                'intrinsics': intrinsics,
                'pose': cam_pose
            }
            rgb, dep, seg = self.take_images(cam_id)

            seg_ids = np.unique(seg)

            for seg_i in seg_ids:
                if seg_i == 0 or seg_i in self.robot.obj_ids['rigid']:
                    continue
                else:
                    seg[seg == seg_i] = 0

            # cv2.imwrite(os.path.join( cam_path, "rgb.png" ), rgb)

            np.save( os.path.join( cam_path, "cam" ), cam )
            np.save( os.path.join( cam_path, "rgb" ), rgb )
            np.save( os.path.join( cam_path, "dep" ), dep )
            np.save( os.path.join( cam_path, "seg" ), seg )
    
    def load_scene(self, save_path, scene_id, block_unit ):

        data_path = os.path.join( save_path, str(scene_id) )

        poses_mat = np.load( os.path.join(data_path, "poses.npy"), allow_pickle=True )
        sizes = np.load( os.path.join(data_path, "sizes.npy"), allow_pickle=True )

        poses = []
        for i in range(len(poses_mat)):
            mat = poses_mat[i]
            size = sizes[i]
            pose = pybullet_utils.mat_to_pose(mat)
            poses.append(pose)

            color = colors[np.random.randint(len(colors))]
            color = pybullet_utils.color_random(color)
            self.add_box(pose, size, color )
            
        
        block_sizes = np.array(sizes / block_unit)
        block_sizes = np.ceil(block_sizes)

        return block_sizes
    