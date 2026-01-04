# generate and save the data
# also blender to render gt mask
# blender file is in assets folder
# 这能提供 segmentation, 好像没必要blender。。。

import os
import sys
import copy
import math
import numpy as np
import time
from tqdm import tqdm

import pybullet as p

from irb6640_pybullet_env.irb6640_envs.task import TAP_Env, pack
from irb6640_pybullet_env.irb6640_envs import models

from rich.traceback import install

from scipy.spatial.transform import Rotation as R
from irb6640_pybullet_env.utils import pybullet_utils
from irb6640_pybullet_env.utils import utils
from PIL import Image
# from detector import Detector     # here ???
import config

# install(show_locals=True)

from block import Block

def get_states(obj_ids, obj_sizes, top_is_clear_dict={}, stacked_dict={}, block_unit=0.03):

    obj_num = len(obj_ids)
    obj_poses = []

    blocks = []

    for i in range(obj_num-1):
        pos, quat = p.getBasePositionAndOrientation(obj_ids[i])
        rot = R.from_quat(quat).as_matrix()
        pose = np.eye(4)
        pose[:3,:3] = rot
        pose[:3,3] = pos

        obj_poses.append(pose)

        size = obj_sizes[i]

        blocks.append(
            Block( size[0], size[1], size[2], pose, basic_unit=block_unit )
        )

    for i, b in enumerate(blocks):
        top_clear = True
        
        top_str = "top_is_clear(object%02d)" % (i)
        
        top_str = "top_is_clear(object%02d)" % (i)
        if top_str not in top_is_clear_dict:
            top_is_clear_dict[top_str] = []
        
        for j, other_b in enumerate(blocks):
            if i==j: continue

            stacked_str = "stacked(object%02d, object%02d)" % (i, j)
            
            if stacked_str not in stacked_dict:
                stacked_dict[stacked_str] = []


            if other_b.is_upon(b):
                top_clear = False
                stacked_dict[stacked_str].append(True)
            else:
                stacked_dict[stacked_str].append(False)
        
        top_is_clear_dict[top_str].append(top_clear)
    
    return obj_poses, top_is_clear_dict, stacked_dict


def generator(
    block_unit = 0.03,
    block_num = 10,
    block_min_size = 1,
    block_max_size = 4,
    container_width = 7,
    task_num = 500,
    mess_min_num = 3,
    mess_max_num = 6,
    save_one=False,
    use_mean=False
    ):
    # detector = Detector()

    save_path = "%d_[%d]_[%d_%d]_%03d_mess_scale" % (block_num, container_width, block_min_size, block_max_size, block_unit*100)

    init_container_origin = np.array([0.4, -0.35, 0])

    if block_num == -1:
        save_path = "mess_real"

    env = TAP_Env( models.get_data_path(), True )
    env.set_plane_offset(config.TAP.plane_offset)

    show_gui = False
    
    env.start(show_gui)
    env.reset(show_gui)

    # seed = env.seed(666)
    # seed = env.seed(777)
    # seed = env.seed(888)
    seed = env.seed(888)
    
    np.random.seed(seed)
    
    if block_num > 15:
        # container_width = 10
        # init_container_origin[1] += -0.05
        init_container_origin[0] += -0.01
    elif block_num > 20:
        # container_width = 10
        init_container_origin[1] += -0.05
        init_container_origin[0] += -0.01

    for i in tqdm(range(task_num)):
        # i += 500
        env.reset()
        # num1 = np.random.randint(1,11)
        # num2 = np.random.randint(1,11)
        
        if save_path == "mess_real":
            block_num = np.random.randint(4, 20)
            mess_min_num = 1
            mess_max_num = int(block_num / 2.0)
        # if block_num < 10:
        #     container_width = 7
        # elif block_num > 15:
        #     container_width = 10
        # elif block_num > 20:
        #     container_width = 12
        # else:
        #     container_width = 8

        num1 = np.random.randint(mess_min_num, mess_max_num)
        num2 = block_num - num1

        env.init_scene(
            block_tasks=[
                { 'unit': block_unit, 'num': num2, 'min_size': block_min_size, 'max_size': block_max_size, 'is_mess': False },
                { 'unit': block_unit, 'num': num1, 'min_size': block_min_size, 'max_size': block_max_size, 'is_mess': True },
            ],
            init_container_origin=init_container_origin,
            init_container_size=[container_width, container_width, block_num * 5],
            use_mean=use_mean
        )
        
        cam_num = len(env.test_cams)
        if save_one:
            cam_num = 1

        for cam_id in range( cam_num ):
            # env.save_scene("./data/%d_[%d]_[%d_%d]_%03d_mess/sim/%d" % 
            #     ( block_num, container_width, block_min_size, block_max_size, block_unit*100, i), cam_id)
            if use_mean:
                env.save_scene("./data/%s/train_mean/%d" %  (save_path, i), cam_id )
            else:
                env.save_scene("./data/%s/train_network/%d" %  (save_path, i), cam_id )


def simple_gen(
    block_unit = 0.03,
    block_num = 4,
    block_min_size = 1,
    block_max_size = 4,
    container_width = 7,
    task_num = 500,
    patch_size = 56,
    task_name='train'
    ):
    # detector = Detector()

    save_path = "/mnt/d/jz/dataset/sornet/tap/%s" % task_name
    os.makedirs(save_path, exist_ok=True)

    env = TAP_Env( models.get_data_path(), True )
    env.set_plane_offset(config.TAP.plane_offset)

    show_gui = False
    
    
    env.start(show_gui)
    env.reset(show_gui)

    # seed = env.seed(666)
    seed = env.seed(777)
    np.random.seed(seed)

    top_is_clear_dict, stacked_dict = {}, {}

    all_rgb = []
    all_dep = []
    all_seg = []

    all_patches_rgb = []
    all_patches_dep = []
    all_patches_seg = []
    
    task_count = 0

    blocks_size = []
    blocks_ids = []

    env.reset()

    env.ompl_and_execute( env.robot.homej2 )

    # for i in tqdm(range(task_num)):
    while True:
        if task_count == task_num: break

        for ids in blocks_ids:
            if ids < 8: continue
            p.removeBody(ids)
        
        task = { 'unit': block_unit, 'num': block_num, 'min_size': block_min_size, 'max_size': block_max_size, 'is_mess': True }
        
        block_unit = task['unit']
        block_num = task['num']
        block_min_size = task['min_size']
        block_max_size = task['max_size']
        is_mess = task['is_mess']

        size_list = [ i for i in range(block_min_size, block_max_size+1) ]
        blocks = np.random.choice(size_list, (block_num, 3) )
        positions, container, stable, ratio, scores = pack.calc_positions_lb_greedy(blocks, [container_width, container_width, 10])

        blocks_size = []
        blocks_ids = []

        
        # add block into scene
        for i in range(block_num):
            texture_id = np.random.randint(0, 3)
            quat = (0, 0, 0, 1)
            
            if is_mess:
                quat = utils.eulerXYZ_to_quatXYZW( np.random.rand(3) * np.math.pi / 6 )
            
            size = blocks[i] * block_unit 
            blocks_size.append(size)
            
            pos = positions[i]
            # pos[0] = container_size[0] - pos[0]
            # pos[1] = container_size[1] - pos[1]
            pos = pos * block_unit + size/2
            pos += [0.4, -0.35, 0] + []
            # 加点高度掉落
            pos += [0, 0, 0.15]

            # 这里 size 减去一点点大小是为了生成间距的数据，这样可能避免碰撞，方便抓取规划？
            # margin = 0.0002
            margin = 0
            box_id = env.add_box( (pos, quat), size - margin, [1,1,1], use_tex=True )
            
            blocks_ids.append(box_id)
            
            p.changeVisualShape(box_id, -1, textureUniqueId=env.tex_uids[texture_id])
            
        pybullet_utils.simulate_step(500)

        # rgb, dep, seg = env.take_images(13)
        rgb, dep, seg = env.take_images(12)

        # check_size
        max_len = 12
        if len(np.unique(seg)) != max_len:
            continue

        # seg[seg == env.robot.ee.base] = env.robot.ee.body
        seg[seg == env.robot.ee.body] = 255

        rgb = np.array(Image.fromarray(rgb).resize((224, 224)))
        dep = np.array(Image.fromarray(dep).resize((224, 224)))
        seg = np.array(Image.fromarray(seg).resize((224, 224)))

        # seg[seg == 255] = env.robot.ee.body
        
        # get each obj
        patches_rgb = []
        patches_dep = []
        patches_seg = []

        # blocks_ids.append(env.robot.ee.body)
        blocks_ids.append(255)

        try:
                
            save_this = True
            all_ids = np.unique(seg)
            for idx in all_ids:
                if idx not in blocks_ids:
                    seg[seg == idx] = 0
                    continue
                
                count = (seg == idx).sum()
                if count < 200 and idx < 100:
                    save_this = False
                    break

            if save_this == False:
                continue

            for idx in all_ids:
                if idx not in blocks_ids:
                    continue

                x, y = np.where(seg == idx)

                width = x[-1] - x[0]
                height = y[-1] - y[0]

                max_size = width if width > height else height
                if max_size < 20:
                    max_size = 45
                max_size += 4
                
                center_x = int(x.mean())
                center_y = int(y.mean())

                # half_patch = int(patch_size/2)
                half_patch = int(max_size/2)
                mask_rgb = rgb[ center_x-half_patch : center_x+half_patch, center_y-half_patch : center_y+half_patch ]
                mask_rgb = np.array(Image.fromarray(mask_rgb).resize((patch_size, patch_size)))
                
                mask_dep = dep[ center_x-half_patch : center_x+half_patch, center_y-half_patch : center_y+half_patch ]
                mask_dep = np.array(Image.fromarray(mask_dep).resize((patch_size, patch_size)))
                
                mask_seg = seg[ center_x-half_patch : center_x+half_patch, center_y-half_patch : center_y+half_patch ]
                mask_seg = np.array(Image.fromarray(mask_seg).resize((patch_size, patch_size)))
                mask_seg[ mask_seg!= idx ] = 0
                # mask_seg[ mask_seg== idx ] -= max_len - 4

                patches_rgb.append([mask_rgb])
                patches_dep.append([mask_dep])
                patches_seg.append([mask_seg])
            
            if save_this == False:
                continue

        except:
            patches_rgb = []
            patches_dep = []
            patches_seg = []
            continue
        
        if len(patches_rgb) < len(blocks_ids):
            continue 

        patches_rgb = np.concatenate(patches_rgb, axis=0)
        patches_dep = np.concatenate(patches_dep, axis=0)
        patches_seg = np.concatenate(patches_seg, axis=0)

        # seg[seg != 0] -= max_len - 4

        all_rgb.append([rgb])
        all_dep.append([dep])
        all_seg.append([seg])
        
        all_patches_rgb.append([patches_rgb])
        all_patches_dep.append([patches_dep])
        all_patches_seg.append([patches_seg])

        blocks_pose, top_is_clear_dict, stacked_dict = get_states(blocks_ids, blocks_size, top_is_clear_dict, stacked_dict, block_unit)

        task_count += 1
        print(task_count)

    state_dict = {
        'top_is_clear': top_is_clear_dict,
        'stacked': stacked_dict,
    }

    all_rgb = np.concatenate(all_rgb, axis=0)
    all_dep = np.concatenate(all_dep, axis=0)
    all_seg = np.concatenate(all_seg, axis=0)
    all_patches_rgb = np.concatenate(all_patches_rgb, axis=0)
    all_patches_dep = np.concatenate(all_patches_dep, axis=0)
    all_patches_seg = np.concatenate(all_patches_seg, axis=0)

    # save
    np.save( os.path.join( save_path, 'all_rgb' ), all_rgb )
    np.save( os.path.join( save_path, 'all_dep' ), all_dep )
    np.save( os.path.join( save_path, 'all_seg' ), all_seg )
    np.save( os.path.join( save_path, 'all_patches_rgb' ), all_patches_rgb )
    np.save( os.path.join( save_path, 'all_patches_dep' ), all_patches_dep )
    np.save( os.path.join( save_path, 'all_patches_seg' ), all_patches_seg )
    np.save( os.path.join( save_path, 'state' ), state_dict )


def generator_unit(
    block_unit = 0.05,
    block_num = 10,
    block_min_size = 20,
    block_max_size = 70,
    task_num = 500,
    unit_length = 1,
    save_one=False,
    use_mean=True,
    note=''
    ):
    # detector = Detector()
    globalScaling = 20

    save_path = "%d_[%d_%d]_%03d_unit_%d%s" % (block_num, block_min_size, block_max_size, block_unit*100, unit_length, note)

    init_container_origin = np.array([0.35, -0.35, 0])

    env = TAP_Env( models.get_data_path(), True, globalScaling=globalScaling )
    env.set_plane_offset(config.TAP.plane_offset)

    show_gui = False
    
    env.start(show_gui)
    env.reset(show_gui)

    # seed = env.seed(666)
    # seed = env.seed(777)
    # seed = env.seed(888)
    seed = env.seed(888)
    
    np.random.seed(seed)
    
    for i in tqdm(range(task_num)):
        # i += 500
        env.reset()
        # num1 = np.random.randint(1,11)
        # num2 = np.random.randint(1,11)
        
        if block_num == 0:
            BN = np.random.randint(5, 20)
        else:
            BN = block_num
            
        # num1 = np.random.randint(mess_min_num, mess_max_num)
        # num2 = block_num - num1

        env.init_scene(
            block_tasks=[
                # { 'unit': block_unit, 'num': num2, 'min_size': block_min_size, 'max_size': block_max_size, 'is_mess': False },
                { 'unit': block_unit, 'num': BN, 'min_size': block_min_size, 'max_size': block_max_size, 'is_mess': False },
            ],
            init_container_origin=init_container_origin,
            init_container_size=[140, 140, BN * block_max_size],
            use_mean=use_mean,
            unit_length = unit_length
        )
        
        cam_num = len(env.test_cams)
        if save_one:
            cam_num = 1

        for cam_id in range( cam_num ):
            # env.save_scene("./data/%d_[%d]_[%d_%d]_%03d_mess/sim/%d" % 
            #     ( block_num, container_width, block_min_size, block_max_size, block_unit*100, i), cam_id)
            if use_mean:
                env.save_scene("./data/%s/train_mean/%d" %  (save_path, i), cam_id )
                # env.save_scene("./data/%s/test/%d" %  (save_path, i), cam_id )
            else:
                env.save_scene("./data/%s/train/%d" %  (save_path, i), cam_id )




if __name__ == "__main__":
    
    # default
    # generator(
    #     block_unit = 0.03,
    #     block_num = 10,
    #     block_min_size = 1,
    #     block_max_size = 4,
    #     container_width = 7,
    #     task_num = 1000,
    #     mess_min_num = 3,
    #     mess_max_num = 6,
    #     save_one=True,
    #     use_mean=False
    # )

    # generator_scale(
    #     block_unit = 0.05,
    #     block_num = 0,
    #     block_min_size = 10,
    #     block_max_size = 80,
    #     task_num = 10,
    #     save_one=False,
    #     use_mean=True,
    #     note=''
    # )

    # generator_scale(
    #     block_unit = 0.04,
    #     block_num = 10,
    #     block_min_size = 20,
    #     block_max_size = 70,
    #     task_num = 1000,
    #     save_one=True,
    #     use_mean=False,
    #     unit_length = 1,
    #     note=''
    # )

    generator_unit(
        block_unit = 0.04,
        block_num = 10,
        block_min_size = 20,
        block_max_size = 70,
        task_num = 1000,
        save_one=True,
        use_mean=False,
        unit_length = 10,
        note=''
    )

    # generator(
    #     block_unit = 0.02,
    #     block_num = 10,
    #     block_min_size = 2,
    #     block_max_size = 8,
    #     container_width = 14,
    #     task_num = 1000,
    #     mess_min_num = 3,
    #     mess_max_num = 6,
    #     save_one=True
    # )
    
    # generator(
    #     # block_unit = 0.02,
    #     block_unit = 0.03,
    #     block_num = 20,
    #     block_min_size = 1,
    #     block_max_size = 4,
    #     container_width = 10,
    #     task_num = 1000,
    #     mess_min_num = 3,
    #     mess_max_num = 6,
    #     save_one=True
    # )

    # generator(save_one=True)

    #     block_unit = 0.03,
    #     block_num = 10,
    #     block_min_size = 1,
    #     block_max_size = 6,
    #     container_width = 12,
    #     task_num = 500,
    #     mess_min_num = 5,
    #     mess_max_num = 10
    # )
    
    # simple_gen(task_num=5, task_name='debug')
    # simple_gen(task_num=100, task_name='valid')

    # task_num = 100

    # generator(
    #     block_unit = 0.03,
    #     block_num = -1,
    #     block_min_size = 1,
    #     block_max_size = 5,
    #     task_num = task_num,
    # )

    # generator( task_num = task_num )
    
    # while True:
    #     env.key_event()
    #     # env.take_images('test')
    #     pybullet_utils.simulate_step(1)
    


    # p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

    # env.generate_boxes( 5, 1, 4, [7, 7, 20], [0.4, -0.3, 0], 0.03, True )
    # pybullet_utils.simulate_step(100)
    # env.generate_boxes( 5, 1, 4, [7, 7, 20], [0.4, -0.3, 0], 0.03, True )
    # pybullet_utils.simulate_step(100)
    # env.generate_boxes( 5, 1, 4, [7, 7, 20], [0.4, -0.3, 0], 0.03, True )

    # pos, quat = env.robot.get_ee_pose()
    # pos[1] -= 0.1
    # pos[2] += 0.1
    # pos_b = [pos, quat]

    # pos_a = [[0.64, 0, 0.15/2], [0,0,0,1]]
    # env.add_box( pos_a, [0.05, 0.1, 0.15], utils.COLORS['red'] + [1] )

    # pick_pose = utils.multiply(pos_a, [[0., 0, 0.15/2], [0,0,0,1]])

    # pybullet_utils.draw_pose(pos_a)
    # pybullet_utils.draw_pose(pick_pose)
    # pybullet_utils.draw_pose(pos_b)

    # env.robot.pick(pick_pose)
    # env.robot.place(pos_b)

    # count = 1
    # sign = 1

    # rot_quat = utils.eulerXYZ_to_quatXYZW([ sign * np.math.pi / 18.0, 0, 0 ])
    # pos_c = utils.multiply(pos_b, [[0,0,0], rot_quat])

    # joints = env.robot.solve_ik( ((100, 0, 0), (0, 0, 0, 1)) )
    # print(joints)

    # try:
    #     while True:
    #         p.stepSimulation()
    #         # time.sleep(1/240.0)
    #         # env.key_event()

    #         # if env.robot.is_static:
    #         #     rot_quat = utils.eulerXYZ_to_quatXYZW([ sign * np.math.pi / 18.0, 0, 0 ])
    #         #     count += 1
    #         #     if count == 36:
    #         #         sign = -sign
    #         #         count = 1
    #         #     pos_c = utils.multiply(pos_c, [[0,0,0.0], rot_quat])
    #         #     env.robot.move_to(pos_c, (0, 0, 0.05))

    #         if show_gui and env.robot.is_static:
    #             # env.robot.update_arm()
    #             rgb, dep, seg = env.take_images('init')

    # except Exception as e:
    #     # env.close()
    #     print(e)
    # pybullet_utils.simulate_step(100)
    # env.save_scene("./data/%d_[%d]_[%d_%d]_mess/%d" % ( block_num, container_width, block_min_size, block_max_size, i))


    # rgb, dep, seg = env.take_images('init')
    # intr = env.cams['init'].get_intrinsics()
    # cam_pose = env.cams['init'].get_pose()
    
    
    # for i in env.robot.obj_ids['rigid']:
    #     # print( p.getBodyInfo(i) )
    #     print( p.getBasePositionAndOrientation(i) )

    # blocks_pose, blocks_size = detector.detect_blocks( rgb, dep, intr, cam_pose, block_unit )
    # print(blocks_size)

