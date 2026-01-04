# -*- coding=utf-8 -*-
import numpy as np
import torch
from tqdm import tqdm
import time
import pybullet as p

from task import Pack_Env
from pct_envs.PctDiscrete0 import PackingDiscrete
from block import Block
from packer.app import Packer, get_policy
from arm_pybullet_env.utils import pybullet_utils

from tools import get_args

import warnings
warnings.filterwarnings("ignore")


'''
    setting1: 不考虑物理仿真，不考虑机械臂 (只考虑geometry 也就是直接静态放置)
'''

if __name__ == '__main__':
    # 开始计时
    start_time = time.time()
    args, args_base = get_args()
    np.random.seed(777)
    
    scene = Pack_Env(args_base, disp=True)
    scene.set_plane_offset(scene.plane_offset)
    scene.start(setting=args.setting, show_gui=False)

    packer = Packer(scene.target_container_size) # 装箱决策算法

    task_num = 1

    # start task loop
    for t in tqdm(range(task_num)):

        scene.reset(args.setting)   # 重置环境，加载场景
        pack_num = 0
        move_num = 0
        
        setting = args.setting
        # check_stability = False
        if setting == 1:
            check_stability = False
        else:
            check_stability = True
        threshold = args_base.Scene.threshold
        is_3d_packing = args.is_3d_packing
        scene.is_3d_packing = is_3d_packing
        data = args.data
        method = args.method
        test_data_config = args.test_data_config

        video_path = "/media/wzf/goodluck/Workspace/rl/video/setting" + str(setting) + "/" + str(data) + "/" + str(method) + "_" + str(test_data_config) + ".mp4"

        scene.new_container(only_base=True)
        container_volume = (scene.target_container_size[0] * scene.block_unit) * (scene.target_container_size[1] * scene.block_unit) * (scene.target_container_size[2] * scene.block_unit)

        # load data: init_sizes...
        is_regular_data = scene.is_regular(method)
        
        if is_regular_data:
            num, init_sizes, init_poses_mat, info = scene.load_data(data, test_data_config, args_base)
        else:
            if is_3d_packing:
                num, init_sizes, init_poses_mat, info = scene.load_irregular_data(data, args_base)    # 用obj导入
            else:
                num, init_sizes, init_poses_mat, info = scene.load_2d_irregular_data(data, args_base)    # 用obj导入
        if num > 79:
            num = 80
        pack_init_sizes = np.ceil(np.array(init_sizes) / scene.block_unit).astype(np.int)    # np.round：四舍五入函数  ceil向上取整

        # 去除0值
        init_sizes = init_sizes[init_sizes.sum(axis=1) != 0]
        init_sizes = init_sizes[:, (init_sizes.sum(axis=0) != 0)]

        pack_init_sizes = pack_init_sizes[pack_init_sizes.sum(axis=1) != 0]
        pack_init_sizes = pack_init_sizes[:, (pack_init_sizes.sum(axis=0) != 0)]

        policy, infos = get_policy(args, method)
        
        # 导入 packing 理想环境
        env = PackingDiscrete(setting=1, check_stability=check_stability, threshold=threshold,container_size=scene.target_container_size, item_set=pack_init_sizes,
                              target_origin=scene.target_origin, block_unit=scene.block_unit, load_test_data=True, args=infos['args'])
        env.space.policy = policy
        obs = None
        if is_regular_data:
            obs = env.reset()
            obs = torch.FloatTensor(obs).unsqueeze(dim=0)
        else:
            if is_3d_packing:
                obs = env.ir_reset(scene)
                obs = torch.FloatTensor(obs).unsqueeze(dim=0)

        static_stability = []
        position_offset_mean_max = []
        planning_times = []
        placed_boxes_volume = []
        positions_target = {}
        dangerous_num = 0

        # 设置录制参数
        logId = p.startStateLogging(loggingType=p.STATE_LOGGING_VIDEO_MP4, fileName=video_path)
        # p.addUserDebugText(f"Position: 11111", [0.1, 1.0, 0], textColorRGB=[1, 0, 0], textSize=1.5)

        for n in range(num):
            env.next_item_ID = n + 1
            infos['block_index'] = n
            if is_regular_data:
                placeable, pack_position, rotation_flag, infos, planning_time = packer.pack_box(env, infos, obs, method, policy)
            else:
                placeable, pack_position, rotation_flag, infos, planning_time = packer.pack_irregular_box(n, env, scene, pack_init_sizes, infos, obs, method, policy)
            
            pack_rotation = [0, 0, 0, 1]   # 此处直接固定正方向摆放
            
            obs = infos['next_obs']
            if obs is None:
                pass
            else:
                obs = torch.FloatTensor(obs).unsqueeze(dim=0)

            if is_regular_data:
                mat, size = scene.load_scene(n, rotation_flag, init_poses_mat, init_sizes)
            else:
                mat, size = scene.load_irregular_scene(n, rotation_flag, init_poses_mat, init_sizes)
            init_pose = pybullet_utils.mat_to_pose(mat)
            
            scene.refresh_tex()
                
            l, w, h = size
            block = Block(l, w, h, mat, basic_unit=scene.block_unit)

            volume = l * w * h
            bodyUniqueId = scene.box_ids[0]     # 选中箱子对应的bodyUniqueId       
            pack_size = np.array(size)
            pack_size = np.ceil(pack_size / scene.block_unit)

            # 有些非常小的mesh计算得到某个size为0，故将数组中为0的值置为1
            pack_size[pack_size == 0] = 1

            height_offset = scene.target_base_size[-1]/2 + 0.01

            positions_init = {}
            for i in range(scene.box_ids[0] - scene.containers[0][-1]):
                id = scene.containers[0][-1] + i + 1
                positions_init[str(id)] = p.getBasePositionAndOrientation(id)[0]
            stability_reward = 0

            if placeable:
                # 抓取box之前确保机械臂静态
                if setting == 3:
                    while not scene.robot.is_static:
                        pybullet_utils.simulate_step(1)
                    for _ in range(50):
                        pybullet_utils.simulate_step(1)
                    # 选择block并用吸盘吸起来(图像变化)
                    pick_success, pre_pose = scene.pick_block(block)
                else:
                    pybullet_utils.simulate_step(50)
            
                scene.packed_positions.append(np.array(pack_position))
                scene.blocks_size.append(np.array(pack_size))       # 此处添加的是扩充后的size，而不是实际的size
                # 更新虚拟容器
                if is_3d_packing:
                    add_success = packer.add_box(pack_size, pack_position)
                else:
                    add_success = True
                if not add_success:
                    print("Add_success fail! Check the heightmap_max! BodyUniqueId: ", bodyUniqueId)
                    scene.failed_object_ids.append(bodyUniqueId)
                    break

                # offset为container的左上角，place的坐标以这个坐标为基准
                offset = scene.target_origin.copy()  # a little up offset
                offset[2] += height_offset

                if setting == 3:
                    # 确保放置前box的速度为0
                    num_joints = p.getNumJoints(bodyUniqueId)
                    # 设置物体的所有关节速度为0
                    for jointIndex in range(num_joints):
                        p.setJointMotorControl2(bodyUniqueId, jointIndex=jointIndex, controlMode=p.VELOCITY_CONTROL,
                                                targetVelocity=0, force=500)

                if is_regular_data:
                    move_success, stability_reward, target_pose, dangerous_flag = scene.place_at(setting, bodyUniqueId, pack_position, pack_rotation, pack_size, offset)
                    # move_success, stability_reward, stop_flag, now_position_offset = scene.place_at(setting, bodyUniqueId, pack_position, pack_rotation, size, offset)
                else:
                    if is_3d_packing:
                        move_success, stability_reward, stop_flag, now_position_offset = scene.place_at_irregular(setting, bodyUniqueId, pack_position, pack_rotation, pack_size, offset)
                    else:
                        move_success, _ = scene.place_at_irregular_2d(bodyUniqueId, pack_position, rotation_flag, size, offset)

                positions_target[str(bodyUniqueId)] = target_pose
                if not move_success:
                    print("Move_ee_to fail! Pose is not reachable: height is too high or height is too low! BodyUniqueId: ", bodyUniqueId)
                    remove_success = scene.remove_current_block(setting, bodyUniqueId, pack_size, height_offset)
                    scene.failed_object_ids.append(bodyUniqueId)
                    break
                else:
                    print("Packing succeed! BodyUniqueId: ", bodyUniqueId)
                    scene.packed_object_ids.append(bodyUniqueId)

            else:
                print("Packing fail! This box is not placeable! BodyUniqueId: ", bodyUniqueId)
                remove_success = scene.remove_current_block(setting, bodyUniqueId, pack_size, height_offset)
                scene.failed_object_ids.append(bodyUniqueId)
                break

            if setting == 3:
                # 移动机械臂到初始定义好的homej初始位置(图像变化)
                scene.ompl_and_execute(False, scene.robot.homej)

            pybullet_utils.simulate_step(1000)
            now_position_offsets = []
            stop_flag = False
            for i in range(bodyUniqueId - scene.containers[0][-1]):
                id = scene.containers[0][-1] + i + 1
                positions_final = p.getBasePositionAndOrientation(id)[0]
                positions_offset = tuple(abs(a - b) for a, b in zip(positions_final, positions_target[str(id)]))
                now_position_offset = positions_offset[0] + positions_offset[1]
                now_position_offsets.append(now_position_offset)

                collapsed_flag = any([positions_offset[0] > scene.threshold, positions_offset[1] > scene.threshold])
                if collapsed_flag is True:
                    print("Packing collapsed! Id: ", id)
                    print("positions_offset[0]: ", positions_offset[0])
                    print("positions_offset[1]: ", positions_offset[1])
                    stop_flag = True

            if stop_flag:
                print("Packing collapsed! Warning! Packing BodyUniqueId: ", bodyUniqueId)
                now_position_offsets = now_position_offsets[:-1]        # 去除最后一个失败的box
                scene.packed_object_ids = scene.packed_object_ids[:-1]  # 去除最后一个失败的box
                break

            if dangerous_flag:
                dangerous_num += 1
                if dangerous_num > 2:
                    print("Dangerous operations of the robotic arm occurred more than 3 times! Packing BodyUniqueId: ", bodyUniqueId)
                    now_position_offsets = now_position_offsets[:-1]  # 去除最后一个失败的box
                    scene.packed_object_ids = scene.packed_object_ids[:-1]  # 去除最后一个失败的box
                    break

            static_stability.append(stability_reward)
            position_offset_mean_max.append([np.mean(now_position_offsets), np.max(now_position_offsets)])
            planning_times.append(planning_time)
            placed_boxes_volume.append(volume)

            scene.box_ids.pop()

            # 更新heightmap 也就是plain
            if is_regular_data is False:
                if is_3d_packing:
                    env.space.shot_whole()
            
            # 更新打包数量和移动次数
            if pack_num != -1:
                pack_num += 1
            move_num += 1

            if scene.ompl_plan_allowed_time < 19:
                scene.ompl_plan_allowed_time += 1
            time.sleep(0.2)

        if setting == 2 or setting == 3:
            # 计算稳定性
            mean_stability_reward = np.mean(static_stability)
            position_offset_mean = np.mean(now_position_offsets)
            position_offset_max = np.max(now_position_offsets)

        # 计算空间利用率
        placed_boxes_volumes = np.sum(placed_boxes_volume)
        space_utilization = placed_boxes_volumes / container_volume

        # 计算平均规划时间
        full_planning_time = np.sum(planning_times)

        # 成功放置数量
        feasible_num = len(scene.packed_object_ids)

        print("*"*75)
        print("Finsh!")
        print("Packed Object_ids: " + ", ".join(map(str, scene.packed_object_ids)))
        print("Failed Object_ids: " + ", ".join(map(str, scene.failed_object_ids)))
        print(f"Feasible Num: {feasible_num}")
        if setting == 2 or setting == 3:
            print(f"Static Stability: {mean_stability_reward}")
            print(f"Position Offset Mean: {position_offset_mean}")
            print(f"Position Offset Max: {position_offset_max}")
        print(f"Space Utilization: {space_utilization}")
        print(f"Planning Time: {full_planning_time}")
        print("*"*75)
        
        # 结束计时
        end_time = time.time()
        # 计算运行时间
        execution_time = end_time - start_time
        print(f"All time is {execution_time:.4f} 秒")
        print("end")

        # 停止录制
        p.stopStateLogging(logId)

        # Define the content to be written to the file
        content = (
                "*" * 75 + "\n"
                "Finsh!\n"
                "Packed Object_ids: " + ", ".join(map(str, scene.packed_object_ids)) + "\n"
                "Failed Object_ids: " + ", ".join(map(str, scene.failed_object_ids)) + "\n"
                f"Feasible Num: {feasible_num}\n"
        )

        # Add conditional content based on the setting
        if setting == 2 or setting == 3:
            content += (
                f"Static Stability: {mean_stability_reward}\n"
                f"Position Offset Mean: {position_offset_mean}\n"
                f"Position Offset Max: {position_offset_max}\n"
            )

        # Add the remaining content
        content += (
            f"Space Utilization: {space_utilization}\n"
            f"Planning Time: {full_planning_time}\n"
            f"All time is {execution_time:.4f} \n"
        )

        # Write the content to a .txt file
        txt_path = video_path.replace("video", "log")
        txt_path = txt_path.replace("mp4", "txt")
        print(txt_path)
        with open(txt_path, "w") as file:
            file.write(content)

        
