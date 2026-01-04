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
from irb6640_pybullet_env.irb6640_envs.task import TAP_Env
from irb6640_pybullet_env.irb6640_envs import models

from rich.traceback import install

from irb6640_pybullet_env.utils import pybullet_utils
from irb6640_pybullet_env.utils import utils

install(show_locals=True)

if __name__ == "__main__":

    globalScaling = 20
    env = TAP_Env( models.get_data_path(), True, globalScaling=globalScaling )
    show_gui = True
    
    env.set_plane_offset([0,0,-0.0001])
    env.start(show_gui)
    env.reset(show_gui)

    seed = env.seed(666)
    np.random.seed(seed)
    
    # env.init_scene(
    #     [
    #         # { 'unit': 0.05, 'num': 2, 'min_size': 5, 'max_size': 5, 'is_mess': True },
    #         # { 'unit': 0.05, 'num': 1, 'min_size': 90, 'max_size': 100, 'is_mess': True },
    #         { 'unit': 0.04, 'num': 10, 'min_size': 20, 'max_size': 70, 'is_mess': True },
    #         # { 'unit': 0.05, 'num': 5, 'min_size': 10, 'max_size': 50, 'is_mess': True },
    #     ],
    #     init_container_size= [120, 120, 500]
    # )
    
    # p.setGravity(0, 0, -9.8)

    all_b = []
    
    def add_b(pos, size):
        offset = [1, 0, 0]
        block_unit = 0.03

        pos = np.array(pos)
        size = np.array(size)

        pos = (pos + size/2) * block_unit  + offset
        size = block_unit * size

        orient = [0,0,0,1]
        print("pos: ", pos)
        print("size: ", size)
        box_id, box_urdf = env.add_box([pos, orient], size)

        all_b.append(box_id)

    add_b( [0,0,0], [2,3,4] )

    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

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

    try:
        while True:
            p.stepSimulation()
            # time.sleep(1/240.0)
            # env.key_event()

            # if env.robot.is_static:
            #     rot_quat = utils.eulerXYZ_to_quatXYZW([ sign * np.math.pi / 18.0, 0, 0 ])
            #     count += 1
            #     if count == 36:
            #         sign = -sign
            #         count = 1
            #     pos_c = utils.multiply(pos_c, [[0,0,0.0], rot_quat])
            #     env.robot.move_to(pos_c, (0, 0, 0.05))

            if show_gui and env.robot.is_static:
                # env.robot.update_arm()
                # rgb, dep, seg = env.take_images('tmp')
                rgb, dep, seg = env.take_images(0)

    except Exception as e:
        # env.close()
        print(e)