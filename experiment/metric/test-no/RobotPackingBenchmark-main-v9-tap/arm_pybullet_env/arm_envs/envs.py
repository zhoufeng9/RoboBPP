import os
import sys
import copy

import numpy as np
import matplotlib.pyplot as plt

import pybullet as p

import os
import pkgutil
import sys
import tempfile
import time

import numpy as np
from arm_pybullet_env.arm_envs import cameras
 
from arm_pybullet_env.utils import pybullet_utils
from arm_pybullet_env.utils import utils
from arm_pybullet_env.arm_envs.arm import ARM
from arm_pybullet_env.arm_envs import pb_ompl
import transform as tf
ARM_WORKSPACE_URDF_PATH = "workspace/workspace.urdf"
PLANE_URDF_PATH = "plane/plane.urdf"
HOUSE_URDF_PATH = "house/house.urdf"
CONVEYOR_URDF_PATH = "conveyor/conveyor.urdf"
SUPPORT_URDF_PATH = "support/support.urdf"
CAM_URDF_PATH = "camera/camera.urdf"

# colors = [utils.COLORS[c] for c in utils.COLORS if c != 'brown']
colors = [[0.8, 0.8, 0.8, 1],
          [0.5, 0.5, 0.5, 1],
          [0.3, 0.3, 0.3, 1],
          [0.6, 0.6, 0.8, 1],
          [0.9, 0.9, 0.7, 1],
          [0.7, 0.6, 0.5, 1],
          [0.9, 0.8, 0.8, 1],
          [0.7, 0.8, 0.7, 1],
          [0.8, 0.7, 0.8, 1],
          [0.9, 0.85, 0.7, 1]]

# from arm_pybullet_env.pybullet_blender_recorder.pyBulletSimRecorder import PyBulletRecorder

class Env:
    def __init__(self,
               assets_root,
               disp=False,
               shared_memory=False,
               hz=240,
               use_egl=False) -> None:
        if use_egl and disp:
            raise ValueError('EGL rendering cannot be used with `disp=True`.')

        self.use_egl = use_egl
        self.disp = disp
        self.hz = hz
        self.shared_memory = shared_memory
        self.assets_root = assets_root
        self.obstacles = []
        self.interpolate_num = 50
        self.step_size = 0.1
        self.arm_urdf_path = "ur5/ur5.urdf"
        self.homej = [-1, -0.7, 0.5, -0.5, -0.5, 0]

        # self.pix_size = 0.003125

        self.seed(666)
        
        self.recorder = None
        # self.recorder = PyBulletRecorder()

        # add robot
        self.robot = ARM(recorder=self.recorder)
        
        self.containers = []

        self.packed_object_ids = []
        
        self.failed_object_ids = []


    def set_plane_offset(self, offset):
        self.robot.offset = offset
        self.offset = offset

    def start(self, setting=1, show_gui=True):
        '''Start PyBullet'''
        
        disp_option = p.DIRECT
        if self.disp:
            disp_option = p.GUI
        if self.shared_memory:
            disp_option = p.SHARED_MEMORY

        client = p.connect(disp_option)

        file_io = p.loadPlugin('fileIOPlugin', physicsClientId=client)
        if file_io < 0:
            raise RuntimeError('pybullet: cannot load FileIO!')
        if file_io >= 0:
            p.executePluginCommand(
                file_io,
                textArgument=self.assets_root,
                intArgs=[p.AddFileIOAction],
                physicsClientId=client)

        self._egl_plugin = None
        if self.use_egl:
            assert sys.platform == 'linux', ('EGL rendering is only supported on Linux.')
            egl = pkgutil.get_loader('eglRenderer')
            if egl:
                self._egl_plugin = p.loadPlugin(egl.get_filename(), '_eglRendererPlugin')
            else:
                self._egl_plugin = p.loadPlugin('eglRendererPlugin')
            print('EGL renderering enabled.')

        p.configureDebugVisualizer(p.COV_ENABLE_GUI, show_gui)
        p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0)
        p.setPhysicsEngineParameter(enableFileCaching=0)
        p.setAdditionalSearchPath(self.assets_root)
        p.setAdditionalSearchPath(tempfile.gettempdir())
        p.setTimeStep(1. / self.hz)

        p.setGravity(0, 0, -9.8)

        self.house_tex = p.loadTexture( os.path.join( self.assets_root, "house/01.png" ) )
        self.conveyor_tex = p.loadTexture( os.path.join( self.assets_root, "conveyor/flipped_conveyor.png" ) )
        self.support_tex = p.loadTexture( os.path.join( self.assets_root, "support/02.png" ) )

        self.tex_uids = []
        for texture_id in range(10):
            text_path = os.path.join( self.assets_root, "box/textures/box_%d.png" % texture_id )
            tex_uid = p.loadTexture(text_path)
            self.tex_uids.append(tex_uid)

        if setting == 1 or setting == 2:
            p.resetDebugVisualizerCamera(
                cameraDistance=5.6,
                cameraYaw=90.4,
                cameraPitch=-59.6,
                cameraTargetPosition=(-0.0253008846193552, -0.17173346877098083, -2.7479984760284424))
        else:
            p.resetDebugVisualizerCamera(
                cameraDistance=4.2,
                cameraYaw=74,
                cameraPitch=-38,
                cameraTargetPosition=(-0.02025381661951542, -0.1773565262556076, 0.39200010895729065))
        
    def reset(self, setting=1, show_gui=True):
        '''load plane and robot'''
        p.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD)
        p.setGravity(0, 0, -9.8)
        # Temporarily disable rendering to load scene faster.
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)

        [ p.removeBody(i) for i in range( p.getNumBodies() ) ]
        
        # logId = p.startStateLogging(p.STATE_LOGGING_PROFILE_TIMINGS, "timings.json")
        
        # reload init scene
        plane_id = pybullet_utils.load_urdf(p, os.path.join(self.assets_root, PLANE_URDF_PATH), self.offset)
        # house_id = pybullet_utils.load_urdf(p, os.path.join(self.assets_root, HOUSE_URDF_PATH))
        # p.changeVisualShape(house_id, -1, textureUniqueId=self.house_tex)

        # workspace_id = pybullet_utils.load_urdf(p, os.path.join(self.assets_root, ARM_WORKSPACE_URDF_PATH),
        #                         [0.5, 0, 0])
        if setting == 3:
            workspace_id = pybullet_utils.load_urdf(p, os.path.join(self.assets_root, ARM_WORKSPACE_URDF_PATH) )

        conveyor_urdf = os.path.join(self.assets_root, CONVEYOR_URDF_PATH)
        conveyor_id = pybullet_utils.load_urdf(p, conveyor_urdf, self.conveyor_pose)
        p.changeVisualShape(conveyor_id, -1, textureUniqueId=self.conveyor_tex)

        # support_urdf = os.path.join(self.assets_root, SUPPORT_URDF_PATH)
        # support_id = pybullet_utils.load_urdf(p, support_urdf, self.support_pose)   # 工作台位置 [0.5, -0.25, 0.0]
        # p.changeVisualShape(support_id, -1, textureUniqueId=self.support_tex)

        self.containers.clear()

        self.obstacles.clear()
        self.obstacles.append(plane_id)
        self.obstacles.append(conveyor_id)
        # self.obstacles.append(support_id)

        # self.support_id = support_id
        self.conveyor_id = conveyor_id

        # reset robot / reload robot
        self.robot.reset(setting, self.arm_urdf_path, self.homej, show_gui)
        
        # self.pb_ompl_interface = pb_ompl.PbOMPL(self.robot.robot_ompl, self.obstacles, 'kBITstar', interpolate_num=50, planning_time=0.1)
        self.pb_ompl_interface = pb_ompl.PbOMPL(self.robot.robot_ompl, self.obstacles, 'RRT', interpolate_num=self.interpolate_num, step_size=self.step_size)
        # self.pb_ompl_interface = pb_ompl.PbOMPL(self.robot.robot_ompl, self.obstacles, 'PRM')

        # Re-enable rendering.
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        self.refresh_tex()

        self.packed_object_ids.clear()


    def refresh_tex(self):
        # p.changeVisualShape(self.support_id, -1, textureUniqueId=self.support_tex)
        p.changeVisualShape(self.conveyor_id, -1, textureUniqueId=self.conveyor_tex)
        
    def take_images(self, cam_name:str):
        # p.changeVisualShape(self.init_cam_id, -1, rgbaColor=[0,0,0,0])
        # p.changeVisualShape(self.tmp_cam_id, -1, rgbaColor=[0,0,0,0])

        rgb, dep, seg = self.cams[cam_name].take_images()
        # if cam_name == 'init':
        #     p.changeVisualShape(self.init_cam_id, -1, rgbaColor=[1,0,0,1])
        #     p.changeVisualShape(self.tmp_cam_id, -1, rgbaColor=[0.8,0.8,0.8,1])
        # elif cam_name == 'check':
        #     p.changeVisualShape(self.tmp_cam_id, -1, rgbaColor=[1,0,0,1])
        #     p.changeVisualShape(self.init_cam_id, -1, rgbaColor=[0.8,0.8,0.8,1])

        return rgb, dep, seg

    def add_object(self, urdf, pose, category='rigid'):
        """List of (fixed, rigid, or deformable) objects in env."""
        fixed_base = 1 if category == 'fixed' else 0
        
        obj_urdf = os.path.join(self.assets_root, urdf)
        new_obj_urdf = obj_urdf.replace('.', '-')
        os.system("cp %s %s" % (obj_urdf, new_obj_urdf))
        obj_id = pybullet_utils.load_urdf(
            p,
            obj_urdf,
            pose[0],
            pose[1],
            useFixedBase=fixed_base
            )
        self.robot.add_object_to_list(category, obj_id)

        self.obstacles.append(obj_id)

        return obj_id, new_obj_urdf

    def add_box(self, pose, size, color=[1, 1, 1, 1], category='rigid', mass=0.5, use_tex=False):
        box_template = 'box/box-template-tex.urdf'
        urdf = pybullet_utils.fill_template( self.assets_root, box_template, {'DIM': size, 'RGBA': color, "FILE_PATH": [os.path.join(self.assets_root, 'box')] })
        box_id, box_urdf = self.add_object(urdf, pose, category)

        p.changeDynamics(box_id, -1, mass=mass)
        return box_id, box_urdf

    def seed(self, seed=None):
        self._random = np.random.RandomState(seed)
        return seed

    def close(self):
        if self._egl_plugin is not None:
            p.unloadPlugin(self._egl_plugin)
        p.disconnect()
    
    def key_event(self):

        def check_key(key, keys):
            key = ord(key)
            return key in keys and keys[key] & p.KEY_WAS_TRIGGERED

        keys = p.getKeyboardEvents()
        
        if check_key('a', keys):
            unit = 0.025
            pos = np.random.rand(3) * 0.3 + [0,0,0.1]
            quat = utils.eulerXYZ_to_quatXYZW(np.random.rand(3) * np.math.pi)
            size = np.random.randint(1,6,3) * unit
            self.add_box( [pos, quat], size )
        
        elif check_key('b', keys):
            pos = np.random.rand(3) * 0.3 + [0,0,0.1]
            quat = utils.eulerXYZ_to_quatXYZW(np.random.rand(3) * np.math.pi)
            obj_id, _ = self.add_object('bottle_21/bottle_21.urdf', [pos, quat], category='fixed')
            p.changeDynamics(obj_id, -1, mass=0.5)
        
        elif check_key('c', keys):
            w, h, view, proj, up, forward, hori, verti, yaw, pitch, dist, target = p.getDebugVisualizerCamera()
            pos = target - np.array(forward) * dist
            z = forward
            y = -np.array(up)
            x = np.cross(y, z)
            mat = np.identity(4)
            mat[:3, 0] = x
            mat[:3, 1] = y
            mat[:3, 2] = z
            mat[:3, 3] = pos
            pose = pybullet_utils.mat_to_pose(mat)
            print(pose)
            pybullet_utils.draw_pose(pose)
            self.cams['test'].config['position'] = pose[0]
            self.cams['test'].config['rotation'] = pose[1]

        
        elif check_key('r', keys):
            self.reset()
        
        else:
            pybullet_utils.key_event(keys)


    
    #============----------------   Robot motion   ----------------============#

    def set_ompl_distance(self, max_distance):
        self.pb_ompl_interface.max_distance = max_distance

    def is_state_collision(self, goal, with_ee_base=True, with_ee_body=True):
        # goal: joint state
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        
        goal = [ float(g) for g in goal ]

        currj = self.robot.get_current_joints()

        attach_objs = []
        if with_ee_base:
            attach_objs.append(self.robot.ee.base)
        if with_ee_body:
            attach_objs.append(self.robot.ee.body)
        if self.robot.ee.attach_obj_ids is not None:
            attach_objs.append(self.robot.ee.attach_obj_ids)
        
        self.pb_ompl_interface.set_obstacles(self.obstacles, attach_objs, True)

        # res = self.pb_ompl_interface.is_state_valid(goal)
        res, collision_with = self.pb_ompl_interface.check_is_state_valid(goal)

        self.robot.robot_ompl.set_state(currj)
        
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        
        return res, collision_with

    def ompl(self, goal, allowed_time=None, with_ee_base=True, with_ee_body=True):
        # goal: joint state
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        
        goal = [float(g) for g in goal]

        currj = self.robot.get_current_joints()

        attach_objs = []
        if with_ee_base:
            attach_objs.append(self.robot.ee.base)  # 末端执行器的基座部分（位置）
        if with_ee_body:
            attach_objs.append(self.robot.ee.body)  # 末端执行器的主体部分（方向）
        if self.robot.ee.attach_obj_ids is not None:
            attach_objs.append(self.robot.ee.attach_obj_ids)
        
        self.pb_ompl_interface.set_obstacles(self.obstacles, attach_objs, True) # obstacles为障碍物

        self.robot.robot_ompl.set_state(currj)
        res, path = self.pb_ompl_interface.plan(goal, allowed_time)
        
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        
        return res, path


    def ompl_and_execute(self, dangerous_flag, goal_pose, allowed_time=None, with_ee_base=True, with_ee_body=True):
        if len(goal_pose) == 6 or len(goal_pose) == 8:
            goal = goal_pose
        else:
            goal = self.robot.solve_ik(goal_pose, 500)  # goal为(6,) 应该是为到达goal_pose所需的最终机械臂关节位置

        self.robot.positions = []
        if dangerous_flag is not True:
            # ompl是求解出机械臂为到达goal关节位置，每一步需要怎么移动。path为一个解(中间机械臂关节位置)的list [[1], [2], ..., [50]]
            res, path = self.ompl(goal, allowed_time=allowed_time, with_ee_base=with_ee_base, with_ee_body=with_ee_body)
            # while True:
            # 找到了路径，然后调用 execute 方法来执行这个路径 ???? 此处源代码为何注释掉了
            # self.pb_ompl_interface.execute(path, dynamics=True)    # here ？？？？？and True or False？
            def path_simple(path, num_samples):
                length = len(path)
                if length == 0: return []
                # Calculate the interval
                interval = max(1, length // (num_samples - 1))
                # Sample elements at the calculated interval
                sampled_path = [path[i] for i in range(0, length, interval)]
                # Ensure the last element is included
                if sampled_path[-1] != path[-1]:
                    sampled_path.append(path[-1])
                return sampled_path

            # sampled_path = path_simple(path, 50)
            # for joints in sampled_path:
            for joints in path:
                self.robot.movej(joints)
        else:
            self.robot.movej(goal)
        total_distance = self.robot.calculate_path_distance()
        # for joint in path:
        #     self.env.robot.movej(joint)
        # 确保放置前box的速度为0
        num_joints = p.getNumJoints(self.box_ids[0])
        # 设置物体的所有关节速度为0
        for jointIndex in range(num_joints):
            p.setJointMotorControl2(self.box_ids[0], jointIndex=jointIndex, controlMode=p.VELOCITY_CONTROL,
                                    targetVelocity=0, force=500)
        for _ in range(10):
            pybullet_utils.simulate_step(1)
        return total_distance
        # return res, sampled_path
        # return True, _


    def ompl_and_execute_backup(self, goal_pose, allowed_time=None, with_ee_base=True, with_ee_body=True):
        if len(goal_pose) == 6 or len(goal_pose) == 8:
            goal = goal_pose
        else:
            goal = self.robot.solve_ik(goal_pose, 500)      # goal为(6,) 应该是为到达goal_pose所需的最终机械臂关节位置
            
        # ompl是求解出机械臂为到达goal关节位置，每一步需要怎么移动。path为一个解(中间机械臂关节位置)的list [[1], [2], ..., [50]]
        res, path = self.ompl(goal, allowed_time=allowed_time, with_ee_base=with_ee_base, with_ee_body=with_ee_body)

        if res:
            # while True:
            # 找到了路径，然后调用 execute 方法来执行这个路径 ???? 此处源代码为何注释掉了
            # self.pb_ompl_interface.execute(path, dynamics=True)    # here ？？？？？and True or False？
            def path_simple(path, num_samples):
                length = len(path)
                if length == 0: return []
                # Calculate the interval
                interval = max(1, length // (num_samples - 1))
                # Sample elements at the calculated interval
                sampled_path = [path[i] for i in range(0, length, interval)]
                # Ensure the last element is included
                if sampled_path[-1] != path[-1]:
                    sampled_path.append(path[-1])
                return sampled_path
            
            # sampled_path = path_simple(path, 50)
            # for joints in sampled_path:
            for joints in path:
                self.robot.movej(joints)
            
            # for joint in path:
            #     self.env.robot.movej(joint)
            # 确保放置前box的速度为0
            num_joints = p.getNumJoints(self.box_ids[0])
            # 设置物体的所有关节速度为0
            for jointIndex in range(num_joints):
                p.setJointMotorControl2(self.box_ids[0], jointIndex=jointIndex, controlMode=p.VELOCITY_CONTROL, targetVelocity=0, force=500)
            for _ in range(10):
                pybullet_utils.simulate_step(1)

        # return res, sampled_path
        return res, path

    def ompl_to_pose(self, goal_pose):
        if len(goal_pose) > 2:
            goal_pose = pybullet_utils.mat_to_pose(goal_pose)

        goal = self.robot.solve_ik(goal_pose)
        return self.ompl(goal)
    
    def check_reachable(self, goal_pose, allowed_time=5, with_ee_base=True, with_ee_body=True, dist_threshold=0.02):

        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)

        if len(goal_pose) > 2:
            goal_pose = pybullet_utils.mat_to_pose(goal_pose)
        
        goal = self.robot.solve_ik(goal_pose)
        goal = [ float(g) for g in goal ]
    
        currj = self.robot.get_current_joints()
        self.robot.robot_ompl.set_state(goal)
        result_pos = self.robot.get_link_pose( self.robot.ee_tip )[0]
        target_pos = goal_pose[0]
        dist = np.linalg.norm( np.array(target_pos - result_pos) )
        self.robot.robot_ompl.set_state(currj)
        
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

        collision_with = []

        if dist > dist_threshold:
            return False, collision_with

        # start_time = time.time()
        if allowed_time == 0:
            res, collision_with = self.is_state_collision(goal, with_ee_base, with_ee_body)

        else:
            res, path = self.ompl(goal, allowed_time, with_ee_base, with_ee_body)
        # end_time = time.time()
        # print("time: ", end_time-start_time)
        return res, collision_with

    
    def move_ee_to(self, target_pose, offset=(0,0,0), keep_moving=True, check_rot_z=False, allowed_time=5 ):
        """Move end effector to pose.
            若设置offset,则先移动到 offset 位置,再不断靠近目标
        Args:
            pose: SE(3) picking pose.

        Returns:
            timeout: robot movement timed out if True.
        """

        self.collision_off_obstacles(self.robot.ee.body, 0)
        self.collision_off_obstacles(self.robot.ee.body, -1)

        # 先移动到 offset 位置，再不断靠近目标
        offset = np.array(offset)

        offset_norm = np.linalg.norm(offset)
        need_offset = (offset_norm != 0)
        
        prepick_to_pick = (offset, (0,0,0,1))

        if check_rot_z:
            pose_mat = pybullet_utils.pose_to_mat(target_pose)
            rot_z_90 = tf.vec_euler_to_matrix([0, 0, 1], np.deg2rad(90))
            for i in range(5):
                pose_mat[:3,:3] = pose_mat[:3,:3] @ rot_z_90
                pose = pybullet_utils.mat_to_pose(pose_mat)
                if self.check_reachable(  pose, allowed_time=allowed_time, with_ee_body=False)[0]:
                    break
                
                prepick_pose = utils.multiply(pose, prepick_to_pick)
                if self.check_reachable(  prepick_pose, allowed_time=allowed_time)[0]:
                    break

                if i == 4:
                    return False, True
        else:
            pose = target_pose

        prepick_pose = utils.multiply(pose, prepick_to_pick)

        with_ee_body = False

        if offset_norm > 0.08:
            split_num = 50.0
        else:
            split_num = 5.0
            # allowed_time = 5  # 5

        delta = ( -offset / split_num , (0,0,0,1))
        delta_step = split_num
        while delta_step > 0:
            delta_step -= 1
            prepick_pose = utils.multiply(prepick_pose, delta)
            
            pybullet_utils.draw_pose(prepick_pose, life_time=2)

            t0 = time.time()
            can_prepose, _ = self.check_reachable(prepick_pose, allowed_time=allowed_time, with_ee_body=False)
            if time.time() - t0 > 1:
                dangerous_flag = True
            else:
                dangerous_flag = False
            plan_result = False
            if can_prepose:
                # plan_result, _ = self.ompl_and_execute(dangerous_flag, prepick_pose, allowed_time=allowed_time, with_ee_body=False)
                total_distance = self.ompl_and_execute(dangerous_flag, prepick_pose, allowed_time=allowed_time, with_ee_body=False)
                plan_result = True
                break
                # if plan_result:
                #     break
        
        # check offset pose is ok
        # can_prepose = self.check_reachable(prepick_pose, allowed_time=1 )
        # if can_prepose:
        #     plan_result, plan_path = self.ompl_and_execute(prepick_pose)
        # else:
        #     plan_result, plan_path = self.ompl_and_execute(pose)

        self.collision_on_obstacles(self.robot.ee.body, 0)
        self.collision_on_obstacles(self.robot.ee.body, -1)

        if plan_result is False:
            return False, True, total_distance

        if not need_offset:
            return plan_result, dangerous_flag, total_distance

        # Move towards pick pose until contact is detected.
        split_num = 15.0
        delta = ( -offset / split_num , (0,0,0,1))

        if keep_moving:
            delta_step = split_num * 1.2
        else:
            delta_step = split_num

        if can_prepose:
            targ_pose = prepick_pose
        else:
            targ_pose = pose
        
        current_pos = self.robot.get_link_pose( self.robot.ee_tip )[0]  # self.robot.ee_tip为 Link ID of suction cup
        target_pos = pose[0]
        dist = np.linalg.norm( np.array(current_pos) - np.array(target_pos) )
        # delta = (-dist / 5, (0, 0, 0, 1))

        # if dist > offset_norm * 2.0:
        #     return False, dangerous_flag

        # 改成 and 判断的话就不会一直往下怼
        # 改成 or 判断的话就一直往下怼
        # while delta_step > 0 and not self.robot.ee.detect_contact():  # and target_pose[2] > 0:
        while not self.robot.ee.detect_contact():  # and target_pose[2] > 0:
            delta_step -= 1
            targ_pose = utils.multiply(targ_pose, delta)
            self.robot.movep(targ_pose)
            
        return plan_result or self.robot.ee.detect_contact(), dangerous_flag, total_distance


    def pick(self, pose, offset = None):
        """Move to pose to pick.

        Args:
            pose: SE(3) picking pose.

        Returns:
            pick_success: pick success if True
        """

        # close_to_pick = ((0, 0, -0.01), (0,0,0,1))
        # close_pose = utils.multiply(pose, close_to_pick)
        use_movej = False

        if offset is None:
            use_movej = True
            offset = [0, 0, 0.02]

        move_success, _, _ = self.move_ee_to( pose, offset, check_rot_z=True )
        # move_success = self.move_ee_to( pose, [0,0,0], check_rot_z=True, allowed_time=5 )
        
        if move_success == False:
            return False

        # Activate end effector, move up, and check picking success.
        self.robot.ee.activate()
        self.robot.update_robot_ompl()
        
        if self.robot.ee.attach_obj_ids is not None and self.robot.ee.attach_obj_ids in self.obstacles:
            self.obstacles.remove(self.robot.ee.attach_obj_ids)

        if use_movej:
            currj = self.robot.get_current_joints()
            currj[1] -= 0.3
            # currj[1] -= 0.5
            self.robot.movej(currj)
        else:
            # pose_pos = [ pos for pos in pose[0]]
            # pose_pos[-1] += 0.1
            # postpick_pose = ( pose_pos, pose[1] )
            postpick_to_pick = ( offset, (0,0,0,1) )
            postpick_pose = utils.multiply(pose, postpick_to_pick)
            self.robot.movep(postpick_pose, self.robot.speed)

        # self.robot.update_ee_pose()
        pick_success = self.robot.ee.check_grasp()

        return pick_success

    def place(self, pose, offset=None):
        """Move end effector to pose to place.

        Args:
            pose: SE(3) picking pose.

        Returns:
            timeout: robot movement timed out if True.
        """
        # self.set_ompl_distance(0.011)

        if offset is None:
            offset = (0, 0, 0.1)
        move_success, _ = self.move_ee_to(pose, offset, False)
        if move_success == False:
            return False
        
        attach_obj_ids = self.robot.ee.attach_obj_ids

        self.collision_off_obstacles(self.robot.ee.body, 0)
        self.collision_off_obstacles(self.robot.ee.body, -1)

        pybullet_utils.simulate_step(50)

        self.robot.ee.release()
        self.robot.update_robot_ompl()

        postplace_to_place = (offset, (0,0,0,1.0))
        postplace_pose = utils.multiply(pose, postplace_to_place)
        # timeout |= self.robot.movep(postplace_pose)
        self.move_ee_to(postplace_pose)

        self.collision_on_obstacles(self.robot.ee.body, 0)
        self.collision_on_obstacles(self.robot.ee.body, -1)

        if attach_obj_ids is not None and attach_obj_ids not in self.obstacles:
            self.obstacles.append(attach_obj_ids)
        
        if attach_obj_ids is not None:
            if len(self.containers) == 0:
                self.containers.append([])
            self.containers[-1].append(attach_obj_ids)
        
        self.packed_object_ids.append(attach_obj_ids)

        return move_success
        

    def collision_off_obstacles(self, obj_id, link_id):
        for o in self.obstacles:
            pybullet_utils.mask_coll_off(obj_id, link_id)
            p.setCollisionFilterPair(obj_id, o, link_id, -1, 0)
        p.stepSimulation()

    def collision_on_obstacles(self, obj_id, link_id):
        for o in self.obstacles:
            pybullet_utils.mask_coll_on(obj_id, link_id)
            p.setCollisionFilterPair(obj_id, o, link_id, -1, 1)
        p.stepSimulation()
        