
import os
import time
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.sys.path.insert(0, currentdir)

import numpy as np
import pybullet as p
from arm_pybullet_env.arm_envs import models
from arm_pybullet_env.arm_envs import pb_ompl
from arm_pybullet_env.arm_envs.grippers import Suction
from arm_pybullet_env.utils import pybullet_utils, utils, ompl_utils

from scipy.spatial.transform import Rotation as R

# PLACE_STEP = 0.0003
# PLACE_DELTA_THRESHOLD = 0.005

# ARM_URDF_PATH = "irb2400/irb2400.urdf"
# ARM_URDF_PATH = "ur5/ur5.urdf"


class ARM:
    '''
    IRB6700 Robot or UR5
    
    Args:
        - `ee_type`: str 末端执行器类型
            * end effector type, we only have 'suction' currently
        - `speed`: float 机器人运动的速度
        - `obj_ids`: dict   字典, 收集了场景中所有可抓取的对象的ID,用于吸盘末端执行器抓取对象
            * collect all the graspable object we have in the scene, for Suction to grasp object
        -  `homej`: list(float) [3] 机器人臂的初始关节位置
            * init joint positions
        - `is_static`: bool 机器人是否处于静止状态
            * True if the robot is moving

    Functions:
        - `reset` -> reload the robot state to init pose
        - `add_object_to_list` -> add obj id into self.obj_ids
        - `movej` -> move end effector to target joints' positions
        - `movep` -> move end effector to target pose
        - `solve_ik` -> solve ik for target pose
        - `move_to` -> controll end effector to approach target pose
        - `pick` -> move to target pose to pick object
        - `place` -> move to target pose to place object
        - `get_ee_pose` -> get ee pose
        - `debug_gui` -> add slide bar gui in the pybullet gui
        - `update_arm` -> update the joints' positions based on slide bar values
    '''
    
    def __init__(self, ee_type='suction', speed=0.01, offset=[0,0,0], recorder=None):
        

        self.obj_ids = {'fixed': [], 'rigid': [], 'deformable': []}

        # init joint posiitons
        
        self.homej = np.array([0, 0, 0, 0, 0, 0]) * np.pi # now
        # self.homej = np.array([0, 0.01367294, -0.04786678, 0, 1.53602555, 0])  # wen
        # self.homej = np.array([0, -0.5, 0.5 , 0, 1.5, 0])
        self.homej = np.array([-1, -0.7, 0.5, -0.5, -0.5, 0]) * np.pi
        # self.homej = np.array([-1, -0.5, 0.5, -0.5, -0.5, 0]) * np.pi
        # self.homej = np.array([0, -0.25, 0.25, 0, 0.25, 0]) * np.pi # now

        # self.homej = np.array([0, -0.5, 0, -0.5, 0, 0.5, 0.25, 0]) * np.pi  # home
        # self.homej = np.array([0, -0.25, 0, -0.5, 0, 0.25, 0.25, 0]) * np.pi  # ready
        # self.homej = np.array([0, -0.25, 0.25, 0, 0.25, 0.25, -0.7, 0]) * np.pi  # my
        # self.homej = np.array([0, -0.25, 0.25, 0, 0.25, 0.25, 0.25, 0]) * np.pi  # now

        self.homej2 = np.array([-3.81010793, -1.9823318 ,  1.66832743, -1.25692833, -1.57112184, -0.66827464])

        self.speed = speed

        self.offset = offset    # 机器人臂的偏移量

        self.ee_type = ee_type

        self.recorder = recorder

        self.arm_urdf_path = 'irb2400/irb2400.urdf'

        self.positions = []
            
    def reset(self, setting, arm_urdf_path, homej, show_gui=True):
        """Performs common reset functionality for all supported tasks. 重载机器人状态到初始姿态"""

        self.obj_ids = {'fixed': [], 'rigid': [], 'deformable': []}

        self.arm_urdf_path = arm_urdf_path
        self.homej = np.array(homej) * np.pi

        # Load IRB6700 robot arm equipped with suction end effector.
        self.arm = pybullet_utils.load_urdf(p, os.path.join( models.get_data_path(), self.arm_urdf_path), self.offset)
        # self.arm = p.loadURDF(os.path.join( models.get_data_path(), self.arm_urdf_path), globalScaling=1.0)
            # self.offset, p.getQuaternionFromEuler( [0,0,-np.pi/2.0] ))
            # p, os.path.join( models.get_data_path(), self.arm_urdf_path), flags=p.URDF_USE_SELF_COLLISION | p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT )
        self.urdf_file = os.path.join( models.get_data_path(), self.arm_urdf_path)
        
        
        if self.ee_type == 'suction':
            self.ee = Suction( models.get_data_path(), self.arm, 9, self.obj_ids)   # 9    11   'link_6-tool0_fixed_joint'
            self.ee_tip = 10  # 10   12 # Link ID of suction cup.   'tool0-tool_tip'
            
        # Get revolute joint indices of robot (skip fixed joints).  JOINT_PRISMATIC
        n_joints = p.getNumJoints(self.arm)
        joints = [p.getJointInfo(self.arm, i) for i in range(n_joints)]
        # self.joints = [j[0] for j in joints if j[2] == p.JOINT_REVOLUTE]
        # self.joint_names = [str(j[1]) for j in joints if j[2] == p.JOINT_REVOLUTE]
        # self.joint_lower_limits = [j[8] for j in joints if j[2] == p.JOINT_REVOLUTE]
        # self.joint_upper_limits = [j[9] for j in joints if j[2] == p.JOINT_REVOLUTE]
        self.joints = [j[0] for j in joints if j[2] == p.JOINT_REVOLUTE or j[2] == p.JOINT_PRISMATIC]
        self.joint_names = [str(j[1]) for j in joints if j[2] == p.JOINT_REVOLUTE or j[2] == p.JOINT_PRISMATIC]
        self.joint_lower_limits = [j[8] for j in joints if j[2] == p.JOINT_REVOLUTE or j[2] == p.JOINT_PRISMATIC]
        self.joint_upper_limits = [j[9] for j in joints if j[2] == p.JOINT_REVOLUTE or j[2] == p.JOINT_PRISMATIC]
        self.joint_ranges = [upper - lower for upper, lower in zip(self.joint_upper_limits, self.joint_lower_limits)]


        # Move robot to home joint configuration.
        self.set_joints_state(self.homej)

        # Reset end effector.
        self.ee.release()

        # No consideration of robotic arms
        if setting == 1 or setting == 2:
            for link_index in range(-1, n_joints):
                p.changeVisualShape(self.arm, link_index, rgbaColor=[1, 1, 1, 0])
            for link_index in range(-1, p.getNumJoints(self.ee.base)):
                p.changeVisualShape(self.ee.base, link_index, rgbaColor=[1, 1, 1, 0])
            for link_index in range(-1, p.getNumJoints(self.ee.body)):
                p.changeVisualShape(self.ee.body, link_index, rgbaColor=[1, 1, 1, 0])


        # add ompl
        self.robot_ompl = pb_ompl.PbOMPLRobot(self.arm, self.homej)
        self.update_robot_ompl()

        if show_gui:
            self.debug_gui()

    @property
    def is_static(self):
        """Return true if objects are no longer moving."""
        v = [np.linalg.norm(p.getBaseVelocity(i)[0])
            for i in self.obj_ids['rigid']]
        return all(np.array(v) < 5e-3)

    def add_object_to_list(self, category, obj_id):
        self.obj_ids[category].append(obj_id)

    #---------------------------------------------------------------------------
    # Robot Movement Functions
    #---------------------------------------------------------------------------

    def record_position(self):
        """记录当前时间步末端执行器的位置"""
        # position, _ = p.getBasePositionAndOrientation(self.ee_tip)
        position, _ = self.get_link_pose(9)
        self.positions.append(np.array(position))  # 将位置转换为 NumPy 数组并存储

    def calculate_path_distance(self):
        """计算末端执行器的移动路径距离"""
        total_distance = 0.0
        for i in range(1, len(self.positions)):
            # 计算相邻两点之间的欧氏距离并累加
            distance = np.linalg.norm(self.positions[i] - self.positions[i - 1])
            total_distance += distance
        return total_distance


    def movej(self, targj, speed=0.01, timeout=0.1):
        """Move ARM to target joint configuration.  移动末端执行器到目标关节位置"""
        t0 = time.time()
        while (time.time() - t0) < timeout:
            currj = [p.getJointState(self.arm, i)[0] for i in self.joints]
            currj = np.array(currj)
            diffj = targj - currj
            if all(np.abs(diffj) < 1e-2):
                return False

            # Move with constant velocity
            norm = np.linalg.norm(diffj)
            v = diffj / norm if norm > 0 else 0
            stepj = currj + v * speed
            gains = np.ones(len(self.joints))
            p.setJointMotorControlArray(
                bodyIndex=self.arm,
                jointIndices=self.joints,
                controlMode=p.POSITION_CONTROL,
                targetPositions=stepj,
                positionGains=gains)
            # p.stepSimulation()
            # time.sleep(1/ 240.0)
            pybullet_utils.simulate_step(1)
            # self.update_ee_pose()
            self.record_position()

        # print(f'Warning: movej exceeded {timeout} second timeout. Skipping.')
        return True

    def movej_backup(self, targj, speed=0.01, timeout=0.1):
        """Move ARM to target joint configuration.  移动末端执行器到目标关节位置"""
        t0 = time.time()
        while (time.time() - t0) < timeout:
            currj = [p.getJointState(self.arm, i)[0] for i in self.joints]
            currj = np.array(currj)
            diffj = targj - currj
            if all(np.abs(diffj) < 1e-2):
                return False

            # Move with constant velocity
            norm = np.linalg.norm(diffj)
            v = diffj / norm if norm > 0 else 0
            stepj = currj + v * speed
            gains = np.ones(len(self.joints))
            p.setJointMotorControlArray(
                bodyIndex=self.arm,
                jointIndices=self.joints,
                controlMode=p.POSITION_CONTROL,
                targetPositions=stepj,
                positionGains=gains)
            # p.stepSimulation()
            # time.sleep(1/ 240.0)
            pybullet_utils.simulate_step(1)
            # self.update_ee_pose()

        # print(f'Warning: movej exceeded {timeout} second timeout. Skipping.')
        return True

    def movep(self, pose, speed=0.01):
        """Move ARM to target end effector pose.    移动末端执行器到目标姿态"""
        targj = self.solve_ik(pose)
        return self.movej(targj, speed)

    def solve_ik(self, pose, max_num_iterations=100):
        """Calculate joint configuration with inverse kinematics.   求解目标姿态的逆运动学"""
        # lowerLimits=[-3 * np.pi / 2, -2.3562, -5, -5, -5, -5, -5],
        # upperLimits=[-np.pi / 2, 0, 5, 5, 5, 5],
        # jointRanges=[np.pi, 2.3562, 10, 10, 10, 10],  # * 6,
        joints = p.calculateInverseKinematics(
            bodyUniqueId=self.arm,
            endEffectorLinkIndex=self.ee_tip,
            targetPosition=pose[0],
            targetOrientation=pose[1],
            lowerLimits=self.joint_lower_limits,
            upperLimits=self.joint_upper_limits,
            jointRanges=self.joint_ranges,
            restPoses=np.float32(self.homej).tolist(),
            maxNumIterations=max_num_iterations,
            residualThreshold=1e-5)
        # 取前7个数
        # new_joints = joints[:7]     # 最后一个为prismatic joint
        joints = np.float32(joints)
        joints[2:] = (joints[2:] + np.pi) % (2 * np.pi) - np.pi
        return joints

    #============----------------   Move and grasp   ----------------============#
    
    def move_to(self, pose, offset=(0,0,0)):
        """Move end effector to pose.   控制末端执行器接近目标姿态

        Args:
            pose: SE(3) picking pose.

        Returns:
            timeout: robot movement timed out if True.
        """

        # 先移动到 offset 位置，再不断靠近目标
        offset = np.array(offset)
        
        prepick_to_pick = (offset, (0,0,0,1))

        prepick_pose = utils.multiply(pose, prepick_to_pick)
        pybullet_utils.draw_pose(prepick_pose, life_time=2)
        timeout = self.movep(prepick_pose)

        if np.linalg.norm(offset) == 0:
            return timeout
        
        # Move towards pick pose until contact is detected.
        delta_step = 103
        delta = ( -offset / 100.0 , (0,0,0,1))

        targ_pose = prepick_pose
        # 改成 and 判断的话就不会一直往下怼
        # 改成 or 判断的话就一直往下怼
        while delta_step > 0 and not self.ee.detect_contact():  # and target_pose[2] > 0:
            delta_step -= 1
            targ_pose = utils.multiply(targ_pose, delta)
            timeout |= self.movep(targ_pose)
            if timeout:
                return True

        return timeout

    def pick(self, pose):
        """Move to pose to pick.    移动到目标姿态以抓取物体

        Args:
            pose: SE(3) picking pose.

        Returns:
            pick_success: pick success if True
        """

        offset = (0, 0, 0.1)
        timeout = self.move_to(pose, offset)
        
        if timeout:
            return False

        # Activate end effector, move up, and check picking success.
        self.ee.activate()
        self.update_robot_ompl()
        
        postpick_to_pick = ( offset, (0,0,0,1) )
        
        postpick_pose = utils.multiply(pose, postpick_to_pick)

        timeout |= self.movep(postpick_pose, self.speed)

        pick_success = self.ee.check_grasp()

        return pick_success

    def place(self, pose):
        """Move end effector to pose to place.  移动到目标姿态以放置物体

        Args:
            pose: SE(3) picking pose.

        Returns:
            timeout: robot movement timed out if True.
        """

        offset = (0, 0, 0.1)
        timeout = self.move_to(pose, offset)
        if timeout:
            return True
        
        self.ee.release()
        self.update_robot_ompl()

        postplace_to_place = (offset, (0,0,0,1.0))
        postplace_pose = utils.multiply(pose, postplace_to_place)
        timeout |= self.movep(postplace_pose)

        return timeout

    #============----------------   robot tools   ----------------============#
    
    def get_ee_pose(self):
        '''获取末端执行器的姿态'''
        pos, orient, *_ = p.getLinkState(self.arm, self.ee_tip)
        return list(pos), list(orient)
        
    def debug_gui(self):
        '''在PyBullet GUI中添加滑动条'''
        debug_items = []
        for i in range(len(self.joints)):
            pos, *_ = p.getJointState(self.arm, self.joints[i])
            item = p.addUserDebugParameter( self.joint_names[i], self.joint_lower_limits[i], self.joint_upper_limits[i], pos )
            debug_items.append(item)
        self.debug_items = debug_items

    def update_arm(self):
        '''根据滑动条的值更新关节位置'''
        joint_values = []
        for item in self.debug_items:
            joint_values.append(p.readUserDebugParameter(item))
        self.movej(joint_values)

    def get_current_joints(self):
        return np.array([p.getJointState(self.arm, i)[0] for i in self.joints])

    def set_joints_state(self, joints):
        for i in range(len(self.joints)):
            # p.resetJointState(self.arm, self.joints[i], joints[i])
            p.resetJointState(self.arm, self.joints[i], joints[i], targetVelocity=0)
        self.update_ee_pose()

        if self.ee.attach_obj_ids is not None:
            self.update_attach_offset(self.ee.attach_obj_ids, self.ee.obj_to_body)

    def update_ee_pose(self):
        tool_pos, tool_quat = self.get_link_pose(9)     # 9 11  self.ee_tip   'link_6-tool0_fixed_joint'

        tool_mat = np.array(p.getMatrixFromQuaternion(tool_quat))
        tool_mat = np.reshape(tool_mat, (3,3))
        tool_mat_z = tool_mat[:,2]
        
        ee_base_pos = np.array([ pos for pos in tool_pos ])
        ee_body_pos = np.array([ pos for pos in tool_pos ])

        ee_base_pos += tool_mat_z * -0.01
        ee_body_pos += tool_mat_z * 0.08
        p.resetBasePositionAndOrientation( self.ee.base, list(ee_base_pos), tool_quat )
        p.resetBasePositionAndOrientation( self.ee.body, list(ee_body_pos), tool_quat )



    def update_attach_offset(self, obj_id, obj_to_body):
        
        link_info = p.getLinkState(self.ee.body, 0)
        tool_pos = np.array(link_info[0])
        tool_quat = np.array(link_info[1])

        tool_mat = pybullet_utils.pose_to_mat([ tool_pos, tool_quat ])
        obj_mat = pybullet_utils.pose_to_mat( obj_to_body )
        
        obj_mat = tool_mat @ obj_mat

        obj_pos = obj_mat[:3, 3]
        obj_quat = R.from_matrix(obj_mat[:3,:3]).as_quat()
        p.resetBasePositionAndOrientation( obj_id, list(obj_pos), list(obj_quat) )
        
    def update_robot_ompl(self):
        self.robot_ompl.ee_base = self.ee.base
        self.robot_ompl.ee_body = self.ee.body

        self.robot_ompl.attach_obj = self.ee.attach_obj_ids
        self.robot_ompl.obj_to_body = self.ee.obj_to_body


    def get_links_dict(self):
        
        gripper_id = self.arm

        link_name_to_index = {p.getBodyInfo(gripper_id)[0].decode('UTF-8'):-1,}
        joint_name_to_index = {}
        joint_name_to_parent = {}

        for _id in range(p.getNumJoints(gripper_id)):
            joint_info = p.getJointInfo(gripper_id, _id)
            joint_index = joint_info[0]
            joint_name = joint_info[1].decode('UTF-8')
            joint_parent_link_index = joint_info[-1]

            joint_name_to_index[joint_name] = joint_index
            joint_name_to_parent[joint_name] = joint_parent_link_index

            link_name = joint_info[12].decode('UTF-8')
            link_name_to_index[link_name] = _id
        
        self.joint_name_to_index = joint_name_to_index
        self.joint_name_to_parent = joint_name_to_parent
        self.link_name_to_index = link_name_to_index

        return joint_name_to_index, joint_name_to_parent, link_name_to_index

    def get_joint_pose(self, joint_id):
        joint_info = p.getJointInfo(self.arm, joint_id)
        
        pos_in_parent = joint_info[-3]
        # quat_in_parent = joint_info[-2]
        link_id = joint_info[-1]

        link_pos, link_quat = self.get_link_pose(link_id)

        joint_m = np.identity(4)
        link_m = np.identity(4)

        joint_m[:3,3] = pos_in_parent
        link_m[:3,3] = link_pos

        link_m[:3,:3] = R.from_quat(link_quat).as_matrix()

        joint_m = link_m @ joint_m

        joint_pos = joint_m[:3, 3]
        joint_quat = R.from_matrix(joint_m[:3,:3]).as_quat()
        return [joint_pos, joint_quat]
    
    def get_link_pose(self, link_id):

        return pybullet_utils.get_link_pose(self.arm, link_id)
    
