# code from https://github.com/lyf44/pybullet_ompl
# https://github.com/ompl/ompl/releases/tag/prerelease

'''
cd /media/wzf/goodluck/Downloads
pip install ompl-1.6.0-cp37-cp37m-manylinux_2_28_x86_64.whl
'''


try:
    from ompl import util as ou
    from ompl import base as ob
    from ompl import geometric as og
except ImportError:
    # if the ompl module is not in the PYTHONPATH assume it is installed in a
    # subdirectory of the parent directory called "py-bindings."
    from os.path import abspath, dirname, join
    import sys
    sys.path.insert(0, join(dirname(dirname(abspath(__file__))), 'ompl/py-bindings'))
    # sys.path.insert(0, join(dirname(abspath(__file__)), '../whole-body-motion-planning/src/ompl/py-bindings'))
    # print(sys.path)
    from ompl import util as ou
    from ompl import base as ob
    from ompl import geometric as og
import pybullet as p
from arm_pybullet_env.utils import ompl_utils as utils
import time
from itertools import product
import copy
import math
from scipy.spatial.transform import Rotation as R
import numpy as np


def pose_to_mat(pose):
    pos, quat = pose
    tool_mat = R.from_quat(quat).as_matrix()

    tool_pose = np.identity(4)
    tool_pose[:3,:3] = tool_mat
    tool_pose[:3, 3] = pos
    return tool_pose

class PbOMPLRobot():
    '''
    To use with Pb_OMPL. You need to construct a instance of this class and pass to PbOMPL.

    Note:
    This parent class by default assumes that all joints are acutated and should be planned. If this is not your desired
    behaviour, please write your own inheritated class that overrides respective functionalities.
    '''
    def __init__(self, id, homej=None, ee_base=None, ee_body=None) -> None:
        # Public attributes
        self.id = id
        self.dynamics = False

        # prune fixed joints
        all_joint_num = p.getNumJoints(id)
        all_joint_idx = list(range(all_joint_num))
        joint_idx = [j for j in all_joint_idx if self._is_not_fixed(j)]
        self.num_dim = len(joint_idx)
        self.joint_idx = joint_idx
        self.joint_bounds = []

        self.homej = homej

        self.ee_base = ee_base
        self.ee_body = ee_body
        self.attach_obj = None
        self.obj_to_body = None

        self.reset()



    def update_ee_offset(self, ee_base, ee_body):
        
        link_info = p.getLinkState(self.id, 9)  # self.ee_tip   6  9
        tool_pos = np.array(link_info[0])
        tool_quat = np.array(link_info[1])

        tool_mat = R.from_quat(tool_quat).as_matrix()
        tool_mat_z = tool_mat[:,2]
        
        ee_base_pos = np.array([ pos for pos in tool_pos ])
        ee_body_pos = np.array([ pos for pos in tool_pos ])

        ee_base_pos += tool_mat_z * -0.01
        ee_body_pos += tool_mat_z * 0.08
        p.resetBasePositionAndOrientation( ee_base, list(ee_base_pos), tool_quat )
        p.resetBasePositionAndOrientation( ee_body, list(ee_body_pos), tool_quat )

    def update_attach_offset(self, ee_body, obj_id, obj_to_body):
        
        link_info = p.getLinkState(ee_body, 0)
        tool_pos = np.array(link_info[0])
        tool_quat = np.array(link_info[1])

        tool_mat = pose_to_mat([ tool_pos, tool_quat ])
        obj_mat = pose_to_mat( obj_to_body )
        
        obj_mat =  tool_mat @ obj_mat

        obj_pos = obj_mat[:3, 3]
        obj_quat = R.from_matrix(obj_mat[:3,:3]).as_quat()
        
        p.resetBasePositionAndOrientation( obj_id, list(obj_pos), list(obj_quat) )

    def _is_not_fixed(self, joint_idx):
        joint_info = p.getJointInfo(self.id, joint_idx)
        return joint_info[2] != p.JOINT_FIXED

    def get_joint_bounds(self):
        '''
        Get joint bounds.
        By default, read from pybullet
        '''
        for i, joint_id in enumerate(self.joint_idx):
            joint_info = p.getJointInfo(self.id, joint_id)
            low = joint_info[8] # low bounds
            high = joint_info[9] # high bounds
            if low < high:
                self.joint_bounds.append([low, high])
        # print("Joint bounds: {}".format(self.joint_bounds))
        return self.joint_bounds

    def get_cur_state(self):

        states = p.getJointStates(self.id, self.joint_idx)

        self.state = [ state[0] for state in states ]
        return self.state
        # return copy.deepcopy(self.state)

    def set_state(self, state):
        '''
        Set robot state.
        To faciliate collision checking
        Args:
            state: list[Float], joint values of robot
        '''
        self._set_joint_positions(self.joint_idx, state)
        self.state = state

    def reset(self):
        '''
        Reset robot state
        Args:
            state: list[Float], joint values of robot
        '''
        if self.homej is None:
            state = [0] * self.num_dim
        else:
            state = self.homej
        self._set_joint_positions(self.joint_idx, state)
        self.state = state


    def _set_joint_positions(self, joints, positions):
        if self.dynamics:
            for joint, value in zip(joints, positions):
                p.setJointMotorControl2(self.id, joint, p.POSITION_CONTROL, value, force=50., positionGain=0.02)

            # p.setJointMotorControlArray(
            #     self.id,
            #     joints,
            #     p.POSITION_CONTROL,
            #     targetPositions=positions,
            #     forces=[300] * len(joints),
            #     positionGains=[0.5] * len(joints)
            # )

        else:
            for joint, value in zip(joints, positions):
                p.resetJointState(self.id, joint, value, targetVelocity=0)
            
            if self.ee_base is not None:
                self.update_ee_offset(self.ee_base, self.ee_body)
            if self.attach_obj is not None:
                self.update_attach_offset(self.ee_body, self.attach_obj, self.obj_to_body)


    def get_joint_pose(self, joint_id):
        joint_info = p.getJointInfo(self.id, joint_id)
        
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
        
        link_info = p.getLinkState(self.id, link_id)
        link_pos = np.array(link_info[0])
        link_quat = np.array(link_info[1])
        return [link_pos, link_quat]

class PbStateSpace(ob.RealVectorStateSpace):
    def __init__(self, num_dim) -> None:
        super().__init__(num_dim)
        self.num_dim = num_dim
        self.state_sampler = None

    def allocStateSampler(self):
        '''
        This will be called by the internal OMPL planner
        '''
        # WARN: This will cause problems if the underlying planner is multi-threaded!!!
        if self.state_sampler:
            return self.state_sampler

        # when ompl planner calls this, we will return our sampler
        return self.allocDefaultStateSampler()

    def set_state_sampler(self, state_sampler):
        '''
        Optional, Set custom state sampler.
        '''
        self.state_sampler = state_sampler

class PbOMPL():
    def __init__(self, robot, obstacles = [], planner="RRT", interpolate_num=100, step_size=2, planning_time=5.0) -> None:
        '''
        Args
            robot: A PbOMPLRobot instance.
            obstacles: list of obstacle ids. Optional.
        '''
        self.robot = robot
        self.robot_id = robot.id
        self.obstacles = obstacles
        # print(self.obstacles)
        self.step_size = step_size

        self.space = PbStateSpace(robot.num_dim)

        bounds = ob.RealVectorBounds(robot.num_dim)
        joint_bounds = self.robot.get_joint_bounds()
        for i, bound in enumerate(joint_bounds):
            bounds.setLow(i, bound[0])
            bounds.setHigh(i, bound[1])

            # bounds.setLow(i, -np.pi * 1.2)
            # bounds.setHigh(i, np.pi * 1.2)

        self.space.setBounds(bounds)

        self.ss = og.SimpleSetup(self.space)
        self.ss.setStateValidityChecker(ob.StateValidityCheckerFn(self.is_state_valid))
        self.si = self.ss.getSpaceInformation()
        # self.si.setStateValidityCheckingResolution(0.005)
        # self.collision_fn = pb_utils.get_collision_fn(self.robot_id, self.robot.joint_idx, self.obstacles, [], True, set(),
        #                                                 custom_limits={}, max_distance=0, allow_collision_links=[])

        self.set_obstacles(self.obstacles)
        self.set_planner(planner, step_size) # RRT by default
        

        # self._force = 300
        # self._speed = 0.5

        self.interpolate_num = interpolate_num
        self.planning_time = planning_time

        self.max_distance = 1e-5

    def new_simplSetup(self):
        self.ss = og.SimpleSetup(self.space)
        self.ss.setStateValidityChecker(ob.StateValidityCheckerFn(self.is_state_valid))
        self.set_planner(self.planner_name, self.step_size)

    def set_obstacles(self, obstacles, attach_objs=[], self_collisions=False):
        self.obstacles = obstacles
        # update collision detection
        self.setup_collision_detection(self.robot, self.obstacles, attach_objs, self_collisions)

    def add_obstacles(self, obstacle_id):
        self.obstacles.append(obstacle_id)

    def remove_obstacles(self, obstacle_id):
        self.obstacles.remove(obstacle_id)

    def is_state_valid(self, state):
        # satisfy bounds TODO
        # Should be unecessary if joint bounds is properly set

        # check self-collision
        self.robot.set_state(self.state_to_list(state))
        for link1, link2 in self.check_link_pairs:
            if utils.pairwise_link_collision(self.robot_id, link1, self.robot_id, link2, max_distance=self.max_distance):
                # print(get_body_name(body), get_link_name(body, link1), get_link_name(body, link2))
                # if link1 ==5 and link2 == 7:
                #     continue
                return False

        # check collision against environment
        for body1, body2 in self.check_body_pairs:
            if utils.pairwise_collision(body1, body2, max_distance=self.max_distance):
                # print('body collision', body1, body2)
                # print(get_body_name(body1), get_body_name(body2))
                return False

        for body1, body2 in self.attach_check_body_pairs:
            if utils.pairwise_collision(body1, body2, max_distance=self.max_distance):
                return False
                
        return True

    def check_is_state_valid(self, state):
        # satisfy bounds TODO
        # Should be unecessary if joint bounds is properly set
        collision_with = []

        # check self-collision
        self.robot.set_state(self.state_to_list(state))
        for link1, link2 in self.check_link_pairs:
            if utils.pairwise_link_collision(self.robot_id, link1, self.robot_id, link2, max_distance=self.max_distance):
                # print(get_body_name(body), get_link_name(body, link1), get_link_name(body, link2))
                return False, collision_with

        # check collision against environment
        for body1, body2 in self.check_body_pairs:
            if utils.pairwise_collision(body1, body2, max_distance=self.max_distance):
                # print('body collision', body1, body2)
                # print(get_body_name(body1), get_body_name(body2))
                collision_with.append(body2)
                return False, collision_with

        if len(collision_with) > 0:
            return False, collision_with

        for body1, body2 in self.attach_check_body_pairs:
            if utils.pairwise_collision(body1, body2, max_distance=self.max_distance):
                return False, collision_with

        return True, collision_with

    def setup_collision_detection(self, robot, obstacles, attach_objs, self_collisions = False, allow_collision_links = []):
        self.check_link_pairs = utils.get_self_link_pairs(robot.id, robot.joint_idx) if self_collisions else []
        moving_links = frozenset(
            [item for item in utils.get_moving_links(robot.id, robot.joint_idx) if not item in allow_collision_links])
        moving_bodies = [(robot.id, moving_links)]
        self.check_body_pairs = list(product(moving_bodies, obstacles))
        self.attach_check_body_pairs = list(product(attach_objs, obstacles))

        if self_collisions:
            robot_pairs = [ ( robot.id, robot.joint_idx ) ]
            self.attach_check_body_pairs.extend(list(product(attach_objs, robot_pairs)))
        
        a = 1

    def set_planner(self, planner_name, step_size):
        '''
        Note: Add your planner here!!
        '''
        self.planner_name = planner_name
        
        if planner_name == "PRM":
            self.planner = og.PRM(self.ss.getSpaceInformation())
        elif planner_name == "RRT":
            self.planner = og.RRT(self.ss.getSpaceInformation())
        elif planner_name == "RRTConnect":
            self.planner = og.RRTConnect(self.ss.getSpaceInformation())
        elif planner_name == "RRTstar":
            self.planner = og.RRTstar(self.ss.getSpaceInformation())
        elif planner_name == "TRRT":
            self.planner = og.TRRT(self.ss.getSpaceInformation())
        elif planner_name == "EST":
            self.planner = og.EST(self.ss.getSpaceInformation())
        elif planner_name == "FMT":
            self.planner = og.FMT(self.ss.getSpaceInformation())
        elif planner_name == "BITstar" or planner_name == "kBITstar" :
            self.planner = og.BITstar(self.ss.getSpaceInformation())
        elif planner_name == "LBKPIECE1":
            self.planner = og.LBKPIECE1(self.ss.getSpaceInformation())
        elif planner_name == "PRM":
            self.planner = og.PRM(self.ss.getSpaceInformation())
        elif planner_name == "SPARS":
            self.planner = og.SPARS(self.ss.getSpaceInformation())
        elif planner_name == "SPARS2":
            self.planner = og.SPARStwo(self.ss.getSpaceInformation())
        else:
            print("{} not recognized, please add it first".format(planner_name))
            return

        self.planner.setRange(2)  # 调整步长
        
        self.ss.setPlanner(self.planner)

    def plan_start_goal(self, start, goal, allowed_time=None):
        '''
        plan a path to goal from the given robot start state
        '''
        print("start_planning")
        # print(self.planner.params())

        if allowed_time is None:
            allowed_time = self.planning_time

        orig_robot_state = self.robot.get_cur_state()

        # set the start and goal states;
        s = ob.State(self.space)
        g = ob.State(self.space)
        for i in range(len(start)):
            s[i] = start[i]
            g[i] = goal[i]
        
        self.new_simplSetup()

        self.ss.setStartAndGoalStates(s, g)
        # self.ss.setStartAndGoalStates(s, s)

        res = False

        self.robot.set_state(orig_robot_state)

        # attempt to solve the problem within allowed planning time  尝试在allowed_time指定的时间内找到一条从当前状态到目标状态的路径
        # 此处如果已经安装ompl包则会报错，因为envs中自己实现了ompl方法，两者同时存在会报 Abnormal program termination: received signal 11 (Segmentation fault) 的错误
        solved = self.ss.solve(allowed_time)        # 此处的solve是如何实现的？？？
        # solved = self.ss.solve(1)
        sol_path_list = []
        if solved:      # 获取解决方案路径，并将其插值为指定数量的段落，以便可以更平滑地执行
            print("Found solution: interpolating into {} segments".format(self.interpolate_num))
            # print the path to screen
            try:
                sol_path_geometric = self.ss.getSolutionPath()
            except:
                self.robot.set_state(orig_robot_state)
                return res, sol_path_list

            sol_path_geometric.interpolate(self.interpolate_num)
            sol_path_states = sol_path_geometric.getStates()
            sol_path_list = [self.state_to_list(state) for state in sol_path_states]

            # for sol_path in sol_path_list:
            #     self.is_state_valid(sol_path)
            res = True
        else:
            print("No solution found")

        # reset robot state
        self.robot.set_state(orig_robot_state)
        
        return res, sol_path_list

    def plan(self, goal, allowed_time=None):
        '''
        plan a path to gaol from current robot state
        '''
        start = self.robot.get_cur_state()
        return self.plan_start_goal(start, goal, allowed_time=allowed_time)


    def execute(self, path, dynamics=False):
        '''
        Execute a planned plan. Will visualize in pybullet.
        Args:
            path: list[state], a list of state
            dynamics: allow dynamic simulation. If dynamics is false, this API will use robot.set_state(),
                      meaning that the simulator will simply reset robot's state WITHOUT any dynamics simulation. Since the
                      path is collision free, this is somewhat acceptable.
        '''
        for q in path:
            if dynamics:
                for i in range(self.robot.num_dim):
                    p.setJointMotorControl2(self.robot.id, i, p.POSITION_CONTROL, q[i], force=240.)
            else:
                self.robot.set_state(q)

            p.stepSimulation()
            time.sleep(1/240.0)
            
    # -------------
    # Configurations
    # ------------

    def set_state_sampler(self, state_sampler):
        self.space.set_state_sampler(state_sampler)

    # -------------
    # Util
    # ------------

    def state_to_list(self, state):
        return [state[i] for i in range(self.robot.num_dim)]