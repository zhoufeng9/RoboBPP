import pybullet as p
import pybullet_data
import time

from os.path import abspath, dirname, join
import sys
sys.path.insert(0, join(dirname(abspath(__file__)), 'ompl/py-bindings'))
from ompl import util as ou
from ompl import base as ob
from ompl import geometric as og
import numpy as np
from utils import ompl_utils as utils

# 连接到 PyBullet 仿真
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
print(pybullet_data.getDataPath())

plane_id = p.loadURDF("/home/wzf/Workspace/rl/RobotPackingBenchmark-main-v4/arm_pybullet_env/arm_envs/models/plane/plane.urdf")
# robot_id = p.loadURDF("franka_panda/panda.urdf")
robot_id = p.loadURDF("/home/wzf/Workspace/rl/RobotPackingBenchmark-main-v4/arm_pybullet_env/arm_envs/models/irb6700/irb6700.urdf")
p.setGravity(0, 0, -9.81)

# 等待一段时间让模型加载完毕
for _ in range(100):
    p.stepSimulation()
    time.sleep(0.01)

# 定义状态空间，假设机械臂有7个关节
space = ob.RealVectorStateSpace(6)


# 设置关节角度的边界（根据实际机械臂的关节限制）
bounds = ob.RealVectorBounds(6)
low_bounds = [-3.1416, -1.7453, -1.0472, -3.49, -2.5944, -6.9813]
high_bounds = [3.1416, 1.9199, 1.1345, 3.49, 2.5944, 6.9813]
joint_ranges = [upper - lower for upper, lower in zip(low_bounds, high_bounds)]

for i in range(6):
    bounds.setLow(i, low_bounds[i])
    bounds.setHigh(i, high_bounds[i])

space.setBounds(bounds)


# 定义简单的空间信息
si = ob.SpaceInformation(space)

joint_id = [2, 3, 4, 5, 6, 7, 11]
pairs = utils.get_self_link_pairs(robot_id, joint_id)

# 定义有效状态采样器
def isStateValid(state):
    # 在这里添加碰撞检测逻辑
    for link1, link2 in pairs:
        if utils.pairwise_link_collision(1, link1, 1, link2, max_distance=0):
            print("link1:",link1)
            print("link2:",link2)
            return False
    
    
    return True

si.setStateValidityChecker(ob.StateValidityCheckerFn(isStateValid))
si.setup()

for link1, link2 in pairs:
    if utils.pairwise_link_collision(3, link1, 3, link2, max_distance=0):
        print("link1:",link1)
        print("link2:",link2)


# 定义起始状态和目标状态
start = ob.State(space)
start_values = [0, 0, 0, 0, 0, 0, 0]
for i in range(6):
    start[i] = start_values[i]

goal = ob.State(space)

def solve_ik(pose, max_num_iterations=100):
    """Calculate joint configuration with inverse kinematics.   求解目标姿态的逆运动学"""
    # lowerLimits=[-3 * np.pi / 2, -2.3562, -5, -5, -5, -5, -5],
    # upperLimits=[-np.pi / 2, 0, 5, 5, 5, 5],
    # jointRanges=[np.pi, 2.3562, 10, 10, 10, 10],  # * 6,
    joints = p.calculateInverseKinematics(
        bodyUniqueId=robot_id,
        endEffectorLinkIndex=10,
        targetPosition=pose[0],
        targetOrientation=pose[1],  # [0,0,0,1]

        # lowerLimits=[-3 * np.pi / 2, -2.3562, -5, -5, -5, -5, -5],
        # upperLimits=[-np.pi / 2, 0, 5, 5, 5, 5],
        # jointRanges=[np.pi, 2.3562, 10, 10, 10, 10],  # * 6,
        lowerLimits=low_bounds,
        upperLimits=high_bounds,
        jointRanges=joint_ranges,  # * 6,
        
        restPoses=np.float32([0, 0, 0, 0, 0, 0]).tolist(),
        maxNumIterations=max_num_iterations,
        residualThreshold=1e-5)
    joints = np.float32(joints)
    joints[2:] = (joints[2:] + np.pi) % (2 * np.pi) - np.pi
    return joints

# goal_pose = [[ 0.82056177, -0.40000001,  0.11197798],[ 4.32978047e-17,  4.32978047e-17, -7.07106781e-01,  7.07106781e-01]]
goal_pose = [[ 1.2059999704360962, 0.7660000324249268, 0.46299999952316284],[ 0, 0, 0, 1]]
goal_values = solve_ik(goal_pose)   # -0.45257324,  0.919198  ,  0.49332452, -0.01405168,  0.1582048 ,  -0.43844485
# goal_values = [-3.5732487e-01,  9.4936633e-01,  1.3976622e-01, -6.2823296e-04,
#         4.8196554e-01, -3.5663962e-01]

# 确保目标状态在边界内
adjusted_goal_values = []
for i in range(7):
    if goal_values[i] < low_bounds[i]:
        adjusted_goal_values.append(low_bounds[i])
    elif goal_values[i] > high_bounds[i]:
        adjusted_goal_values.append(high_bounds[i])
    else:
        adjusted_goal_values.append(goal_values[i])

# formatted_adjusted_goal_values = ["{:.14e}".format(num) for num in adjusted_goal_values]

adjusted_goal_values = [float(num) for num in adjusted_goal_values]

for i in range(7):
    goal[i] = adjusted_goal_values[i]

# 检查目标状态是否在边界内
for i in range(7):
    if goal[i] < low_bounds[i] or goal[i] > high_bounds[i]:
        raise ValueError(f"Goal state {goal[i]} out of bounds for dimension {i}")

# 设置问题定义
pdef = ob.ProblemDefinition(si)
pdef.setStartAndGoalStates(start, goal)

# 选择规划器
planner = og.RRTConnect(si)
planner.setProblemDefinition(pdef)
planner.setRange(0.1)  # 调整步长

# 执行规划
solved = planner.solve(10.0)  # 设置最大规划时间为1秒

if solved:
    print("Found solution:")
    path = pdef.getSolutionPath()
    print(path.printAsMatrix())

    # 获取路径中的状态
    path_states = path.getStates()
    # joint_indices = [0, 1, 2, 3, 4, 5]  # 机械臂的关节索引
    joint_indices = [2, 3, 4, 5, 6, 7, 11]  # 机械臂的关节索引

    for state in path_states:
        target_positions = [state[i] for i in range(6)]
        for i in range(7):
            p.setJointMotorControl2(robot_id, jointIndex=joint_indices[i], controlMode=p.POSITION_CONTROL, targetPosition=target_positions[i])
        for _ in range(100):
            p.stepSimulation()
            time.sleep(0.01)
else:
    print("No solution found")