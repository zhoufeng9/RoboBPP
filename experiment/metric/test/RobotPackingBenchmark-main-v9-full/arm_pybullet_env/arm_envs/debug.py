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

# 连接到 PyBullet 仿真
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
plane_id = p.loadURDF("plane.urdf")
robot_id = p.loadURDF("franka_panda/panda.urdf")
p.setGravity(0, 0, -9.81)

# 等待一段时间让模型加载完毕
for _ in range(100):
    p.stepSimulation()
    time.sleep(0.01)

# 定义状态空间，假设机械臂有7个关节
space = ob.RealVectorStateSpace(7)

# 设置关节角度的边界（根据实际机械臂的关节限制）
bounds = ob.RealVectorBounds(7)
low_bounds = [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]
high_bounds = [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]

for i in range(7):
    bounds.setLow(i, low_bounds[i])
    bounds.setHigh(i, high_bounds[i])

space.setBounds(bounds)


# 定义简单的空间信息
si = ob.SpaceInformation(space)

# 定义有效状态采样器
def isStateValid(state):
    # 在这里添加碰撞检测逻辑
    return True

si.setStateValidityChecker(ob.StateValidityCheckerFn(isStateValid))
si.setup()

# 定义起始状态和目标状态
start = ob.State(space)
start_values = [0, -0.5, 0, -1.5, 0, 1.5, 0]
for i in range(7):
    start[i] = start_values[i]

goal = ob.State(space)
goal_values = [0, 0.5, 0, 1.5, 0, -1.5, 0]

# 确保目标状态在边界内
adjusted_goal_values = []
for i in range(7):
    if goal_values[i] < low_bounds[i]:
        adjusted_goal_values.append(low_bounds[i])
    elif goal_values[i] > high_bounds[i]:
        adjusted_goal_values.append(high_bounds[i])
    else:
        adjusted_goal_values.append(goal_values[i])

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

# 执行规划
solved = planner.solve(1.0)  # 设置最大规划时间为1秒

if solved:
    print("Found solution:")
    path = pdef.getSolutionPath()
    print(path.printAsMatrix())

    # 获取路径中的状态
    path_states = path.getStates()
    joint_indices = [0, 1, 2, 3, 4, 5, 6]  # 机械臂的关节索引

    for state in path_states:
        target_positions = [state[i] for i in range(7)]
        for i in range(7):
            p.setJointMotorControl2(robot_id, jointIndex=joint_indices[i], controlMode=p.POSITION_CONTROL, targetPosition=target_positions[i])
        for _ in range(100):
            p.stepSimulation()
            time.sleep(0.01)
else:
    print("No solution found")