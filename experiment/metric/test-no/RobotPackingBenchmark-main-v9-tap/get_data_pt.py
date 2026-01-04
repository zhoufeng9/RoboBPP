import math
import os

import torch
import pandas as pd
import numpy as np

# time_series
'''
file_path = '/home/wzf/Workspace/rl/RobotPackingBenchmark-main-v5/packing_dataset/宝洁-Simulation+data.xlsx'
df = pd.read_excel(file_path, engine='openpyxl', index_col=None)
# Extract the relevant columns
sizes = df[['Length_cm', 'Width_cm', 'Height_cm']].to_numpy() / 100    # 将单位由cm转换为m
# block_unit = 0.01
# pack_init_sizes = np.ceil(np.array(sizes) / block_unit).astype(np.int)
for i in range(0, len(sizes), 100):
    if i/100 == 30:
        break
    data = sizes[i:i+100]
    # 将数据保存到字典中
    data_dict = {'data': data}
    # 保存字典到 `.pt` 文件
    torch.save(data_dict, '/media/wzf/goodluck/Workspace/rl/data/time_series/data_time_series_' + str(int(i/100)) + '.pt')
    print('save to ' + '/media/wzf/goodluck/Workspace/rl/data/time_series/data_time_series_' + str(int(i/100)) + '.pt!')
'''


# occupancy
'''
file_path = '/home/wzf/Workspace/rl/RobotPackingBenchmark-main-v5/packing_dataset/得力选品desk.xlsx'
df = pd.read_excel(file_path, sheet_name='道口物料清单', engine='openpyxl', index_col=None)
# Extract the relevant columns
sizes = df[['尺寸（L）', '尺寸（W）', '尺寸（H）']].to_numpy() / 100    # 将单位由cm转换为m
# block_unit = 0.01
# pack_init_sizes = np.ceil(np.array(sizes) / block_unit).astype(np.int)
for i in range(30):
    random_indices = np.random.choice(len(sizes), size=100, replace=False)  # 随机选择 100 个索引
    random_chunk = sizes[random_indices]  # 根据随机索引获取数据

    # 将数据保存到字典中
    data_dict = {'data': random_chunk}
    # 保存字典到 `.pt` 文件
    torch.save(data_dict, '/media/wzf/goodluck/Workspace/rl/data/occupancy/data_occupancy_' + str(i) + '.pt')
    print('save to ' + '/media/wzf/goodluck/Workspace/rl/data/occupancy/data_occupancy_' + str(i) + '.pt!')
'''


# flat_long
# '''
# 读取文件并去重数据
data = []
with open('/home/wzf/Workspace/rl/RobotPackingBenchmark-main-v5/packing_dataset/欧派家居opai/opai.txt', 'r') as file:
    for line in file:
        line_data = tuple(map(float, line.split()))
        if any(val > 200 for val in line_data):  # 检查是否有数据大于 200
            continue
        processed_data = tuple(val / 200 for val in line_data)  # 对每个数据进行处理并向上取整
        if processed_data not in data:
            data.append(processed_data)

# 随机读取并保存数据
for i in range(30):
    random_indices = np.random.choice(len(data), size=100, replace=False)  # 随机选择 100 行数据
    random_chunk = [data[idx] for idx in random_indices]  # 根据随机索引获取数据
    # 将数据保存到字典中
    data_dict = {'data': random_chunk}
    # 保存字典到 `.pt` 文件
    torch.save(data_dict, '/media/wzf/goodluck/Workspace/rl/data/flat_long/data_flat_long_' + str(i) + '.pt')
    print('save to ' + '/media/wzf/goodluck/Workspace/rl/data/flat_long/data_flat_long_' + str(i) + '.pt!')
# '''


# np.array(data['data'])



# backup
'''
def load_data(self, data, args):
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
        file_path = args.Data.Time_series.data_path
        df = pd.read_excel(file_path, engine='openpyxl', index_col=None)
        # Extract the relevant columns
        sizes = df[['Length_cm', 'Width_cm', 'Height_cm']].to_numpy() / 100  # 将单位由cm转换为m
        # sizes = sizes / 2   # 此处缩小四倍方便packing验证
        poses_mat = np.tile(pose_mat, (sizes.shape[0], 1, 1))
        info = df[['Material', 'Case_pick_start_time']].to_numpy()

    elif data == 'occupancy':
        file_path = args.Data.Occupancy.data_path
        df = pd.read_excel(file_path, sheet_name='道口物料清单', engine='openpyxl', index_col=None)
        # Extract the relevant columns
        sizes = df[['尺寸（L）', '尺寸（W）', '尺寸（H）']].to_numpy() / 100  # 将单位由cm转换为m
        # sizes = sizes / 3   # 此处缩小四倍方便packing验证
        poses_mat = np.tile(pose_mat, (sizes.shape[0], 1, 1))
        info = df[['商品code', '名称']].to_numpy()

    elif data == 'flat_long':
        folder_path = args.Data.Flat_long.data_path
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
        sizes = sizes / 2  # 此处缩小四倍方便packing验证
        poses_mat = np.tile(pose_mat, (sizes.shape[0], 1, 1))
        # 创建info数组，包含从1开始的连续整数
        info = np.arange(1, sizes.shape[0] + 1).reshape(-1, 1)

    return len(sizes), sizes, poses_mat, info
'''
