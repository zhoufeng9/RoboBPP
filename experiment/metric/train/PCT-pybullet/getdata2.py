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
block_unit = 0.01
pack_init_sizes = np.ceil(np.array(sizes) / block_unit).astype(int)
for i in range(0, len(pack_init_sizes), 100):
    if i/100 == 30:
        break
    data = pack_init_sizes[i:i+100]
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
block_unit = 0.01
pack_init_sizes = np.ceil(np.array(sizes) / block_unit).astype(int)
for i in range(30):
    random_indices = np.random.choice(len(pack_init_sizes), size=100, replace=False)  # 随机选择 100 个索引
    random_chunk = pack_init_sizes[random_indices]  # 根据随机索引获取数据

    # 将数据保存到字典中
    data_dict = {'data': random_chunk}
    # 保存字典到 `.pt` 文件
    torch.save(data_dict, '/media/wzf/goodluck/Workspace/rl/data/occupancy/data_occupancy_' + str(i) + '.pt')
    print('save to ' + '/media/wzf/goodluck/Workspace/rl/data/occupancy/data_occupancy_' + str(i) + '.pt!')
'''


# flat_long
'''
# 读取文件并去重数据
data = []
with open('/home/wzf/Workspace/rl/RobotPackingBenchmark-main-v5/packing_dataset/欧派家居opai/opai.txt', 'r') as file:
    for line in file:
        line_data = tuple(map(float, line.split()))
        if any(val > 200 for val in line_data):  # 检查是否有数据大于 200
            continue
        processed_data = tuple(math.ceil(val / (200 * 0.01)) for val in line_data)  # 对每个数据进行处理并向上取整
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
'''


# np.array(data['data'])


