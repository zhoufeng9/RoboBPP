import math
import os
import random

import torch
import pandas as pd
import numpy as np

# time_series
'''
file_path = 'data/time_series/pg.xlsx'
df = pd.read_excel(file_path, engine='openpyxl', index_col=None)
# Extract the relevant columns
sizes = df[['Length_cm', 'Width_cm', 'Height_cm']].to_numpy() / 100  # 将单位由cm转换为m
# block_unit = 0.01
# pack_init_sizes = np.ceil(np.array(sizes) / block_unit).astype(np.int).tolist()  # np.round：四舍五入函数  ceil向上取整
# 使用列表推导式过滤掉包含0的数组
pack_init_sizes = [list(box) for box in sizes if not np.any(box == 0)]

for i in range(30):
    # 随机选择一个起始位置，确保有足够的条目进行提取
    start_index = random.randint(0, len(pack_init_sizes) - 80)
    # 从随机位置提取连续的80条数据
    data = pack_init_sizes[start_index:start_index + 80]
    # 将数据保存到字典中
    data = np.array(data)
    data_dict = {'data': data}
    # 保存字典到 `.pt` 文件
    torch.save(data_dict,'/media/wzf/goodluck/Workspace/rl/data/time_series/data_time_series_' + str(i) + '.pt')
    print('save to ' + '/media/wzf/goodluck/Workspace/rl/data/time_series/data_time_series_' + str(i) + '.pt!')
'''


# occupancy
'''
file_path = 'data/occupancy/deli.xlsx'
df = pd.read_excel(file_path, sheet_name='道口物料清单', engine='openpyxl', index_col=None)
# Extract the relevant columns
sizes = df[['尺寸（L）', '尺寸（W）', '尺寸（H）']].to_numpy() / 100  # 将单位由cm转换为m
init_occupancy = df['占比'].to_numpy()
max_occupancy = max(init_occupancy)
normalized_occupancy = [o / max_occupancy for o in init_occupancy]
# block_unit = 0.01
# init_sizes = np.ceil(np.array(sizes) / block_unit).astype(np.int)
filtered_box_trajs, filtered_occupancy = [], []
for box, occ in zip(sizes, normalized_occupancy):
    if not np.any(box == 0):  # 只有在 box 中不含 0 的情况下才保留
        filtered_box_trajs.append(list(box))
        filtered_occupancy.append(occ)
pack_init_sizes = filtered_box_trajs
occupancy = filtered_occupancy

for i in range(30):
    data = random.choices(pack_init_sizes, weights=occupancy, k=80)
    for box in data:
        index = next(i for i, traj in enumerate(pack_init_sizes) if np.array_equal(traj, box))  # 找到选中值在 box_trajs 中的索引
        weight = occupancy[index]  # 获取对应的权重
        print(f"选中的数据: {box}, 对应的权重: {weight}")

    data = np.array(data)
    # 将数据保存到字典中
    data_dict = {'data': data}
    # 保存字典到 `.pt` 文件
    torch.save(data_dict, '/media/wzf/goodluck/Workspace/rl/data/occupancy/data_occupancy_' + str(i) + '.pt')
    print('save to ' + '/media/wzf/goodluck/Workspace/rl/data/occupancy/data_occupancy_' + str(i) + '.pt!')
'''


# flat_long
# '''
data = []                   # 'data/flat_long/opai.txt'
with open('data/flat_long/opai.txt', 'r') as file:
    for line in file:
        line_data = list(map(float, line.split()))
        if any(val > 250 for val in line_data):  # 检查是否有数据大于 200
            continue
        processed_data = list(val / 100 for val in line_data)  # 对每个数据进行处理并向上取整
        # if processed_data not in data:
        data.append(processed_data)

sizes = np.array(data)
# block_unit = 0.01
# pack_init_sizes = np.ceil(np.array(sizes) / block_unit).astype(np.int)
# 使用列表推导式过滤掉包含0的数组
pack_init_sizes = [list(box) for box in sizes if not np.any(box == 0)]

# 随机读取并保存数据
for i in range(30):
    data = random.sample(pack_init_sizes, 80)
    data = np.array(data)
    # 将数据保存到字典中
    data_dict = {'data': data}
    # 保存字典到 `.pt` 文件
    torch.save(data_dict, '/media/wzf/goodluck/Workspace/rl/data/flat_long/data_flat_long_' + str(i) + '.pt')
    print('save to ' + '/media/wzf/goodluck/Workspace/rl/data/flat_long/data_flat_long_' + str(i) + '.pt!')
# '''

