import numpy as np
import pandas as pd
import os
from irb6640_pybullet_env.utils import pybullet_utils


sizes_oral = np.load( "dataset/10_[7]_[1_4]_003_mess_real/train/0/sizes.npy", allow_pickle=True )
# (10, 3)
data_type = 'random'

if data_type == 'random':
    file_path = 'dataset/10_[7]_[1_4]_003_mess_real/train/0/poses.npy'
    poses_mat = np.load( file_path, allow_pickle=True )[0]
    mat = pybullet_utils.mat_to_pose(poses_mat)

elif data_type == 'Manufacturer':
    file_path = 'packing_dataset/得力选品desk.xlsx'
    df = pd.read_excel(file_path, sheet_name='选品', engine='openpyxl')
    # Extract the relevant columns
    sizes = df[['尺寸（L）', '尺寸（W）', '尺寸（H）']].to_numpy()
    info = df[['商品code', '名称']].to_numpy()

elif data_type == 'flat_long':
    folder_path = 'packing_dataset/欧派家居opai'
    # 初始化一个集合来存储去重后的数据
    unique_data = set()
    # 读取文件夹内的每个txt文件
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.txt'):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'r') as file:
                for line in file:
                    # 将每一行数据转换为元组，并加入集合进行去重
                    data_tuple = tuple(map(float, line.strip().split()))
                    unique_data.add(data_tuple)
    # 将集合转换为numpy数组
    sizes = np.array(list(unique_data))
    # 创建info数组，包含从1开始的连续整数
    info = np.arange(1, sizes.shape[0] + 1).reshape(-1, 1)


elif data_type == 'time_sequence':
    file_path = 'packing_dataset/宝洁-Simulation+data.xlsx'
    df = pd.read_excel(file_path, engine='openpyxl')
    # Extract the relevant columns
    sizes = df[['Length_cm', 'Width_cm', 'Height_cm']].to_numpy()
    info = df[['Material', 'Case_pick_start_time']].to_numpy()

print("here")