import random

import numpy as np
import copy

import pandas as pd
import torch

class BoxCreator(object):
    def __init__(self):
        self.box_list = []

    def reset(self):
        self.box_list.clear()

    def generate_box_size(self, **kwargs):
        pass

    def preview(self, length):
        while len(self.box_list) < length:
            self.generate_box_size()
        return copy.deepcopy(self.box_list[:length])

    def drop_box(self, idx=0):
        assert len(self.box_list) >= 0
        self.box_list.pop(idx)

class RandomBoxCreator(BoxCreator):
    default_box_set = []
    for i in range(5):
        for j in range(5):
            for k in range(5):
                default_box_set.append((2+i, 2+j, 2+k))

    def __init__(self, box_size_set=None, item_prob = None):
        super().__init__()
        self.box_set = box_size_set
        self.item_prob = item_prob
        if self.box_set is None:
            self.box_set = RandomBoxCreator.default_box_set

    def generate_box_size(self, **kwargs):
        if self.item_prob is not None:
            idx = np.random.choice(len(self.box_set), p=self.item_prob)
            self.box_list.append(self.box_set[idx].tolist())
        else:
            idx = np.random.randint(0, len(self.box_set))
            self.box_list.append(self.box_set[idx])

class LoadBoxCreator(BoxCreator):
    def __init__(self, data_name=None):
        super().__init__()
        self.data_name = data_name
        self.data_type = data_name.split('/')[1].split('/', 1)[0]

        self.index = 0
        self.box_index = 0
        self.occupancy = None

        if self.data_type == 'time_series':
            file_path = self.data_name  # 'data/time_series/pg.xlsx'
            df = pd.read_excel(file_path, engine='openpyxl', index_col=None)
            # Extract the relevant columns
            sizes = df[['Length_cm', 'Width_cm', 'Height_cm']].to_numpy() / 100  # 将单位由cm转换为m
            block_unit = 0.01
            pack_init_sizes = np.ceil(np.array(sizes) / block_unit).astype(int).tolist()  # np.round：四舍五入函数  ceil向上取整
            # 使用列表推导式过滤掉包含0的数组
            pack_init_sizes = [box for box in pack_init_sizes if not np.any(box == 0)]

        elif self.data_type == 'occupancy':
            file_path = self.data_name  # 'data/occupancy/deli.xlsx'
            df = pd.read_excel(file_path, sheet_name='道口物料清单', engine='openpyxl', index_col=None)
            # Extract the relevant columns
            sizes = df[['尺寸（L）', '尺寸（W）', '尺寸（H）']].to_numpy() / 100  # 将单位由cm转换为m
            init_occupancy = df['占比'].to_numpy()
            max_occupancy = max(init_occupancy)
            normalized_occupancy = [o / max_occupancy for o in init_occupancy]

            block_unit = 0.01
            init_sizes = np.ceil(np.array(sizes) / block_unit).astype(int)

            filtered_box_trajs, filtered_occupancy, filtered_size = [], [], []
            for box, occ, size in zip(init_sizes, normalized_occupancy, sizes):
                if not np.any(box == 0):  # 只有在 box 中不含 0 的情况下才保留
                    filtered_box_trajs.append(box)
                    filtered_occupancy.append(occ)
                    filtered_size.append(size)
            pack_init_sizes = filtered_box_trajs
            self.occupancy = filtered_occupancy
            sizes = filtered_size


        elif self.data_type == 'flat_long':
            data = []                   # 'data/flat_long/opai.txt'
            with open(self.data_name, 'r') as file:
                for line in file:
                    line_data = list(map(float, line.split()))
                    # if any(val > 200 for val in line_data):  # 检查是否有数据大于 200
                    if line_data[0] > 249 or line_data[1] > 119:  # 检查是否有数据大于 200
                        # print(line_data)
                        continue
                    # processed_data = list(val / 200 for val in line_data)  # 对每个数据进行处理并向上取整
                    processed_data = list(val / 100 for val in line_data)  # 对每个数据进行处理并向上取整
                    # if processed_data not in data:
                    data.append(processed_data)

            sizes = np.array(data)
            block_unit = 0.01
            pack_init_sizes = np.ceil(np.array(sizes) / block_unit).astype(int)
            # 使用列表推导式过滤掉包含0的数组
            pack_init_sizes = [box for box in pack_init_sizes if not np.any(box == 0)]
            sizes = [box for box in sizes if not np.any(box == 0)]


        print("load data set successfully! data type:", self.data_type)
        self.traj_nums = len(pack_init_sizes)
        self.box_trajs = pack_init_sizes
        self.init_box_trajs = sizes

        # self.traj_nums = len(torch.load(self.data_name))
        # self.box_trajs = torch.load(self.data_name)

    def reset(self, index=None):
        self.box_list.clear()
        self.recorder = []
        if index is None:
            # self.index += 1
            self.index = 'data'
        else:
            self.index = index
        # self.boxes = np.array(self.box_trajs[self.index])
        # self.boxes = self.boxes.tolist()

        if self.data_type == 'time_series':
            # 随机选择一个起始位置，确保有足够的条目进行提取
            start_index = random.randint(0, len(self.box_trajs) - 150)
            # 从随机位置提取连续的150条数据
            self.boxes = self.box_trajs[start_index:start_index + 150]
            self.init_sizes = self.init_box_trajs[start_index:start_index + 150]

        elif self.data_type == 'occupancy':
            # Step 1: 使用相同的随机索引来抽样
            indices = random.choices(range(len(self.box_trajs)), weights=self.occupancy, k=150)
            # Step 2: 根据这些索引从 box_trajs 和 init_box_trajs 中抽取对应的元素
            self.boxes = [self.box_trajs[i] for i in indices]
            self.init_sizes = [self.init_box_trajs[i] for i in indices]

        elif self.data_type == 'flat_long':
            # Step 1: 从索引范围中抽取 150 个唯一索引
            indices = random.sample(range(len(self.box_trajs)), 150)
            # Step 2: 使用这些索引同步抽取 self.box_trajs 和 self.init_box_trajs 的元素
            self.boxes = [self.box_trajs[i] for i in indices]
            self.init_sizes = [self.init_box_trajs[i] for i in indices]



        # self.boxes = self.box_trajs
        self.box_index = 0
        self.box_set = self.boxes
        self.box_set.append([100, 100, 100])
        self.init_sizes = np.vstack((self.init_sizes, np.array([100, 100, 100])))

    def generate_box_size(self, **kwargs):
        if self.box_index < len(self.box_set):
            self.box_list.append(self.box_set[self.box_index])
            self.recorder.append(self.box_set[self.box_index])
            self.box_index += 1
        else:
            self.box_list.append((10, 10, 10))
            self.recorder.append((10, 10, 10))
            self.box_index += 1