import numpy as np
import copy
import torch

import pandas as pd
import random
import time

class BoxCreator(object):
    def __init__(self):
        self.box_list = []

    def reset(self):
        self.box_list.clear()

    def generate_box(self, **kwargs):
        pass

    def preview(self, box_num=1):
        while len(self.box_list) < box_num:
            self.generate_box()
        return copy.deepcopy(self.box_list[:box_num])

    def drop_box(self, index=0):
        assert len(self.box_list) >= 0
        self.box_list.pop(index)


class RandomSeqCreator(BoxCreator):
    default_box_set = []
    default_max_size = {'width': 5, 'length': 5, 'height': 5}
    for i in range(default_max_size['width']):
        for j in range(default_max_size['length']):
            for k in range(default_max_size['height']):
                default_box_set.append((2+i, 2+j, 2+k))

    def __init__(self, box_set=None, setting=1):
        super(RandomSeqCreator, self).__init__()
        self.box_set = box_set
        if self.box_set is None:
            self.box_set = RandomSeqCreator.default_box_set

        if setting == 3:
            self.box_density = lambda: -np.random.random((150, 1)) + 1
        else:
            self.box_density = lambda: np.ones((150, 1), dtype=np.float)

    def reset(self):
        self.box_list.clear()
        idx = np.random.choice(np.arange(len(self.box_set)), size=150, replace=True)
        self.boxes = np.concatenate((np.array(self.box_set)[idx, ...], self.box_density()), axis=-1)
        self.box_idx = 0

    def generate_box(self, **kwargs):
        self.box_list.append(self.boxes[self.box_idx])
        self.box_idx += 1



class LoadBoxCreator(BoxCreator):
    def __init__(self, box_set=None):
        super(LoadBoxCreator, self).__init__()
        self.box_set = box_set
        self.box_density = lambda: np.ones((80, 1), dtype=np.float)

    def reset(self):
        self.box_list.clear()

        self.box_idx = 0
        self.boxes = np.concatenate((np.array(self.box_set), self.box_density()), axis=-1)

    def generate_box(self, **kwargs):
        if self.box_idx < 80:
            self.box_list.append(self.boxes[self.box_idx])
        else:
            self.box_list.append(np.array([100., 100., 100., 1.]))
        self.box_idx += 1

