import numpy as np
import copy
import matplotlib.pyplot as plt
import os
from . import ems_tools as ET


# ===============================

class Container(object):
    def __init__(self, container_size):
        '''
        Args:
            container_size: List(int) [3]
        '''

        self.current_box_num = 0

        self.container_size = container_size

        self.start_to_stop = False

        self.reset_container()

    def reset_container(self):
        self.boxes = []
        self.positions = []
        self.heightmap = np.zeros(self.container_size[:-1], dtype=np.float32)
        self.idmap = np.zeros(self.container_size[:-1], dtype=np.int32)
        
        self.empty_max_spaces = [[[0,0,0], list(self.container_size), [1]*3]]
        self.start_to_stop = False
        
        self.current_box_num = 0

    def add_new_box(self, box_size, box_pos):
        '''
        Add a box into current container
        ---
        params:
        ---
            box_size: float * 3 array
            box_pos: int * 3 array
        '''
        
        ems_xy = box_pos[:2]
        
        bx, by, bz = box_size
        bx = int(bx)
        by = int(by)

        heightmap_max = 50000

        while heightmap_max > self.container_size[-1] and self.start_to_stop == False:
            
            heightmap = self.heightmap.copy()
            id_map = self.idmap.copy()


            lx = int(ems_xy[0])
            ly = int(ems_xy[1])
            lz = heightmap[lx:lx+bx, ly:ly+by].max()
                        
            final_z = bz + lz
            
            heightmap[ lx:lx+bx, ly:ly+by ] = final_z            
            id_map[ lx:lx+bx, ly:ly+by ] = self.current_box_num + 1

            pos = np.array([lx, ly, lz])
            
            heightmap_max = heightmap.max()
            
            stop_current_pack = False

            if heightmap_max > self.container_size[-1]:
                stop_current_pack = True

            if stop_current_pack:
                self.start_to_stop = True
                pos = None
                break
        
        if self.start_to_stop == True:
            return None
        
         
        if pos is not None:
            self.boxes.append(box_size)
            self.positions.append(pos)

            self.heightmap = heightmap
            self.idmap = id_map
        
        self.update_ems()

        return pos

    def update_ems(self):
        self.empty_max_spaces = ET.compute_ems(self.idmap, self.heightmap, self.container_size)


    def get_ems(self, ems_dim=7, for_test=False, max_ems_num=None):

        ems = self.empty_max_spaces
        ems = np.array(ems).reshape(-1, 9)[:, :ems_dim]

        # pos | right_top_pos
        left_botton_corner = ems[:, :3]
        right_top_corner = ems[:, 3:6]
        ems[:, 3:6] = right_top_corner - left_botton_corner

        origin_ems = copy.deepcopy(ems)
        
        # pos | size
        if len(ems) > 0:
            size_min_height = ems[:, 5].min()
            ems[:, 5] = ems[:, 5] - size_min_height

        ems_len = len(ems)
        
        if for_test or max_ems_num is None:
            max_ems_num = ems_len
        else:
            max_ems_num = max_ems_num
        # max_ems_num = self.max_ems_num

        # ems_per_box = 5
        all_ems = np.zeros(( max_ems_num, ems_dim), dtype=np.float32 )
        origin_all_ems = np.zeros(( max_ems_num, ems_dim), dtype=np.float32 )
        # all_ems = np.zeros(( ems_len, 6), dtype=np.float32 )
        all_ems[:ems_len] = ems
        origin_all_ems[:ems_len] = origin_ems

        ems_masks = np.zeros(( max_ems_num ), dtype=bool)
        ems_masks[:ems_len] = 1

        return origin_all_ems, all_ems, ems_masks


