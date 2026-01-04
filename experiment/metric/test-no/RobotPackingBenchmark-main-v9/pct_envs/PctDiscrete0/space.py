import os

import numpy as np
from functools import reduce
import copy

import trimesh
from .convex_hull import ConvexHull, point_in_polygen
from .PctTools import AddNewEMSZ, maintainEventBottom, smallBox, extreme2D, corners2D
import pybullet as p
from concurrent.futures import ThreadPoolExecutor, as_completed
from packer.irrgular_tools import shot_item, gen_ray_origin_direction

class Stack(object):
    def __init__(self, centre, mass):
        self.centre = centre
        self.mass = mass

class DownEdge(object):
    def __init__(self, box):
        self.box = box
        self.area = None
        self.centre2D = None

def IsUsableEMS(xlow, ylow, zlow, x1, y1, z1, x2, y2, z2):
    xd = x2 - x1
    yd = y2 - y1
    zd = z2 - z1
    if ((xd >= xlow) and (yd >= ylow) and (zd >= zlow)):
        return True
    return False

class Box(object):
    def __init__(self, x, y, z, lx, ly, lz, density, virtual=False):
        self.x = x
        self.y = y
        self.z = z
        self.lx = lx
        self.ly = ly
        self.lz = lz

        self.centre = np.array([self.lx + self.x / 2, self.ly + self.y / 2, self.lz + self.z / 2])
        self.vertex_low = np.array([self.lx, self.ly, self.lz])
        self.vertex_high = np.array([self.lx + self.x, self.ly + self.y, self.lz + self.z])
        self.mass = x * y * z * density
        if virtual: self.mass *= 1.0
        self.bottom_edges = []
        self.bottom_whole_contact_area = None

        self.up_edges = {}
        self.up_virtual_edges = {}

        self.thisStack = Stack(self.centre, self.mass)
        self.thisVirtualStack = Stack(self.centre, self.mass)
        self.involved = False


    def calculate_new_com(self, virtual=False):
        new_stack_centre = self.centre * self.mass
        new_stack_mass = self.mass

        for ue in self.up_edges.keys():
            if not ue.involved:
                new_stack_centre += self.up_edges[ue].centre * self.up_edges[ue].mass
                new_stack_mass += self.up_edges[ue].mass

        for ue in self.up_virtual_edges.keys():
            if ue.involved:
                new_stack_centre += self.up_virtual_edges[ue].centre * self.up_virtual_edges[ue].mass
                new_stack_mass += self.up_virtual_edges[ue].mass

        new_stack_centre /= new_stack_mass
        if virtual:
            self.thisVirtualStack.mass = new_stack_mass
            self.thisVirtualStack.centre = new_stack_centre
        else:
            self.thisStack.mass = new_stack_mass
            self.thisStack.centre = new_stack_centre

    def calculated_impact(self):
        if len(self.bottom_edges) == 0:
            return True
        elif not point_in_polygen(self.thisStack.centre[0:2],
                                  self.bottom_whole_contact_area):
            return False
        else:
            if len(self.bottom_edges) == 1:
                stack = self.thisStack
                self.bottom_edges[0].box.up_edges[self] = stack
                self.bottom_edges[0].box.calculate_new_com()
                if not self.bottom_edges[0].box.calculated_impact():
                    return False
            else:
                direct_edge = None
                for e in self.bottom_edges:
                    if self.thisStack.centre[0] > e.area[0] and self.thisStack.centre[0] < e.area[2] \
                            and self.thisStack.centre[1] > e.area[1] and self.thisStack.centre[1] < e.area[3]:
                        direct_edge = e
                        break

                if direct_edge is not None:
                    for edge in self.bottom_edges:
                        if edge == direct_edge:
                            edge.box.up_edges[self] = self.thisStack
                            edge.box.calculate_new_com()
                        else:
                            edge.box.up_edges[self] = Stack(self.thisStack.centre, 0)
                            edge.box.calculate_new_com()

                    for edge in self.bottom_edges:
                        if not edge.box.calculated_impact():
                            return False

                elif len(self.bottom_edges) == 2:
                    com2D = self.thisStack.centre[0:2]

                    tri_base_line = self.bottom_edges[0].centre2D - self.bottom_edges[1].centre2D
                    tri_base_len = np.linalg.norm(tri_base_line)
                    tri_base_line /= tri_base_len ** 2

                    ratio0 = abs(np.dot(com2D - self.bottom_edges[1].centre2D, tri_base_line))
                    ratio1 = abs(np.dot(com2D - self.bottom_edges[0].centre2D, tri_base_line))

                    com0 = np.array([*self.bottom_edges[0].centre2D, self.thisStack.centre[2]])
                    com1 = np.array([*self.bottom_edges[1].centre2D, self.thisStack.centre[2]])

                    stack0 = Stack(com0, self.thisStack.mass * ratio0)
                    stack1 = Stack(com1, self.thisStack.mass * ratio1)

                    self.bottom_edges[0].box.up_edges[self] = stack0
                    self.bottom_edges[0].box.calculate_new_com()

                    self.bottom_edges[1].box.up_edges[self] = stack1
                    self.bottom_edges[1].box.calculate_new_com()

                    if not self.bottom_edges[0].box.calculated_impact():
                        return False
                    if not self.bottom_edges[1].box.calculated_impact():
                        return False

                else:
                    com2D = self.thisStack.centre[0:2]
                    length = len(self.bottom_edges)
                    coefficient = np.zeros((int(length * (length - 1) / 2 + 1), length))
                    value = np.zeros((int(length * (length - 1) / 2 + 1), 1))
                    counter = 0
                    for i in range(length - 1):
                        for j in range(i + 1, length):
                            tri_base_line = self.bottom_edges[i].centre2D - self.bottom_edges[j].centre2D
                            molecular = np.dot(com2D - self.bottom_edges[i].centre2D, tri_base_line)
                            if molecular != 0:
                                ratioI2J = abs(np.dot(com2D - self.bottom_edges[j].centre2D, tri_base_line)) / molecular
                                coefficient[counter, i] = 1
                                coefficient[counter, j] = - ratioI2J
                            counter += 1

                    coefficient[-1, :] = 1
                    value[-1, 0] = 1
                    assgin_ratio = np.linalg.lstsq(coefficient, value, rcond=None)[0]

                    for i in range(length):
                        e = self.bottom_edges[i]
                        newAdded_mass = self.thisStack.mass * assgin_ratio[i][0]
                        newAdded_com = np.array([*e.centre2D, self.thisStack.centre[2]])
                        e.box.up_edges[self] = Stack(newAdded_com, newAdded_mass)
                        e.box.calculate_new_com()

                    for e in self.bottom_edges:
                        if not e.box.calculated_impact():
                            return False
            return True

    def calculated_impact_virtual(self, first=False):
        self.involved = True
        if len(self.bottom_edges) == 0:
            self.involved = False
            return True
        elif not point_in_polygen(self.thisVirtualStack.centre[0:2], self.bottom_whole_contact_area):
            self.involved = False
            return False
        else:
            if len(self.bottom_edges) == 1:
                stack = self.thisVirtualStack
                self.bottom_edges[0].box.up_virtual_edges[self] = stack
                self.bottom_edges[0].box.calculate_new_com(True)
                if not self.bottom_edges[0].box.calculated_impact_virtual():
                    self.involved = False
                    return False
            else:
                direct_edge = None
                for e in self.bottom_edges:
                    if self.thisVirtualStack.centre[0] > e.area[0] and self.thisVirtualStack.centre[0] < e.area[2] \
                            and self.thisVirtualStack.centre[1] > e.area[1] and self.thisVirtualStack.centre[1] < e.area[3]:
                        direct_edge = e
                        break

                if direct_edge is not None:
                    for edge in self.bottom_edges:
                        if edge == direct_edge:
                            edge.box.up_virtual_edges[self] = self.thisVirtualStack
                            edge.box.calculate_new_com(True)
                        else:
                            edge.box.up_virtual_edges[self] = Stack(self.centre, 0)
                            edge.box.calculate_new_com(True)

                    for edge in self.bottom_edges:
                        if not edge.box.calculated_impact_virtual():
                            self.involved = False
                            return False

                elif len(self.bottom_edges) == 2:
                    com2D = self.thisVirtualStack.centre[0:2]

                    tri_base_line = self.bottom_edges[0].centre2D - self.bottom_edges[1].centre2D
                    tri_base_len = np.linalg.norm(tri_base_line)
                    tri_base_line /= tri_base_len ** 2

                    ratio0 = abs(np.dot(com2D - self.bottom_edges[1].centre2D, tri_base_line))
                    ratio1 = abs(np.dot(com2D - self.bottom_edges[0].centre2D, tri_base_line))

                    com0 = np.array([*self.bottom_edges[0].centre2D, self.thisVirtualStack.centre[2]])
                    com1 = np.array([*self.bottom_edges[1].centre2D, self.thisVirtualStack.centre[2]])

                    stack0 = Stack(com0, self.thisVirtualStack.mass * ratio0)
                    stack1 = Stack(com1, self.thisVirtualStack.mass * ratio1)

                    self.bottom_edges[0].box.up_virtual_edges[self] = stack0
                    self.bottom_edges[0].box.calculate_new_com(True)
                    self.bottom_edges[1].box.up_virtual_edges[self] = stack1
                    self.bottom_edges[1].box.calculate_new_com(True)

                    if not self.bottom_edges[0].box.calculated_impact_virtual() \
                            or not self.bottom_edges[1].box.calculated_impact_virtual():
                        self.involved = False
                        return False

                else:
                    com2D = self.thisVirtualStack.centre[0:2]
                    length = len(self.bottom_edges)
                    coefficient = np.zeros((int(length * (length - 1) / 2 + 1), length))
                    value = np.zeros((int(length * (length - 1) / 2 + 1), 1))
                    counter = 0
                    for i in range(length - 1):
                        for j in range(i + 1, length):
                            tri_base_line = self.bottom_edges[i].centre2D - self.bottom_edges[j].centre2D
                            molecular = np.dot(com2D - self.bottom_edges[i].centre2D, tri_base_line)
                            if molecular != 0:
                                ratioI2J = abs(np.dot(com2D - self.bottom_edges[j].centre2D, tri_base_line)) / molecular
                                coefficient[counter, i] = 1
                                coefficient[counter, j] = -ratioI2J
                            counter += 1

                    coefficient[-1, :] = 1
                    value[-1, 0] = 1
                    x = np.linalg.lstsq(coefficient, value, rcond=None)
                    assgin_ratio = x[0]
                    for i in range(length):
                        e = self.bottom_edges[i]
                        newAdded_mass = self.thisVirtualStack.mass * assgin_ratio[i][0]
                        newAdded_com = np.array([*e.centre2D, self.thisVirtualStack.centre[2]])
                        e.box.up_virtual_edges[self] = Stack(newAdded_com, newAdded_mass)
                        e.box.calculate_new_com(True)

                    for e in self.bottom_edges:
                        if not e.box.calculated_impact_virtual():
                            self.involved = False
                            return False

            if first:
                for e in self.bottom_edges:
                    e.box.up_virtual_edges.pop(self)
            self.involved = False
            return True


# irregular
def gen_ray_origin_direction_real(xRange, yRange, target_origin, block_unit, shift = 0.01):
    block_gap = 0.004
    bottom = np.arange(0, xRange * yRange)      # array([0, 1, 2, 3, 4, 5, 6, 7, 8])
    bottom = bottom.reshape((xRange, yRange))   # array([[0, 1, 2],
                                                #        [3, 4, 5],
                                                #        [6, 7, 8]])

    origin = np.zeros((xRange, yRange, 3))
    origin_up = np.zeros((xRange, yRange, 3))
    origin_down = np.zeros((xRange, yRange, 3))
    origin_left = np.zeros((xRange, yRange, 3))
    origin_right = np.zeros((xRange, yRange, 3))

    # array([[[    0.,     0., -1000.],
    #         [    0.,     1., -1000.],
    #         [    0.,     2., -1000.]],

    #        [[    1.,     0., -1000.],
    #         [    1.,     1., -1000.],
    #         [    1.,     2., -1000.]],

    #        [[    2.,     0., -1000.],
    #         [    2.,     1., -1000.],
    #         [    2.,     2., -1000.]]])
    
    for x in range(xRange):
        for y in range(yRange):
            x_temp = xRange - (x + 1)
            x_temp = x_temp * (block_unit + block_gap)
            x_real = target_origin[0] + x_temp + block_unit * 0.5
            x_real_up = target_origin[0] + x_temp + block_unit * 0.5 - shift
            x_real_down = target_origin[0] + x_temp + block_unit * 0.5 + shift


            y_temp = yRange - (y + 1)
            y_temp = y_temp * (block_unit + block_gap)
            y_real = target_origin[1] + y_temp + block_unit * 0.5 + shift
            y_real_left = target_origin[0] + x_temp + block_unit * 0.5 - shift
            y_real_right = target_origin[0] + x_temp + block_unit * 0.5 + shift
            
            origin[x][y][0] = x_real
            origin[x][y][1] = y_real
            origin[x][y][2] = -10e2

            origin_up[x][y][0] = x_real_up
            origin_up[x][y][1] = y_real
            origin_up[x][y][2] = -10e2

            origin_down[x][y][0] = x_real_down
            origin_down[x][y][1] = y_real
            origin_down[x][y][2] = -10e2

            origin_left[x][y][0] = x_real
            origin_left[x][y][1] = y_real_left
            origin_left[x][y][2] = -10e2

            origin_right[x][y][0] = x_real
            origin_right[x][y][1] = y_real_right
            origin_right[x][y][2] = -10e2
    
    
    
    origin = origin.reshape((xRange, yRange, 3))
    origin_up = origin_up.reshape((xRange, yRange, 3))
    origin_down = origin_down.reshape((xRange, yRange, 3))
    origin_left = origin_left.reshape((xRange, yRange, 3))
    origin_right = origin_right.reshape((xRange, yRange, 3))
    
    direction = np.zeros_like(origin)
    direction[:,:,2] = 1

    # origins = [origin, origin_up, origin_down, origin_left, origin_right]
    origins = [origin, origin_down, origin_right]
    return origins, direction


def gen_init_sdf(width, length, height):
    sdf = np.zeros(shape=(width, length, height), dtype=np.int32)
    # 设置每一层的 sdf 值为当前层的高度，做初始化
    for h in range(height):
        sdf[:, :, h] = h
    
    return sdf
    


def gen_real_position(xRange, yRange, zRange, target_origin, block_unit):
    block_gap = 0.004
    real_position_x = np.zeros((xRange, yRange, zRange))
    real_position_y = np.zeros((xRange, yRange, zRange))
    real_position_z = np.zeros((xRange, yRange, zRange))
    
    for x in range(xRange):
        for y in range(yRange):
            for z in range(zRange):
                x_temp = xRange - (x + 1)
                x_temp = x_temp * (block_unit + block_gap)
                x_real = target_origin[0] + x_temp + block_unit * 0.5

                y_temp = yRange - (y + 1)
                y_temp = y_temp * (block_unit + block_gap)
                y_real = target_origin[1] + y_temp + block_unit * 0.5
                
                z_temp = z + 0.5
                z_temp = z_temp * (block_unit + block_gap)
                z_real = target_origin[2] + z_temp + block_unit * 0.5
                
                real_position_x[x][y][z] = x_real
                real_position_y[x][y][z] = y_real
                real_position_z[x][y][z] = z_real
                
    
    return real_position_x, real_position_y ,real_position_z
            




class Space(object):
    def __init__(self, width=5, length=5, height=10, size_minimum=0, holder=60, target_origin=np.array([0.35, 0.28, 0.025]), block_unit=0.03, item_set=None, check_stability=False, threshold=0.01):
        self.plain_size = np.array([width, length, height])
        self.max_axis = max(width, length)
        self.width = width
        self.length = length
        self.height = height
        self.low_bound = size_minimum
        self.target_origin = target_origin
        self.block_unit = block_unit
        self.item_set = item_set
        self.check_stability = check_stability

        # init needed
        # self.plain = np.zeros(shape=(self.max_axis, self.max_axis), dtype=np.int32)
        # self.space_mask = np.zeros(shape=(self.max_axis, self.max_axis), dtype=np.int32)
        # self.left_space = np.zeros(shape=(self.max_axis, self.max_axis), dtype=np.int32)
        # self.plain = np.zeros(shape=(width, length), dtype=np.int32)
        self.plain = np.zeros(shape=(width, length), dtype=np.float32)
        self.space_mask = np.zeros(shape=(width, length), dtype=np.int32)
        self.left_space = np.zeros(shape=(width, length), dtype=np.int32)
        self.box_vec = np.zeros((holder, 9))
        self.box_vec[0][-1] = 1

        self.reset()
        self.alleps = []

        self.EMS3D = dict()
        self.EMS3D[0] = np.array([0, 0, 0, width, length, height, self.serial_number])

        self.block_unit = 0.01
        self.threshold = threshold
        self._connected = False
        self.plane_id = None  # 存储地面的ID
        # self.initialize_simulation()
        
        # irgular
        self.ray_origins, self.ray_direction = gen_ray_origin_direction_real(self.width, self.length, self.target_origin, self.block_unit, shift = 0.005)
        self.sdf = gen_init_sdf(self.width, self.length, self.height)
        self.real_position_x, self.real_position_y, self.real_position_z = gen_real_position(self.width, self.length, self.height, self.target_origin, self.block_unit)
        self.posZmap = np.zeros((1, self.width, self.length))
        self.posZValid = np.zeros((1, self.width, self.length))
        self.policy = None
        
        
        

    def reset(self):
        self.plain[:] = 0
        self.space_mask[:] = 0
        self.left_space[:] = 0
        self.box_vec[:] = 0
        self.box_vec[0][-1] =1

        self.NOEMS = 1
        self.EMS = [np.array([0, 0, 0, *self.plain_size])]

        self.boxes = []
        self.box_idx = 0
        self.serial_number = 0

        self.ZMAP = dict()
        self.ZMAP[0] = dict()

        r = self.ZMAP[0]
        r['x_up'] = [0]
        r['y_left'] = [0]
        r['x_bottom'] = [self.plain_size[0]]
        r['y_right'] = [self.plain_size[1]]

        self.EMS3D = dict()
        self.EMS3D[0] = np.array([0, 0, 0, self.plain_size[0], self.plain_size[1], self.plain_size[2], self.serial_number])

    @staticmethod
    def update_height_graph(plain, box):
        plain = copy.deepcopy(plain)
        le = box.lx
        ri = box.lx + box.x
        up = box.ly
        do = box.ly + box.y
        max_h = np.max(plain[le:ri, up:do])
        max_h = max(max_h, box.lz + box.z)
        plain[le:ri, up:do] = max_h
        return plain

    def get_plain(self):
        return copy.deepcopy(self.plain)

    def get_action_space(self):
        return self.plain_size[0] * self.plain_size[1]

    def get_ratio(self):
        vo = reduce(lambda x, y: x + y, [box.x * box.y * box.z for box in self.boxes], 0.0)
        mx = self.plain_size[0] * self.plain_size[1] * self.plain_size[2]
        ratio = vo / mx
        assert ratio <= 1.0
        return ratio

    def scale_down(self, bottom_whole_contact_area):
        centre2D = np.mean(bottom_whole_contact_area, axis=0)
        dirction2D = bottom_whole_contact_area - centre2D
        bottom_whole_contact_area -= dirction2D * 0.1
        return bottom_whole_contact_area.tolist()

    def drop_box(self, box_size, idx, flag, density, setting, packed):
        if not flag:
            x, y, z = box_size
        else:
            y, x, z = box_size

        lx, ly = idx
        rec = self.plain[lx:lx + x, ly:ly + y]
        max_h = np.max(rec)
        box_now = Box(x, y, z, lx, ly, max_h, density)

        # if setting != 2:
        if self.check_stability:
            combine_contact_points = []
            for tmp in self.boxes:
                if tmp.lz + tmp.z == max_h:
                    x1 = max(box_now.vertex_low[0], tmp.vertex_low[0])
                    y1 = max(box_now.vertex_low[1], tmp.vertex_low[1])
                    x2 = min(box_now.vertex_high[0], tmp.vertex_high[0])
                    y2 = min(box_now.vertex_high[1], tmp.vertex_high[1])
                    if x1 >= x2 or y1 >= y2:
                        continue
                    else:
                        newEdge = DownEdge(tmp)
                        newEdge.area = (x1, y1, x2, y2)
                        newEdge.centre2D = np.array([x1 + x2, y1 + y2]) / 2
                        box_now.bottom_edges.append(newEdge)
                        combine_contact_points.append([x1, y1])
                        combine_contact_points.append([x1, y2])
                        combine_contact_points.append([x2, y1])
                        combine_contact_points.append([x2, y2])

            if len(combine_contact_points) > 0:
                box_now.bottom_whole_contact_area = self.scale_down(ConvexHull(combine_contact_points))

        sta_flag = self.check_box_true(x, y, lx, ly, z, max_h, box_now, setting)
        '''
        if self.check_stability:
            sta_flag = self.check_box(x, y, lx, ly, z, max_h, box_now, setting)
            if sta_flag is True:
                sta_flag = self.check_box_in_pybullet(x, y, lx, ly, z, max_h, box_now, setting, packed)
                # sta_flag = self.check_box_true(x, y, lx, ly, z, max_h, box_now, setting)
        else:
            sta_flag = self.check_box_true(x, y, lx, ly, z, max_h, box_now, setting)
        '''


        if sta_flag:
            self.boxes.append(box_now)  # record rotated box
            self.plain = self.update_height_graph(self.plain, self.boxes[-1])
            self.height = max(self.height, max_h + z)
            self.box_vec[self.box_idx] = np.array(
                        [lx, ly, max_h, lx + x, ly + y, max_h + z, density, 0, 1])
            self.box_idx += 1
            return True
        return False

    # Virtually place an item into the bin,
    # this function is used to check whether the placement is feasible for the current item
    def drop_box_virtual(self, box_size, idx, flag, density, setting,  returnH = False, returnMap = False):
        if not flag:
            x, y, z = box_size
        else:
            y, x, z = box_size

        lx, ly = int(idx[0]), int(idx[1])
        rec = self.plain[lx:lx + x, ly:ly + y]
        max_h = np.max(rec)

        box_now = Box(x, y, z, lx, ly, max_h, density, True)

        # if setting != 2:
        if self.check_stability:
            combine_contact_points = []
            for tmp in self.boxes:
                if tmp.lz + tmp.z == max_h:
                    x1 = max(box_now.vertex_low[0], tmp.vertex_low[0])
                    y1 = max(box_now.vertex_low[1], tmp.vertex_low[1])
                    x2 = min(box_now.vertex_high[0], tmp.vertex_high[0])
                    y2 = min(box_now.vertex_high[1], tmp.vertex_high[1])
                    if x1 >= x2 or y1 >= y2:
                        continue
                    else:
                        newEdge = DownEdge(tmp)
                        newEdge.area = (x1, y1, x2, y2)
                        newEdge.centre2D = np.array([x1 + x2, y1 + y2]) / 2
                        box_now.bottom_edges.append(newEdge)
                        combine_contact_points.append([x1, y1])
                        combine_contact_points.append([x1, y2])
                        combine_contact_points.append([x2, y1])
                        combine_contact_points.append([x2, y2])

            if len(combine_contact_points) > 0:
                box_now.bottom_whole_contact_area = self.scale_down(ConvexHull(combine_contact_points))

        if returnH:
            return self.check_box(x, y, lx, ly, z, max_h, box_now, setting, True), max_h
        elif returnMap:
            return self.check_box(x, y, lx, ly, z, max_h, box_now, setting, True), self.update_height_graph(self.plain, box_now)
        else:
            return self.check_box(x, y, lx, ly, z, max_h, box_now, setting, True)

    # Check if the placement is feasible
    def check_box(self, x, y, lx, ly, z, max_h, box_now, setting, virtual=False):
        assert isinstance(setting, int), 'The environment setting should be integer.'
        if lx + x > self.plain_size[0] or ly + y > self.plain_size[1]:
            return False
        if lx < 0 or ly < 0:
            return False
        if max_h + z > self.height:
            return False

        if isinstance(self.policy, str):
            return True

        if setting == 2:
            return True
        else:
            if max_h == 0:
                return True
            # 根据 virtual 决定调用哪个 impact 计算函数
            result = box_now.calculated_impact_virtual(True) if virtual else box_now.calculated_impact()
            return result

    def check_box_true(self, x, y, lx, ly, z, max_h, box_now, setting, virtual=False):
        assert isinstance(setting, int), 'The environment setting should be integer.'
        if lx + x > self.plain_size[0] or ly + y > self.plain_size[1]:
            return False
        if lx < 0 or ly < 0:
            return False
        if max_h + z > self.height:
            return False

        return True


    def check_box_in_pybullet(self, x, y, lx, ly, z, max_h, box_now, setting, packed, virtual=False):
        assert isinstance(setting, int), 'The environment setting should be integer.'
        if lx + x > self.plain_size[0] or ly + y > self.plain_size[1]:
            return False
        if lx < 0 or ly < 0:
            return False
        if max_h + z > self.height:
            return False

        # return True
        # '''  # 考虑理想状态, 直接return True
        if setting == 2:
            return True
        else:
            if max_h == 0:
                return True
            # 根据 virtual 决定调用哪个 impact 计算函数
            result = self.check_in_pybullet(box_now, packed)
            return result


    def initialize_simulation(self):
        """初始化PyBullet仿真环境，并加载地面"""
        if not self._connected:
            # 仅建立一次连接
            self._connected = True
            p.connect(p.DIRECT)  # 使用DIRECT模式，不显示GUI
            p.setGravity(0, 0, -10)  # 设置重力

            # # 加载地面，仅加载一次
            # self.plane_id = p.loadURDF("pct_envs/PctDiscrete0/plane/plane.urdf")
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)  # 禁用渲染以加快仿真速度
            p.setPhysicsEngineParameter(numSolverIterations=10, solverResidualThreshold=0.01)
            # 设置物理仿真的时间步长，例如从默认的1/240秒增加到1/120秒
            p.setTimeStep(1 / 120)

            p.configureDebugVisualizer(flag=p.COV_ENABLE_MOUSE_PICKING, enable=0)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 0)

    def reset_pybullet(self):
        physicsClient = p.connect(p.DIRECT)  # 使用DIRECT模式，不显示GUI
        p.setGravity(0, 0, -10, physicsClientId=physicsClient)  # 设置重力

        # # 加载地面，仅加载一次
        # self.plane_id = p.loadURDF("arm_pybullet_env/arm_envs/models/plane/plane.urdf")
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0, physicsClientId=physicsClient)  # 禁用渲染以加快仿真速度
        p.setPhysicsEngineParameter(numSolverIterations=10, solverResidualThreshold=0.01, physicsClientId=physicsClient)
        # 设置物理仿真的时间步长，例如从默认的1/240秒增加到1/120秒
        p.setTimeStep(1 / 120, physicsClientId=physicsClient)

        p.configureDebugVisualizer(flag=p.COV_ENABLE_MOUSE_PICKING, enable=0, physicsClientId=physicsClient)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=physicsClient)
        p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 0, physicsClientId=physicsClient)
        p.resetSimulation(physicsClientId=physicsClient)
        return physicsClient


    def close_simulation(self):
        """关闭PyBullet仿真环境"""
        if self._connected:
            p.disconnect()
            self._connected = False


    def check_in_pybullet(self, box_now, packed):
        """每次调用时仅清空仿真环境中的箱子，并检查当前物体的稳定性"""
        # 清空仿真环境中的箱子，保留地面
        physicsClient = self.reset_pybullet()

        # 记录所有箱子的 target_pose
        target_poses = []
        body_ids = []

        # 将已经摆放好的箱子加入仿真
        for box in packed:
            size = [box[0] * self.block_unit / 2, box[1] * self.block_unit / 2, box[2] * self.block_unit / 2]  # 缩放尺寸
            position = [box[3] * self.block_unit + size[0],
                        box[4] * self.block_unit + size[1],
                        box[5] * self.block_unit + size[2]]  # 中心位置

            collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=size, physicsClientId=physicsClient)
            body_id = p.createMultiBody(baseMass=1, baseCollisionShapeIndex=collision_shape, basePosition=position, physicsClientId=physicsClient)
            body_ids.append(body_id)  # 记录当前箱子ID
            target_poses.append(position)  # 记录目标位置

        # 加入当前需要摆放的 box_now
        size_now = [box_now.x * self.block_unit / 2, box_now.y * self.block_unit / 2,
                    box_now.z * self.block_unit / 2]  # 缩放尺寸
        position_now = [box_now.lx * self.block_unit + size_now[0],
                        box_now.ly * self.block_unit + size_now[1],
                        box_now.lz * self.block_unit + size_now[2]]  # 中心位置

        collision_shape_now = p.createCollisionShape(p.GEOM_BOX, halfExtents=size_now, physicsClientId=physicsClient)
        box_now_id = p.createMultiBody(baseMass=1, baseCollisionShapeIndex=collision_shape_now,
                                       basePosition=position_now, physicsClientId=physicsClient)
        body_ids.append(box_now_id)  # 记录当前 box_now 的ID
        target_poses.append(position_now)  # 记录目标位置

        # 仿真几步观察稳定性
        for _ in range(50):  # 仿真50步
            p.stepSimulation(physicsClientId=physicsClient)

        # 获取所有箱子的 final_pose
        final_poses = []
        for body_id in body_ids:  # 获取所有箱子的 final pose
            final_pos, _ = p.getBasePositionAndOrientation(body_id, physicsClientId=physicsClient)
            final_poses.append(final_pos)

        # 比较 target_pose 和 final_pose，计算 positions_offset
        stable = True
        for target, final in zip(target_poses, final_poses):
            positions_offset = [abs(t - f) for t, f in zip(target, final)]  # 计算偏移量
            if any(offset > self.threshold for offset in positions_offset):  # 检查偏移量是否超过阈值
                stable = False
                break
        p.disconnect(physicsClientId=physicsClient)

        return stable


    # Calculate the incrementally generated empty maximal spaces during the packing.
    def GENEMS(self, itemLocation):
        numofemss = len(self.EMS)
        delflag = []
        for emsIdx in range(numofemss):
            xems1, yems1, zems1, xems2, yems2, zems2 = self.EMS[emsIdx]
            xtmp1, ytmp1, ztmp1, xtmp2, ytmp2, ztmp2 = itemLocation

            if (xems1 > xtmp1): xtmp1 = xems1
            if (yems1 > ytmp1): ytmp1 = yems1
            if (zems1 > ztmp1): ztmp1 = zems1
            if (xems2 < xtmp2): xtmp2 = xems2
            if (yems2 < ytmp2): ytmp2 = yems2
            if (zems2 < ztmp2): ztmp2 = zems2

            if (xtmp1 > xtmp2): xtmp1 = xtmp2
            if (ytmp1 > ytmp2): ytmp1 = ytmp2
            if (ztmp1 > ztmp2): ztmp1 = ztmp2
            if (xtmp1 == xtmp2 or ytmp1 == ytmp2 or ztmp1 == ztmp2):
                continue

            self.Difference(emsIdx, (xtmp1, ytmp1, ztmp1, xtmp2, ytmp2, ztmp2))
            delflag.append(emsIdx)

        if len(delflag) != 0:
            NOEMS = len(self.EMS)
            self.EMS = [self.EMS[i] for i in range(NOEMS) if i not in delflag]
        self.EliminateInscribedEMS()

        # maintain the event point by the way
        cx_min, cy_min, cz_min, cx_max, cy_max, cz_max = itemLocation
        # bottom
        if cz_min < self.plain_size[2]:
            # bottomRecorder = self.ZMAP[cz_min]
            bottomRecorder = self.ZMAP[float(str(np.round(cz_min, 1)))]
            # bottomRecorder = self.ZMAP[round(cz_min, 6)]
            cbox2d = [cx_min, cy_min, cx_max, cy_max]
            maintainEventBottom(cbox2d, bottomRecorder['x_up'], bottomRecorder['y_left'], bottomRecorder['x_bottom'],
                                bottomRecorder['y_right'], self.plain_size)

        if cz_max < self.plain_size[2]:
            AddNewEMSZ(itemLocation, self)

    # Split an EMS when it intersects a placed item
    def Difference(self, emsID, intersection):
        x1, y1, z1, x2, y2, z2 = self.EMS[emsID]
        x3, y3, z3, x4, y4, z4, = intersection
        if self.low_bound == 0:
            self.low_bound = 0.1
        if IsUsableEMS(self.low_bound, self.low_bound, self.low_bound, x1, y1, z1, x3, y2, z2):
            self.AddNewEMS(x1, y1, z1, x3, y2, z2)
        if IsUsableEMS(self.low_bound, self.low_bound, self.low_bound, x4, y1, z1, x2, y2, z2):
            self.AddNewEMS(x4, y1, z1, x2, y2, z2)
        if IsUsableEMS(self.low_bound, self.low_bound, self.low_bound, x1, y1, z1, x2, y3, z2):
            self.AddNewEMS(x1, y1, z1, x2, y3, z2)
        if IsUsableEMS(self.low_bound, self.low_bound, self.low_bound, x1, y4, z1, x2, y2, z2):
            self.AddNewEMS(x1, y4, z1, x2, y2, z2)
        if IsUsableEMS(self.low_bound, self.low_bound, self.low_bound, x1, y1, z4, x2, y2, z2):
            self.AddNewEMS(x1, y1, z4, x2, y2, z2)

    def AddNewEMS(self, a, b, c, x, y, z):
        self.EMS.append(np.array([a, b, c, x, y, z]))

    # Eliminate redundant ems
    def EliminateInscribedEMS(self):
        NOEMS = len(self.EMS)
        delflags = np.zeros(NOEMS)
        for i in range(NOEMS):
            for j in range(NOEMS):
                if i == j:
                    continue
                if (self.EMS[i][0] >= self.EMS[j][0] and self.EMS[i][1] >= self.EMS[j][1]
                        and self.EMS[i][2] >= self.EMS[j][2] and self.EMS[i][3] <= self.EMS[j][3]
                        and self.EMS[i][4] <= self.EMS[j][4] and self.EMS[i][5] <= self.EMS[j][5]):
                    delflags[i] = 1
                    break
        self.EMS = [self.EMS[i] for i in range(NOEMS) if delflags[i] != 1]
        return len(self.EMS)

    # Convert EMS to placement (leaf node) for the current item.
    def EMSPoint(self, next_box, setting):
        posVec = set()
        if setting == 2: orientation = 6
        else: orientation = 2

        for ems in self.EMS:
            for rot in range(orientation):  # 0 x y z, 1 y x z, 2 x z y,  3 y z x, 4 z x y, 5 z y x
                if rot == 0:
                    sizex, sizey, sizez = next_box[0], next_box[1], next_box[2]
                elif rot == 1:
                    sizex, sizey, sizez = next_box[1], next_box[0], next_box[2]
                    if sizex == sizey:
                        continue
                elif rot == 2:
                    sizex, sizey, sizez = next_box[0], next_box[2], next_box[1]
                    if sizex == sizey and sizey == sizez:
                        continue
                elif rot == 3:
                    sizex, sizey, sizez = next_box[1], next_box[2], next_box[0]
                    if sizex == sizey and sizey == sizez:
                        continue
                elif rot == 4:
                    sizex, sizey, sizez = next_box[2], next_box[0], next_box[1]
                    if sizex == sizey:
                        continue
                elif rot == 5:
                    sizex, sizey, sizez = next_box[2], next_box[1], next_box[0]
                    if sizex == sizey:
                        continue

                if ems[3] - ems[0] >= sizex and ems[4] - ems[1] >= sizey and ems[5] - ems[2] >= sizez:
                    posVec.add((ems[0], ems[1], ems[2], ems[0] + sizex, ems[1] + sizey, ems[2] + sizez))
                    posVec.add((ems[3] - sizex, ems[1], ems[2], ems[3], ems[1] + sizey, ems[2] + sizez))
                    posVec.add((ems[0], ems[4] - sizey, ems[2], ems[0] + sizex, ems[4], ems[2] + sizez))
                    posVec.add((ems[3] - sizex, ems[4] - sizey, ems[2], ems[3], ems[4], ems[2] + sizez))
        posVec = np.array(list(posVec))
        return posVec

    # Find all placement that can accommodate the current item in the full coordinate space
    def FullCoord(self, next_box, setting):
        posVec = set()
        if setting == 2: orientation = 6
        else: orientation = 2

        for rot in range(orientation):  # 0 x y z, 1 y x z, 2 x z y,  3 y z x, 4 z x y, 5 z y x
            if rot == 0:
                sizex, sizey, sizez = next_box[0], next_box[1], next_box[2]
            elif rot == 1:
                sizex, sizey, sizez = next_box[1], next_box[0], next_box[2]
                if sizex == sizey:
                    continue
            elif rot == 2:
                sizex, sizey, sizez = next_box[0], next_box[2], next_box[1]
                if sizex == sizey and sizey == sizez:
                    continue
            elif rot == 3:
                sizex, sizey, sizez = next_box[1], next_box[2], next_box[0]
                if sizex == sizey and sizey == sizez:
                    continue
            elif rot == 4:
                sizex, sizey, sizez = next_box[2], next_box[0], next_box[1]
                if sizex == sizey:
                    continue
            elif rot == 5:
                sizex, sizey, sizez = next_box[2], next_box[1], next_box[0]
                if sizex == sizey:
                    continue

            for lx in range(self.plain_size[0]):
                for ly in range(self.plain_size[1]):
                    lz = self.plain[lx, ly]
                    if lx + sizex <= self.plain_size[0] and ly + sizey <= self.plain_size[1] \
                            and lz + sizez <= self.plain_size[2]:
                        posVec.add((lx, ly, lz, lx + sizex, ly + sizey, lz + sizez))

        posVec = np.array(list(posVec))
        return posVec

    # Find event points.
    def EventPoint(self, next_box, setting):
        if setting == 2: orientation = 6
        else: orientation = 2

        allPostion = []
        for k in self.ZMAP.keys():
            posVec = set()
            validEms = []
            for ems in self.EMS:
                if ems[2] == k:
                    validEms.append([ems[0], ems[1], -1, ems[3], ems[4], -1])
            if len(validEms) == 0:
                continue
            validEms = np.array(validEms)
            r = self.ZMAP[k]

            for rot in range(orientation):  # 0 x y z, 1 y x z, 2 x z y,  3 y z x, 4 z x y, 5 z y x
                if rot == 0:
                    sizex, sizey, sizez = next_box[0], next_box[1], next_box[2]
                elif rot == 1:
                    sizex, sizey, sizez = next_box[1], next_box[0], next_box[2]
                    if sizex == sizey:
                        continue
                elif rot == 2:
                    sizex, sizey, sizez = next_box[0], next_box[2], next_box[1]
                    if sizex == sizey and sizey == sizez:
                        continue
                elif rot == 3:
                    sizex, sizey, sizez = next_box[1], next_box[2], next_box[0]
                    if sizex == sizey and sizey == sizez:
                        continue
                elif rot == 4:
                    sizex, sizey, sizez = next_box[2], next_box[0], next_box[1]
                    if sizex == sizey:
                        continue
                elif rot == 5:
                    sizex, sizey, sizez = next_box[2], next_box[1], next_box[0]
                    if sizex == sizey:
                        continue

                check_list = []

                for xs in r['x_up']:
                    for ys in r['y_left']:
                        xe = xs + sizex
                        ye = ys + sizey
                        posVec.add((xs, ys, k, xe, ye, k + sizez))

                    for ye in r['y_right']:
                        ys = ye - sizey
                        xe = xs + sizex
                        posVec.add((xs, ys, k, xe, ye, k + sizez))

                for xe in r['x_bottom']:
                    xs = xe - sizex
                    for ys in r['y_left']:
                        ye = ys + sizey
                        posVec.add((xs, ys, k, xe, ye, k + sizez))

                    for ye in r['y_right']:
                        ys = ye - sizey
                        posVec.add((xs, ys, k, xe, ye, k + sizez))

            posVec  = np.array(list(posVec))
            emsSize = validEms.shape[0]

            cmpPos = posVec.repeat(emsSize, axis=0)
            cmpPos = cmpPos.reshape((-1, *validEms.shape))
            cmpPos = cmpPos - validEms
            cmpPos[:, :, 3] *= -1
            cmpPos[:, :, 4] *= -1
            cmpPos[cmpPos >= 0] = 1
            cmpPos[cmpPos < 0] = 0
            cmpPos = cmpPos.cumprod(axis=2)
            cmpPos = cmpPos[:, :, -1]
            cmpPos = np.sum(cmpPos, axis=1)
            validIdx = np.argwhere(cmpPos > 0)
            tmpVec = posVec[validIdx, :].squeeze(axis=1)
            allPostion.extend(tmpVec.tolist())

        return allPostion

    # Find extre empoints on each distinct two-dimensional plane.
    def ExtremePoint2D(self, next_box, setting):
        if setting == 2: orientation = 6
        else: orientation = 2
        cboxList = self.boxes
        if len(cboxList) == 0: return [(0, 0, 0, next_box[0], next_box[1], next_box[2]),
                                       (0, 0, 0, next_box[1], next_box[0], next_box[2])]
        Tset = {0}
        for box in cboxList:
            Tset.add(box.z + box.lz)
        Tset = sorted(list(Tset))

        CI = []
        lastCik = []
        for k in Tset:
            IK = []
            for box in cboxList:
                if box.lz + box.z > k:
                    IK.append(smallBox(box.lx, box.ly, box.lx + box.x, box.ly + box.y))
            Cik = extreme2D(IK)
            for p in Cik:
                if p not in lastCik:
                    CI.append((p[0], p[1], k))
            lastCik = copy.deepcopy(Cik)

        posVec = set()
        for pos in CI:
            for rot in range(orientation):  # 0 x y z, 1 y x z, 2 x z y,  3 y z x, 4 z x y, 5 z y x
                if rot == 0:
                    sizex, sizey, sizez = next_box[0], next_box[1], next_box[2]
                elif rot == 1:
                    sizex, sizey, sizez = next_box[1], next_box[0], next_box[2]
                    if sizex == sizey:
                        continue
                elif rot == 2:
                    sizex, sizey, sizez = next_box[0], next_box[2], next_box[1]
                    if sizex == sizey and sizey == sizez:
                        continue
                elif rot == 3:
                    sizex, sizey, sizez = next_box[1], next_box[2], next_box[0]
                    if sizex == sizey and sizey == sizez:
                        continue
                elif rot == 4:
                    sizex, sizey, sizez = next_box[2], next_box[0], next_box[1]
                    if sizex == sizey:
                        continue
                elif rot == 5:
                    sizex, sizey, sizez = next_box[2], next_box[1], next_box[0]
                    if sizex == sizey:
                        continue

                if pos[0] + sizex <= self.plain_size[0] and pos[1] + sizey <= self.plain_size[1] \
                        and pos[2] + sizez <= self.plain_size[2]:
                    posVec.add((pos[0], pos[1], pos[2], pos[0] + sizex, pos[1] + sizey, pos[2] + sizez))
        posVec = np.array(list(posVec))
        return posVec

    def CornerPoint(self, next_box, setting):
        if setting == 2: orientation = 6
        else: orientation = 2
        cboxList = self.boxes
        if len(cboxList) == 0: return [(0, 0, 0, next_box[0], next_box[1], next_box[2]),
                                       (0, 0, 0, next_box[1], next_box[0], next_box[2])]
        Tset = {0}
        for box in cboxList:
            Tset.add(box.z + box.lz)
        Tset = sorted(list(Tset))

        CI = []
        lastCik = []
        for k in Tset:
            IK = []
            for box in cboxList:
                if box.lz + box.z > k:
                    IK.append((box.lx, box.ly, box.lx + box.x, box.ly + box.y))
            Cik = corners2D(IK)
            for p in Cik:
                if p not in lastCik:
                    CI.append((p[0], p[1], k))
            lastCik = copy.deepcopy(Cik)

        posVec = set()
        for pos in CI:
            for rot in range(orientation):  # 0 x y z, 1 y x z, 2 x z y,  3 y z x, 4 z x y, 5 z y x
                if rot == 0:
                    sizex, sizey, sizez = next_box[0], next_box[1], next_box[2]
                elif rot == 1:
                    sizex, sizey, sizez = next_box[1], next_box[0], next_box[2]
                    if sizex == sizey:
                        continue
                elif rot == 2:
                    sizex, sizey, sizez = next_box[0], next_box[2], next_box[1]
                    if sizex == sizey and sizey == sizez:
                        continue
                elif rot == 3:
                    sizex, sizey, sizez = next_box[1], next_box[2], next_box[0]
                    if sizex == sizey and sizey == sizez:
                        continue
                elif rot == 4:
                    sizex, sizey, sizez = next_box[2], next_box[0], next_box[1]
                    if sizex == sizey:
                        continue
                elif rot == 5:
                    sizex, sizey, sizez = next_box[2], next_box[1], next_box[0]
                    if sizex == sizey:
                        continue

                if pos[0] + sizex <= self.plain_size[0] and pos[1] + sizey <= self.plain_size[1] \
                        and pos[2] + sizez <= self.plain_size[2]:
                    posVec.add((pos[0], pos[1], pos[2], pos[0] + sizex, pos[1] + sizey, pos[2] + sizez))
        posVec = np.array(list(posVec))
        return posVec

    def shot_whole(self):
        print("orinal plain is :", self.plain)
        plains = []
        for ray_origins in self.ray_origins:
            ray_origins = ray_origins.reshape((-1, 3))
            ray_ends    = ray_origins.copy().reshape((-1, 3))

            ray_origins[:, 2] = self.height * 2 + self.target_origin[2]
            ray_ends[:, 2] = 0 + self.target_origin[2]

            intersections = p.rayTestBatch(ray_origins, ray_ends, numThreads=16)    
            intersections = np.array(intersections, dtype=object)

            maskH = intersections[:, 0]
            maskH = np.where(maskH >= 0, 1, 0)

            fractions = intersections[:, 2]
            heightMapH = ray_origins[:, 2] + (ray_ends[:, 2] - ray_origins[:, 2]) * fractions - ray_ends[:, 2]
            heightMapH *= maskH

            
            heightMapH = heightMapH.reshape((self.width, self.length))
            # self.plain = heightMapH.astype(np.int32)
            heightMapH_real = (np.array(heightMapH) / self.block_unit).astype(float)
            plain = np.round(heightMapH_real).astype(np.int32)
            plains.append(plain)
        # max_plain = np.maximum.reduce(plains)
        plains = np.array(plains)
        # 初始化存储最大值的数组
        max_plain = np.zeros((self.plain.shape[0], self.plain.shape[1]))
        
        for i in range(self.plain.shape[0]):
            for j in range(self.plain.shape[1]):
                # 取出每个位置的所有值
                values = plains[:, i, j]
                # 排序并选择次大值
                if len(values) > 1:
                    sorted_values = np.sort(values)
                    max_plain[i, j] = sorted_values[-2]
                else:
                    # 如果只有一个值，直接使用该值
                    max_plain[i, j] = values[0]
        
        self.plain = max_plain.astype(np.int32)
        print("change plain to :", self.plain)

    def update_sdf(self, obstacles):
        # 获取 plain 的形状
        container_width, container_length, container_height = self.width, self.length, self.height
        sdf = np.zeros((container_width, container_length, container_height), dtype=np.float32)
        
        # 遍历 plain 中的每个元素
        for i in range(container_width):
            for j in range(container_length):
                height = self.plain[i, j]
                # 如果 plain 的值大于 0，表示该位置被占用
                if height > 0:
                    # 将 sdf 中相应位置的所有高度层设置为 -1
                    sdf[i, j, :height] = -1

        # self.sdf = sdf
        for i in range(container_width):
            for j in range(container_length):
                for k in range(container_height):
                    if sdf[i][j][k] < 0:    # 该位置已被占用，跳过
                        continue
                    
                    real_position = [self.real_position_x[i][j][k], self.real_position_y[i][j][k], self.real_position_z[i][j][k]]
                    distance = []
                    for packed_id in obstacles:
                        # 使用一个虚拟对象（球体）来模拟点位置
                        virtual_sphere_radius = 0.001  # 虚拟球体的半径，设为很小的值来模拟点
                        virtual_sphere_collision_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=virtual_sphere_radius)
                        virtual_sphere_id = p.createMultiBody(baseCollisionShapeIndex=virtual_sphere_collision_shape, basePosition=real_position)

                        # 计算最近点
                        closest_points = p.getClosestPoints(virtual_sphere_id, packed_id, distance=0.2)

                        # 提取最短距离
                        if closest_points:
                            shortest_distance = closest_points[0][8]  # 结果中的第9个元素是距离
                            contact_point_on_object = closest_points[0][5]  # 结果中的第6个元素是物体表面的接触点
                            print(f"Shortest distance: {shortest_distance}")
                            print(f"Contact point on object: {contact_point_on_object}")
                            distance.append(shortest_distance)
                        else:
                            print("No contact points found.")

                        # 移除虚拟球体
                        p.removeBody(virtual_sphere_id)
                    # 找到最小值
                    min_distance = min(distance)
                    sdf[i][j][k] = min_distance

        print('done')

    def update_sdf_thread(self, obstacles):
        def get_min_distance(position):
            real_position = position[0]
            id = position[1]
            distance = []
            for packed_id in obstacles:
                # 使用一个虚拟对象（球体）来模拟点位置
                virtual_sphere_radius = 0.001  # 虚拟球体的半径，设为很小的值来模拟点
                virtual_sphere_collision_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=virtual_sphere_radius)
                virtual_sphere_id = p.createMultiBody(baseCollisionShapeIndex=virtual_sphere_collision_shape, basePosition=real_position)

                # 计算最近点
                closest_points = p.getClosestPoints(virtual_sphere_id, packed_id, distance=0.2)

                # 提取最短距离
                if closest_points:
                    shortest_distance = closest_points[0][8]  # 结果中的第9个元素是距离
                    contact_point_on_object = closest_points[0][5]  # 结果中的第6个元素是物体表面的接触点
                    print(f"Shortest distance: {shortest_distance}")
                    print(f"Contact point on object: {contact_point_on_object}")
                    distance.append(shortest_distance)
                else:
                    print("No contact points found.")

                # 移除虚拟球体
                p.removeBody(virtual_sphere_id)
            # 找到最小值
            min_distance = min(distance)
            
            result = [min_distance, id]
            return result
        
        
        
        # 获取 plain 的形状
        container_width, container_length, container_height = self.width, self.length, self.height
        sdf = np.zeros((container_width, container_length, container_height), dtype=np.float32)
        
        # 遍历 plain 中的每个元素
        for i in range(container_width):
            for j in range(container_length):
                height = self.plain[i, j]
                # 如果 plain 的值大于 0，表示该位置被占用
                if height > 0:
                    # 将 sdf 中相应位置的所有高度层设置为 -1
                    sdf[i, j, :height] = -1

        # self.sdf = sdf
        real_positions = []
        for i in range(container_width):
            for j in range(container_length):
                for k in range(container_height):
                    if sdf[i][j][k] < 0:    # 该位置已被占用，跳过
                        continue
                    
                    real_position = [[self.real_position_x[i][j][k], self.real_position_y[i][j][k], self.real_position_z[i][j][k]],[i, j, k]]
                    real_positions.append(real_position)
                    
                    
                    # sdf[i][j][k] = min_distance

        
        # 使用线程池并行处理多个点
        min_distances = []
        num_threads = 24  # 这里设置为4个线程，可以根据需要调整
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            future_to_position = {executor.submit(get_min_distance, real_position): real_position for real_position in real_positions}
            for future in as_completed(future_to_position):
                real_position = future_to_position[future]
                try:
                    result = future.result()
                    min_distance, id = result[0], result[1]
                    sdf[id[0]][id[1]][id[2]] = min_distance
                    
                except Exception as exc:
                    print(f"Target point {real_position} generated an exception: {exc}")

        
        print('done')


    def get_possible_position(self, next_item_ID, scene, selectedAction):
        
        name = scene.box_names[next_item_ID]
        size = self.item_set[next_item_ID]
        # objPath = 'dataset/blockout/shape_vhacd/'+name+'_0.obj'
        objPath = os.path.join(scene.objPath, name) + '_0.obj'
        mesh = trimesh.load(objPath)
        scale = scene.scale
        mesh.apply_scale(scale)
        # 目标四元数
        quaternion = [0, 0, 0, 1]
        # 创建旋转矩阵
        rotation_matrix = trimesh.transformations.quaternion_matrix(quaternion)
        # 应用旋转矩阵
        mesh.apply_transform(rotation_matrix)
        
        origin, direction = gen_ray_origin_direction(self.width, self.length, self.block_unit)
        
        heightMapH, heightMapB, maskH, maskB = shot_item(mesh, origin, direction, size[0], size[1], self.block_unit)
        

        rotNum = 1
        naiveMask = np.zeros((rotNum, self.width, self.length))
        self.posZmap[:] = 1e3

        for lx in range(self.width- size[0] + 1):
            for ly in range(self.length - size[1] + 1):
                # get real Z            
                posZ = np.max((self.plain[lx: lx + size[0], ly: ly + size[1]] - heightMapB) * maskB)
                if np.round(posZ + size[2] - self.height, decimals=6) <= 0:
                    naiveMask[0, lx, ly] = 1
                self.posZmap[0, lx, ly] = posZ

        self.naiveMask = naiveMask.copy()
        invalidIndex = np.where(naiveMask==0)
        self.posZValid[:] = self.posZmap[:]
        self.posZValid[invalidIndex] = 1e3

        return naiveMask





