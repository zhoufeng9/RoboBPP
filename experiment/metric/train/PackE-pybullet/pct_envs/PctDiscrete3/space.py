import math

import numpy as np
from functools import reduce
import copy
from .convex_hull import ConvexHull, point_in_polygen
from .PctTools import AddNewEMSZ, maintainEventBottom, smallBox, extreme2D, corners2D
from .Interface import Interface
import pybullet as p
from numpy.lib.stride_tricks import sliding_window_view
import string
import random
import os
import time

colors = [[0.8, 0.8, 0.8, 1],
          [0.5, 0.5, 0.5, 1],
          [0.3, 0.3, 0.3, 1],
          [0.6, 0.6, 0.8, 1],
          [0.9, 0.9, 0.7, 1],
          [0.7, 0.6, 0.5, 1],
          [0.9, 0.8, 0.8, 1],
          [0.7, 0.8, 0.7, 1],
          [0.8, 0.7, 0.8, 1],
          [0.9, 0.85, 0.7, 1]]


class Stack(object):
    def __init__(self, centre, mass):
        self.centre = centre
        self.mass = mass

class DownEdge(object):
    def __init__(self, box):
        self.box = box
        self.area = None
        self.centre2D = None

class EMSNode(object):
    def __init__(self, parent, node_vec):
        x1, y1, z1, x2, y2, z2 = node_vec
        self.lx = x1 
        self.ly = y1
        self.lz = z1
        self.x = x2 - x1
        self.y = y2 - y1
        self.z = z2 - z1
        self.node_vec = node_vec

        self.x_low_bound = 0.1
        self.y_low_bound = 0.1
        self.z_low_bound = 0.1
        self.is_usable = True   # 是否是有效的 EMS （有些 EMS 太小了以至于可以忽略）

        self.parent = parent
        self.children = []
        self.placed_items = []
        self.valid_children = 0
        self.is_valid = True    # 是否被其他 leaf EMS 包含

        self.V = self.x * self.y * self.z
        self.volume = 0
        self.td_target = self.volume + self.V

    def _isUsableEMS(self, node_vec):
        x1, y1, z1, x2, y2, z2 = node_vec
        xd = x2 - x1
        yd = y2 - y1
        zd = z2 - z1
        if (xd >= self.x_low_bound) and (yd >= self.y_low_bound)\
            and (zd >= self.z_low_bound):
            return True
        return False

    def addChild(self, bounds, next_vec):
        self.x_low_bound, self.y_low_bound, self.z_low_bound = bounds
        if self._isUsableEMS(next_vec):
            child = EMSNode(self, next_vec)
            self.children.append(child)
            self.valid_children += 1

    def disable(self):
        self.is_valid = False
        if self.parent is None:
            return
        self.parent.valid_children -= 1
        self.parent.children.remove(self)
        if self.parent.valid_children == 0:
            self.parent.disable()


class Box(object):
    def __init__(self, x, y, z, lx, ly, lz, density, virtual=False, category=None):
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
        self.density = density
        self.category = category

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
            self.thisVirtualStack.mass   = new_stack_mass
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
        elif not point_in_polygen(self.thisVirtualStack.centre[0:2],
                                  self.bottom_whole_contact_area):
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


def fill_template(template, replace):
    """Read a file and replace key strings."""
    with open(template, 'r') as file:
        fdata = file.read()
    for field in replace:
        for i in range(len(replace[field])):
            fdata = fdata.replace(f'{field}{i}', str(replace[field][i]))
    alphabet = string.ascii_lowercase + string.digits
    rname = ''.join(random.choices(alphabet, k=16))
    template_filename = 'box' + str(time.time())
    fname = os.path.join('urdfs', f'{template_filename}-{rname}.urdf')
    with open(fname, 'w') as file:
        file.write(fdata)
    return fname


def add_object(obj_urdf, pose, category='rigid', physicsClient=1):
    fixed_base = 1 if category == 'fixed' else 0
    obj_id = p.loadURDF(
        obj_urdf,
        pose[0],
        pose[1],
        useFixedBase=fixed_base,
        physicsClientId=physicsClient
    )

    return obj_id


class Space(object):
    def __init__(self, width=10, length=10, height=10,
                 size_minimum=0, holder = 60, threshold=0.01, init_sizes=np.array([0.43, 0.20,  0.07]), physics = False,
                 robot_in_roop = False, check_area = False,
                 loading = False, draw = False, distribution = None, args = None):

        self.plain_size = np.array([width, length, height]).astype(np.int32)
        self.max_axis = int(max(width, length))
        self.height = height
        self.low_bound = size_minimum
        self.args = args

        # init needed
        self.plain = np.zeros(shape=(self.max_axis, self.max_axis), dtype=np.int32)
        self.frontier = np.zeros(shape=(width, height), dtype=np.int32)
        self.space_mask = np.zeros(shape=(self.max_axis, self.max_axis), dtype=np.int32)
        self.left_space = np.zeros(shape=(self.max_axis, self.max_axis), dtype=np.int32)
        self.box_vec = np.zeros((holder, 9))
        self.box_vec[0][-1] = 1

        self.physics = physics
        self.loading = loading
        self.robot_in_roop   = robot_in_roop
        self.distribution = distribution
        self.space_scale = 0.70
        self.check_area  = check_area
        self.init_sizes = init_sizes

        self.reset()
        self.alleps = []

        self.gripper_size = [140, 130, 68]

        self.block_unit = 0.01
        self.block_gap = 0.001
        if length == 125:
            self.offset = np.array([0.9, -0.6, 0.025])
        elif length == 100:
            self.offset = np.array([1.1, -0.5, 0.025])
        elif length == 120:
            self.offset = np.array([0.9, -1.0, 0.025])

        self.threshold = threshold
        self.plane_id = None  # 存储地面的ID

    def reset(self):
        self.plain[:] = 0
        self.frontier[:] = 0
        self.last_frontier = copy.deepcopy(self.frontier)
        self.space_mask[:] = 0
        self.left_space[:] = 0
        self.box_vec[:] = 0
        self.box_vec[0][-1] =1

        self.root_ems = EMSNode(None, np.array([0, 0, 0, *self.plain_size]))
        self.leaf_emss = [self.root_ems]

        self.boxes = []
        self.box_infos = []
        self.box_idx = 0
        self.serial_number = 0

        self.ZMAP = dict()
        self.ZMAP[0] = dict()

        r = self.ZMAP[0]
        r['x_up'] = [0]
        r['y_left'] = [0]
        r['x_bottom'] = [self.plain_size[0]]
        r['y_right'] = [self.plain_size[1]]

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

    @staticmethod
    def update_frontier(plain, box):
        plain = copy.deepcopy(plain)
        le = box.lx
        ri = box.lx + box.x
        up = box.lz
        do = box.lz + box.z
        max_y = np.max(plain[le:ri, up:do])
        max_y = max(max_y, box.ly + box.y)
        plain[le:ri, up:do] = max_y
        return plain

    def get_frontier_y(self, lx, next_box):
        x, y, z = next_box
        frontier = self.frontier
        ly_list = []
        for lz in range(0, self.height - z):
            max_y = np.max(frontier[lx:lx + x, lz:lz + z])
            if max_y + y > self.plain_size[1]:
                max_y = 1e10
            ly_list.append(max_y)
        return np.argmin(ly_list)

    @staticmethod
    def update_massmap(massmap, box):
        le = box.lx
        ri = box.lx + box.x
        up = box.ly
        do = box.ly + box.y
        massmap[le:ri, up:do] += box.density * box.z
        return massmap

    def cal_left_space(self, x,y,z,lx,ly,lz):
        if lx + x < self.max_axis:
            for right_idx in range(ly, ly + y):
                if z + lz > np.max(self.plain[lx + x:self.max_axis,right_idx]):
                    self.space_mask[lx + x:self.max_axis, right_idx] = self.plain_size[2]
        self.left_space[:] = self.plain[:]
        self.left_space[ self.left_space < self.space_mask ] = self.plain_size[2]

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

    def get_bridge_number(self):
        all_box_info = self.box_infos
        if len(all_box_info) <= 1:
            return 1

        all_box_info = np.array(all_box_info)
        box = all_box_info[-1]
        box_z = np.max(box[-2])
        if box_z == 0:
            return 1

        all_box_info = all_box_info[:-1]

        intersect = np.around(np.minimum(box, all_box_info), 6)
        signal = (intersect[:, 0] + intersect[:, 2] > 0) * (intersect[:, 1] + intersect[:, 3] > 0) # 等于零的地方表示不相交
        index = np.where(signal)[0]
        bridge_num = len(np.where((box_z - all_box_info[index][:, -1]) == 0)[0]) if len(index) != 0 else 0
        return bridge_num

    def get_variance(self):
        variance = np.var(self.plain/self.plain_size[2])
        return variance

    def get_loading_variance(self):
        variance = np.var(self.massmap/self.plain_size[2])
        return variance

    def get_category_distance(self):
        if len(self.boxes) < 1:
            return 0

        packed_box = self.boxes[-1]
        packed_cx = packed_box.lx + packed_box.x / 2
        packed_cy = packed_box.ly + packed_box.y / 2
        packed_cz = packed_box.lz + packed_box.z / 2

        distance_list = []
        for box in self.boxes[0:-1]:
            if packed_box.category == box.category:
                box_cx = box.lx + box.x / 2
                box_cy = box.ly + box.y / 2
                box_cz = box.lz + box.z / 2
                distance_list.append(np.sqrt((packed_cx - box_cx)**2 + (packed_cy - box_cy)**2 + (packed_cz - box_cz)**2))
        if len(distance_list) == 0:
            return 0
        else:
            return np.mean(distance_list)

    def get_loading_force(self):
        for _ in range(100):
            p.stepSimulation()

        forces = []
        for box_id in self.interface.objs:
            contact_points = p.getContactPoints(bodyA=box_id)
            total_force = 0
            box_forces = []
            for point in contact_points:
                force_components = point[9]
                box_forces.append(force_components)
                # total_force += force_components
            # forces.append(total_force)
            mean_item_force = np.mean(box_forces)
            if not np.isnan(mean_item_force):
                forces.append(mean_item_force)
            # print('forces', forces)
        mean = np.mean(forces)
        std = np.std(forces)
        # 计算Z-score并筛除异常值
        threshold = 3  # 一般选择3作为Z-score的阈值
        forces = [x for x in forces if (x - mean) / std < threshold]
        mean_force = np.mean(forces)
        return mean_force

    # drop a box and check feasibility
    def get_robot_feasibility(self, place_position,  pack_size):
        block_unit = 0.03
        block_unit /= 2
        generate_origin = np.array([0.35, -0.24, 0.049])
        # pack_size = np.array([2, 2, 2])
        cube_id, pick_pose = self.scene.load_box(pack_size, block_unit, generate_origin)

        # 摆放容器左上角
        target_origin = np.array([0.35, 0.28, 0.025])
        target_width = 5
        target_width *= 2
        self.scene.ompl_and_execute(self.scene.robot.homej)

        self.scene.pick_block(pick_pose)  # 捡起来物体
        self.scene.ompl_and_execute(self.scene.robot.homej)  # 移动到一个位置上去
        place_rotation = np.identity(3)

        place_position[-1] += pack_size[-1] / 2
        place_position[0] = target_width - (place_position[0] + pack_size[0])
        place_position[1] = target_width - (place_position[1] + pack_size[1])

        move_success, _ = self.scene.place_at(place_position, place_rotation, pack_size * block_unit,
                                              target_origin, block_unit, 'place', False, cube_id = cube_id)

    def drop_box(self, box_size, idx, flag, density, setting, category = None, model_architecture = 'PCT', packed=[]):
        if not flag:
            x, y, z = box_size
        else:
            y, x, z = box_size

        lx, ly = idx

        if model_architecture == 'Attend2Pack':
            if lx + x > self.plain_size[0] or ly + y > self.plain_size[1]:
                return False

        rec = self.plain[lx:lx + x, ly:ly + y]

        max_h = np.max(rec)
        box_now = Box(x, y, z, lx, ly, max_h, density, category=category)
        box_info = np.array([-lx, -ly, lx + x, ly + y, max_h, max_h + z])

        if setting != 2:
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

        if model_architecture == 'PackE':
            sta_flag = self.check_box_rule(x, y, lx, ly, z)
            sta_flag = sta_flag >= 0
            self.boxes.append(box_now)  # record rotated box
        else:
            sta_flag = self.check_box(x, y, lx, ly, z, max_h, box_now, setting)

        stability_reward, positions_offset = 0, 1  # myhere
        if sta_flag is True:
            sta_flag, stability_reward, positions_offset = self.check_box_in_pybullet(x, y, lx, ly, z, max_h, box_now, setting, packed)       # myhere

        if sta_flag:
            if model_architecture != 'PackE':
                self.boxes.append(box_now)  # record rotated box
            self.box_infos.append(box_info)  # record the box info -lx, -ly, lx + x, ly + y, max_h + z
            self.plain = self.update_height_graph(self.plain, self.boxes[-1])
            self.last_frontier = copy.deepcopy(self.frontier)
            self.frontier = self.update_frontier(self.frontier, self.boxes[-1])
            self.height = max(self.height, max_h + z)
            if self.robot_in_roop:
                self.cal_left_space(x,y,z,lx,ly,max_h)
            self.box_vec[self.box_idx] = np.array(
                        [lx, ly, max_h, lx + x, ly + y, max_h + z, density, category, 1])
            self.box_idx += 1
            return True, stability_reward, positions_offset  # myhere
        return False, stability_reward, positions_offset  # myhere

    def gripper_feasibility(self, ex, ey, ez, _ex, _ey, _ez, sizex, sizey, sizez):
        ex = int(ex)
        ey = int(ey)
        real_z = ez

        gx, gy, gz = self.gripper_size
        x_left  = max(0, int(ex + sizex / 2 - gx / 2))
        x_right = int(ex + sizex / 2 + gx / 2)

        y_left  = max(0, int(ey + sizey / 2 - gy / 2))
        y_right = int(ey + sizey / 2 + gy / 2)
        max_h = np.max(self.plain[x_left:x_right, y_left:y_right])

        return max_h - (real_z + sizez) <= gz


    # Virtually place an item into the bin,
    # this function is used to check whether the placement is feasible for the current item
    def drop_box_virtual(self, box_size, idx, flag, density, setting,  returnH = False, returnMap = False):
        if not flag:
            x, y, z = box_size
        else:
            y, x, z = box_size

        lx, ly = idx
        lx, ly, x, y, z = int(lx), int(ly), int(x), int(y), int(z)
        rec = self.plain[lx:lx + x, ly:ly + y]
        max_h = np.max(rec)

        if self.distribution == 'deli':
            if not self.gripper_feasibility(lx, ly, max_h, lx + x, ly + y, max_h + z, x, y, z):
                return False

        box_now = Box(x, y, z, lx, ly, max_h, density, True)

        if setting != 2:
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

    def check_box_rule(self, x, y, lx, ly, z):
        if lx + x > self.plain_size[0] or ly + y > self.plain_size[1]:
            return -1
        if lx < 0 or ly < 0:
            return -1

        rec = self.plain[lx:lx + x, ly:ly + y]
        r00 = rec[0, 0]
        r10 = rec[x - 1, 0]
        r01 = rec[0, y - 1]
        r11 = rec[x - 1, y - 1]
        rm = max(r00, r10, r01, r11)
        sc = int(r00 == rm) + int(r10 == rm) + int(r01 == rm) + int(r11 == rm)
        if sc < 3:
            return -1
        # get the max height
        max_h = np.max(rec)
        # check area and corner
        max_area = np.sum(rec == max_h)
        area = x * y

        # check boundary
        assert max_h >= 0
        if max_h + z > self.height:
            return -1

        if max_area / area > 0.95:
            return max_h
        if rm == max_h and sc == 3 and max_area / area > 0.85:
            return max_h
        if rm == max_h and sc == 4 and max_area / area > 0.50:
            return max_h

        return -1

    def check_box_rule_batch(self, x, y, z, without_stable_check = False):
        plain = self.plain[0: self.plain_size[0], 0: self.plain_size[1]]
        windows = sliding_window_view(plain, (x, y))
        M, N = windows.shape[:2]  # 窗口起始点的数量
        area = x * y  # 窗口面积

        r00 = windows[:, :, 0, 0]  # 左上角
        r10 = windows[:, :, x - 1, 0]  # 左下角
        r01 = windows[:, :, 0, y - 1]  # 右上角
        r11 = windows[:, :, x - 1, y - 1]  # 右下角

        # 获取窗口内的最大高度
        max_h = windows.max(axis=(2, 3))  # 在 x, y 方向上取最大值
        # 判断最大高度是否超出允许范围
        valid_mask = (max_h + z <= self.height)

        if without_stable_check:
            return valid_mask, M, N

        rm = np.maximum(np.maximum(r00, r10), np.maximum(r01, r11))
        sc = ((r00 == rm).astype(int) +
              (r10 == rm).astype(int) +
              (r01 == rm).astype(int) +
              (r11 == rm).astype(int))

        # 判断是否符合规则：sc < 3 或 lx/ly 越界
        valid_mask &= (sc >= 3)  # 至少有 3 个角的值等于 rm
        # 计算窗口内的最大值区域占比
        max_area = (windows == max_h[:, :, None, None]).sum(axis=(2, 3))  # 最大值的数量
        area_ratio = max_area / area

        final_mask = (
                (area_ratio > 0.95) |
                ((rm == max_h) & (sc == 3) & (area_ratio > 0.85)) |
                ((rm == max_h) & (sc == 4) & (area_ratio > 0.50))
        )
        final_result = np.where(valid_mask & final_mask, max_h, -1).reshape(M, N)
        return final_result, M, N


    # Check if the placement is feasible
    def check_box(self, x, y, lx, ly, z, max_h, box_now, setting, virtual=False):
        assert isinstance(setting, int), 'The environment setting should be integer.'
        if lx + x > self.plain_size[0] or ly + y > self.plain_size[1]:
            return False
        if lx < 0 or ly < 0:
            return False
        if max_h + z > self.height:
            return False

        if setting == 2:
            return True
        else:
            if max_h == 0:
                return True
            if not virtual:
                result = box_now.calculated_impact()
                return result
            else:
                return box_now.calculated_impact_virtual(True)
        # # For test only
        # return True

    def check_box_in_pybullet(self, x, y, lx, ly, z, max_h, box_now, setting, packed, virtual=False):
        stability_reward, positions_offset = 0.5, 0  # myhere
        assert isinstance(setting, int), 'The environment setting should be integer.'
        if lx + x > self.plain_size[0] or ly + y > self.plain_size[1]:
            return False, stability_reward, positions_offset  # myhere
        if lx < 0 or ly < 0:
            return False, stability_reward, positions_offset  # myhere
        if max_h + z > self.height:
            return False, stability_reward, positions_offset  # myhere

        # return True
        # '''  # 考虑理想状态, 直接return True
        if setting == 2:
            return True, stability_reward, positions_offset  # myhere
        else:
            if max_h == 0:
                return True, stability_reward, positions_offset  # myhere
            # 根据 virtual 决定调用哪个 impact 计算函数
            result, stability_reward, positions_offset = self.check_in_pybullet(box_now, packed)        # myhere
            # result = self.check_in_pybullet_backup(box_now, packed)
            return result, stability_reward, positions_offset



    def reset_pybullet(self):
        physicsClient = p.connect(p.DIRECT)  # 使用DIRECT模式，不显示GUI
        # physicsClient = p.connect(p.GUI)  # 使用DIRECT模式，不显示GUI
        # p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        p.setGravity(0, 0, -10, physicsClientId=physicsClient)  # 设置重力

        self.obstacles = []

        # # 加载地面，仅加载一次
        self.plane_id = p.loadURDF("pct_envs/PctDiscrete3/plane/plane.urdf", physicsClientId=physicsClient)
        self.obstacles.append(self.plane_id)
        # 设置物理仿真的时间步长，例如从默认的1/240秒增加到1/120秒
        p.setTimeStep(1 / 120, physicsClientId=physicsClient)

        # p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0, physicsClientId=physicsClient)  # 禁用渲染以加快仿真速度
        # p.setPhysicsEngineParameter(numSolverIterations=10, solverResidualThreshold=0.01, physicsClientId=physicsClient)
        #
        #
        # p.configureDebugVisualizer(flag=p.COV_ENABLE_MOUSE_PICKING, enable=0, physicsClientId=physicsClient)
        # p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=physicsClient)
        # p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 0, physicsClientId=physicsClient)
        # p.resetSimulation(physicsClientId=physicsClient)
        return physicsClient



    def check_in_pybullet(self, box_now, packed):
        """每次调用时仅清空仿真环境中的箱子，并检查当前物体的稳定性"""
        # 清空仿真环境中的箱子，保留地面
        physicsClient = self.reset_pybullet()

        # 记录所有箱子的 target_pose
        target_poses = []
        body_ids = []
        urdf_all = []
        rotation = [0, 0, 0, 1]
        box_template = 'box/box-template-tex.urdf'

        # 将已经摆放好的箱子加入仿真
        for i, box in enumerate(packed):
            size = [box[0], box[1], box[2]]
            size_pack_pre = self.init_sizes[i]
            size_pack_pre_int = np.ceil(np.array(size_pack_pre) / 0.01).astype(int)
            if size[0] == size_pack_pre_int[0] and size[1] == size_pack_pre_int[1]:
                size_pack = size_pack_pre
            else:
                size_pack = [size_pack_pre[1], size_pack_pre[0], size_pack_pre[2]]
            # size_pack = [box[0] * self.block_unit, box[1] * self.block_unit, box[2] * self.block_unit]
            position = [box[3] * 1.0, box[4] * 1.0, box[5] * 1.0]
            position[-1] += size[-1] / 2  # 此处做了修改，之后check这个高度对不对
            position[0] = self.plain_size[0] - (position[0] + size[0])  # here
            position[1] = self.plain_size[1] - (position[1] + size[1])

            position = np.array(position) * (self.block_unit + self.block_gap)
            size = np.array(size) * (self.block_unit + self.block_gap)

            real_position = [
                self.offset[0] + position[0] + size[0] / 2,
                self.offset[1] + position[1] + size[1] / 2,
                self.offset[2] + position[2] + size[2] / 2
            ]


            np.random.seed(int(time.time() * 1000) % 2 ** 32)
            color = colors[np.random.randint(len(colors))]

            attempt = 0
            while attempt < 3:
                try:
                    urdf = fill_template(box_template, {'DIM': size_pack, 'RGBA': color, "FILE_PATH": ['box']})
                    body_id = add_object(urdf, [real_position, rotation], 'rigid', physicsClient)
                    urdf_all.append(urdf)
                    break  # 结束循环，避免重复尝试
                except Exception as e:
                    attempt += 1
                    time.sleep(0.0001)  # 等待一定时间后再重试
            else:
                # 如果超过最大尝试次数，抛出异常
                raise RuntimeError("多次尝试后仍未能成功添加对象。")


            # urdf = fill_template(box_template, {'DIM': size_pack, 'RGBA': color, "FILE_PATH": ['box']})
            # body_id = add_object(urdf, [real_position, rotation], 'rigid', physicsClient)
            # urdf_all.append(urdf)

            body_ids.append(body_id)  # 记录当前箱子ID
            target_poses.append(real_position)  # 记录目标位置

            # # 循环直到box与其他物体接触
            # while True:
            #     real_position[-1] -= 0.002
            #     p.resetBasePositionAndOrientation(body_id, real_position, rotation, physicsClientId=physicsClient)
            #     p.stepSimulation(physicsClientId=physicsClient)
            #     contact_points = []
            #     # 检测当前物体是否与任何其他物品接触
            #     for packed_id in self.obstacles:
            #         contact_points = p.getContactPoints(body_id, packed_id, physicsClientId=physicsClient)
            #         if contact_points:  # 如果有接触点
            #             # print(f"contact_points: {contact_points}")
            #             break  # 找到接触点，退出循环
            #         else:
            #             continue
            #     if contact_points:  # 如果有接触点
            #         break  # 找到接触点，退出循环
            # self.obstacles.append(body_id)

        size_now = [box_now.x, box_now.y, box_now.z]
        # size_now_pack = self.init_sizes[len(packed)]
        size_now_pack_pre = self.init_sizes[len(packed)]
        size_now_pack_pre_int = np.ceil(np.array(size_now_pack_pre) / 0.01).astype(int)
        if size_now[0] == size_now_pack_pre_int[0] and size_now[1] == size_now_pack_pre_int[1]:
            size_now_pack = size_now_pack_pre
        else:
            size_now_pack = [size_now_pack_pre[1], size_now_pack_pre[0], size_now_pack_pre[2]]


        # size_now_pack = [box_now.x * self.block_unit, box_now.y * self.block_unit, box_now.z * self.block_unit]
        position_now = [box_now.lx * 1.0, box_now.ly * 1.0, box_now.lz * 1.0]
        position_now[-1] += size_now[-1] / 2  # 此处做了修改，之后check这个高度对不对
        position_now[0] = self.plain_size[0] - (position_now[0] + size_now[0])  # here
        position_now[1] = self.plain_size[1] - (position_now[1] + size_now[1])

        position_now = np.array(position_now) * (self.block_unit + self.block_gap)
        size_now = np.array(size_now) * (self.block_unit + self.block_gap)

        real_position_now = [
            self.offset[0] + position_now[0] + size_now[0] / 2,
            self.offset[1] + position_now[1] + size_now[1] / 2,
            self.offset[2] + position_now[2] + size_now[2] / 2
        ]

        np.random.seed(int(time.time() * 1000) % 2 ** 32)
        color_now = colors[np.random.randint(len(colors))]

        attempt = 0
        while attempt < 3:
            try:
                urdf_now = fill_template(box_template, {'DIM': size_now_pack, 'RGBA': color_now, "FILE_PATH": ['box']})
                box_now_id = add_object(urdf_now, [real_position_now, rotation], 'rigid', physicsClient)
                urdf_all.append(urdf_now)
                break  # 结束循环，避免重复尝试
            except Exception as e:
                attempt += 1
                time.sleep(0.0001)  # 等待一定时间后再重试
        else:
            # 如果超过最大尝试次数，抛出异常
            raise RuntimeError("多次尝试后仍未能成功添加对象。")


        # urdf_now = fill_template(box_template, {'DIM': size_now_pack, 'RGBA': color_now, "FILE_PATH": ['box']})
        # urdf_all.append(urdf_now)
        # box_now_id = add_object(urdf_now, [real_position_now, rotation], 'rigid', physicsClient)

        body_ids.append(box_now_id)  # 记录当前 box_now 的ID
        target_poses.append(real_position_now)  # 记录目标位置

        # # 循环直到box与其他物体接触
        # while True:
        #     real_position_now[-1] -= 0.001
        #     p.resetBasePositionAndOrientation(box_now_id, real_position_now, rotation, physicsClientId=physicsClient)
        #     p.stepSimulation(physicsClientId=physicsClient)
        #     contact_points = []
        #     # 检测当前物体是否与任何其他物品接触
        #     for packed_id in self.obstacles:
        #         contact_points = p.getContactPoints(box_now_id, packed_id, physicsClientId=physicsClient)
        #         if contact_points:  # 如果有接触点
        #             # print(f"contact_points: {contact_points}")
        #             break  # 找到接触点，退出循环
        #         else:
        #             continue
        #     if contact_points:  # 如果有接触点
        #         break  # 找到接触点，退出循环



        # 仿真几步观察稳定性
        # for _ in range(200):  # 仿真200步
        #     p.stepSimulation(physicsClientId=physicsClient)

        num_steps = 20  # myhere
        velocities = {}
        for i in body_ids:
            velocities[str(i)] = []
        # 运行20个仿真步长并测量速度
        for step in range(num_steps):
            p.stepSimulation(physicsClientId=physicsClient)
            for i in body_ids:
                linear_vel, angular_vel = p.getBaseVelocity(i)
                velocities[str(i)].append([linear_vel, angular_vel])

        # print(velocities)
        # For each object in the bin we select the maximum measured velocity
        max_linear_vel = {}
        max_angular_vel = {}
        for key, values in velocities.items():
            p_linear, p_angular = [], []
            for value in values:
                now_linear = math.sqrt(sum(x ** 2 for x in value[0]))
                now_angular = math.sqrt(sum(x ** 2 for x in value[1]))
                p_linear.append(now_linear)
                p_angular.append(now_angular)
            p_linear_max = max(p_linear)
            p_angular_max = max(p_angular)
            max_linear_vel[key] = p_linear_max
            max_angular_vel[key] = p_angular_max

        # Take the max of all objects in the bin for linear and angular velocities respectively.
        max_linear_vel_key = max(max_linear_vel, key=max_linear_vel.get)
        max_linear_vel_value = max_linear_vel[max_linear_vel_key]
        max_angular_vel_key = max(max_angular_vel, key=max_angular_vel.get)
        max_angular_vel_value = max_angular_vel[max_angular_vel_key]

        # 计算linear_stability_reward
        linear_stability_reward = max(0, min(1, -(max_linear_vel_value ** 0.4) + 1))
        angular_stability_reward = max(0, min(1, -(max_angular_vel_value ** 0.3) + 1))
        stability_reward = linear_stability_reward * 0.5 + angular_stability_reward * 0.5



        # 获取所有箱子的 final_pose
        final_poses = []
        for body_id in body_ids:  # 获取所有箱子的 final pose
            final_pos, _ = p.getBasePositionAndOrientation(body_id, physicsClientId=physicsClient)
            final_poses.append(final_pos)

        # 比较 target_pose 和 final_pose，计算 positions_offset
        stable = True
        positions_offsets = []  # myhere
        for target, final in zip(target_poses, final_poses):
            positions_offset = [abs(t - f) for t, f in zip(target, final)]  # 计算偏移量
            now_position_offset = positions_offset[0] + positions_offset[1]  # myhere
            positions_offsets.append(now_position_offset)  # myhere
            if any(offset > self.threshold for offset in positions_offset[0:2]):  # 检查偏移量是否超过阈值
                stable = False
                break
        p.disconnect(physicsClientId=physicsClient)

        final_positions_offset = np.mean(positions_offsets)  # myhere

        for u in urdf_all:
            if os.path.exists(u):
                # 如果文件存在，则删除
                os.remove(u)
            else:
                pass

        return stable, stability_reward, final_positions_offset     # myhere



    # Calculate the incrementally generated empty maximal spaces during the packing.
    def GENEMS(self, itemLocation):
        numofemss = len(self.leaf_emss)
        delflag = []
        for emsIdx in range(numofemss):
            xems1, yems1, zems1, xems2, yems2, zems2 = self.leaf_emss[emsIdx].node_vec
            xtmp1, ytmp1, ztmp1, xtmp2, ytmp2, ztmp2 = itemLocation

            # 计算物体是否和当前 ems 存在 intersection，如果存在，使用 tmp 存储 intersection
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

            # 若相交，将当前 EMS 的 idx 存入 delflag，并生成新的 EMS
            self.Difference(emsIdx, (xtmp1, ytmp1, ztmp1, xtmp2, ytmp2, ztmp2))
            delflag.append(emsIdx)

        # 实际上相当于每插入一个物体就需要重写一遍整个 EMS
        if len(delflag) != 0:
            numofemss = len(self.leaf_emss)
            self.leaf_emss = [self.leaf_emss[i] for i in range(numofemss) if i not in delflag]
        self.EliminateInscribedEMS()

        # maintain the event point by the way
        cx_min, cy_min, cz_min, cx_max, cy_max, cz_max = itemLocation
        # bottom
        if cz_min < self.plain_size[2]:
            bottomRecorder = self.ZMAP[cz_min]
            cbox2d = [cx_min, cy_min, cx_max, cy_max]
            maintainEventBottom(cbox2d, bottomRecorder['x_up'], bottomRecorder['y_left'], bottomRecorder['x_bottom'],
                                bottomRecorder['y_right'], self.plain_size)

        if cz_max < self.plain_size[2]:
            AddNewEMSZ(itemLocation, self)

    # Split an EMS when it intersects a placed item
    def Difference(self, emsID, intersection):
        ems_node = self.leaf_emss[emsID]
        x1, y1, z1, x2, y2, z2 = ems_node.node_vec
        x3, y3, z3, x4, y4, z4, = intersection
        if self.low_bound == 0:
            self.low_bound = 0.1
        self.bounds = np.array([self.low_bound, self.low_bound, self.low_bound])
        self.AddNewEMS(ems_node, np.array([x1, y1, z1, x3, y2, z2]))
        self.AddNewEMS(ems_node, np.array([x4, y1, z1, x2, y2, z2]))
        self.AddNewEMS(ems_node, np.array([x1, y1, z1, x2, y3, z2]))
        self.AddNewEMS(ems_node, np.array([x1, y4, z1, x2, y2, z2]))
        self.AddNewEMS(ems_node, np.array([x1, y1, z4, x2, y2, z2]))
        if ems_node.valid_children:
            self.leaf_emss.extend(ems_node.children)

    def AddNewEMS(self, cur_node, next_vec):
        cur_node.addChild(self.bounds, next_vec)

    # Eliminate redundant ems
    def EliminateInscribedEMS(self):
        numofemss = len(self.leaf_emss)
        delflags = np.zeros(numofemss)
        for i in range(numofemss):
            for j in range(numofemss):
                if i == j:
                    continue
                if (self.leaf_emss[i].node_vec[0] >= self.leaf_emss[j].node_vec[0] and self.leaf_emss[i].node_vec[1] >= self.leaf_emss[j].node_vec[1]
                        and self.leaf_emss[i].node_vec[2] >= self.leaf_emss[j].node_vec[2] and self.leaf_emss[i].node_vec[3] <= self.leaf_emss[j].node_vec[3]
                        and self.leaf_emss[i].node_vec[4] <= self.leaf_emss[j].node_vec[4] and self.leaf_emss[i].node_vec[5] <= self.leaf_emss[j].node_vec[5]):
                    # EMS[i] 被 EMS[j] 完全包含
                    delflags[i] = 1
                    self.leaf_emss[i].disable()
                    break
        self.leaf_emss = [self.leaf_emss[i] for i in range(numofemss) if delflags[i] != 1]
        return len(self.leaf_emss)

    # Convert EMS to placement (leaf node) for the current item.
    def EMSPoint(self, next_box, setting):
        posVec = set()
        if setting == 2: orientation = 6
        else: orientation = 2

        for valid_ems in self.leaf_emss:
            ems = valid_ems.node_vec
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
                    # if self.args.with_four_corner:
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
            for ems_node in self.leaf_emss:
                ems = ems_node.node_vec
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

    # Consider the extreme points without next box.
    def ExtremePointClean(self):
        cboxList = self.boxes
        if len(cboxList) == 0: return [np.array((0, 0))]
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
                posVec.add((pos[0], pos[1], pos[2]))
        posVec = np.array(list(posVec))[:, 0:2]
        posVec = np.unique(np.round(posVec, 6), axis=0)
        return posVec.tolist()

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
        if setting == 2: orientation = 2
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