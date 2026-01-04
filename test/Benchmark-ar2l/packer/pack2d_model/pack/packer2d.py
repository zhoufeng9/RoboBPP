# -*- coding: utf-8 -*-

import copy
import io
import pyclipper
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from shapely import affinity
from shapely.geometry import Polygon
import time
import cv2
import functools
from .tool_2d import *


# 参考链接：https://end0tknr.hateblo.jp/entry/20221122/1669057478

# NOTE 算法的一些处理规则
# 
# 每个零件图的像素都是1mm，算法先将其缩小为1/10的大小然后在cm单位下进行规划，最终才返回的mm单位的数据
# 
# 默认零件都进行一步规划，当零件的最长边 > 容器最短边/2，则进行两步规划，尝试摆放同个零件两次
#
# 如果同时允许多个容器组合，当零件放不下当前所有容器的时候，则将这些容器进行拼接，拼为大号矩形容器进行规划
# 
# remove_location的函数输入的是抓手的目标位置，寻找距离误差小于3cm的记录进行删除

class Packer_2d():
    def __init__(self, config_path, simplify_level=5) -> None:

        self.simplify_level = simplify_level

        self.search_step = None

        self.load_config(config_path)
        self.new_containers(self.cids)

    def load_config(self, config_path):
        assert os.path.exists(config_path), f"配置文件{config_path}不存在"
        config = Config()
        containers, cids, corner = config.load_config(config_path)
        self.containers = containers
        self.cids = cids
        self.corner = corner

        self.pack_id = 0

    def new_containers(self, cids):
        '''
        Args:
            cids: list(int), 要刷新的容器列表
        '''
        if type(cids) == int:
            cids = [cids]
        # 刷新容器
        for c in self.containers:
            if c.cid in cids:
                c.new_container()

    def get_containers(self, container_ids, obj_size=None):
        def reorder(data, order):
            ret = []
            for o in order:
                ret.append(data[o])
            return ret

        container_list = []

        for ids in container_ids:
            each_container_list = []
            each_area_list = []
            combine_them = 0

            for ci in ids:
                if ci not in self.cids:
                    return None, ci

                index = self.cids.index(ci)

                container = self.containers[index]

                # NOTE 根据物体与容器的长度关系判断是否能放下，考虑最后将容器拼接
                if len(ids) > 1 and obj_size is not None:
                    max_length = max(obj_size)

                    min_container_len = max(container.length, container.width)
                    if min_container_len < max_length:
                        combine_them += 1

                each_area_list.append(container.area_rate)

                each_container_list.append(container)

            order = np.argsort(each_area_list)[::-1]  # 按降序排列的索引
            each_container_list = reorder(each_container_list, order)

            container_list.append(each_container_list)

        return container_list, -1

    def update_packed_poly(self, obj: Part, container: Container_2d):
        # 将要装箱的零件更新到容器
        container_ids = container.cid
        if type(container_ids) == int:
            container_ids = [container_ids]

        to_packed_poly = obj.poly

        for ci in container_ids:
            index = self.cids.index(ci)
            container = self.containers[index]

            if to_packed_poly.intersects(container.poly):
                container.update_packed(obj, 'add')
                obj.cids.append(ci)

    def get_packed_polys(self, i=None):
        ret = []
        all_pid = []
        if i is None:
            for c in self.containers:
                for ci, obj in enumerate(c.packed_objs):
                    pid = obj.pid
                    if pid not in all_pid:
                        all_pid.append(pid)
                        ret.append(obj)
        else:
            for c in self.containers:
                if c.cid == i:
                    ret = c.packed_objs
                    return ret
        return ret

    def find_pos(self, nfp, poly_part, container_poly):
        # 返回零件的目标位置和执行代码

        bin_center = get_bbox_center(container_poly)

        minx, miny, maxx, maxy = container_poly.bounds
        if (self.corner == [0, 0]).all():
            bin_corner = np.array([minx, miny])
        elif (self.corner == [0, 1]).all():
            bin_corner = np.array([minx, maxy])
        elif (self.corner == [1, 1]).all():
            bin_corner = np.array([maxx, maxy])
        elif (self.corner == [1, 0]).all():
            bin_corner = np.array([maxx, miny])

        cross_points = get_coords(nfp)

        if len(cross_points) == 0:
            return None, 402

        cross_points = np.array(cross_points)

        poly_center = get_bbox_center(poly_part)
        centroids = cross_points + poly_center

        if self.pos_rule == 'close':  # 距离预设的角点最近

            dist_to_origin = np.linalg.norm(centroids - bin_corner, axis=1)  # 计算每个（centroid）到某个参考点（bin_corner）的欧氏距离
            rule_order = np.argsort(dist_to_origin)

            def x_first(a, b):
                x = 0
                y = 1

                if a[x] < b[x]:
                    return -1
                elif a[x] == b[x]:
                    if a[y] < b[y]:
                        return -1
                    elif a[y] == b[y]:
                        return 0
                    else:
                        return 1
                else:
                    return 1

            # vec_to_origin = centroids - bin_corner
            # order_list = [i for i in range(len(vec_to_origin))]
            # order_list = np.array(order_list)[:,None]

            # pos_data = np.concatenate([vec_to_origin, order_list], axis=1)
            # pos_data = sorted( pos_data, key=functools.cmp_to_key(x_first) )
            # pos_data = np.array(pos_data)
            # rule_order = pos_data[:,-1].astype('int')

            ret = cross_points[rule_order[0]]

        elif self.pos_rule == 'far':  # 距离中心最远

            vec_to_center = centroids - bin_center
            dist_to_center = np.linalg.norm(vec_to_center, axis=1)
            rule_order = np.argsort(dist_to_center)[::-1]

            ret = cross_points[rule_order[0]]

        return list(ret), 200

    def set_search_step(self, search_step=None):
        # 如果search_step为空，那么，算法根据零件大小自己决定规划一步还是两步
        self.search_step = search_step

    def pack(self, packed_poly, container_poly, obj_poly, search_step=None):

        use_self_poly = False

        best_area = np.inf
        best_rot = None
        best_part = None
        best_offset = None
        best_success_step = -1

        best_code = None

        if search_step is None:
            use_self_poly = True
            if self.search_step is None:
                minx, miny, maxx, maxy = container_poly.bounds
                width = maxx - minx
                length = maxy - miny

                minx, miny, maxx, maxy = obj_poly.bounds
                w = maxx - minx
                l = maxy - miny

                poly_len = max(w, l)

                # 根据物体大小判断是否二次规划
                if poly_len >= min(width, length) / 2:
                    search_step = 2
                else:
                    search_step = 1
            else:
                search_step = self.search_step * 1

        search_step -= 1

        # 尝试多种旋转
        for ri, rot_ang in enumerate(self.rotate_angle):

            rot_part = affinity.rotate(obj_poly, rot_ang, origin=(0, 0))
            # 用于找位置的形状
            if self.shape_rule == 'polygon':
                pack_shape = rot_part
                offset = np.array([0, 0])

            elif self.shape_rule == 'rect':
                minx, miny, maxx, maxy = rot_part.bounds
                points = np.array([
                    [minx, miny],
                    [minx, maxy],
                    [maxx, maxy],
                    [maxx, miny]
                ])
                offset = points[0]
                pack_shape = Polygon(shell=points - offset)

            elif self.shape_rule == 'convex':
                convex_hull = rot_part.convex_hull
                convex_points = np.array(get_coords(convex_hull))
                offset = convex_points[0]
                pack_shape = Polygon(convex_points - offset)

            rot_part = affinity.translate(rot_part, -offset[0], -offset[1])  # 旋转后再移至理想空间中的原点位置

            # 计算与容器的可摆放界限 NFP
            # poly_inner = inner_fit_rect(container_poly, pack_shape)
            b = minkowski_diff(container_poly, pack_shape, False)
            poly_inner = container_poly.intersection(b)

            if not poly_inner or poly_inner.area == 0:
                code = 401
                if best_code is None:
                    best_code = code
                continue

            if packed_poly.area:
                # 与已有物体计算摆放界限 NFP
                poly_nfp = minkowski_diff(packed_poly, pack_shape)
                nfp = poly_inner.difference(poly_nfp)

                if not poly_nfp:
                    code = 402
                    if best_code is None:
                        best_code = code
                    continue
            else:
                nfp = poly_inner

            # 在这个NFP轮廓中取最接近的点
            new_part_pos, code = self.find_pos(nfp, pack_shape, container_poly)

            if best_code is None:
                best_code = code
            elif best_code == 402 and code == 403:
                best_code = code

            if code != 200:
                continue

            # 物体进行位移，加入摆放集合中
            # NOTE 上取整
            poly_offset = np.array([
                new_part_pos[0],
                new_part_pos[1]
            ])
            # poly_offset = np.array([
            #     new_part_pos[0] - offset[0],
            #     new_part_pos[1] - offset[1]
            # ])

            rot_part = affinity.translate(rot_part,
                                          xoff=poly_offset[0],
                                          yoff=poly_offset[1])

            # 加入已装箱合集，看摆放后的最小包围盒面积
            tmp_packed_poly = packed_poly.union(rot_part)

            bin_area = get_bbox_area(tmp_packed_poly)
            success_step = search_step

            if best_success_step < success_step or \
                    (bin_area < best_area and best_success_step == success_step):
                best_success_step = success_step  # 越大越好，说明可摆放的越多

                best_area = bin_area
                best_rot = rot_ang
                best_part = rot_part
                best_offset = poly_offset
                best_code = code

        return best_code, best_part, best_rot, best_area

    def calculate_location(
            self,
            container_ids,
            poly,
            reach_area=None,
            obj_gap=0,
            obj_shape=1,
            check_collision=1,
            rotate_angle=None,
            balance=0,
            search_next=1,
            img_path=None):
        '''
        Args:
            container_ids: list(int), 二维数组, 容器的编号[[1, 2, 3]]
            poly: Polygon, 零件的Polygon
            reach_area: [ min_point, max_point ], 垳架的可执行区域, 包括左下角和右上角的顶点位置
            obj_gap: 零件的膨胀大小, 单位mm
            obj_shape: int, 零件进行规划使用的形状
                0: 矩形
                1: 多边形
                2: 凸包
            check_collision: int, 是否考虑抓手与其他零件的碰撞
                0: 不考虑碰撞
                1: 考虑碰撞
            rotate_angle: float, 零件摆放时的旋转角度
            balance: int, 摆放是否考虑平衡
                0: 不考虑平衡, 优先往角落放置
                1: 考虑平衡, 优先填充4个角落
            search_next: int, 前置搜索，是否允许两步规划
                0: 不允许, 只进行一步规划
                1: 允许，当零件的最长边 > 容器最短边/2, 进行二次规划
            img_path: 保存结果图片的文件夹路径，默认不保存

        Returns:
            ret: dict, 包含执行结果码, 抓手中心位置和旋转角度, 信息

                计算结果的编号：200代表成功，4xx代表失败，200的情况返回零件位置，4xx代表异常
                抓手中心（bounding box）的放置位置、角度
                异常编号（补充）：
                    400：程序异常
                    401：零件长度大于料框
                    402：料框无位置存放零件
                    403：料框有位置但无可达区域
        '''

        try:
            # if True:
            if obj_shape == 0:
                self.shape_rule = 'rect'
                self.rotate_angle = [0, 90]

            elif obj_shape == 1:
                self.shape_rule = 'polygon'
                self.rotate_angle = [0, 90, 180, 270]

            elif obj_shape == 2:
                self.shape_rule = 'convex'
                self.rotate_angle = [0, 90, 180, 270]

            if check_collision == 0:
                self.check_collision = 0
            else:
                self.check_collision = 1

            if rotate_angle is not None:
                self.rotate_angle = [rotate_angle]

            if balance == 0:
                self.pos_rule = 'close'
            elif balance == 1:
                self.pos_rule = 'far'

            if search_next == 0:
                self.set_search_step(1)
            else:
                self.set_search_step(None)

            # obj_poly = img_to_poly(obj_img, self.simplify_level, obj_gap=obj_gap, scale=0.1 )
            obj = Part(self.pack_id, poly)

            container_list, c_index = self.get_containers(container_ids, obj.obj_size)

            if container_list is None:
                ret = {
                    'code': 400,
                    'center': [-1, -1, -1],
                    'msg': f"无法找到料框编号{c_index}对应的信息"
                }
                return ret, obj, None

            # 抓手可执行区域对应矩形
            min_point, max_point = reach_area
            minx, miny = min_point
            maxx, maxy = max_point
            self.reach_poly = Polygon(shell=[
                [minx, miny], [minx, maxy], [maxx, maxy], [maxx, miny]
            ])
            reach_area = [minx, miny, maxx, maxy]

            packed_containers = None
            best_part = None
            code = None

            for ci in range(len(container_list)):
                containers = container_list[ci]

                for i in range(len(containers)):
                    if not containers[i].can_pack:
                        code = 402
                        continue

                    container_poly = containers[i].poly
                    container_border_poly = containers[i].border_poly
                    packed_poly = containers[i].packed_poly
                    self.check_border = containers[i].check_border

                    code, best_part, best_rot, best_area = self.pack(packed_poly=packed_poly,
                                                                     container_poly=container_poly, obj_poly=obj.poly)

                    if best_part is not None:
                        packed_containers = containers[i]
                        break

                if best_part is not None:
                    break

            if best_part is not None:
                obj.update_poly(best_part)  # 这里确定摆放位置后，更新obj的信息：size and center
                self.update_packed_poly(obj, packed_containers)  # 将要装箱的零件更新到容器

                if packed_containers.can_pack == False:
                    msg = f"规划成功，容器{packed_containers.cid}已满"
                else:
                    msg = f"规划成功"

                # get_bbox_center(self.poly)
                # cm 转 mm 返回
                ret = {
                    'code': 200,
                    'center': [obj.obj_center[0], obj.obj_center[1], best_rot],
                    'msg': msg
                }

                self.pack_id += 1

                # 保存图片
                # if img_path is not None:
                #     os.makedirs(img_path, exist_ok=True)
                #     if type(packed_containers.cid) == int:
                #         cid_list = [packed_containers.cid]
                #     else:
                #         cid_list = packed_containers.cid
                #     for cid in cid_list:
                #         packed_gripper_polys, packed_gripper_center = self.get_packed_grippers(cid)
                #         c, _ = self.get_containers([[cid]])
                #         if len(packed_gripper_polys) > 0:
                #             packed_gripper_poly = [packed_gripper_polys[-1]]
                #         else:
                #             packed_gripper_poly = []
                #         plot_polygons(c[0], self.get_packed_polys(cid), packed_gripper_poly, color_last=True)
                #         plt.title(self.shape_rule + '_' + f"{cid}")
                #         if len(packed_gripper_center) > 0:
                #             plt.scatter(packed_gripper_center[-1, 0], packed_gripper_center[-1, 1])
                #         plt.savefig( os.path.join(img_path, f"{self.shape_rule}_{cid}.png"), dpi=100, bbox_inches='tight')
                #         plt.close()
                #
                #     plot_polygons(self.containers + [self.reach_poly], self.get_packed_polys(), [], color_last=True)
                #     plt.title(self.shape_rule)
                #     plt.savefig( os.path.join(img_path, f"{self.shape_rule}_global.png"), dpi=100, bbox_inches='tight')
                #     plt.close()


            else:
                if code == 400:
                    msg = "程序异常"
                if code == 401:
                    msg = "零件长度大于料框"
                elif code == 402:
                    msg = "料框无位置存放零件"
                elif code == 403:
                    msg = "料框有位置但无可达区域"

                ret = {
                    'code': code,
                    'center': [-1, -1, -1],
                    'msg': msg
                }
            return ret, obj, best_rot

        except Exception as e:
            print(e)
            ret = {
                'code': 400,
                'center': [-1, -1, -1],
                'msg': "程序异常"
            }
            return ret, None, None

    def remove_location(
            self,
            position
    ):
        '''
        Args:
            poistion: list(float), 单位mm
                calculate_location得到的抓手执行摆放的目标位置
        '''

        try:
            match_obj = False

            # mm 转 cm
            position = np.array(position) / 10.0

            for c in self.containers:
                match_obj = c.update_packed(None, 'remove')
                if match_obj:
                    break

            if not match_obj:
                ret = {
                    "code": 401,
                    "msg": "坐标不存在"
                }
            else:
                ret = {
                    "code": 200,
                    "msg": "成功撤销"
                }
        except Exception as e:
            print(e)
            ret = {
                "code": 400,
                "msg": "程序异常"
            }

        return ret

    def get_ratio(self):
        ret = []
        for c in self.containers:
            ret.append(c.area_rate)
            ret.append(c.convex_rate)
        return ret

    # def show(self):
    #     packed_objs = self.get_packed_polys()
    #     packed_grippers, packed_gripper_centers = self.get_packed_grippers()
    #     plot_polygons(self.containers + [self.reach_poly], packed_objs, packed_grippers, color_last=True)
    #     plt.scatter(packed_gripper_centers[:,0], packed_gripper_centers[:,1])
    #     plt.show()
    #     plt.close()


# algorithm

def inner_fit_rect(poly_a, poly_b):
    (min_x_a, min_y_a, max_x_a, max_y_a) = poly_a.bounds  # bin
    (min_x_b, min_y_b, max_x_b, max_y_b) = poly_b.bounds  # part

    if max_x_a - min_x_a < max_x_b - min_x_b or \
            max_y_a - min_y_a < max_y_b - min_y_b:
        return None

    coords_b = poly_b.exterior.coords[0]
    min_x = min_x_a - min_x_b + coords_b[0]
    min_y = min_y_a - min_y_b + coords_b[1]
    max_x = max_x_a - max_x_b + coords_b[0]
    max_y = max_y_a - max_y_b + coords_b[1]

    poly = Polygon(
        shell=[[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]],
        holes=[])
    return poly


def minkowski_diff(poly_a, poly_b, is_exterior=True):
    '''
    Args:
        poly_a: Polygon, 固定不动的多边形
        poly_b: Polygon, 寻找和poly_a相切的移动多边形
        is_exterior: 寻找外切/内切的NFP NOTE: 
            如果poly_b放得进poly_a, 那么get_minkowskis里面至少获得2个多边形,最大的是外切的NFP，剩下的是内切
            但是poly_b放不进poly_a, 那么get_minkowskis*应该*里面只能获得一个多边形
    
    Returns:
        NFP: Polygon, 相切多边形no-fit polygon 
    '''

    def get_minkowski(shell_a, shell_b):
        minkowskis = pyclipper.MinkowskiDiff(shell_b, shell_a)
        polys = [
            Polygon(shell=mink, holes=[]) for mink in minkowskis
        ]
        order_polys = sorted(polys, key=lambda x: x.area)
        if is_exterior:
            ret_poly = order_polys[-1]
            ret_poly = ret_poly.buffer(0)
        else:
            if len(polys) < 2:
                return None
            # ret_poly = min(polys, key=lambda x: x.area)
            # ret_poly = order_polys[-2]
            ret_poly = Polygon()
            for op in order_polys[:-1]:
                ret_poly = ret_poly.union(op.buffer(0))

        return ret_poly

    shell_b = get_coords(poly_b)

    # no-fit plane
    NFP = Polygon(shell=[], holes=[])

    if poly_a.type == 'MultiPolygon':
        for geo in poly_a.geoms:
            shell_a = get_coords(geo)
            minkowskis = get_minkowski(shell_a, shell_b)

            if minkowskis is None:
                return Polygon(shell=[], holes=[])

            NFP = NFP.union(minkowskis)
    else:
        shell_a = get_coords(poly_a)
        minkowskis = get_minkowski(shell_a, shell_b)

        if minkowskis is None:
            return Polygon(shell=[], holes=[])

        NFP = NFP.union(minkowskis)

    # NFP = NFP.simplify(3)
    # print(len(NFP.exterior.coords))

    return NFP


# test

def load_imgs(data_path, num, case_id=0):
    if case_id == 0:
        candidate_sets = os.listdir(data_path)[:300]
    elif case_id == 1:
        # candidate_sets = ['13130030.png'] # 内凹的

        # candidate_sets = ['14240456.png']
        # candidate_sets = ['04022006(3)@1605.png']
        # candidate_sets = ['02022901-4@72.png']

        candidate_sets = ['20220622021.png']  # 点多的
        # candidate_sets = ['XFC005502370.png']  # 凹凸点多的
        # candidate_sets = ['003077117A0221021.png']  # 点多的

        # candidate_sets = ['BCB005344676.png']
        # candidate_sets = ['BCB007803989.png']
        candidate_sets = ['BCB006490571.png']  # 突出的

        # candidate_sets = ['BCB006154566.png']

        # candidate_sets = ['22.png'] #大的
        # candidate_sets = ['BCB006504119.png'] #大长的
        # candidate_sets = ['BCB006503558_2.png'] #大长的
        # candidate_sets = ['11645840.png'] #大长的
        # candidate_sets = ['BCB005343276.png'] #的

    else:
        candidate_sets = ['BCB006504119.png', "BCB006490571.png", '20220622021.png', 'BCB005343276.png']
        # candidate_sets = [
        #     "10.png", "5.png", "6.png", "1.png", "16.png",
        # ]

    obj_ids = np.random.choice(len(candidate_sets), num)
    imgs = [
        os.path.join(data_path, candidate_sets[i]) for i in obj_ids
    ]
    return imgs
