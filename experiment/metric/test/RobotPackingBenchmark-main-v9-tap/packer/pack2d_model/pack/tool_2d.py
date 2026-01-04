import numpy as np
from shapely.geometry import Polygon
from shapely import affinity
import yaml
import cv2
import copy

import matplotlib.pyplot as plt
import trimesh
from scipy.spatial import ConvexHull

from functools import cmp_to_key
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.collections import PatchCollection


def get_poly(obj_path):
    # 导入 .obj 文件
    mesh = trimesh.load(obj_path)
    # 获取顶面的顶点
    z_max = mesh.vertices[:, 2].max()
    top_vertices_indices = mesh.vertices[:, 2] == z_max
    top_vertices = mesh.vertices[top_vertices_indices][:, :2]  # 获取x, y坐标

    # 使用scipy的ConvexHull来获取顶面顶点的凸包
    hull = ConvexHull(top_vertices)
    hull_vertices = top_vertices[hull.vertices]

    # 计算质心
    centroid = np.mean(hull_vertices, axis=0)

    # 以质心为参考点进行顺时针排序
    def angle_from_centroid(p):
        return np.arctan2(p[1] - centroid[1], p[0] - centroid[0])

    sorted_vertices = sorted(hull_vertices, key=angle_from_centroid, reverse=True)

    # 转换为列表并检查是否包含[0, 0]
    sorted_vertices = [vertex.tolist() for vertex in sorted_vertices]

    # 确保第一个顶点为[0, 0]，如果[0, 0]不在顶点列表中，插入[0, 0]
    if not any(np.all(np.isclose(vertex, [0, 0])) for vertex in sorted_vertices):
        sorted_vertices = [[0, 0]] + sorted_vertices

    # 确保最后一个顶点回到[0, 0]
    if not np.all(np.isclose(sorted_vertices[-1], [0, 0])):
        sorted_vertices.append([0, 0])

    poly = Polygon(sorted_vertices)
    # print(poly)
    # print("done!")
    return poly


def get_bbox_area(poly):
    poly_bbox_area = 0

    if poly.type=='Polygon':
        minx, miny, maxx, maxy = poly.bounds
        poly_bbox_area = (maxx - minx) * (maxy - miny)
        
    if poly.type=='MultiPolygon':
        for geo in poly.geoms:
            minx, miny, maxx, maxy = geo.bounds
            geo_area = (maxx - minx) * (maxy - miny)
            poly_bbox_area += geo_area
    
    return poly_bbox_area

def get_bbox_center(poly):
    minx, miny, maxx, maxy = poly.bounds
    center = [ minx + (maxx - minx)/2, miny + (maxy - miny)/2 ]

    return np.array(center)

def get_bbox_size(poly):
    minx, miny, maxx, maxy = poly.bounds
    length = maxx - minx
    width = maxy - miny
    return length, width

def get_coords(geometry):
    if geometry.type=='Polygon':
        return list( geometry.exterior.coords )
    if geometry.type=='MultiPolygon':
        all_xy = []
        for ea in geometry.geoms:
            all_xy += list( ea.exterior.coords )
        return all_xy
    if geometry.type=='LineString':
        return list( geometry.coords )
        # return list( poly.exterior.coords )

def img_to_poly(img, simplify_level=3, scale=0.1, obj_gap=None, align_to='origin'):
    # 图像转多边形

    binary = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(binary,128,255,cv2.THRESH_BINARY)
    
    contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_TC89_KCOS)
    for ct in contours:
        if len(ct) > 2:
            points = ct[:,0]
            break
    
    points = np.array(points)
    # 图像（Y朝下）转为世界坐标系（Y朝上）
    points[:,1] = -points[:,1]

    poly = Polygon(points)
    
    if obj_gap is not None:
        poly = poly.buffer(obj_gap, resolution=-2, single_sided=True)
        
    poly = poly.simplify(simplify_level)

    # 每个像素1毫米，缩小为厘米单位的
    poly = affinity.scale(poly, scale, scale)

    poly = poly.simplify(simplify_level)
        
    points = get_coords(poly)

    if align_to == 'origin':
        # 保证第一个点是（0，0）
        poly = affinity.translate(poly, -points[0][0], -points[0][1])
    elif align_to == 'center':
        # 将包围盒中心设置为原点
        center = get_bbox_center(poly)
        poly = affinity.translate(poly, -center[0], -center[1])
    
    # print(np.array(get_coords(poly)).shape)
    return poly


# 可视化

def plot_polygons(poly_bins, poly_parts, poly_other=None, ax=None, title='', show_id=True, color_last=False):
    if ax is None:
        fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_aspect('equal')
    
    for pi, poly_bin in enumerate(poly_bins):

        if type(poly_bin) == Container_2d:
            c = [0,0,0]
            poly = poly_bin.poly
            # print(poly_bin.area_rate, poly_bin.convex_rate)
            minx, miny, maxx, maxy = poly_bin.poly.bounds
            ax.text( minx+30, miny+40, poly_bin.cid, color='black', bbox ={'edgecolor':c, 'facecolor':[0,0,0,0]} )
            linestyle = '-'
        else:
            c = [0.6, 0.2, 0.3]
            poly = poly_bin
            linestyle = '--'

        plot_polygon(ax,
                     poly,
                     facecolor='None',
                     edgecolor=c,
                     linestyle=linestyle,
                     alpha=1)
    poly_part = None
    max_poly_id = 0
    max_pi = 0
    for pi, poly_part in enumerate(poly_parts):
        if type(poly_part) == Part:
            if max_poly_id < poly_part.pid:
                max_poly_id = poly_part.pid
                max_pi = pi
        
        if color_last:
            c = (0.7, 0.7, 0.7)
        else:
            c = np.random.rand(3)

        if show_id == False:
            pi = None
        plot_polygon(ax,
                     poly_part,
                     facecolor=c,
                     alpha=0.5, poly_id=pi)

    if color_last and poly_part is not None:
        plot_polygon(ax,
                     poly_parts[max_pi],
                     facecolor='blue',
                     alpha=1, poly_id=max_poly_id)
    if poly_other:
        for pi, poly_part in enumerate(poly_other):
            plot_polygon(ax,
                        poly_part,
                        facecolor='red',
                        alpha=0.3)
    return ax

def plot_polygon(ax, poly, poly_id=None, **kwargs):
    # Plots a Polygon to pyplot `ax`
    # cf. https://stackoverflow.com/questions/55522395

    if type(poly) == Part:
        poly_id = poly.pid
        poly = poly.poly
        
    if poly.type == "MultiPolygon":
        for polygon_tmp in poly.geoms:
            plot_polygon(ax, polygon_tmp, **kwargs)
        return

    path = Path.make_compound_path(
        Path(np.asarray(poly.exterior.coords)[:, :2]),
        *[Path(np.asarray(ring.coords)[:, :2]) for ring in poly.interiors])

    patch = PathPatch(path, **kwargs)
    collection = PatchCollection([patch], **kwargs)

    if poly_id is not None:
        center = np.mean(np.asarray(poly.exterior.coords)[:, :2], axis=0)
        ax.text( center[0], center[1], poly_id )
    
    ax.add_collection(collection, autolim=True)
    ax.autoscale_view()

class Part():
    def __init__(self, pid, poly) -> None:
        self.pid = pid
        self.update_poly(poly)

        # 属于哪个容器
        self.cids = []

    def update_poly(self, poly):
        self.poly = poly
        self.obj_size = get_bbox_size(self.poly)
        self.obj_center = get_bbox_center(self.poly)


class Container_2d():
    def __init__(self, cid, img, pos, check_border, poly=None, border_poly=None, max_rate=1):
        self.cid = cid
        self.img = img

        self.pos = pos
        self.check_border = check_border
        self.new_container(poly, border_poly)

        self.max_rate = max_rate
        self.area_rate = 0
        self.convex_rate = 0

    def update_packed(self, obj, update_type='add'):
        if update_type == 'add':
            self.packed_objs.append(obj)

            self.packed_poly = self.packed_poly.union(obj.poly)

        self.check_rate()
        return True

    def check_rate(self):
        
        self.area_rate = self.packed_poly.area / self.poly.area
        packed_convex = self.packed_poly.convex_hull
        self.convex_rate = self.poly.intersection(packed_convex).area / self.poly.area
    
        if self.convex_rate >= self.max_rate:
            self.can_pack = False
        else:
            self.can_pack = True

        return self.can_pack

    def new_container(self, poly=None, border_poly=None):
        if poly is None:
            poly = img_to_poly(self.img, scale=1, align_to='origin')
            
            minx, miny, maxx, maxy = poly.bounds
            points = np.array([
                [minx, miny],
                [minx, maxy],
                [maxx, maxy],
                [maxx, miny],
            ])
            poly = Polygon(points - points[0])
            poly = affinity.translate(poly, self.pos[0], self.pos[1])

        minx, miny, maxx, maxy = poly.bounds
        self.length = maxx - minx
        self.width = maxy - miny

        self.poly = poly
        self.border_poly = border_poly

        self.packed_poly = Polygon()
        
        self.packed_objs = []
        
        self.can_pack = True
    

class Config():
    def __init__(self) -> None:
        pass

    def load_config(self, config_path):
        f = open(config_path)
        data = yaml.load(f, yaml.BaseLoader)

        containers_data = data['container']

        if 'max_rate' in data:
            max_rate = float(data['max_rate'])
        else:
            max_rate = 1
        
        if 'cornre' in data:
            corner_data = data['corner']
            pos = corner_data[1:-1]
            x, y = pos.split(',')
            pos = np.array([int(x), int(y)])
            corner = pos
        else:
            corner = np.array([0,0])


        containers = []
        cids = []

        for c in containers_data:
            cid = int(c['id'])
            img = cv2.imread(c['img'])

            pos = c['pos'][1:-1]
            x, y = pos.split(',')
            pos = np.array([int(x), int(y)])
            
            check_border = int(c['border'])
            container = Container_2d(cid, img, pos, check_border, max_rate=max_rate)

            containers.append(container)
            cids.append(cid)
        

        return containers, cids, corner

