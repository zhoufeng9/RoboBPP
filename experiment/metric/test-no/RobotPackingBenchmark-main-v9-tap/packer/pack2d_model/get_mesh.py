import os

import numpy as np
import trimesh
import cv2

from shapely.geometry import Polygon, MultiPolygon
from shapely import affinity
from shapely.ops import unary_union, cascaded_union

def get_mesh(num, polygon, height):
    # 获取多边形的外环和内环（如果有的话）
    exterior_coords = list(polygon.exterior.coords)
    interior_coords = [list(interior.coords) for interior in polygon.interiors]



    # 创建 3D 的顶点
    vertices = []
    for z in [0, height]:
        for coord in exterior_coords:
            vertices.append([coord[0], coord[1], z])
        for interior in interior_coords:
            for coord in interior:
                vertices.append([coord[0], coord[1], z])

    # 创建面
    faces = []

    # 外环的面
    num_ext_coords = len(exterior_coords)
    for i in range(num_ext_coords - 1):
        faces.append([i, i + 1, i + num_ext_coords])
        faces.append([i + num_ext_coords, i + 1, i + num_ext_coords + 1])

    # 内环的面（如果有的话）
    offset = num_ext_coords * 2
    for interior in interior_coords:
        num_int_coords = len(interior)
        for i in range(num_int_coords - 1):
            faces.append([offset + i, offset + i + 1, offset + i + num_int_coords])
            faces.append([offset + i + num_int_coords, offset + i + 1, offset + i + num_int_coords + 1])
        offset += num_int_coords * 2

    # 顶面和底面的面
    for i in range(1, num_ext_coords - 1):
        faces.append([0, i + 1, i])     # 反向添加，正常为 faces.append([0, i, i + 1])
        faces.append([num_ext_coords, num_ext_coords + i, num_ext_coords + i + 1])


    # 创建 Trimesh 的 mesh
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    # # 可视化 mesh
    # mesh.show()

    # 保存 mesh 为 obj 文件
    # mesh.export('polygon_mesh.obj')
    mesh.export(mesh_path + str(num) +'.obj')


def img_to_poly(img, simplify_level=3, scale=0.1, obj_gap=None, align_to='origin'):
    # 图像转多边形

    binary = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(binary, 128, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
    for ct in contours:
        if len(ct) > 2:
            points = ct[:, 0]
            break

    points = np.array(points)
    # 图像（Y朝下）转为世界坐标系（Y朝上）
    points[:, 1] = -points[:, 1]

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


def get_bbox_center(poly):
    minx, miny, maxx, maxy = poly.bounds
    center = [ minx + (maxx - minx)/2, miny + (maxy - miny)/2 ]

    return np.array(center)


# # 假设你有一个 shapely 的多边形
# polygon = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
# # 定义延伸的高度
# height = 1
# mesh_path = '/home/wzf/Workspace/rl/pack2d'
#
# get_mesh(999, polygon, height)


data_paths = "./data/large_template"
height = 2
mesh_path = '/home/wzf/Workspace/rl/pack2d/data/mesh/'
for i, data in enumerate(os.listdir(data_paths)):
    img_path = data_paths + '/' + data
    print(img_path)
    img = cv2.imread(img_path)
    poly = img_to_poly(img, simplify_level=5, obj_gap=20, scale=0.1, align_to='origin')
    if isinstance(poly, MultiPolygon):
        outer_polygon = cascaded_union(poly)
    elif isinstance(poly, Polygon):
        outer_polygon = poly
    else:
        raise TypeError("The geometry must be a Polygon or MultiPolygon")

    # print("here")
    get_mesh(i, outer_polygon, height)
print('done')


# img_path = './data/large_template/BCB006277996.png'
# img = cv2.imread(img_path)
# poly = img_to_poly(img, simplify_level=5, obj_gap=20, scale=0.1, align_to='origin')
# # print("here")
# if isinstance(poly, MultiPolygon):
#     outer_polygon = cascaded_union(poly)
# elif isinstance(poly, Polygon):
#     outer_polygon = poly
# else:
#     raise TypeError("The geometry must be a Polygon or MultiPolygon")
#
#
# print(outer_polygon)
# # outer_polygon = cascaded_union(multi_polygon)
# get_mesh(99999, outer_polygon, height)