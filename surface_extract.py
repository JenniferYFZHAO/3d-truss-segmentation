import open3d as o3d
import numpy as np
from copy import deepcopy

if __name__ == '__main__':
    file_path = 'rabbit.pcd'
    pcd = o3d.io.read_point_cloud(file_path)
    pcd = pcd.uniform_down_sample(50)  # 每50个点采样一次
    pcd.paint_uniform_color([0.5, 0.5, 0.5])  # 指定显示为灰色
    print(pcd)

    pcd3 = deepcopy(pcd)
    pcd3.translate((0, 20, 0))  # 整体进行y轴方向平移20
    radius = 0.01  # 搜索半径
    max_nn = 10  # 邻域内用于估算法线的最大点数
    pcd3.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius, max_nn))
    mesh3, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd3, depth=9)
    vertices_to_remove = densities < np.quantile(densities, 0.35)
    mesh3.remove_vertices_by_mask(vertices_to_remove)
    mesh3.paint_uniform_color([1, 0, 0])

    o3d.visualization.draw_geometries([pcd, mesh3],  # 点云列表
                                      window_name="Poisson 重建",
                                      point_show_normal=False,
                                      width=800,  # 窗口宽度
                                      height=600,
                                      mesh_show_wireframe=True,
                                      mesh_show_back_face=True,
                                      )  # 窗口高度