import open3d as o3d
import numpy as np
from copy import deepcopy
from generation import generate_truss_point_cloud

if __name__ == '__main__':
    # 1. 定义桁架节点和构件
    nodes_coords_dict = {
        0: [0.0, 0.0, 0.0],
        1: [5.0, 0.0, 2.5],
        2: [5.0, 5.0, 5.0],
        3: [0.0, 5.0, 7.5]
    }
    
    member_connectivity = [
        (0, 1),  # Member 0
        (1, 2),  # Member 1
        (2, 3),  # Member 2
    ]
    
    # 2. 使用 generation.py 生成圆管点云
    print("Generating cylindrical truss point cloud...")
    point_cloud_np, _ = generate_truss_point_cloud(
        nodes_coords_dict=nodes_coords_dict,
        member_connectivity=member_connectivity,
        points_per_member=50,      # 沿杆件长度的点数
        radius=0.3,                # 圆管半径
        points_per_circle=15,       # 每个圆周上的点数
        noise_std=0.01,            # 减小噪声
        num_noise_points=0         # 不添加额外噪声点
    )
    
    # 3. 将 numpy 数组转换为 Open3D 点云格式
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud_np)
    pcd.paint_uniform_color([0.5, 0.5, 0.5])  # 指定显示为灰色
    print(pcd)

    pcd3 = deepcopy(pcd)
    # 调整参数以适合桁架结构
    radius = 0.5  # 增大搜索半径，适应桁架结构
    max_nn = 30  # 增加邻域点数
    pcd3.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius, max_nn))
    # 调整法线方向，使其朝外
    pcd3.orient_normals_consistent_tangent_plane(30)
    
    # Poisson 重建 - 调整深度参数
    mesh3, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd3, depth=8)
    # 调整移除顶点的阈值
    vertices_to_remove = densities < np.quantile(densities, 0.1)  # 降低阈值，保留更多顶点
    mesh3.remove_vertices_by_mask(vertices_to_remove)
    mesh3.paint_uniform_color([1, 0.7, 0])  # 橙色显示

    # 可视化1：先只展示原始点云
    print("\n=== 第一步：展示原始点云 ===")
    print("关闭此窗口后将展示表面重建结果...")
    o3d.visualization.draw_geometries([pcd],
                                      window_name="Original Point Cloud",
                                      point_show_normal=False,
                                      width=1200,
                                      height=800)
    
    # 可视化2：展示原始点云和重建的网格
    print("\n=== 第二步：展示表面重建结果 ===")
    print("灰色：原始点云")
    print("橙色：重建的表面网格")
    o3d.visualization.draw_geometries([pcd, mesh3],
                                      window_name="Truss Poisson Reconstruction",
                                      point_show_normal=False,
                                      width=1200,
                                      height=800,
                                      mesh_show_wireframe=True,
                                      mesh_show_back_face=True)