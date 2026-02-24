import open3d as o3d
import numpy as np
from copy import deepcopy
from read_cloudpoints import read_las_point_cloud, get_las_info


def extract_surface_from_las(las_file_path, subsample_factor=10, 
                            normal_radius=0.5, normal_max_nn=30,
                            poisson_depth=8, density_quantile=0.1,
                            show_progress=True, scale_factor=1.0):
    """
    从 .las 文件读取点云并提取表面。
    
    Args:
        las_file_path (str): .las 文件路径
        subsample_factor (int): 下采样因子，用于控制点云数量
        normal_radius (float): 法线估计的搜索半径
        normal_max_nn (int): 法线估计的最大邻域点数
        poisson_depth (int): Poisson 重建的深度参数（越大越精细但计算量越大）
        density_quantile (float): 密度阈值分位数（移除低密度顶点）
        show_progress (bool): 是否显示进度
        scale_factor (float): 缩放参数，0 表示自动计算，1.0 表示默认缩放
        
    Returns:
        tuple: (原始点云, 重建的网格)
    """
    
    # 存储缩放参数以便后续使用
    global current_scale_factor
    current_scale_factor = scale_factor
    
    # 1. 读取点云
    print("="*60)
    print("第一步：读取点云")
    print("="*60)
    
    # 先获取文件信息
    info = get_las_info(las_file_path)
    print(f"文件信息:")
    print(f"  点数: {info['point_count']:,}")
    print(f"  X 范围: [{info['x_min']:.3f}, {info['x_max']:.3f}]")
    print(f"  Y 范围: [{info['y_min']:.3f}, {info['y_max']:.3f}]")
    print(f"  Z 范围: [{info['z_min']:.3f}, {info['z_max']:.3f}]")
    
    # 根据点数自动调整下采样比例
    estimated_points = info['point_count']
    if subsample_factor is None:
        if estimated_points > 100_000_000:
            subsample_factor = 100
        elif estimated_points > 50_000_000:
            subsample_factor = 50
        elif estimated_points > 10_000_000:
            subsample_factor = 20
        elif estimated_points > 1_000_000:
            subsample_factor = 10
        else:
            subsample_factor = 5
    
    print(f"\n使用下采样比例: {subsample_factor}")
    print(f"预计读取点数: ~{estimated_points // subsample_factor:,}")
    
    # 读取点云
    point_cloud_np = read_las_point_cloud(
        las_file_path, 
        subsample_factor=subsample_factor,
        show_progress=show_progress
    )
    
    # 2. 转换为 Open3D 点云格式
    print("\n" + "="*60)
    print("第二步：转换点云格式")
    print("="*60)
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud_np)
    print(f"点云对象创建成功，点数: {len(pcd.points):,}")
    
    # 预处理：移除重复点和离群点
    print("\n预处理点云...")
    
    # 移除重复点
    original_points = len(pcd.points)
    pcd = pcd.remove_duplicated_points()
    removed_duplicates = original_points - len(pcd.points)
    if removed_duplicates > 0:
        print(f"移除重复点: {removed_duplicates:,} 个")
    
    # 移除统计离群点
    pcd_clean, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    removed_outliers = len(pcd.points) - len(pcd_clean.points)
    if removed_outliers > 0:
        print(f"移除离群点: {removed_outliers:,} 个")
    
    pcd = pcd_clean
    pcd.paint_uniform_color([0.5, 0.5, 0.5])
    print(f"预处理完成，最终点数: {len(pcd.points):,}")
    
    # 3. 估计法线
    print("\n" + "="*60)
    print("第三步：估计法线")
    print("="*60)
    print(f"搜索半径: {normal_radius}")
    print(f"最大邻域点数: {normal_max_nn}")
    
    pcd_normals = deepcopy(pcd)
    
    # 先尝试估计法线
    try:
        pcd_normals.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(normal_radius, normal_max_nn)
        )
        
        # 尝试一致性法线方向（可能失败）
        try:
            pcd_normals.orient_normals_consistent_tangent_plane(30)
            print("法线估计完成")
        except RuntimeError as e:
            if "qhull" in str(e).lower() or "topology error" in str(e).lower():
                print("警告: 一致性法线方向调整失败（点云可能存在重复点）")
                print("使用替代方法：基于视点方向")
                # 使用基于视点的方向调整
                pcd_normals.orient_normals_towards_camera_location(
                    camera_location=pcd_normals.get_center() + np.array([0, 0, 100])
                )
                print("法线估计完成（使用替代方法）")
            else:
                raise
    except Exception as e:
        print(f"法线估计出错: {e}")
        raise
    
    # 4. Poisson 表面重建
    print("\n" + "="*60)
    print("第四步：Poisson 表面重建")
    print("="*60)
    print(f"重建深度: {poisson_depth}")
    print("正在重建网格（这可能需要一些时间）...")
    
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd_normals, 
        depth=poisson_depth
    )
    
    print(f"重建完成!")
    print(f"  顶点数: {len(mesh.vertices):,}")
    print(f"  三角面数: {len(mesh.triangles):,}")
    
    # 5. 移除低密度顶点
    print("\n" + "="*60)
    print("第五步：移除低密度顶点")
    print("="*60)
    print(f"密度阈值分位数: {density_quantile}")
    
    vertices_to_remove = densities < np.quantile(densities, density_quantile)
    mesh.remove_vertices_by_mask(vertices_to_remove)
    print(f"移除顶点数: {np.sum(vertices_to_remove):,}")
    print(f"剩余顶点数: {len(mesh.vertices):,}")
    
    mesh.paint_uniform_color([1, 0.7, 0])
    
    return pcd, mesh


if __name__ == '__main__':
    # 默认文件路径
    default_las_file = r"D:\1-学术\预拼装\结构模型\泉州项目点云及模型\扫描点云数据\44-unset.las"
    
    import os
    import sys
    
    # 检查命令行参数
    if len(sys.argv) > 1:
        las_file_path = sys.argv[1]
    else:
        las_file_path = default_las_file
    
    # 检查文件是否存在
    if not os.path.exists(las_file_path):
        print(f"错误: 文件不存在 - {las_file_path}")
        sys.exit(1)
    
    print(f"处理文件: {las_file_path}")
    print()
    
    # 设置参数（可根据文件大小调整）
    # 对于大文件，建议：
    # - subsample_factor: 10-100（控制点云数量）
    # - poisson_depth: 6-10（控制网格精细度，越大越慢）
    # - normal_radius: 根据点云密度调整
    # - scale_factor: 0.5-5.0（控制可视化缩放）
    
    # 询问用户缩放参数
    print("\n" + "="*60)
    print("选择缩放参数")
    print("="*60)
    print("缩放参数控制可视化时的初始视角缩放:")
    print("建议值:")
    print("  - 紧密视图: 0.5-1.0")
    print("  - 适中视图: 1.0-2.0")
    print("  - 全局视图: 2.0-5.0")
    print("  - 自动缩放: 0 (自动计算)")
    
    recommended_scale = 1.0
    
    print(f"\n推荐缩放参数: {recommended_scale}")
    
    # 获取用户输入
    try:
        user_input = input(f"请输入缩放参数 [默认: {recommended_scale}]: ").strip()
        if user_input:
            scale_factor = float(user_input)
        else:
            scale_factor = recommended_scale
    except KeyboardInterrupt:
        print("\n用户取消操作")
        sys.exit(0)
    except:
        print(f"输入无效，使用默认值: {recommended_scale}")
        scale_factor = recommended_scale
    
    try:
        # 执行表面提取
        pcd, mesh = extract_surface_from_las(
            las_file_path=las_file_path,
            subsample_factor=5,          # 下采样比例（增大以减少点数）
            normal_radius=0.1,            # 法线估计半径（增大以适应稀疏点云）
            normal_max_nn=30,             # 法线估计最大邻域点数（增大）
            poisson_depth=10,              # Poisson 重建深度（降低以减少计算量）
            density_quantile=0.005,        # 密度阈值（降低以保留更多顶点）
            show_progress=True,
            scale_factor=scale_factor     # 缩放参数
        )
        
        # 可视化1：展示原始点云
        print("\n" + "="*60)
        print("可视化：原始点云")
        print("="*60)
        print("提示: 按 'Q' 或 'ESC' 键关闭窗口")
        print("关闭此窗口后将展示表面重建结果...")
        
        o3d.visualization.draw_geometries(
            [pcd],
            window_name="Original Point Cloud",
            point_show_normal=False,
            width=1200,
            height=800
        )
        
        # 可视化2：展示原始点云和重建的网格
        print("\n" + "="*60)
        print("可视化：表面重建结果")
        print("="*60)
        print("灰色：原始点云")
        print("橙色：重建的表面网格")
        print("提示: 按 'Q' 或 'ESC' 键关闭窗口")
        
        o3d.visualization.draw_geometries(
            [pcd, mesh],
            window_name="Surface Reconstruction",
            point_show_normal=False,
            width=1200,
            height=800,
            mesh_show_wireframe=True,
            mesh_show_back_face=True
        )
        
        print("\n" + "="*60)
        print("处理完成!")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n用户取消操作")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
