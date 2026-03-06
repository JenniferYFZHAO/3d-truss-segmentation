import open3d as o3d
import numpy as np
from copy import deepcopy
import os
import sys

from read_cloudpoints import read_las_point_cloud, get_las_info, calculate_centroid, translate_to_origin, visualize_point_cloud


# ============================================================
# 参数配置区域 - 手动修改以下参数以适应不同的点云
# ============================================================

# 离群点移除参数
OUTLIER_NB_NEIGHBORS = 20      # 邻域点数，值越大越严格 (推荐: 10-50)
OUTLIER_STD_RATIO = 5        # 标准差比率，值越小越严格 (推荐: 1.0-3.0)

# 法线估计参数
NORMAL_RADIUS = 0.005            # 搜索半径，高密度点云用小值 (推荐: 0.01-0.5)
NORMAL_MAX_NN = 10             # 最大邻域点数 (推荐: 20-50)

# Poisson 表面重建参数
POISSON_DEPTH = 12             # 八叉树深度，值越大网格越精细但计算量越大 (推荐: 6-12)

# 密度过滤参数
DENSITY_QUANTILE = 0.005       # 密度阈值分位数，值越小保留越多顶点 (推荐: 0.001-0.1)

# ============================================================


def main():
    default_las_file = r"D:\1-学术\预拼装\结构模型\泉州项目点云及模型\扫描点云数据\44-unset.las"
    
    if len(sys.argv) > 1:
        las_file_path = sys.argv[1]
    else:
        las_file_path = default_las_file
    
    if not os.path.exists(las_file_path):
        print(f"错误: 文件不存在 - {las_file_path}")
        sys.exit(1)
    
    print(f"处理文件: {las_file_path}")
    print()
    
    try:
        # 1. 获取文件信息
        print("="*60)
        print("第一步：获取文件信息")
        print("="*60)
        info = get_las_info(las_file_path)
        print(f"点数: {info['point_count']:,}")
        print(f"X 范围: [{info['x_min']:.3f}, {info['x_max']:.3f}]")
        print(f"Y 范围: [{info['y_min']:.3f}, {info['y_max']:.3f}]")
        print(f"Z 范围: [{info['z_min']:.3f}, {info['z_max']:.3f}]")
        
        # 2. 询问下采样比例
        print("\n" + "="*60)
        print("第二步：选择下采样比例")
        print("="*60)
        print("建议值: 快速预览(100-200), 中等质量(50-100), 高质量(10-50)")
        
        estimated_points = info['point_count']
        if estimated_points > 50_000_000:
            recommended = 100
        elif estimated_points > 10_000_000:
            recommended = 50
        elif estimated_points > 1_000_000:
            recommended = 20
        else:
            recommended = 10
        
        user_input = input(f"请输入采样比例 [默认: {recommended}]: ").strip()
        subsample_factor = int(user_input) if user_input else recommended
        
        # 3. 读取点云
        print("\n" + "="*60)
        print("第三步：读取点云")
        print("="*60)
        point_cloud_np = read_las_point_cloud(
            las_file_path, 
            subsample_factor=subsample_factor,
            show_progress=True
        )
        
        # 4. 质心计算与平移
        print("\n" + "="*60)
        print("第四步：质心计算与平移")
        print("="*60)
        centroid = calculate_centroid(point_cloud_np)
        print(f"点云质心坐标: ({centroid[0]:.3f}, {centroid[1]:.3f}, {centroid[2]:.3f})")
        
        translate = input("是否将质心平移到坐标原点？(y/n) [默认: n]: ").strip().lower()
        if translate == 'y' or translate == 'yes':
            point_cloud_np, original_centroid = translate_to_origin(point_cloud_np, centroid)
            print(f"已将点云平移，质心移至原点 (0, 0, 0)")
            print(f"平移后范围: X[{point_cloud_np[:, 0].min():.1f}, {point_cloud_np[:, 0].max():.1f}], "
                  f"Y[{point_cloud_np[:, 1].min():.1f}, {point_cloud_np[:, 1].max():.1f}], "
                  f"Z[{point_cloud_np[:, 2].min():.1f}, {point_cloud_np[:, 2].max():.1f}]")
        else:
            print("保持原始坐标不变")
        
        # 5. 转换为 Open3D 点云并预处理
        print("\n" + "="*60)
        print("第五步：预处理点云")
        print("="*60)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud_np)
        print(f"原始点数: {len(pcd.points):,}")
        
        pcd = pcd.remove_duplicated_points()
        pcd_clean, _ = pcd.remove_statistical_outlier(
            nb_neighbors=OUTLIER_NB_NEIGHBORS, 
            std_ratio=OUTLIER_STD_RATIO
        )
        pcd = pcd_clean
        print(f"预处理后点数: {len(pcd.points):,}")
        
        # 6. 估计法线
        print("\n" + "="*60)
        print("第六步：估计法线")
        print("="*60)
        print(f"参数: radius={NORMAL_RADIUS}, max_nn={NORMAL_MAX_NN}")
        pcd_normals = deepcopy(pcd)
        pcd_normals.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(NORMAL_RADIUS, NORMAL_MAX_NN)
        )
        try:
            pcd_normals.orient_normals_consistent_tangent_plane(30)
            print("法线估计完成")
        except RuntimeError:
            print("使用替代方法调整法线方向")
            pcd_normals.orient_normals_towards_camera_location(
                camera_location=pcd_normals.get_center() + np.array([0, 0, 100])
            )
        
        # 7. Poisson 表面重建
        print("\n" + "="*60)
        print("第七步：Poisson 表面重建")
        print("="*60)
        print(f"参数: depth={POISSON_DEPTH}")
        print("正在重建网格...")
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd_normals, depth=POISSON_DEPTH
        )
        print(f"重建完成: 顶点数 {len(mesh.vertices):,}, 三角面数 {len(mesh.triangles):,}")
        
        # 移除低密度顶点
        print(f"参数: density_quantile={DENSITY_QUANTILE}")
        vertices_to_remove = densities < np.quantile(densities, DENSITY_QUANTILE)
        mesh.remove_vertices_by_mask(vertices_to_remove)
        print(f"移除低密度顶点后: 剩余顶点数 {len(mesh.vertices):,}")
        mesh.paint_uniform_color([1, 0.7, 0])
        
        # 8. 可视化原始点云
        print("\n" + "="*60)
        print("第八步：可视化原始点云")
        print("="*60)
        visualize_point_cloud(
            pcd=pcd,
            title="Original Point Cloud",
            width=1200,
            height=800
        )
        
        # 9. 可视化表面重建结果
        print("\n" + "="*60)
        print("第九步：可视化表面重建结果")
        print("="*60)
        print("灰色：原始点云，橙色：重建的表面网格")
        visualize_point_cloud(
            geometries=[pcd, mesh],
            title="Surface Reconstruction",
            width=1200,
            height=800
        )
        
        # 10. 保存网格模型
        print("\n" + "="*60)
        print("第十步：保存网格模型")
        print("="*60)
        save_mesh = input("是否保存重建的网格模型？(y/n) [默认: n]: ").strip().lower()
        
        if save_mesh == 'y' or save_mesh == 'yes':
            default_save_path = las_file_path.replace('.las', '_mesh.ply')
            save_path = input(f"请输入保存路径 [默认: {default_save_path}]: ").strip()
            if not save_path:
                save_path = default_save_path
            
            print("\n支持的文件格式:")
            print("  1. PLY (.ply) - 推荐格式")
            print("  2. OBJ (.obj) - 通用3D模型格式")
            print("  3. STL (.stl) - 3D打印格式")
            
            format_choice = input("请选择格式 (1/2/3) [默认: 1]: ").strip()
            
            if format_choice == '2':
                save_path = save_path.rsplit('.', 1)[0] + '.obj'
            elif format_choice == '3':
                save_path = save_path.rsplit('.', 1)[0] + '.stl'
            else:
                save_path = save_path.rsplit('.', 1)[0] + '.ply'
            
            save_dir = os.path.dirname(save_path)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            success = o3d.io.write_triangle_mesh(save_path, mesh)
            if success:
                print(f"\n网格模型已保存到: {save_path}")
                print(f"顶点数: {len(mesh.vertices):,}, 三角面数: {len(mesh.triangles):,}")
            else:
                print("\n警告: 网格模型保存失败")
        
        print("\n" + "="*60)
        print("处理完成!")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n用户取消操作")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
