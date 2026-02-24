import numpy as np
import open3d as o3d
import laspy
import time
from tqdm import tqdm
import threading
import sys


def read_las_point_cloud(file_path, subsample_factor=1, show_progress=True):
    """
    读取 .las 格式的点云文件，支持大文件处理。
    
    Args:
        file_path (str): .las 文件路径
        subsample_factor (int): 下采样因子，1 表示不采样，2 表示每2个点取1个，以此类推
        show_progress (bool): 是否显示进度条
        
    Returns:
        numpy.ndarray: 点云数据，形状为 (N, 3)，格式为 [[x, y, z], ...]
    """
    print(f"正在读取点云文件: {file_path}")
    
    # 使用 laspy 读取 .las 文件
    start_time = time.time()
    las = laspy.read(file_path)
    
    # 获取点坐标
    total_points = len(las.x)
    print(f"原始点云数量: {total_points:,}")
    
    if show_progress:
        # 使用 tqdm 显示进度
        if subsample_factor > 1:
            sample_size = total_points // subsample_factor
            print(f"正在下采样 (每 {subsample_factor} 个点取 1 个)...")
            
            # 分批读取以显示进度
            points_list = []
            batch_size = 1000000  # 每批处理 100 万个点
            
            for i in tqdm(range(0, total_points, batch_size * subsample_factor), 
                         desc="读取进度", unit="点"):
                end_idx = min(i + batch_size * subsample_factor, total_points)
                x_batch = las.x[i:end_idx:subsample_factor]
                y_batch = las.y[i:end_idx:subsample_factor]
                z_batch = las.z[i:end_idx:subsample_factor]
                batch_points = np.column_stack((x_batch, y_batch, z_batch))
                points_list.append(batch_points)
            
            points = np.vstack(points_list)
        else:
            print("正在读取全部点...")
            # 对于不下采样的情况，也分批读取显示进度
            points_list = []
            batch_size = 1000000
            
            for i in tqdm(range(0, total_points, batch_size), 
                         desc="读取进度", unit="点"):
                end_idx = min(i + batch_size, total_points)
                x_batch = las.x[i:end_idx]
                y_batch = las.y[i:end_idx]
                z_batch = las.z[i:end_idx]
                batch_points = np.column_stack((x_batch, y_batch, z_batch))
                points_list.append(batch_points)
            
            points = np.vstack(points_list)
    else:
        # 不显示进度条的快速读取
        x = las.x
        y = las.y
        z = las.z
        points = np.column_stack((x, y, z))
        
        if subsample_factor > 1:
            points = points[::subsample_factor]
    
    elapsed_time = time.time() - start_time
    print(f"读取完成! 用时: {elapsed_time:.2f} 秒")
    print(f"最终点云数量: {len(points):,}")
    
    return points


def visualize_point_cloud(points, title="LAS Point Cloud", point_size=1.0, auto_close_time=None, scale_factor=1.0):
    """
    使用 Open3D 可视化点云。
    
    Args:
        points (numpy.ndarray): 点云数据，形状为 (N, 3)
        title (str): 窗口标题
        point_size (float): 点的大小
        auto_close_time (float): 自动关闭时间（秒），None 表示不自动关闭
        scale_factor (float): 缩放参数，0 表示自动计算，1.0 表示默认缩放
    """
    print(f"正在可视化点云 (点数: {len(points):,})...")
    print("提示: 按 'Q' 或 'ESC' 键关闭窗口")
    
    # 创建 Open3D 点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # 设置默认颜色（灰色）
    pcd.paint_uniform_color([0.5, 0.5, 0.5])
    
    # 可视化
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title, width=1400, height=900)
    vis.add_geometry(pcd)
    
    # 设置点大小
    vis.get_render_option().point_size = point_size
    vis.get_render_option().background_color = np.array([0.1, 0.1, 0.1])  # 深色背景
    
    # 设置缩放
    if scale_factor > 0:
        print(f"使用缩放参数: {scale_factor}")
        # 获取当前视图控制
        ctr = vis.get_view_control()
        # 设置视图中心为点云中心
        ctr.set_lookat(pcd.get_center())
        # 设置相机方向
        ctr.set_up([0, 1, 0])
        ctr.set_front([1, 0, 0])
        # 应用缩放（注意：set_zoom 的值越大，视图越近；所以需要反向）
        # scale_factor 越大，视图应该越远，所以使用 1.0/scale_factor
        ctr.set_zoom(1.0 / scale_factor)
    
    # 如果设置了自动关闭时间，启动定时器
    if auto_close_time is not None:
        print(f"窗口将在 {auto_close_time} 秒后自动关闭...")
        
        def close_visualizer():
            time.sleep(auto_close_time)
            try:
                vis.close()
            except:
                pass
        
        close_thread = threading.Thread(target=close_visualizer, daemon=True)
        close_thread.start()
    
    try:
        # 运行可视化
        vis.run()
    except Exception as e:
        print(f"可视化出错: {e}")
    finally:
        try:
            vis.destroy_window()
        except:
            pass
    
    print("可视化窗口已关闭")


def read_and_visualize_las(file_path, subsample_factor=50, point_size=1.0, auto_close_time=None, scale_factor=1.0):
    """
    读取并可视化 .las 点云文件（便捷函数）。
    
    Args:
        file_path (str): .las 文件路径
        subsample_factor (int): 下采样因子，对于大文件建议使用 10-100
        point_size (float): 点的大小
        auto_close_time (float): 自动关闭时间（秒），None 表示不自动关闭
        scale_factor (float): 缩放参数，0 表示自动计算，1.0 表示默认缩放
    """
    # 读取点云
    points = read_las_point_cloud(file_path, subsample_factor=subsample_factor)
    
    # 可视化
    visualize_point_cloud(
        points, 
        title=f"LAS Point Cloud: {file_path}", 
        point_size=point_size,
        auto_close_time=auto_close_time,
        scale_factor=scale_factor
    )
    
    return points


def get_las_info(file_path):
    """
    获取 .las 文件的基本信息（不读取全部数据）。
    
    Args:
        file_path (str): .las 文件路径
        
    Returns:
        dict: 包含点云信息的字典
    """
    las = laspy.read(file_path)
    
    info = {
        'point_count': len(las.x),
        'x_min': float(las.x.min()),
        'x_max': float(las.x.max()),
        'y_min': float(las.y.min()),
        'y_max': float(las.y.max()),
        'z_min': float(las.z.min()),
        'z_max': float(las.z.max()),
        'version': f"{las.header.version.major}.{las.header.version.minor}"
    }
    
    # 尝试获取点格式信息（兼容不同的 laspy 版本）
    try:
        if hasattr(las.header.point_format, 'id'):
            info['point_format'] = las.header.point_format.id
        elif hasattr(las.header, 'point_data_record_format'):
            info['point_format'] = las.header.point_data_record_format
        else:
            info['point_format'] = str(las.header.point_format)
    except:
        info['point_format'] = 'unknown'
    
    return info


if __name__ == '__main__':
    # 默认使用用户指定的文件
    default_file_path = r"D:\1-学术\预拼装\结构模型\泉州项目点云及模型\扫描点云数据\44-unset.las"
    
    import sys
    
    # 检查是否有命令行参数
    if len(sys.argv) > 1:
        las_file_path = sys.argv[1]
    else:
        las_file_path = default_file_path
        print(f"使用默认文件: {las_file_path}")
        print("提示: 也可以使用命令行参数指定其他文件: python read_cloudpoints.py <your_file.las>")
    
    # 检查文件是否存在
    import os
    if not os.path.exists(las_file_path):
        print(f"错误: 文件不存在 - {las_file_path}")
        sys.exit(1)
    
    # 1. 先获取文件信息（快速预览）
    print("\n" + "="*60)
    print("第一步：获取文件信息")
    print("="*60)
    try:
        info = get_las_info(las_file_path)
        for key, value in info.items():
            if key == 'point_count':
                print(f"{key}: {value:,}")
            elif key in ['x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max']:
                print(f"{key}: {value:.3f}")
            else:
                print(f"{key}: {value}")
    except Exception as e:
        print(f"获取文件信息时出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # 2. 询问用户下采样比例
    print("\n" + "="*60)
    print("第二步：选择下采样比例")
    print("="*60)
    print("建议值:")
    print("  - 快速预览: 100-200")
    print("  - 中等质量: 50-100")
    print("  - 高质量: 10-50")
    print("  - 完整质量: 1 (不推荐大文件)")
    
    # 根据文件大小推荐采样比例
    estimated_point_count = info.get('point_count', 0)
    if estimated_point_count > 50_000_000:
        recommended = 100
    elif estimated_point_count > 10_000_000:
        recommended = 50
    elif estimated_point_count > 1_000_000:
        recommended = 20
    else:
        recommended = 10
    
    print(f"\n推荐采样比例: {recommended}")
    
    # 获取用户输入
    try:
        user_input = input(f"请输入采样比例 [默认: {recommended}]: ").strip()
        if user_input:
            subsample_factor = int(user_input)
        else:
            subsample_factor = recommended
    except KeyboardInterrupt:
        print("\n用户取消操作")
        sys.exit(0)
    except:
        print(f"输入无效，使用默认值: {recommended}")
        subsample_factor = recommended
    
    # 3. 询问用户缩放参数
    print("\n" + "="*60)
    print("第三步：选择缩放参数")
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
    
    # 3. 读取和可视化
    print("\n" + "="*60)
    print("第四步：读取和可视化点云")
    print("="*60)
    print(f"采样比例: {subsample_factor}")
    print(f"缩放参数: {scale_factor}")
    
    try:
        # 读取并可视化
        points = read_and_visualize_las(
            las_file_path, 
            subsample_factor=subsample_factor,
            point_size=1.0,
            scale_factor=scale_factor
        )
        
        print("\n" + "="*60)
        print("完成!")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n用户取消操作")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
