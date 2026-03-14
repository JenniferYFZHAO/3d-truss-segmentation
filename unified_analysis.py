import numpy as np
import open3d as o3d
import os
import csv
import re
from scipy.spatial.distance import cdist
from scipy.spatial import KDTree


# ==================== 文件路径配置 ====================
# 用户可以根据实际情况修改这些路径
DEFAULT_OBJ_FILE = r"D:\1-学术\预拼装\结构模型\泉州项目点云及模型\Final\44品-separated-complete.obj"
DEFAULT_LAS_FILE = r"D:\1-学术\预拼装\结构模型\泉州项目点云及模型\Final\44-unset-部分去噪.las"
DEFAULT_CSV_PATH = r"D:\1-学术\预拼装\结构模型\泉州项目点云及模型\Final\mesh_analysis_results.csv"
DEFAULT_SURROUNDING_OBJ = r"D:\1-学术\预拼装\结构模型\泉州项目点云及模型\Final\周边.obj"
DEFAULT_SURROUNDING_NODES = r"D:\1-学术\预拼装\结构模型\泉州项目点云及模型\Final\sphere_extraction_results1.csv"


# ==================== 读取函数 ====================

def read_las_point_cloud(file_path, subsample_factor=10):
    """
    读取 LAS 点云文件。
    
    Args:
        file_path: LAS 文件路径
        subsample_factor: 下采样因子
        
    Returns:
        points: 点云坐标 (N, 3)
    """
    print(f"正在读取 LAS 文件: {file_path}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    import laspy
    las = laspy.read(file_path)
    
    print(f"原始点数: {len(las.points):,}")
    
    # 下采样
    if subsample_factor > 1:
        points = np.vstack((las.x[::subsample_factor], las.y[::subsample_factor], las.z[::subsample_factor])).T
        print(f"下采样因子: {subsample_factor}")
    else:
        points = np.vstack((las.x, las.y, las.z)).T
    
    print(f"读取完成!")
    print(f"  点数: {len(points):,}")
    print(f"  范围:")
    print(f"    X: [{points[:, 0].min():.3f}, {points[:, 0].max():.3f}]")
    print(f"    Y: [{points[:, 1].min():.3f}, {points[:, 1].max():.3f}]")
    print(f"    Z: [{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]")
    
    return points


def get_las_info(file_path):
    """
    获取 LAS 文件信息。
    """
    import laspy
    las = laspy.read(file_path)
    return {
        'point_count': len(las.points),
        'x_min': las.header.x_min,
        'x_max': las.header.x_max,
        'y_min': las.header.y_min,
        'y_max': las.header.y_max,
        'z_min': las.header.z_min,
        'z_max': las.header.z_max
    }


def read_obj_mesh(file_path):
    """
    读取 OBJ 网格文件。
    支持 Open3D 直接读取和手动解析两种方式。
    """
    print(f"正在读取 OBJ 文件: {file_path}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    # 首先尝试使用 Open3D 直接读取
    try:
        mesh = o3d.io.read_triangle_mesh(file_path)
        if not mesh.is_empty() and len(mesh.vertices) > 0:
            vertices = np.asarray(mesh.vertices)
            print(f"使用 Open3D 读取成功!")
            print(f"  顶点数: {len(mesh.vertices):,}")
            print(f"  三角面数: {len(mesh.triangles):,}")
            return mesh, vertices
    except Exception as e:
        print(f"Open3D 读取失败: {e}")
    
    # 如果 Open3D 读取失败，尝试手动解析
    print("尝试手动解析 OBJ 文件...")
    mesh = parse_obj_file_manual(file_path)
    
    if mesh is None or mesh.is_empty() or len(mesh.vertices) == 0:
        raise ValueError(f"无法读取 OBJ 文件或文件为空: {file_path}")
    
    vertices = np.asarray(mesh.vertices)
    
    print(f"手动解析成功!")
    print(f"  顶点数: {len(mesh.vertices):,}")
    print(f"  三角面数: {len(mesh.triangles):,}")
    
    return mesh, vertices


def parse_obj_file_manual(file_path):
    """
    手动解析 OBJ 文件，处理编码问题。
    """
    vertices = []
    faces = []
    
    encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1', 'cp1252']
    lines = None
    
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
                lines = f.readlines()
            print(f"使用 {encoding} 编码读取文件")
            break
        except Exception as e:
            continue
    
    if lines is None:
        try:
            with open(file_path, 'rb') as f:
                raw_content = f.read()
                for encoding in encodings:
                    try:
                        content = raw_content.decode(encoding, errors='ignore')
                        lines = content.splitlines()
                        print(f"使用二进制模式 + {encoding} 编码读取文件")
                        break
                    except:
                        continue
        except Exception as e:
            print(f"无法读取文件: {e}")
            return None
    
    if lines is None:
        return None
    
    vertex_count = 0
    face_count = 0
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        
        parts = line.split()
        if not parts:
            continue
        
        if parts[0] == 'v':
            try:
                if len(parts) >= 4:
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    vertices.append([x, y, z])
                    vertex_count += 1
            except (ValueError, IndexError) as e:
                continue
        elif parts[0] == 'f':
            try:
                face_indices = []
                for part in parts[1:]:
                    idx = part.split('/')[0]
                    face_indices.append(int(idx) - 1)
                
                if len(face_indices) >= 3:
                    for i in range(1, len(face_indices) - 1):
                        faces.append([face_indices[0], face_indices[i], face_indices[i+1]])
                        face_count += 1
            except (ValueError, IndexError) as e:
                continue
    
    print(f"解析完成: {vertex_count} 个顶点, {face_count} 个三角面")
    
    mesh = o3d.geometry.TriangleMesh()
    
    if vertices:
        mesh.vertices = o3d.utility.Vector3dVector(np.array(vertices))
    
    if faces:
        mesh.triangles = o3d.utility.Vector3iVector(np.array(faces))
    
    if len(mesh.triangles) > 0:
        mesh.compute_vertex_normals()
    
    return mesh


def read_csv_model_info(csv_path):
    """
    读取 CSV 文件中的模型信息（球节点和圆管）。
    CSV 结构：
    - 第一列: Object Type (Sphere 或 Tube)
    - 球节点: X, Y, Z, Radius (第2-5列)
    - 圆管: Radius, Start_X, Start_Y, Start_Z, End_X, End_Y, End_Z (第5-11列)
    """
    spheres = []
    tubes = []
    
    print(f"读取模型信息文件: {csv_path}")
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                type_ = row['Object Type'].strip()
                
                if type_ == 'Sphere':
                    x = float(row['X'])
                    y = float(row['Y'])
                    z = float(row['Z'])
                    radius = float(row['Radius'])
                    spheres.append((x, y, z, radius))
                    
                elif type_ == 'Tube':
                    start_x = float(row['Start_X'])
                    start_y = float(row['Start_Y'])
                    start_z = float(row['Start_Z'])
                    end_x = float(row['End_X'])
                    end_y = float(row['End_Y'])
                    end_z = float(row['End_Z'])
                    radius = float(row['Radius'])
                    tubes.append((start_x, start_y, start_z, end_x, end_y, end_z, radius))
                    
            except Exception as e:
                print(f"解析行时出错: {e}")
                continue
    
    print(f"成功读取: {len(spheres)} 个球节点, {len(tubes)} 个圆管")
    return spheres, tubes


def pick_points_on_point_cloud(points, title="Select Control Points", num_points=3):
    """
    在点云上交互式选取控制点。
    
    Args:
        points (numpy.ndarray): 点云坐标数组 (N, 3)
        title (str): 窗口标题
        num_points (int): 需要选取的点数
        
    Returns:
        numpy.ndarray: 选取的控制点坐标数组 (num_points, 3)
    """
    print(f"\n请在可视化窗口中选取 {num_points} 个控制点")
    print("操作说明:")
    print("  - 按住 Shift + 左键点击: 选取点")
    print("  - 按住 Shift + 右键点击: 取消选取")
    print("  - 选取完成后按 'Q' 或 'ESC' 键关闭窗口")
    print(f"  - 请确保选取恰好 {num_points} 个点")
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color([0.5, 0.5, 0.5])
    
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name=title, width=1400, height=900)
    vis.add_geometry(pcd)
    
    vis.run()
    vis.destroy_window()
    
    picked_indices = vis.get_picked_points()
    
    if len(picked_indices) != num_points:
        print(f"\n警告: 您选取了 {len(picked_indices)} 个点，但需要 {num_points} 个点")
        if len(picked_indices) < num_points:
            print("请重新运行程序并选取足够的点")
            return None
    
    picked_points = points[picked_indices]
    
    print(f"\n已选取 {len(picked_points)} 个控制点:")
    for i, pt in enumerate(picked_points):
        print(f"  点 {i+1}: ({pt[0]:.3f}, {pt[1]:.3f}, {pt[2]:.3f})")
    
    return picked_points


def pick_points_on_mesh(mesh, title="Select Control Points", num_points=3):
    """
    在网格模型上交互式选取控制点。
    
    Args:
        mesh (open3d.geometry.TriangleMesh): 网格模型
        title (str): 窗口标题
        num_points (int): 需要选取的点数
        
    Returns:
        numpy.ndarray: 选取的控制点坐标数组 (num_points, 3)
    """
    print(f"\n请在可视化窗口中选取 {num_points} 个控制点")
    print("操作说明:")
    print("  - 按住 Shift + 左键点击: 选取点")
    print("  - 按住 Shift + 右键点击: 取消选取")
    print("  - 选取完成后按 'Q' 或 'ESC' 键关闭窗口")
    print(f"  - 请确保选取恰好 {num_points} 个点")
    
    # 提取网格顶点作为点云
    vertices = np.asarray(mesh.vertices)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)
    pcd.paint_uniform_color([0.5, 0.5, 0.5])
    
    # 使用点云进行控制点选取
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name=title, width=1400, height=900)
    vis.add_geometry(pcd)
    
    vis.run()
    vis.destroy_window()
    
    picked_indices = vis.get_picked_points()
    
    if len(picked_indices) != num_points:
        print(f"\n警告: 您选取了 {len(picked_indices)} 个点，但需要 {num_points} 个点")
        if len(picked_indices) < num_points:
            print("请重新运行程序并选取足够的点")
            return None
    
    picked_points = vertices[picked_indices]
    
    print(f"\n已选取 {len(picked_points)} 个控制点:")
    for i, pt in enumerate(picked_points):
        print(f"  点 {i+1}: ({pt[0]:.3f}, {pt[1]:.3f}, {pt[2]:.3f})")
    
    return picked_points


def compute_similarity_transform(source_points, target_points):
    """
    计算从源点集到目标点集的相似变换（旋转、平移、缩放）。
    
    Args:
        source_points (numpy.ndarray): 源点集 (N, 3)
        target_points (numpy.ndarray): 目标点集 (N, 3)
        
    Returns:
        dict: 包含旋转矩阵R、平移向量t、缩放因子s和变换矩阵的字典
    """
    assert source_points.shape == target_points.shape, "源点集和目标点集形状必须相同"
    
    n = source_points.shape[0]
    
    # 计算质心
    source_centroid = np.mean(source_points, axis=0)
    target_centroid = np.mean(target_points, axis=0)
    
    # 中心化
    source_centered = source_points - source_centroid
    target_centered = target_points - target_centroid
    
    # 计算缩放因子
    source_norm = np.linalg.norm(source_centered)
    target_norm = np.linalg.norm(target_centered)
    s = target_norm / source_norm if source_norm > 0 else 1.0
    
    # 归一化
    source_normalized = source_centered / (source_norm + 1e-10)
    target_normalized = target_centered / (target_norm + 1e-10)
    
    # 计算旋转矩阵（使用Kabsch算法）
    H = source_normalized.T @ target_normalized
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    # 确保右手坐标系
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # 计算平移向量
    t = target_centroid - s * R @ source_centroid
    
    # 构造4x4变换矩阵
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = s * R
    transform_matrix[:3, 3] = t
    
    return {
        'R': R,
        't': t,
        's': s,
        'transform_matrix': transform_matrix
    }


def apply_transform_to_mesh(mesh, R, t, s=1.0):
    """
    应用变换到网格模型。
    
    Args:
        mesh (open3d.geometry.TriangleMesh): 输入网格
        R (numpy.ndarray): 旋转矩阵 (3, 3)
        t (numpy.ndarray): 平移向量 (3,)
        s (float): 缩放因子
        
    Returns:
        open3d.geometry.TriangleMesh: 变换后的网格
    """
    vertices = np.asarray(mesh.vertices)
    transformed_vertices = s * (R @ vertices.T).T + t
    
    transformed_mesh = o3d.geometry.TriangleMesh()
    transformed_mesh.vertices = o3d.utility.Vector3dVector(transformed_vertices)
    transformed_mesh.triangles = mesh.triangles
    transformed_mesh.compute_vertex_normals()
    
    return transformed_mesh


def visualize_registration_result(mesh, points, title="Registration Result"):
    """
    可视化配准结果。
    
    Args:
        mesh (open3d.geometry.TriangleMesh): 结构模型网格
        points (numpy.ndarray): 点云坐标 (N, 3)
        title (str): 窗口标题
    """
    mesh_copy = o3d.geometry.TriangleMesh(mesh)
    mesh_copy.paint_uniform_color([1.0, 0.0, 0.0])
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color([0.0, 1.0, 0.0])
    
    # 设置点云渲染选项，调小点的大小
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title, width=1400, height=900)
    vis.add_geometry(mesh_copy)
    vis.add_geometry(pcd)
    
    # 获取渲染选项并设置点大小
    render_option = vis.get_render_option()
    render_option.point_size = 1.0  # 调小点的大小
    
    vis.run()
    vis.destroy_window()


def similarity_icp(source, target, max_iterations=20, threshold=1e-4, initial_correspondence_distance=10.0, final_correspondence_distance=0.5, voxel_size=0.2):
    """
    相似变换 ICP（Similarity ICP）
    支持旋转、平移和缩放
    
    Args:
        source: 源点云（o3d.geometry.PointCloud）
        target: 目标点云（o3d.geometry.PointCloud）
        max_iterations: 最大迭代次数（默认20，减少迭代次数）
        threshold: 收敛阈值（默认1e-4，放宽收敛条件）
        initial_correspondence_distance: 初始对应点最大距离
        final_correspondence_distance: 最终对应点最大距离
        voxel_size: 体素下采样大小（默认0.2，减少计算量）
        
    Returns:
        dict: 包含变换参数的字典
    """
    print("执行相似变换 ICP (s-ICP)...")
    
    # 对源点云和目标点云进行下采样，减少计算量
    if voxel_size > 0:
        source_down = source.voxel_down_sample(voxel_size=voxel_size)
        target_down = target.voxel_down_sample(voxel_size=voxel_size)
        print(f"源点云下采样后: {len(source_down.points)} 个点")
        print(f"目标点云下采样后: {len(target_down.points)} 个点")
    else:
        source_down = source
        target_down = target
    
    # 初始化源点云
    source_points = np.asarray(source_down.points)
    target_points = np.asarray(target_down.points)
    
    # 初始变换参数
    R = np.eye(3)
    t = np.zeros(3)
    s = 1.0
    
    prev_error = float('inf')
    
    for i in range(max_iterations):
        # 动态调整对应点距离阈值（从大到小）
        alpha = i / max_iterations
        current_correspondence_distance = initial_correspondence_distance * (1 - alpha) + final_correspondence_distance * alpha
        
        # 应用当前变换
        transformed_source = s * (R @ source_points.T).T + t
        transformed_source_pcd = o3d.geometry.PointCloud()
        transformed_source_pcd.points = o3d.utility.Vector3dVector(transformed_source)
        
        # 计算对应点（使用快速对应点搜索）
        registration = o3d.pipelines.registration.registration_icp(
            transformed_source_pcd,
            target_down,
            current_correspondence_distance,
            init=np.eye(4),
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1)
        )
        
        # 获取对应点
        correspondences = registration.correspondence_set
        if len(correspondences) < 3:
            # 如果对应点不足，尝试使用更大的距离阈值
            if i == 0:
                print("初始对应点不足，尝试使用更大的距离阈值...")
                current_correspondence_distance = initial_correspondence_distance * 2
                registration = o3d.pipelines.registration.registration_icp(
                    transformed_source_pcd,
                    target_down,
                    current_correspondence_distance,
                    init=np.eye(4),
                    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                    criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1)
                )
                correspondences = registration.correspondence_set
        
        if len(correspondences) < 3:
            print(f"对应点不足 ({len(correspondences)} 个)，无法继续")
            break
        
        # 提取对应点
        source_corr = np.asarray([source_points[idx[0]] for idx in correspondences])
        target_corr = np.asarray([target_points[idx[1]] for idx in correspondences])
        
        # 使用 Umeyama 算法计算相似变换
        transform = compute_similarity_transform(source_corr, target_corr)
        R = transform['R']
        t = transform['t']
        s = transform['s']
        
        # 计算误差
        current_error = registration.inlier_rmse
        print(f"迭代 {i+1}/{max_iterations}: RMSE = {current_error:.6f}, 缩放因子 = {s:.6f}, 对应距离 = {current_correspondence_distance:.2f}")
        
        # 检查收敛（增加早期收敛检测）
        if abs(prev_error - current_error) < threshold:
            # 如果误差已经很小，提前收敛
            if current_error < 0.8:
                print(f"s-ICP 提前收敛于第 {i+1} 次迭代")
                break
        
        prev_error = current_error
    
    # 构建变换矩阵
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = s * R
    transform_matrix[:3, 3] = t
    
    return {
        'R': R,
        't': t,
        's': s,
        'transform_matrix': transform_matrix,
        'rmse': prev_error
    }


def read_surrounding_nodes(surrounding_nodes_path):
    """
    读取周边结构模型的节点信息。
    
    Args:
        surrounding_nodes_path: 周边结构节点信息文件路径
    
    Returns:
        list: 周边结构节点信息列表
    """
    print(f"读取周边结构节点信息: {surrounding_nodes_path}")
    
    surrounding_nodes = []
    
    try:
        # 尝试使用不同编码读取
        encodings = ['utf-8', 'gbk', 'latin-1']
        for encoding in encodings:
            try:
                with open(surrounding_nodes_path, 'r', encoding=encoding) as f:
                    reader = csv.DictReader(f)
                    
                    # 检查列名
                    fieldnames = reader.fieldnames
                    print(f"CSV列名: {fieldnames}")
                    
                    for row in reader:
                        try:
                            # 尝试不同的列名格式
                            if 'x' in row and 'y' in row and 'z' in row and 'radius' in row:
                                # 格式1: id, x, y, z, radius
                                node_id = row.get('id', f"node_{len(surrounding_nodes)}")
                                x = float(row['x'])
                                y = float(row['y'])
                                z = float(row['z'])
                                radius = float(row['radius'])
                            elif 'Center_X' in row and 'Center_Y' in row and 'Center_Z' in row and 'Radius' in row:
                                # 格式3: Object Type, Center_X, Center_Y, Center_Z, Radius
                                if row.get('Object Type') == 'Sphere':
                                    node_id = f"node_{len(surrounding_nodes)}"
                                    x = float(row['Center_X'])
                                    y = float(row['Center_Y'])
                                    z = float(row['Center_Z'])
                                    radius = float(row['Radius'])
                                else:
                                    continue
                            elif len(row) >= 5:
                                # 格式2: 物件名称, X, Y, Z, 半径
                                values = list(row.values())
                                if values[0] == 'Sphere':
                                    node_id = f"node_{len(surrounding_nodes)}"
                                    x = float(values[1])
                                    y = float(values[2])
                                    z = float(values[3])
                                    radius = float(values[4])
                                else:
                                    continue
                            else:
                                # 尝试其他可能的列名
                                x = None
                                y = None
                                z = None
                                radius = None
                                
                                for key, value in row.items():
                                    key_lower = key.lower()
                                    if 'x' in key_lower:
                                        x = float(value)
                                    elif 'y' in key_lower:
                                        y = float(value)
                                    elif 'z' in key_lower:
                                        z = float(value)
                                    elif 'radius' in key_lower or 'r' in key_lower:
                                        radius = float(value)
                                
                                if x is None or y is None or z is None:
                                    continue
                                
                                if radius is None:
                                    radius = 0.1
                                
                                node_id = row.get('id', f"node_{len(surrounding_nodes)}")
                            
                            surrounding_nodes.append({
                                'id': node_id,
                                'x': x,
                                'y': y,
                                'z': z,
                                'radius': radius
                            })
                        except (ValueError, KeyError) as e:
                            print(f"解析行时出错: {e}, 行数据: {row}")
                            continue
                print(f"成功使用 {encoding} 编码读取文件")
                break
            except UnicodeDecodeError:
                continue
    except Exception as e:
        print(f"读取周边结构节点文件时出错: {e}")
    
    print(f"成功读取 {len(surrounding_nodes)} 个周边结构节点")
    return surrounding_nodes


# ==================== 变换函数 ====================

def apply_transform_to_spheres(spheres, transform_matrix):
    """
    将变换矩阵应用到球节点，包括半径缩放。
    """
    transformed_spheres = []
    
    scale = np.linalg.norm(transform_matrix[:3, 0])
    
    for x, y, z, radius in spheres:
        point = np.array([x, y, z, 1.0])
        transformed_point = transform_matrix @ point
        tx, ty, tz, _ = transformed_point
        scaled_radius = radius * scale
        transformed_spheres.append((tx, ty, tz, scaled_radius))
    
    return transformed_spheres


def apply_transform_to_tubes(tubes, transform_matrix):
    """
    将变换矩阵应用到圆管，包括半径缩放。
    """
    transformed_tubes = []
    
    scale = np.linalg.norm(transform_matrix[:3, 0])
    
    for sx, sy, sz, ex, ey, ez, radius in tubes:
        start_point = np.array([sx, sy, sz, 1.0])
        transformed_start = transform_matrix @ start_point
        tsx, tsy, tsz, _ = transformed_start
        
        end_point = np.array([ex, ey, ez, 1.0])
        transformed_end = transform_matrix @ end_point
        tex, tey, tez, _ = transformed_end
        
        scaled_radius = radius * scale
        transformed_tubes.append((tsx, tsy, tsz, tex, tey, tez, scaled_radius))
    
    return transformed_tubes


# ==================== RANSAC球面拟合 ====================

def fit_sphere_from_4_points(points):
    """
    从4个点拟合球面。
    
    Args:
        points: 4个点的坐标 (4, 3)
        
    Returns:
        center: 球心坐标
        radius: 球半径
    """
    if len(points) != 4:
        return None, None
    
    A = np.zeros((4, 4))
    b = np.zeros(4)
    
    for i in range(4):
        x, y, z = points[i]
        A[i, 0] = -2 * x
        A[i, 1] = -2 * y
        A[i, 2] = -2 * z
        A[i, 3] = 1
        b[i] = -(x**2 + y**2 + z**2)
    
    try:
        solution = np.linalg.solve(A, b)
        center = solution[:3]
        radius = np.sqrt(center[0]**2 + center[1]**2 + center[2]**2 - solution[3])
        
        if radius < 0 or np.isnan(radius) or np.isinf(radius):
            return None, None
        
        return center, radius
    except np.linalg.LinAlgError:
        return None, None


def refine_sphere_fit(points):
    """
    使用最小二乘法精化球面拟合。
    
    Args:
        points: 点云坐标 (N, 3)
        
    Returns:
        center: 精化后的球心
        radius: 精化后的半径
    """
    if len(points) < 4:
        return None, None
    
    centroid = np.mean(points, axis=0)
    
    A = 2 * (points - centroid)
    b = np.sum((points - centroid) ** 2, axis=1)
    
    try:
        center_offset = np.linalg.lstsq(A, b, rcond=None)[0]
        center = centroid + center_offset
        radius = np.sqrt(np.mean(np.sum((points - center) ** 2, axis=1)))
        
        if radius < 0 or np.isnan(radius) or np.isinf(radius):
            return None, None
        
        return center, radius
    except Exception:
        return None, None


def calculate_fit_quality_score(result, distance_threshold):
    """
    计算拟合质量评分 (0-100)。
    
    评分维度：
    1. 内点比例 (30分)
    2. 残差均值 (25分)
    3. 残差标准差 (25分)
    4. 最大残差 (20分)
    """
    score = 0
    
    # 1. 内点比例评分 (30分)
    inlier_ratio_score = min(30, result['inlier_ratio'] * 30)
    score += inlier_ratio_score
    
    # 2. 残差均值评分 (25分)
    if result['residual_mean'] <= distance_threshold:
        residual_mean_score = 25
    elif result['residual_mean'] <= distance_threshold * 2:
        residual_mean_score = 20
    elif result['residual_mean'] <= distance_threshold * 5:
        residual_mean_score = 15
    elif result['residual_mean'] <= distance_threshold * 10:
        residual_mean_score = 10
    else:
        residual_mean_score = 5
    score += residual_mean_score
    
    # 3. 残差标准差评分 (25分)
    if result['residual_std'] <= distance_threshold * 0.5:
        residual_std_score = 25
    elif result['residual_std'] <= distance_threshold:
        residual_std_score = 20
    elif result['residual_std'] <= distance_threshold * 2:
        residual_std_score = 15
    elif result['residual_std'] <= distance_threshold * 5:
        residual_std_score = 10
    else:
        residual_std_score = 5
    score += residual_std_score
    
    # 4. 最大残差评分 (20分)
    if result['residual_max'] <= distance_threshold * 2:
        residual_max_score = 20
    elif result['residual_max'] <= distance_threshold * 5:
        residual_max_score = 15
    elif result['residual_max'] <= distance_threshold * 10:
        residual_max_score = 10
    elif result['residual_max'] <= distance_threshold * 20:
        residual_max_score = 5
    else:
        residual_max_score = 0
    score += residual_max_score
    
    return score


def get_quality_level(score):
    """
    根据评分获取质量等级。
    """
    if score >= 90:
        return "优秀"
    elif score >= 75:
        return "良好"
    elif score >= 60:
        return "合格"
    elif score >= 40:
        return "较差"
    else:
        return "不合格"


def get_stability_level(score):
    """
    根据评分获取稳定性等级。
    """
    if score >= 90:
        return "优秀"
    elif score >= 75:
        return "良好"
    elif score >= 60:
        return "合格"
    elif score >= 40:
        return "较差"
    else:
        return "不合格"


def ransac_sphere_fitting(points, max_iterations=100, distance_threshold=0.01, min_inliers=100, 
                         num_runs=5, random_seed=None):
    """
    使用改进的 RANSAC 方法拟合球面，包含稳定性保证和质量评价。
    
    Args:
        points: 点云坐标 (N, 3)
        max_iterations: 每次运行的最大迭代次数
        distance_threshold: 内点距离阈值
        min_inliers: 最小内点数量
        num_runs: 运行次数，用于评估稳定性
        random_seed: 随机种子，用于结果复现
        
    Returns:
        center: 球心坐标
        radius: 球半径
        inlier_indices: 内点索引
        quality_score: 拟合质量评分 (0-100)
        stability_score: 稳定性评分 (0-100)
        fit_info: 详细的拟合信息字典
    """
    if len(points) < 4:
        return None, None, None, 0, 0, {'error': '点数不足'}
    
    # 存储多次运行的结果
    all_results = []
    
    # 设置随机种子以保证可复现性
    if random_seed is not None:
        np.random.seed(random_seed)
    
    for run in range(num_runs):
        # 每次运行使用不同的随机种子
        if random_seed is not None:
            np.random.seed(random_seed + run)
        
        best_center = None
        best_radius = None
        best_inliers = []
        best_score = 0
        
        for _ in range(max_iterations):
            indices = np.random.choice(len(points), 4, replace=False)
            sample_points = points[indices]
            
            center, radius = fit_sphere_from_4_points(sample_points)
            
            if center is None or radius is None:
                continue
            
            if radius <= 0 or radius > 100:
                continue
            
            distances = np.linalg.norm(points - center, axis=1)
            inlier_mask = np.abs(distances - radius) < distance_threshold
            inlier_indices = np.where(inlier_mask)[0]
            
            # 综合评分：内点数量 + 内点比例
            inlier_ratio = len(inlier_indices) / len(points)
            score = len(inlier_indices) + inlier_ratio * 100
            
            if score > best_score:
                best_score = score
                best_inliers = inlier_indices
                best_center = center
                best_radius = radius
        
        if len(best_inliers) >= min_inliers:
            # 精化拟合结果
            inlier_points = points[best_inliers]
            refined_center, refined_radius = refine_sphere_fit(inlier_points)
            
            if refined_center is not None:
                # 计算拟合质量指标
                distances = np.linalg.norm(points - refined_center, axis=1)
                residuals = np.abs(distances - refined_radius)
                
                # 内点重新判定
                inlier_mask = residuals < distance_threshold
                final_inliers = np.where(inlier_mask)[0]
                
                all_results.append({
                    'center': refined_center,
                    'radius': refined_radius,
                    'inliers': final_inliers,
                    'inlier_count': len(final_inliers),
                    'inlier_ratio': len(final_inliers) / len(points),
                    'residual_std': np.std(residuals[final_inliers]) if len(final_inliers) > 0 else float('inf'),
                    'residual_mean': np.mean(residuals[final_inliers]) if len(final_inliers) > 0 else float('inf'),
                    'residual_max': np.max(residuals[final_inliers]) if len(final_inliers) > 0 else float('inf')
                })
    
    if len(all_results) == 0:
        return None, None, None, 0, 0, {'error': '所有运行均未找到足够内点'}
    
    # 选择最佳结果（内点数量最多且残差最小的）
    best_result = max(all_results, key=lambda x: x['inlier_count'] - x['residual_mean'] * 100)
    
    # 计算稳定性评分
    if len(all_results) >= 3:
        centers = np.array([r['center'] for r in all_results])
        radii = np.array([r['radius'] for r in all_results])
        
        center_std = np.std(centers, axis=0)
        radius_std = np.std(radii)
        
        # 稳定性评分：基于多次运行的标准差
        center_stability = max(0, 100 - np.mean(center_std) * 1000)
        radius_stability = max(0, 100 - radius_std * 1000)
        stability_score = (center_stability + radius_stability) / 2
    else:
        stability_score = 50
    
    # 计算拟合质量评分
    quality_score = calculate_fit_quality_score(best_result, distance_threshold)
    
    # 组装详细信息
    fit_info = {
        'num_runs': len(all_results),
        'inlier_count': best_result['inlier_count'],
        'inlier_ratio': best_result['inlier_ratio'],
        'residual_mean': best_result['residual_mean'],
        'residual_std': best_result['residual_std'],
        'residual_max': best_result['residual_max'],
        'center_std': np.std([r['center'] for r in all_results], axis=0).tolist() if len(all_results) > 1 else [0, 0, 0],
        'radius_std': np.std([r['radius'] for r in all_results]).tolist() if len(all_results) > 1 else 0,
        'quality_level': get_quality_level(quality_score),
        'stability_level': get_stability_level(stability_score)
    }
    
    return best_result['center'], best_result['radius'], best_result['inliers'], quality_score, stability_score, fit_info


# ==================== 装配误差分析 ====================

def template_based_sphere_fitting(points, model_center, model_radius, search_radius_factor=2.0, 
                                  max_iterations=50, position_step=0.01, radius_step=0.005):
    """
    基于模板的球节点拟合（针对RANSAC质量较低的节点）。
    
    使用模板半径作为初始半径，在点云邻域中寻找最佳球心位置，
    然后迭代微调半径，直到找到最佳拟合。
    
    Args:
        points: 点云坐标 (N, 3)
        model_center: 模型球心坐标
        model_radius: 模型半径（作为初始半径）
        search_radius_factor: 搜索半径因子
        max_iterations: 最大迭代次数
        position_step: 球心位置调整步长
        radius_step: 半径调整步长
        
    Returns:
        best_center: 最佳球心坐标
        best_radius: 最佳半径
        best_inlier_count: 最佳内点数量
        best_score: 最佳评分
    """
    kd_tree = KDTree(points)
    search_radius = model_radius * search_radius_factor
    
    indices = kd_tree.query_ball_point(model_center, search_radius)
    
    if len(indices) < 30:
        return None, None, 0, 0
    
    region_points = points[indices]
    
    # 初始设置：使用模板半径和模型球心
    best_center = model_center.copy()
    best_radius = model_radius
    best_inlier_count = 0
    best_score = 0
    
    # 计算初始内点
    distances = np.linalg.norm(region_points - best_center, axis=1)
    inlier_mask = np.abs(distances - best_radius) < 0.02
    best_inlier_count = np.sum(inlier_mask)
    
    # 评分：内点数量 + 球心与模型中心的距离惩罚
    best_score = best_inlier_count - np.linalg.norm(best_center - model_center) * 10
    
    print(f"    模板拟合初始: 球心={best_center}, 半径={best_radius:.4f}m, 内点={best_inlier_count}, 评分={best_score:.1f}")
    
    # 迭代优化
    for iteration in range(max_iterations):
        improved = False
        
        # 尝试调整球心位置（6个方向）
        directions = [
            np.array([1, 0, 0]), np.array([-1, 0, 0]),
            np.array([0, 1, 0]), np.array([0, -1, 0]),
            np.array([0, 0, 1]), np.array([0, 0, -1])
        ]
        
        for direction in directions:
            test_center = best_center + direction * position_step
            
            # 计算内点
            distances = np.linalg.norm(region_points - test_center, axis=1)
            inlier_mask = np.abs(distances - best_radius) < 0.02
            inlier_count = np.sum(inlier_mask)
            
            # 评分
            score = inlier_count - np.linalg.norm(test_center - model_center) * 10
            
            if score > best_score:
                best_center = test_center
                best_inlier_count = inlier_count
                best_score = score
                improved = True
                print(f"    迭代{iteration+1}: 球心调整, 新球心={best_center}, 内点={best_inlier_count}, 评分={best_score:.1f}")
        
        # 尝试调整半径
        for delta in [-radius_step, radius_step]:
            test_radius = best_radius + delta
            if test_radius <= 0:
                continue
            
            distances = np.linalg.norm(region_points - best_center, axis=1)
            inlier_mask = np.abs(distances - test_radius) < 0.02
            inlier_count = np.sum(inlier_mask)
            
            # 评分：半径偏离模板的惩罚
            radius_penalty = abs(test_radius - model_radius) * 20
            score = inlier_count - radius_penalty
            
            if score > best_score:
                best_radius = test_radius
                best_inlier_count = inlier_count
                best_score = score
                improved = True
                print(f"    迭代{iteration+1}: 半径调整, 新半径={best_radius:.4f}m, 内点={best_inlier_count}, 评分={best_score:.1f}")
        
        if not improved:
            print(f"    迭代{iteration+1}: 收敛")
            break
    
    return best_center, best_radius, best_inlier_count, best_score


def visualize_template_fitting_pointcloud(poor_results, improved_results, all_points):
    """
    使用点云可视化模板拟合前后的对比（只显示节点邻域内的点）。
    为每个不合格节点分别创建可视化窗口。
    
    Args:
        poor_results: 不合格的节点结果
        improved_results: 模板拟合后的结果
        all_points: 所有点云数据（用于KD树查找）
    """
    print("\n可视化模板拟合前后对比（点云模式）...")
    
    if len(poor_results) == 0:
        print("  没有可用的不合格节点")
        return
    
    # 创建KD树用于快速查询
    from scipy.spatial import KDTree
    kd_tree = KDTree(all_points)
    
    # 为每个不合格节点创建单独的可视化窗口
    for i, poor_result in enumerate(poor_results):
        # 找到对应的改进结果
        improved_result = None
        for imp in improved_results:
            if imp['index'] == poor_result['index']:
                improved_result = imp
                break
        
        if improved_result is None:
            continue
        
        # 获取改进后的球心和半径
        center = improved_result['cloud_center']
        radius = improved_result['cloud_radius']
        
        if center is None or radius is None:
            print(f"  节点{poor_result['index']+1}: 没有有效的球心和半径")
            continue
        
        # 重新从原始点云中获取邻域点（使用2倍半径作为搜索范围）
        search_radius = radius * 2.0
        indices = kd_tree.query_ball_point(center, search_radius)
        
        if len(indices) < 10:
            print(f"  节点{poor_result['index']+1}: 邻域内点太少 ({len(indices)}个)")
            continue
        
        # 使用原始点
        region_points = all_points[indices]
        
        # 计算每个点到球心的距离
        distances = np.linalg.norm(region_points - center, axis=1)
        
        # 区分内点和外点（距离球面0.02m以内为内点）
        inlier_mask = np.abs(distances - radius) < 0.02
        inlier_points = region_points[inlier_mask]
        outlier_points = region_points[~inlier_mask]
        
        # 创建点云
        geometries = []
        
        # 内点点云（绿色）
        if len(inlier_points) > 0:
            inlier_pcd = o3d.geometry.PointCloud()
            inlier_pcd.points = o3d.utility.Vector3dVector(inlier_points)
            inlier_pcd.paint_uniform_color([0.0, 1.0, 0.0])
            geometries.append(inlier_pcd)
        
        # 外点点云（灰色）
        if len(outlier_points) > 0:
            outlier_pcd = o3d.geometry.PointCloud()
            outlier_pcd.points = o3d.utility.Vector3dVector(outlier_points)
            outlier_pcd.paint_uniform_color([0.5, 0.5, 0.5])
            geometries.append(outlier_pcd)
        
        # 创建圆环（黄色）
        circle = create_sphere_cross_section_circle(center, radius, [0, 0, 1])
        geometries.append(circle)
        
        print(f"  节点{poor_result['index']+1}: 内点{len(inlier_points)}个, 外点{len(outlier_points)}个")
        
        # 显示可视化窗口，设置点大小
        vis = o3d.visualization.Visualizer()
        vis.create_window(
            window_name=f"节点{poor_result['index']+1}_模板拟合_质量{improved_result['quality_score']:.1f}",
            width=1400,
            height=900
        )
        
        # 添加几何体
        for geometry in geometries:
            vis.add_geometry(geometry)
        
        # 设置点大小
        render_option = vis.get_render_option()
        render_option.point_size = 2  # 调小点的大小
        
        # 运行可视化
        vis.run()
        vis.destroy_window()
    
    print("显示说明:")
    print("  绿色点云: 邻域内被识别成球的点（内点）")
    print("  灰色点云: 邻域内其他点（外点）")
    print("  黄色圆环: 球切面圆环示意")


def analyze_sphere_errors(points, spheres, search_radius_factor=2.0, distance_threshold=0.02):
    """
    分析球节点的装配误差。
    
    Args:
        points: 点云坐标 (N, 3)
        spheres: 球节点列表 [(x, y, z, radius), ...]
        search_radius_factor: 搜索半径因子（相对于球半径）
        distance_threshold: RANSAC 距离阈值
        
    Returns:
        results: 误差分析结果列表
    """
    print("\n分析球节点装配误差...")
    
    kd_tree = KDTree(points)
    
    results = []
    
    for i, (cx, cy, cz, radius) in enumerate(spheres):
        model_center = np.array([cx, cy, cz])
        search_radius = radius * search_radius_factor
        
        indices = kd_tree.query_ball_point(model_center, search_radius)
        
        if len(indices) < 50:
            print(f"  球节点 {i+1}: 搜索区域内点数不足 ({len(indices)} 个)")
            results.append({
                'index': i,
                'model_center': model_center,
                'model_radius': radius,
                'cloud_center': None,
                'cloud_radius': None,
                'center_error': None,
                'radius_error': None,
                'inlier_count': 0,
                'status': 'insufficient_points'
            })
            continue
        
        region_points = points[indices]
        
        # 使用改进的RANSAC拟合，包含质量评价
        cloud_center, cloud_radius, inlier_indices, quality_score, stability_score, fit_info = ransac_sphere_fitting(
            region_points,
            max_iterations=100,
            distance_threshold=distance_threshold,
            min_inliers=30,
            num_runs=5,
            random_seed=42
        )
        
        if cloud_center is None:
            print(f"  球节点 {i+1}: RANSAC 拟合失败 ({fit_info.get('error', '未知错误')})")
            results.append({
                'index': i,
                'model_center': model_center,
                'model_radius': radius,
                'cloud_center': None,
                'cloud_radius': None,
                'center_error': None,
                'radius_error': None,
                'inlier_count': len(indices),
                'status': 'fitting_failed',
                'quality_score': 0,
                'stability_score': 0
            })
            continue
        
        center_error = np.linalg.norm(cloud_center - model_center)
        radius_error = abs(cloud_radius - radius)
        
        # 输出拟合结果和质量评价
        print(f"  球节点 {i+1}: 球心误差 = {center_error:.4f}m, 半径误差 = {radius_error:.4f}m, "
              f"质量评分 = {quality_score:.1f}({fit_info['quality_level']}), "
              f"稳定性 = {stability_score:.1f}({fit_info['stability_level']})")
        
        results.append({
            'index': i,
            'model_center': model_center,
            'model_radius': radius,
            'cloud_center': cloud_center,
            'cloud_radius': cloud_radius,
            'center_error': center_error,
            'radius_error': radius_error,
            'inlier_count': len(inlier_indices) if inlier_indices is not None else 0,
            'status': 'success',
            'quality_score': quality_score,
            'stability_score': stability_score,
            'fit_info': fit_info,
            'region_points': region_points
        })
    
    return results


def visualize_errors(points, sphere_results, error_scale=100.0, show_error_vectors=True):
    """
    可视化装配误差。
    只显示识别的球节点，用颜色表示误差大小。
    
    Args:
        points: 点云坐标（不再显示）
        sphere_results: 球节点分析结果
        error_scale: 误差矢量显示缩放因子
        show_error_vectors: 是否显示误差矢量
    """
    print("\n可视化装配误差...")
    
    geometries = []
    
    # 计算误差范围
    center_errors = [r['center_error'] for r in sphere_results if r['status'] == 'success' and r['center_error'] is not None]
    
    if center_errors:
        min_error = min(center_errors)
        max_error = max(center_errors)
        error_range = max_error - min_error
    else:
        min_error = 0
        max_error = 0
        error_range = 0
    
    # 只显示识别的球节点（颜色根据误差大小）
    for result in sphere_results:
        if result['status'] != 'success':
            continue
        
        cloud_center = result['cloud_center']
        cloud_radius = result['cloud_radius']
        center_error = result['center_error']
        
        # 计算颜色（绿色->红色，表示误差从小到大）
        if error_range > 0:
            color_ratio = (center_error - min_error) / error_range
            color_ratio = np.clip(color_ratio, 0, 1)
            color = np.array([color_ratio, 1.0 - color_ratio, 0.0])
        else:
            color = np.array([0.0, 1.0, 0.0])
        
        sphere = o3d.geometry.TriangleMesh.create_sphere(
            radius=cloud_radius,
            resolution=20
        )
        sphere.translate(cloud_center)
        sphere.paint_uniform_color(color.tolist())
        sphere.compute_vertex_normals()
        geometries.append(sphere)
    
    print("显示说明:")
    print("  绿色->红色: 识别的球节点（颜色表示误差大小，绿色=误差小，红色=误差大）")
    if center_errors:
        print(f"  误差范围: {min_error:.3f}m - {max_error:.3f}m")
    print(f"  成功识别的球节点: {len([r for r in sphere_results if r['status'] == 'success'])}/{len(sphere_results)}")
    
    o3d.visualization.draw_geometries(
        geometries,
        window_name="Assembly Error Visualization",
        width=1400,
        height=900
    )


# ==================== 接口错边分析 ====================

def analyze_interface_misalignment(sphere_results, surrounding_nodes, transform_matrix, tolerance=0.5):
    """
    分析接口错边误差。
    
    Args:
        sphere_results: 球节点分析结果
        surrounding_nodes: 周边结构节点
        transform_matrix: 变换矩阵
        tolerance: 匹配容差
        
    Returns:
        dict: 分析结果
    """
    print("\n执行接口错边误差分析...")
    
    # 准备重构节点
    reconstructed_nodes = []
    for result in sphere_results:
        if result['status'] == 'success':
            reconstructed_nodes.append({
                'id': f"node_{result['index']}",
                'coords': result['cloud_center'],
                'radius': result['cloud_radius']
            })
    
    print(f"已准备 {len(reconstructed_nodes)} 个重构节点")
    
    # 处理周边结构节点
    processed_surrounding = []
    for node in surrounding_nodes:
        point = np.array([node['x'], node['y'], node['z'], 1.0])
        transformed_point = transform_matrix @ point
        processed_surrounding.append({
            'id': node['id'],
            'coords': transformed_point[:3],
            'radius': node['radius']
        })
    
    print(f"已处理 {len(processed_surrounding)} 个周边结构节点")
    
    # 打印坐标范围
    if reconstructed_nodes:
        recon_coords = np.array([n['coords'] for n in reconstructed_nodes])
        print(f"重构节点坐标范围:")
        print(f"  X: [{recon_coords[:, 0].min():.3f}, {recon_coords[:, 0].max():.3f}]")
        print(f"  Y: [{recon_coords[:, 1].min():.3f}, {recon_coords[:, 1].max():.3f}]")
        print(f"  Z: [{recon_coords[:, 2].min():.3f}, {recon_coords[:, 2].max():.3f}]")
    
    if processed_surrounding:
        surround_coords = np.array([n['coords'] for n in processed_surrounding])
        print(f"周边结构节点坐标范围（变换后）:")
        print(f"  X: [{surround_coords[:, 0].min():.3f}, {surround_coords[:, 0].max():.3f}]")
        print(f"  Y: [{surround_coords[:, 1].min():.3f}, {surround_coords[:, 1].max():.3f}]")
        print(f"  Z: [{surround_coords[:, 2].min():.3f}, {surround_coords[:, 2].max():.3f}]")
    
    # 计算距离矩阵
    if reconstructed_nodes and processed_surrounding:
        recon_coords = np.array([n['coords'] for n in reconstructed_nodes])
        surround_coords = np.array([n['coords'] for n in processed_surrounding])
        
        distance_matrix = cdist(recon_coords, surround_coords)
        
        print(f"距离矩阵统计信息:")
        print(f"  最小距离: {distance_matrix.min():.3f} m")
        print(f"  最大距离: {distance_matrix.max():.3f} m")
        print(f"  平均距离: {distance_matrix.mean():.3f} m")
        print(f"  当前容差: {tolerance:.3f} m")
        
        # 找到匹配对
        matched_pairs = []
        for i, recon_node in enumerate(reconstructed_nodes):
            min_dist_idx = np.argmin(distance_matrix[i])
            min_dist = distance_matrix[i, min_dist_idx]
            
            if min_dist <= tolerance:
                matched_pairs.append({
                    'reconstructed_node_id': recon_node['id'],
                    'reconstructed_node': recon_node,
                    'surrounding_node_id': processed_surrounding[min_dist_idx]['id'],
                    'surrounding_node': processed_surrounding[min_dist_idx],
                    'distance': min_dist
                })
        
        print(f"找到 {len(matched_pairs)} 对匹配节点")
        
        # 计算接口错边和拼装偏差
        interface_misalignments = []
        assembly_deviations = []
        
        for pair in matched_pairs:
            misalignment_distance = pair['distance']
            interface_misalignments.append({
                'reconstructed_node_id': pair['reconstructed_node_id'],
                'surrounding_node_id': pair['surrounding_node_id'],
                'misalignment_distance': misalignment_distance,
                'recon_coords': pair['reconstructed_node']['coords'],
                'surround_coords': pair['surrounding_node']['coords']
            })
            
            assembly_deviations.append({
                'reconstructed_node_id': pair['reconstructed_node_id'],
                'surrounding_node_id': pair['surrounding_node_id'],
                'deviation': misalignment_distance,
                'recon_coords': pair['reconstructed_node']['coords'],
                'surround_coords': pair['surrounding_node']['coords']
            })
        
        print(f"分析完成：{len(interface_misalignments)} 个接口错边，{len(assembly_deviations)} 个拼装偏差")
        
        return {
            'matched_pairs': matched_pairs,
            'interface_misalignments': interface_misalignments,
            'assembly_deviations': assembly_deviations
        }
    else:
        return {
            'matched_pairs': [],
            'interface_misalignments': [],
            'assembly_deviations': []
        }


def visualize_interface_misalignment(analysis_results):
    """
    可视化接口错边误差分析结果。
    
    Args:
        analysis_results: 接口错边分析结果
    """
    print("\n可视化接口错边误差分析结果...")
    
    geometries = []
    
    # 显示重构的球节点（红色）
    for misalignment in analysis_results['interface_misalignments']:
        center = misalignment['recon_coords']
        radius = 0.1
        
        sphere = o3d.geometry.TriangleMesh.create_sphere(
            radius=radius,
            resolution=20
        )
        sphere.translate(center)
        sphere.paint_uniform_color([1.0, 0.0, 0.0])
        sphere.compute_vertex_normals()
        geometries.append(sphere)
    
    # 显示周边结构节点（蓝色）
    for misalignment in analysis_results['interface_misalignments']:
        center = misalignment['surround_coords']
        radius = 0.1
        
        sphere = o3d.geometry.TriangleMesh.create_sphere(
            radius=radius,
            resolution=20
        )
        sphere.translate(center)
        sphere.paint_uniform_color([0.0, 0.0, 1.0])
        sphere.compute_vertex_normals()
        geometries.append(sphere)
    
    # 显示匹配对之间的连接线（绿色）
    for match in analysis_results['matched_pairs']:
        recon_coords = match['reconstructed_node']['coords']
        surround_coords = match['surrounding_node']['coords']
        
        # 创建圆柱体表示连接线
        cylinder = o3d.geometry.TriangleMesh.create_cylinder(
            radius=0.05,
            height=match['distance'],
            resolution=20
        )
        
        # 计算圆柱体的方向和位置
        direction = surround_coords - recon_coords
        length = np.linalg.norm(direction)
        
        if length > 0:
            direction_normalized = direction / length
            z_axis = np.array([0, 0, 1])
            
            # 计算旋转矩阵
            if np.dot(direction_normalized, z_axis) < 0.999:
                cross = np.cross(z_axis, direction_normalized)
                angle = np.arccos(np.dot(z_axis, direction_normalized))
                rotation = o3d.geometry.get_rotation_matrix_from_axis_angle(cross * angle)
                cylinder.rotate(rotation, center=np.array([0, 0, 0]))
            
            # 放置圆柱体
            cylinder.translate(recon_coords + direction / 2)
            cylinder.paint_uniform_color([0.2, 0.8, 0.2])
            cylinder.compute_vertex_normals()
            geometries.append(cylinder)
    
    print("显示说明:")
    print("  红色球体: 重构的球节点（实际位置）")
    print("  蓝色球体: 周边结构节点（设计位置）")
    print("  绿色圆柱体: 接口错边量")
    print("\n观察要点:")
    print("  1. 红色球与蓝色球的偏差表示接口错边")
    print("  2. 绿色圆柱体的长度表示错边距离")
    print("  3. 关注错边量较大的接口")
    
    o3d.visualization.draw_geometries(
        geometries,
        window_name="Interface Misalignment Visualization",
        width=1400,
        height=900
    )


# ==================== 切面可视化 ====================

def extract_sphere_cross_section(points, center, radius, normal, thickness=0.02):
    """
    提取球面的切面点云。
    
    Args:
        points: 原始点云 (N, 3)
        center: 球心坐标
        radius: 球半径
        normal: 切面法向量
        thickness: 切面厚度（用于投影）
        
    Returns:
        on_surface_points: 在球面上的点
        off_surface_points: 不在球面上的点
    """
    # 计算点到切面的距离
    points_array = np.array(points)
    center_array = np.array(center)
    
    # 点到球心的向量
    vec_to_points = points_array - center_array
    
    # 计算点到球面的距离
    distances_to_center = np.linalg.norm(vec_to_points, axis=1)
    distances_to_surface = np.abs(distances_to_center - radius)
    
    # 计算点到切面的距离
    normal_normalized = normal / np.linalg.norm(normal)
    distances_to_plane = np.abs(np.dot(vec_to_points, normal_normalized))
    
    # 判断点是否在切面上（距离切面很近且在球面上）
    on_surface_mask = (distances_to_plane < thickness) & (distances_to_surface < thickness)
    off_surface_mask = ~on_surface_mask
    
    on_surface_points = points_array[on_surface_mask]
    off_surface_points = points_array[off_surface_mask]
    
    return on_surface_points, off_surface_points


def create_sphere_cross_section_circle(center, radius, normal, tube_radius_factor=0.02):
    """
    创建球切面圆环（使用圆柱体管道增加厚度）。
    
    Args:
        center: 球心坐标
        radius: 球半径
        normal: 切面法向量
        tube_radius_factor: 管道半径相对于球半径的比例
        
    Returns:
        tube_mesh: 圆环管道网格
    """
    # 归一化法向量
    normal = np.array(normal)
    normal = normal / np.linalg.norm(normal)
    
    # 找到两个正交向量，与法向量构成坐标系
    if abs(normal[2]) < 0.9:
        u = np.cross(normal, [0, 0, 1])
    else:
        u = np.cross(normal, [0, 1, 0])
    u = u / np.linalg.norm(u)
    v = np.cross(normal, u)
    v = v / np.linalg.norm(v)
    
    # 创建圆环管道
    tube_radius = radius * tube_radius_factor
    torus = o3d.geometry.TriangleMesh.create_torus(
        torus_radius=radius,
        tube_radius=tube_radius,
        radial_resolution=30,
        tubular_resolution=20
    )
    
    # 旋转圆环使其与切面对齐
    z_axis = np.array([0, 0, 1])
    if np.dot(normal, z_axis) < 0.9999:
        rotation_axis = np.cross(z_axis, normal)
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        angle = np.arccos(np.dot(z_axis, normal))
        R = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_axis * angle)
        torus.rotate(R, center=np.array([0, 0, 0]))
    
    # 平移到球心位置
    torus.translate(center)
    
    # 设置为亮黄色
    torus.paint_uniform_color([1.0, 1.0, 0.0])
    torus.compute_vertex_normals()
    
    return torus


def create_cross_section_visualization(points, center, radius, normal, thickness=0.03, 
                                   node_id="Node", quality_level="Unknown"):
    """
    创建切面可视化（2D平面视图）。
    
    Args:
        points: 点云数据
        center: 球心坐标
        radius: 球半径
        normal: 切面法向量
        thickness: 切面厚度
        node_id: 节点ID
        quality_level: 质量等级
        
    Returns:
        geometries: 可视化几何体列表
    """
    geometries = []
    
    # 提取切面点云
    on_surface_points, off_surface_points = extract_sphere_cross_section(
        points, center, radius, normal, thickness
    )
    
    # 1. 只显示在球面上的点（绿色）
    if len(on_surface_points) > 0:
        on_surface_pcd = o3d.geometry.PointCloud()
        on_surface_pcd.points = o3d.utility.Vector3dVector(on_surface_points)
        on_surface_pcd.paint_uniform_color([0.0, 1.0, 0.0])
        geometries.append(on_surface_pcd)
    
    # 2. 只显示不在球面上的点（红色）
    if len(off_surface_points) > 0:
        off_surface_pcd = o3d.geometry.PointCloud()
        off_surface_pcd.points = o3d.utility.Vector3dVector(off_surface_points)
        off_surface_pcd.paint_uniform_color([1.0, 0.0, 0.0])
        geometries.append(off_surface_pcd)
    
    # 3. 显示球切面圆环（黄色）
    circle = create_sphere_cross_section_circle(center, radius, normal)
    geometries.append(circle)
    
    # 4. 显示球心（黄色）
    center_sphere = o3d.geometry.TriangleMesh.create_sphere(
        radius=radius * 0.05, resolution=10
    )
    center_sphere.translate(center)
    center_sphere.paint_uniform_color([1.0, 1.0, 0.0])
    center_sphere.compute_vertex_normals()
    geometries.append(center_sphere)
    
    return geometries


def visualize_selected_nodes_cross_sections(sphere_results, num_poor=3, num_excellent=3):
    """
    可视化选中节点的切面分析。
    
    Args:
        sphere_results: 球节点分析结果列表
        num_poor: 选取的不合格节点数量
        num_excellent: 选取的优秀节点数量
    """
    # 按质量评分排序
    sorted_results = sorted(
        [r for r in sphere_results if r['status'] == 'success'],
        key=lambda x: x['quality_score']
    )
    
    # 选取不合格节点（质量评分最低的）
    poor_nodes = sorted_results[:num_poor]
    
    # 选取优秀节点（质量评分最高的）
    excellent_nodes = sorted_results[-num_excellent:]
    
    print(f"\n" + "="*60)
    print("切面可视化分析")
    print("="*60)
    print(f"选取 {num_poor} 个不合格节点和 {num_excellent} 个优秀节点进行切面分析")
    print()
    
    # 为每个节点生成切面可视化
    for i, result in enumerate(poor_nodes + excellent_nodes):
        node_type = "不合格" if i < num_poor else "优秀"
        print(f"节点 {result['index']+1} ({node_type}, 质量评分: {result['quality_score']:.1f})")
        
        center = result['cloud_center']
        radius = result['cloud_radius']
        points = result['region_points']
        
        # 使用球心到模型中心的向量作为切面法向量
        model_center = result['model_center']
        normal = center - model_center
        
        if np.linalg.norm(normal) < 0.001:
            normal = np.array([0, 0, 1])
        
        # 提取切面点云
        on_surface_points, off_surface_points = extract_sphere_cross_section(
            points, center, radius, normal, thickness=0.03
        )
        
        # 每次都重新创建点云对象，确保颜色设置正确
        geometries = []
        
        # 1. 只显示在球面上的点（绿色）
        if len(on_surface_points) > 0:
            on_surface_pcd = o3d.geometry.PointCloud()
            on_surface_pcd.points = o3d.utility.Vector3dVector(on_surface_points)
            on_surface_pcd.paint_uniform_color([0.0, 1.0, 0.0])
            geometries.append(on_surface_pcd)
        
        # 2. 只显示不在球面上的点（红色）
        if len(off_surface_points) > 0:
            off_surface_pcd = o3d.geometry.PointCloud()
            off_surface_pcd.points = o3d.utility.Vector3dVector(off_surface_points)
            off_surface_pcd.paint_uniform_color([1.0, 0.0, 0.0])
            geometries.append(off_surface_pcd)
        
        # 3. 显示球切面圆环（黄色）
        circle = create_sphere_cross_section_circle(center, radius, normal)
        geometries.append(circle)
        
        # 4. 显示球心（黄色）
        center_sphere = o3d.geometry.TriangleMesh.create_sphere(
            radius=radius * 0.05, resolution=10
        )
        center_sphere.translate(center)
        center_sphere.paint_uniform_color([1.0, 1.0, 0.0])
        center_sphere.compute_vertex_normals()
        geometries.append(center_sphere)
        
        # 显示
        window_name = f"节点{result['index']+1}_切面分析_{node_type}_质量{result['quality_score']:.1f}"
        
        # 创建可视化器并设置背景色
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=window_name, width=1200, height=800)
        
        # 重置可视化器状态
        vis.clear_geometries()
        
        # 添加几何体
        for geometry in geometries:
            vis.add_geometry(geometry)
        
        # 设置渲染选项 - 暗色背景
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0.15, 0.15, 0.15])
        opt.point_size = 2.0
        opt.light_on = True
        
        # 重置视图并更新
        vis.reset_view_point(True)
        vis.poll_events()
        vis.update_renderer()
        
        # 运行可视化
        vis.run()
        vis.destroy_window()
        
        # 显式清理资源
        del vis, geometries, on_surface_points, off_surface_points


# ==================== 保存结果 ====================

def save_error_results(sphere_results, tube_results, output_path):
    """
    保存误差分析结果到CSV文件。
    
    Args:
        sphere_results: 球节点分析结果
        tube_results: 圆管分析结果
        output_path: 输出文件路径
    """
    print(f"\n保存误差分析结果到: {output_path}")
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        writer.writerow(['球节点装配误差分析结果'])
        writer.writerow([
            '节点编号', '模型球心X', '模型球心Y', '模型球心Z', '模型半径',
            '实际球心X', '实际球心Y', '实际球心Z', '实际半径',
            '球心误差(m)', '半径误差(m)', '内点数量', '质量评分', '稳定性评分', '质量等级', '稳定性等级', '状态'
        ])
        
        for result in sphere_results:
            writer.writerow([
                result['index'] + 1,
                f"{result['model_center'][0]:.6f}",
                f"{result['model_center'][1]:.6f}",
                f"{result['model_center'][2]:.6f}",
                f"{result['model_radius']:.6f}",
                f"{result['cloud_center'][0]:.6f}" if result['cloud_center'] is not None else 'N/A',
                f"{result['cloud_center'][1]:.6f}" if result['cloud_center'] is not None else 'N/A',
                f"{result['cloud_center'][2]:.6f}" if result['cloud_center'] is not None else 'N/A',
                f"{result['cloud_radius']:.6f}" if result['cloud_radius'] is not None else 'N/A',
                f"{result['center_error']:.6f}" if result['center_error'] is not None else 'N/A',
                f"{result['radius_error']:.6f}" if result['radius_error'] is not None else 'N/A',
                result['inlier_count'],
                f"{result['quality_score']:.2f}" if 'quality_score' in result else 'N/A',
                f"{result['stability_score']:.2f}" if 'stability_score' in result else 'N/A',
                result['fit_info']['quality_level'] if 'fit_info' in result else 'N/A',
                result['fit_info']['stability_level'] if 'fit_info' in result else 'N/A',
                result['status']
            ])
    
    print("结果保存完成!")


# ==================== 主函数 ====================

def main():
    """
    主函数：完整的配准、分割、误差分析和接口错边分析流程。
    """
    print("="*60)
    print("3D 桁架结构点云配准、分割与误差分析工具")
    print("="*60)
    
    # ========== 文件路径配置 ==========
    print("\n" + "="*60)
    print("文件路径配置")
    print("="*60)
    
    obj_file = input(f"请输入 OBJ 文件路径 [默认: {DEFAULT_OBJ_FILE}]: ").strip()
    if not obj_file:
        obj_file = DEFAULT_OBJ_FILE
    
    las_file = input(f"请输入 LAS 文件路径 [默认: {DEFAULT_LAS_FILE}]: ").strip()
    if not las_file:
        las_file = DEFAULT_LAS_FILE
    
    csv_path = input(f"请输入 CSV 模型信息文件路径 [默认: {DEFAULT_CSV_PATH}]: ").strip()
    if not csv_path:
        csv_path = DEFAULT_CSV_PATH
    
    surrounding_obj = input(f"请输入周边结构 OBJ 文件路径 [默认: {DEFAULT_SURROUNDING_OBJ}]: ").strip()
    if not surrounding_obj:
        surrounding_obj = DEFAULT_SURROUNDING_OBJ
    
    surrounding_nodes_path = input(f"请输入周边结构节点 CSV 文件路径 [默认: {DEFAULT_SURROUNDING_NODES}]: ").strip()
    if not surrounding_nodes_path:
        surrounding_nodes_path = DEFAULT_SURROUNDING_NODES
    
    try:
        # ========== 第一步：读取并处理 OBJ 文件 ==========
        print("\n" + "="*60)
        print("第一步：读取 OBJ 文件")
        print("="*60)
        
        mesh, vertices = read_obj_mesh(obj_file)
        
        print(f"\nOBJ 模型原始信息:")
        print(f"  顶点范围:")
        print(f"    X: [{vertices[:, 0].min():.3f}, {vertices[:, 0].max():.3f}]")
        print(f"    Y: [{vertices[:, 1].min():.3f}, {vertices[:, 1].max():.3f}]")
        print(f"    Z: [{vertices[:, 2].min():.3f}, {vertices[:, 2].max():.3f}]")
        
        # 计算质心并询问是否平移
        obj_centroid = np.mean(vertices, axis=0)
        print(f"\nOBJ 模型质心坐标: ({obj_centroid[0]:.3f}, {obj_centroid[1]:.3f}, {obj_centroid[2]:.3f})")
        
        translate_obj = input("是否将 OBJ 模型质心平移到坐标原点？(y/n) [默认: y]: ").strip().lower()
        obj_translation_matrix = np.eye(4)
        if translate_obj != 'n':
            # 平移网格顶点
            vertices_centered = vertices - obj_centroid
            mesh.vertices = o3d.utility.Vector3dVector(vertices_centered)
            vertices = vertices_centered
            # 记录平移矩阵
            obj_translation_matrix[:3, 3] = -obj_centroid
            print(f"已将 OBJ 模型平移至原点")
            print(f"  平移后范围:")
            print(f"    X: [{vertices[:, 0].min():.3f}, {vertices[:, 0].max():.3f}]")
            print(f"    Y: [{vertices[:, 1].min():.3f}, {vertices[:, 1].max():.3f}]")
            print(f"    Z: [{vertices[:, 2].min():.3f}, {vertices[:, 2].max():.3f}]")
        else:
            print("保持 OBJ 模型原始坐标不变")
        
        # ========== 第二步：读取 LAS 文件 ==========
        print("\n" + "="*60)
        print("第二步：读取 LAS 点云文件")
        print("="*60)
        
        info = get_las_info(las_file)
        print(f"点云数量: {info['point_count']:,}")
        
        print("\n建议下采样比例:")
        print("  - 快速预览: 100-200")
        print("  - 中等质量: 50-100")
        print("  - 高质量: 10-50")
        
        if info['point_count'] > 50_000_000:
            recommended = 100
        elif info['point_count'] > 10_000_000:
            recommended = 50
        elif info['point_count'] > 1_000_000:
            recommended = 20
        else:
            recommended = 10
        
        subsample_input = input(f"请输入下采样比例 [默认: {recommended}]: ").strip()
        subsample_factor = int(subsample_input) if subsample_input else recommended
        
        las_points = read_las_point_cloud(las_file, subsample_factor=subsample_factor)
        
        print(f"\nLAS 点云原始信息:")
        print(f"  点云范围:")
        print(f"    X: [{las_points[:, 0].min():.3f}, {las_points[:, 0].max():.3f}]")
        print(f"    Y: [{las_points[:, 1].min():.3f}, {las_points[:, 1].max():.3f}]")
        print(f"    Z: [{las_points[:, 2].min():.3f}, {las_points[:, 2].max():.3f}]")
        
        # 计算质心并询问是否平移
        las_centroid = np.mean(las_points, axis=0)
        print(f"\n点云质心坐标: ({las_centroid[0]:.3f}, {las_centroid[1]:.3f}, {las_centroid[2]:.3f})")
        
        las_translation_matrix = np.eye(4)
        translate_las = input("是否将 LAS 点云质心平移到坐标原点？(y/n) [默认: y]: ").strip().lower()
        if translate_las != 'n':
            las_points = las_points - las_centroid
            # 记录平移矩阵
            las_translation_matrix[:3, 3] = -las_centroid
            print(f"已将 LAS 点云平移至原点")
            print(f"  平移后范围:")
            print(f"    X: [{las_points[:, 0].min():.3f}, {las_points[:, 0].max():.3f}]")
            print(f"    Y: [{las_points[:, 1].min():.3f}, {las_points[:, 1].max():.3f}]")
            print(f"    Z: [{las_points[:, 2].min():.3f}, {las_points[:, 2].max():.3f}]")
        else:
            print("保持 LAS 点云原始坐标不变")
        
        # ========== 第三步：在 LAS 点云上选取控制点 ==========
        print("\n" + "="*60)
        print("第三步：在 LAS 点云上选取控制点")
        print("="*60)
        
        las_control_points = pick_points_on_point_cloud(las_points, title="LAS Point Cloud - Select 3 Control Points", num_points=3)
        
        if las_control_points is None:
            print("控制点选取失败，程序退出")
            return
        
        # ========== 第四步：在 OBJ 上选取控制点 ==========
        print("\n" + "="*60)
        print("第四步：在 OBJ 模型上选取控制点")
        print("="*60)
        print("\n重要提示: 请按照与 LAS 点云相同的顺序选取对应的控制点！")
        print("例如: 如果 LAS 上选的是 A->B->C，这里也要选对应的 A'->B'->C'")
        
        obj_control_points = pick_points_on_mesh(mesh, title="OBJ Model - Select 3 Corresponding Control Points", num_points=3)
        
        if obj_control_points is None:
            print("控制点选取失败，程序退出")
            return
        
        # ========== 第五步：计算变换矩阵 ==========
        print("\n" + "="*60)
        print("第五步：计算变换矩阵")
        print("="*60)
        
        print("\n控制点对应关系:")
        print("-"*50)
        for i in range(3):
            print(f"点 {i+1}:")
            print(f"  LAS:  ({las_control_points[i][0]:.3f}, {las_control_points[i][1]:.3f}, {las_control_points[i][2]:.3f})")
            print(f"  OBJ:  ({obj_control_points[i][0]:.3f}, {obj_control_points[i][1]:.3f}, {obj_control_points[i][2]:.3f})")
        
        # 计算从 OBJ 到 LAS 的变换（将结构模型配准到点云上）
        transform = compute_similarity_transform(obj_control_points, las_control_points)
        
        print("\n变换参数:")
        print(f"  缩放因子 s: {transform['s']:.6f}")
        print(f"  平移向量 t: ({transform['t'][0]:.3f}, {transform['t'][1]:.3f}, {transform['t'][2]:.3f})")
        print(f"\n旋转矩阵 R:")
        print(transform['R'])
        
        # ========== 第六步：应用变换 ==========
        print("\n" + "="*60)
        print("第六步：应用变换到结构模型")
        print("="*60)
        
        # 应用变换到 OBJ 模型
        transformed_mesh = apply_transform_to_mesh(mesh, transform['R'], transform['t'], transform['s'])
        
        # 可视化粗配准结果
        print("红色：结构模型（已配准）")
        print("绿色：点云")
        visualize_registration_result(transformed_mesh, las_points, title="Coarse Registration Result: OBJ (Red, Registered) + LAS (Green)")
        
        # ========== 第六步.1：ICP 精配准 ==========
        print("\n" + "="*60)
        print("第六步.1：ICP 精配准")
        print("="*60)
        
        # 转换为 Open3D 格式
        source_pcd = o3d.geometry.PointCloud()
        source_pcd.points = o3d.utility.Vector3dVector(np.asarray(transformed_mesh.vertices))
        
        target_pcd = o3d.geometry.PointCloud()
        target_pcd.points = o3d.utility.Vector3dVector(las_points)
        
        # 计算 ICP 精配准
        print("执行 ICP 精配准...")
        
        # 为了加速 ICP，对点云进行下采样
        if len(las_points) > 10000:
            target_pcd = target_pcd.voxel_down_sample(voxel_size=0.1)
            print(f"目标点云下采样后: {len(target_pcd.points)}")
        
        if len(source_pcd.points) > 10000:
            source_pcd = source_pcd.voxel_down_sample(voxel_size=0.1)
            print(f"源点云下采样后: {len(source_pcd.points)}")
        
        # 执行相似变换 ICP（支持缩放）
        icp_result = similarity_icp(
            source_pcd,
            target_pcd,
            max_iterations=20,
            initial_correspondence_distance=10.0,
            final_correspondence_distance=0.5
        )
        
        print(f"ICP 精配准完成，RMSE: {icp_result['rmse']:.6f}")
        print(f"缩放因子: {icp_result['s']:.6f}")
        
        # 应用 ICP 精配准变换到结构模型
        icp_transform = icp_result['transform_matrix']
        final_transformed_mesh = apply_transform_to_mesh(transformed_mesh, icp_result['R'], icp_result['t'], icp_result['s'])
        
        # 计算组合变换矩阵（粗配准 + ICP 精配准）
        combined_transform = icp_transform @ transform['transform_matrix']
        
        # ========== 第七步：可视化配准结果 ==========
        print("\n" + "="*60)
        print("第七步：可视化配准结果")
        print("="*60)
        
        visualize_registration_result(final_transformed_mesh, las_points, title="Registration Result: OBJ (Red, Registered) + LAS (Green)")
        
        # ========== 第八步：读取 CSV 模型信息 ==========
        print("\n" + "="*60)
        print("第八步：读取模型信息")
        print("="*60)
        
        spheres, tubes = read_csv_model_info(csv_path)
        
        # ========== 第九步：应用变换到模型信息 ==========
        print("\n" + "="*60)
        print("第九步：应用变换到模型信息")
        print("="*60)
        
        # 打印变换前的模型信息示例
        if spheres:
            print(f"变换前第一个球节点: {spheres[0]}")
        if tubes:
            print(f"变换前第一个圆管: {tubes[0]}")
        
        # 首先应用 OBJ 中心化变换
        transformed_spheres = apply_transform_to_spheres(spheres, obj_translation_matrix)
        transformed_tubes = apply_transform_to_tubes(tubes, obj_translation_matrix)
        
        # 打印 OBJ 中心化变换后的模型信息示例
        if transformed_spheres:
            print(f"OBJ 中心化变换后第一个球节点: {transformed_spheres[0]}")
        if transformed_tubes:
            print(f"OBJ 中心化变换后第一个圆管: {transformed_tubes[0]}")
        
        # 然后应用粗配准和 ICP 精配准变换
        transformed_spheres = apply_transform_to_spheres(transformed_spheres, combined_transform)
        transformed_tubes = apply_transform_to_tubes(transformed_tubes, combined_transform)
        
        # 打印最终变换后的模型信息示例
        if transformed_spheres:
            print(f"最终变换后第一个球节点: {transformed_spheres[0]}")
        if transformed_tubes:
            print(f"最终变换后第一个圆管: {transformed_tubes[0]}")
        
        # 打印配准后点云的范围
        print(f"点云范围:")
        print(f"  X: [{las_points[:, 0].min():.3f}, {las_points[:, 0].max():.3f}]")
        print(f"  Y: [{las_points[:, 1].min():.3f}, {las_points[:, 1].max():.3f}]")
        print(f"  Z: [{las_points[:, 2].min():.3f}, {las_points[:, 2].max():.3f}]")
        
        # 打印变换后模型信息的范围
        if transformed_spheres:
            sphere_coords = np.array([s[:3] for s in transformed_spheres])
            print(f"变换后球节点范围:")
            print(f"  X: [{sphere_coords[:, 0].min():.3f}, {sphere_coords[:, 0].max():.3f}]")
            print(f"  Y: [{sphere_coords[:, 1].min():.3f}, {sphere_coords[:, 1].max():.3f}]")
            print(f"  Z: [{sphere_coords[:, 2].min():.3f}, {sphere_coords[:, 2].max():.3f}]")
        
        # ========== 第十步：球节点装配误差分析（RANSAC） ==========
        print("\n" + "="*60)
        print("第十一步：球节点装配误差分析（RANSAC）")
        print("="*60)
        
        sphere_results = analyze_sphere_errors(
            las_points,
            transformed_spheres,
            search_radius_factor=2.0,
            distance_threshold=0.02
        )
        
        # ========== 第十二步：统计误差结果 ==========
        print("\n" + "="*60)
        print("第十二步：统计误差结果")
        print("="*60)
        
        successful_spheres = [r for r in sphere_results if r['status'] == 'success']
        
        if successful_spheres:
            center_errors = [r['center_error'] for r in successful_spheres]
            radius_errors = [r['radius_error'] for r in successful_spheres]
            
            print(f"球节点成功拟合: {len(successful_spheres)}/{len(sphere_results)}")
            print(f"  球心误差:")
            print(f"    平均值: {np.mean(center_errors):.4f} m")
            print(f"    最大值: {np.max(center_errors):.4f} m")
            print(f"    最小值: {np.min(center_errors):.4f} m")
            print(f"    标准差: {np.std(center_errors):.4f} m")
            print(f"  半径误差:")
            print(f"    平均值: {np.mean(radius_errors):.4f} m")
            print(f"    最大值: {np.max(radius_errors):.4f} m")
            print(f"    最小值: {np.min(radius_errors):.4f} m")
            print(f"    标准差: {np.std(radius_errors):.4f} m")
        else:
            print("无成功拟合的球节点")
        
        # ========== 第十三步：可视化误差结果 ==========
        print("\n" + "="*60)
        print("第十三步：可视化误差结果")
        print("="*60)
        
        print("显示识别的球节点（颜色表示误差大小）")
        
        visualize_errors(None, sphere_results, error_scale=100.0, show_error_vectors=False)

        # ========== 第十五步：切面可视化分析 ==========
        print("\n" + "="*60)
        print("第十五步：切面可视化分析")
        print("="*60)
        
        # 对不合格和优秀节点进行切面分析
        visualize_selected_nodes_cross_sections(sphere_results, num_poor=3, num_excellent=3)        
        
        # ========== 第十三步.1：模板拟合改进（针对稳定性不合格的节点） ==========
        print("\n" + "="*60)
        print("第十三步.1：模板拟合改进（针对稳定性不合格的节点）")
        print("="*60)
        
        # 识别稳定性不合格的节点（稳定性等级为"不合格"）
        poor_results = [r for r in sphere_results if r['status'] == 'success' and r['fit_info']['stability_level'] == '不合格']
        
        if poor_results:
            print(f"发现 {len(poor_results)} 个稳定性不合格的节点")
            print("对这些节点使用模板拟合方法进行改进...")
            
            improved_results = []
            
            for result in poor_results:
                print(f"\n  处理节点 {result['index']+1}:")
                print(f"    原始质量评分: {result['quality_score']:.1f}")
                print(f"    原始稳定性等级: {result['fit_info']['stability_level']}")
                print(f"    原始球心: {result['cloud_center']}")
                print(f"    原始半径: {result['cloud_radius']:.4f}m")
                
                # 使用模板拟合
                best_center, best_radius, best_inlier_count, best_score = template_based_sphere_fitting(
                    las_points,
                    result['model_center'],
                    result['model_radius'],
                    search_radius_factor=2.0,
                    max_iterations=50,
                    position_step=0.01,
                    radius_step=0.005
                )
                
                if best_center is not None:
                    # 计算改进后的误差
                    center_error = np.linalg.norm(best_center - result['model_center'])
                    radius_error = abs(best_radius - result['model_radius'])
                    
                    # 创建改进后的结果
                    improved_result = result.copy()
                    improved_result['cloud_center'] = best_center
                    improved_result['cloud_radius'] = best_radius
                    improved_result['center_error'] = center_error
                    improved_result['radius_error'] = radius_error
                    improved_result['inlier_count'] = best_inlier_count
                    improved_result['quality_score'] = best_score
                    
                    print(f"    改进后球心: {best_center}")
                    print(f"    改进后半径: {best_radius:.4f}m")
                    print(f"    改进后质量评分: {best_score:.1f}")
                    
                    improved_results.append(improved_result)
                else:
                    print(f"    模板拟合失败")
                    improved_results.append(result)
            
            # 可视化模板拟合前后的对比（使用点云）
            print("\n可视化模板拟合前后对比...")
            visualize_template_fitting_pointcloud(poor_results, improved_results, las_points)
            
            # 更新sphere_results中的不合格节点，并重新计算内点信息
            from scipy.spatial import KDTree
            kd_tree = KDTree(las_points)
            
            for improved in improved_results:
                for i, result in enumerate(sphere_results):
                    if result['index'] == improved['index']:
                        # 重新从原始点云中获取邻域点并计算内点
                        center = improved['cloud_center']
                        radius = improved['cloud_radius']
                        
                        if center is not None and radius is not None:
                            search_radius = radius * 2.0
                            indices = kd_tree.query_ball_point(center, search_radius)
                            
                            if len(indices) > 0:
                                region_points = las_points[indices]
                                distances = np.linalg.norm(region_points - center, axis=1)
                                inlier_mask = np.abs(distances - radius) < 0.02
                                inlier_indices = np.where(inlier_mask)[0]
                                
                                # 更新内点信息
                                improved['inlier_count'] = len(inlier_indices)
                                improved['region_points'] = region_points
                                improved['region_indices'] = indices
                                improved['inlier_indices'] = inlier_indices
                                
                                print(f"    节点{improved['index']+1}: 重新计算内点 = {len(inlier_indices)}个")
                        
                        sphere_results[i] = improved
                        break
            
            print(f"\n已完成 {len(improved_results)} 个稳定性不合格节点的模板拟合改进")
        else:
            print(f"所有节点稳定性良好，无需模板拟合改进")
        

        
        # ========== 第十六步：周围结构拼装误差分析 ==========
        print("\n" + "="*60)
        print("第十六步：周围结构拼装误差分析")
        print("="*60)
        
        # 读取周围结构文件
        print(f"读取周围结构文件: {surrounding_obj}")
        
        try:
            surrounding_mesh, surrounding_vertices = read_obj_mesh(surrounding_obj)
            
            # 应用变换到周围结构
            print("应用变换到周围结构...")
            # 首先应用 OBJ 中心化变换
            surrounding_mesh_centered = apply_transform_to_mesh(surrounding_mesh, np.eye(3), -obj_centroid, 1.0)
            # 然后应用粗配准和 ICP 精配准变换
            surrounding_mesh_transformed = apply_transform_to_mesh(surrounding_mesh_centered, transform['R'], transform['t'], transform['s'])
            # 应用 ICP 精配准变换
            icp_transform = icp_result['transform_matrix']
            final_surrounding_mesh = surrounding_mesh_transformed.transform(icp_transform)
            
            # 读取周边结构节点信息
            surrounding_nodes = read_surrounding_nodes(surrounding_nodes_path)
            
            # 应用变换到周边结构节点
            if surrounding_nodes:
                print("应用变换到周边结构节点...")
                # 首先应用 OBJ 中心化变换
                transformed_surrounding_nodes = []
                for node in surrounding_nodes:
                    center = np.array([node['x'], node['y'], node['z']])
                    # 应用中心化变换
                    center_centered = center - obj_centroid
                    # 应用粗配准变换
                    center_registered = transform['s'] * (transform['R'] @ center_centered) + transform['t']
                    # 应用 ICP 精配准变换
                    center_final = (icp_transform[:3, :3] @ center_registered) + icp_transform[:3, 3]
                    
                    transformed_node = {
                        'id': node['id'],
                        'x': center_final[0],
                        'y': center_final[1],
                        'z': center_final[2],
                        'radius': node['radius'] * transform['s'] * icp_result['s']
                    }
                    transformed_surrounding_nodes.append(transformed_node)
                
                print(f"已变换 {len(transformed_surrounding_nodes)} 个周边结构节点")
            
            # 执行接口错边误差分析
            identity_matrix = np.eye(4)
            
            if surrounding_nodes:
                analysis_results = analyze_interface_misalignment(
                    sphere_results, 
                    transformed_surrounding_nodes, 
                    identity_matrix, 
                    tolerance=0.5
                )
                
                # 可视化接口错边误差分析结果
                visualize_interface_misalignment(analysis_results)
                
                # 保存接口错边分析结果
                output_path = os.path.join(os.getcwd(), "interface_misalignment_analysis.csv")
                
                try:
                    with open(output_path, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow(['接口错边分析结果'])
                        writer.writerow([
                            '重构节点ID', '周边节点ID', '错边距离(m)', 
                            '重构节点X', '重构节点Y', '重构节点Z', 
                            '周边节点X', '周边节点Y', '周边节点Z'
                        ])
                        
                        for misalignment in analysis_results['interface_misalignments']:
                            writer.writerow([
                                misalignment['reconstructed_node_id'],
                                misalignment['surrounding_node_id'],
                                f"{misalignment['misalignment_distance']:.6f}",
                                f"{misalignment['recon_coords'][0]:.6f}",
                                f"{misalignment['recon_coords'][1]:.6f}",
                                f"{misalignment['recon_coords'][2]:.6f}",
                                f"{misalignment['surround_coords'][0]:.6f}",
                                f"{misalignment['surround_coords'][1]:.6f}",
                                f"{misalignment['surround_coords'][2]:.6f}"
                            ])
                    
                    print(f"接口错边分析结果已保存到: {output_path}")
                except Exception as e:
                    print(f"保存接口错边分析结果时出错: {e}")
            
        except Exception as e:
            print(f"读取或处理周围结构文件时出错: {e}")
        
        # # ========== 第十七步：保存误差结果 ==========
        # print("\n" + "="*60)
        # print("第十七步：保存误差结果")
        # print("="*60)
        
        # # 保存误差分析结果
        # output_path = os.path.join(os.getcwd(), "assembly_error_analysis.csv")
        # save_error_results(sphere_results, [], output_path)
        
        # # ========== 第十七步.1：保存除去节点点云的点云 ==========
        # print("\n" + "="*60)
        # print("第十七步.1：保存除去节点点云的点云（用于轴线装配误差分析）")
        # print("="*60)
        
        # # 读取原始LAS文件（不下采样）
        # print("读取原始点云（不下采样）...")
        # import laspy
        # original_las = laspy.read(las_file)
        # original_points = np.vstack((original_las.x, original_las.y, original_las.z)).T
        # print(f"原始点云点数: {len(original_points):,}")
        
        # # 应用与处理时相同的质心平移变换
        # # sphere_results中的节点坐标是平移后的，需要将原始点云也平移
        # print("应用质心平移变换...")
        # original_points_transformed = original_points - las_centroid
        # print(f"平移后点云范围:")
        # print(f"  X: [{original_points_transformed[:, 0].min():.3f}, {original_points_transformed[:, 0].max():.3f}]")
        # print(f"  Y: [{original_points_transformed[:, 1].min():.3f}, {original_points_transformed[:, 1].max():.3f}]")
        # print(f"  Z: [{original_points_transformed[:, 2].min():.3f}, {original_points_transformed[:, 2].max():.3f}]")
        
        # # 收集所有识别成球的点（在球半径范围内的点）
        # from scipy.spatial import KDTree
        # kd_tree = KDTree(original_points_transformed)
        
        # all_sphere_indices = set()
        # for result in sphere_results:
        #     if result['status'] == 'success' and result['cloud_center'] is not None and result['cloud_radius'] is not None:
        #         center = result['cloud_center']
        #         radius = result['cloud_radius']
                
        #         # 搜索球半径范围内的点（识别成球的点）
        #         indices = kd_tree.query_ball_point(center, radius)
                
        #         if len(indices) > 0:
        #             # 将球半径范围内的点标记为需要删除
        #             all_sphere_indices.update(indices)
        #             print(f"  节点{result['index']+1}: 球内点数 = {len(indices)}个, 半径 = {radius:.4f}m")
        
        # print(f"所有识别成球的点总数: {len(all_sphere_indices):,}")
        
        # # 创建除去识别成球的点后的点云
        # all_indices = set(range(len(original_points)))
        # remaining_indices = list(all_indices - all_sphere_indices)
        # remaining_points = original_points[remaining_indices]  # 使用原始坐标保存
        
        # print(f"除去球节点后的点云点数: {len(remaining_points):,}")
        # print(f"除去球节点后保留比例: {len(remaining_points)/len(original_points)*100:.2f}%")
        
        # # 保存除去节点点云的点云到LAS文件
        # output_dir = os.path.dirname(las_file)
        # base_name = os.path.splitext(os.path.basename(las_file))[0]
        # output_las_path = os.path.join(output_dir, f"{base_name}_without_nodes.las")
        
        # # 创建新的LAS文件
        # print(f"保存除去节点点云的点云到: {output_las_path}")
        
        # # 创建新的header，点数设置为剩余点数
        # new_header = original_las.header.copy()
        # new_header.point_count = len(remaining_points)
        
        # # 创建新的LasData对象
        # new_las = laspy.LasData(new_header)
        
        # # 设置点云数据（使用原始坐标）
        # new_las.x = remaining_points[:, 0]
        # new_las.y = remaining_points[:, 1]
        # new_las.z = remaining_points[:, 2]
        
        # # 如果原始点云有其他属性，也复制过来
        # if hasattr(original_las, 'intensity') and original_las.intensity is not None:
        #     new_las.intensity = original_las.intensity[remaining_indices]
        # if hasattr(original_las, 'red') and original_las.red is not None:
        #     new_las.red = original_las.red[remaining_indices]
        # if hasattr(original_las, 'green') and original_las.green is not None:
        #     new_las.green = original_las.green[remaining_indices]
        # if hasattr(original_las, 'blue') and original_las.blue is not None:
        #     new_las.blue = original_las.blue[remaining_indices]
        
        # # 保存LAS文件
        # new_las.write(output_las_path)
        # print(f"已成功保存除去节点点云的点云!")
        
        # # ========== 第十八步：保存配准结果 ==========
        # print("\n" + "="*60)
        # print("第十八步：保存配准结果")
        # print("="*60)
        
        # save = input("是否保存配准结果？(y/n) [默认: y]: ").strip().lower()
        
        # if save != 'n':
        #     output_dir = os.path.dirname(las_file)
        #     base_name = os.path.splitext(os.path.basename(las_file))[0] + "_registered"
            
        #     # 1. 保存配准后的结构模型
        #     mesh_path = os.path.join(output_dir, f"{base_name}_mesh.obj")
        #     o3d.io.write_triangle_mesh(mesh_path, final_transformed_mesh)
        #     print(f"配准后的结构模型已保存: {mesh_path}")
            
        #     # 2. 保存所有变换矩阵
        #     transform_path = os.path.join(output_dir, f"{base_name}_transform.txt")
        #     with open(transform_path, 'w') as f:
        #         f.write("Registration Transformation Parameters\n")
        #         f.write("="*60 + "\n\n")
                
        #         f.write("1. OBJ Model Translation (to origin):\n")
        #         f.write(f"   Centroid: {obj_centroid}\n")
        #         f.write(f"   Translation Matrix:\n{obj_translation_matrix}\n\n")
                
        #         f.write("2. LAS Point Cloud Translation (to origin):\n")
        #         f.write(f"   Centroid: {las_centroid}\n")
        #         f.write(f"   Translation Matrix:\n{las_translation_matrix}\n\n")
                
        #         f.write("3. Similarity Transform (OBJ to LAS - 粗配准):\n")
        #         f.write(f"   Scale: {transform['s']:.10f}\n")
        #         f.write(f"   Translation:\n{transform['t']}\n")
        #         f.write(f"   Rotation Matrix:\n{transform['R']}\n")
        #         f.write(f"   4x4 Transform Matrix:\n{transform['transform_matrix']}\n\n")
                
        #         f.write("4. ICP Refinement Transform (精配准):\n")
        #         f.write(f"   RMSE: {icp_result['rmse']:.6f}\n")
        #         f.write(f"   ICP Transform Matrix:\n{icp_transform}\n\n")
                
        #         f.write("5. Combined Transform (OBJ original -> LAS space - 粗配准):\n")
        #         # 计算组合变换：OBJ原始坐标 -> OBJ中心 -> 配准变换 -> LAS坐标
        #         combined_transform_coarse = transform['transform_matrix'] @ obj_translation_matrix
        #         f.write(f"   Combined Matrix:\n{combined_transform_coarse}\n\n")
                
        #         f.write("6. Final Combined Transform (OBJ original -> LAS space - 精配准):\n")
        #         # 计算最终组合变换：OBJ原始坐标 -> OBJ中心 -> 粗配准变换 -> ICP精配准 -> LAS坐标
        #         final_combined_transform = icp_transform @ transform['transform_matrix'] @ obj_translation_matrix
        #         f.write(f"   Final Combined Matrix:\n{final_combined_transform}\n")
            
        #     print(f"\n所有变换参数已保存: {transform_path}")
        
        print("\n" + "="*60)
        print("分析完成!")
        print("="*60)
        
    except FileNotFoundError as e:
        print(f"\n错误: {e}")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
