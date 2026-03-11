import numpy as np
import open3d as o3d
import os
import csv
import re


def load_obj_file(file_path):
    """
    读取 OBJ 文件，返回网格对象和顶点数组。
    支持多种编码格式（UTF-8, GBK, GB2312, Latin-1）。
    """
    print(f"正在读取 OBJ 文件: {file_path}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1', 'cp1252']
    mesh = None
    
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()
            mesh = o3d.io.read_triangle_mesh(file_path)
            if not mesh.is_empty():
                print(f"成功使用 {encoding} 编码读取文件")
                break
        except UnicodeDecodeError:
            continue
        except Exception as e:
            if "codec" in str(e).lower():
                continue
            raise
    
    if mesh is None or mesh.is_empty():
        print("尝试手动解析 OBJ 文件...")
        mesh = parse_obj_file_manual(file_path)
    
    if mesh.is_empty():
        raise ValueError(f"无法读取 OBJ 文件或文件为空: {file_path}")
    
    vertices = np.asarray(mesh.vertices)
    
    print(f"读取完成!")
    print(f"  顶点数: {len(mesh.vertices):,}")
    print(f"  三角面数: {len(mesh.triangles):,}")
    
    return mesh, vertices


def parse_obj_file_manual(file_path):
    """手动解析 OBJ 文件，处理编码问题。"""
    vertices = []
    faces = []
    
    encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1', 'cp1252']
    lines = None
    
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
                lines = f.readlines()
            break
        except:
            continue
    
    if lines is None:
        with open(file_path, 'rb') as f:
            raw_content = f.read()
            for encoding in encodings:
                try:
                    content = raw_content.decode(encoding, errors='ignore')
                    lines = content.splitlines()
                    break
                except:
                    continue
    
    if lines is None:
        raise ValueError("无法解析 OBJ 文件，所有编码尝试均失败")
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        
        parts = line.split()
        if not parts:
            continue
        
        if parts[0] == 'v':
            try:
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                vertices.append([x, y, z])
            except (ValueError, IndexError):
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
            except (ValueError, IndexError):
                continue
    
    mesh = o3d.geometry.TriangleMesh()
    
    if vertices:
        mesh.vertices = o3d.utility.Vector3dVector(np.array(vertices))
    
    if faces:
        mesh.triangles = o3d.utility.Vector3iVector(np.array(faces))
    
    if len(mesh.triangles) > 0:
        mesh.compute_vertex_normals()
    
    return mesh


def pick_points_on_mesh(mesh, title="Select Control Points", num_points=3):
    """
    在网格上交互式选取控制点。
    直接使用网格顶点进行选点。
    
    Args:
        mesh (o3d.geometry.TriangleMesh): 网格对象
        title (str): 窗口标题
        num_points (int): 需要选取的点数
        
    Returns:
        numpy.ndarray: 选取的控制点坐标数组 (num_points, 3)
    """
    print(f"\n请在可视化窗口中选取 {num_points} 个控制点")
    print("操作说明:")
    print("  - 按住 Shift + 左键点击: 选取点")
    print("  - 按住 Shift + 右键点击: 取消选取")
    print("  - 按 'Q' 或 'ESC' 键: 完成选取并关闭窗口")
    print(f"  - 请确保选取恰好 {num_points} 个点")
    print("  - 提示: 可以正常使用鼠标旋转、平移、缩放视角")
    
    mesh_copy = o3d.geometry.TriangleMesh(mesh)
    mesh_copy.compute_vertex_normals()
    if not mesh_copy.has_vertex_colors():
        mesh_copy.paint_uniform_color([0.7, 0.7, 0.7])
    
    vertices = np.asarray(mesh_copy.vertices)
    
    # 使用网格顶点创建点云
    vertex_pcd = o3d.geometry.PointCloud()
    vertex_pcd.points = o3d.utility.Vector3dVector(vertices)
    vertex_pcd.paint_uniform_color([0.0, 0.7, 1.0])  # 蓝色，更明显
    
    print(f"已使用网格的 {len(vertices):,} 个顶点进行选点")
    
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name=title, width=1400, height=900)
    # 只添加顶点的点云用于选点
    vis.add_geometry(vertex_pcd)
    
    render_option = vis.get_render_option()
    render_option.point_size = 3  # 增大点大小
    render_option.mesh_show_back_face = True
    
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


def compute_similarity_transform(source_points, target_points):
    """
    计算两组 3D 点之间的相似变换（旋转 + 平移 + 缩放）。
    将 source_points 变换到 target_points 的坐标系。
    
    Args:
        source_points (numpy.ndarray): 源点集（LAS点云的控制点）(N, 3)
        target_points (numpy.ndarray): 目标点集（OBJ模型的控制点）(N, 3)
        
    Returns:
        dict: 包含变换参数的字典
    """
    assert source_points.shape == target_points.shape
    assert source_points.shape[0] >= 3
    
    centroid_source = np.mean(source_points, axis=0)
    centroid_target = np.mean(target_points, axis=0)
    
    source_centered = source_points - centroid_source
    target_centered = target_points - centroid_target
    
    norm_source = np.sqrt(np.sum(source_centered ** 2))
    norm_target = np.sqrt(np.sum(target_centered ** 2))
    
    scale = norm_target / norm_source if norm_source > 0 else 1.0
    
    H = source_centered.T @ target_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T
    
    t = centroid_target - scale * R @ centroid_source
    
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = scale * R
    transform_matrix[:3, 3] = t
    
    return {
        'R': R,
        't': t,
        's': scale,
        'transform_matrix': transform_matrix
    }


def apply_transform_to_point_cloud(points, R, t, s=1.0):
    """
    将变换应用到点云。
    
    Args:
        points (numpy.ndarray): 点云坐标 (N, 3)
        R (numpy.ndarray): 3x3 旋转矩阵
        t (numpy.ndarray): 3x1 平移向量
        s (float): 缩放因子
        
    Returns:
        numpy.ndarray: 变换后的点云坐标
    """
    transformed_points = s * (R @ points.T).T + t
    return transformed_points


def apply_transform_to_mesh(mesh, R, t, s=1.0):
    """
    将变换应用到网格。
    
    Args:
        mesh (o3d.geometry.TriangleMesh): 网格对象
        R (numpy.ndarray): 3x3 旋转矩阵
        t (numpy.ndarray): 3x1 平移向量
        s (float): 缩放因子
        
    Returns:
        o3d.geometry.TriangleMesh: 变换后的网格
    """
    vertices = np.asarray(mesh.vertices)
    transformed_vertices = s * (R @ vertices.T).T + t
    mesh.vertices = o3d.utility.Vector3dVector(transformed_vertices)
    return mesh


def visualize_registration_result(mesh, points, title="Registration Result"):
    """
    可视化配准结果（同时显示网格和点云）。
    
    Args:
        mesh (o3d.geometry.TriangleMesh): 网格对象
        points (numpy.ndarray): 点云坐标 (N, 3)
        title (str): 窗口标题
    """
    mesh_copy = o3d.geometry.TriangleMesh(mesh)
    mesh_copy.paint_uniform_color([0.8, 0.2, 0.2])  # 红色网格
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color([0.2, 0.8, 0.2])  # 绿色点云
    
    print(f"\n可视化配准结果:")
    print("  红色: OBJ 网格模型")
    print("  绿色: LAS 点云（已配准）")
    print("  按 'Q' 或 'ESC' 键关闭窗口")
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title, width=1400, height=900)
    vis.add_geometry(mesh_copy)
    vis.add_geometry(pcd)
    
    vis.get_render_option().point_size = 2.0
    vis.get_render_option().background_color = np.array([0.1, 0.1, 0.1])
    
    vis.run()
    vis.destroy_window()


def save_transformed_data(mesh, points, output_dir, base_name):
    """
    保存配准后的数据。
    
    Args:
        mesh: 网格对象
        points: 点云坐标
        output_dir: 输出目录
        base_name: 基础文件名
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    mesh_path = os.path.join(output_dir, f"{base_name}_mesh.obj")
    o3d.io.write_triangle_mesh(mesh_path, mesh)
    print(f"网格已保存: {mesh_path}")
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd_path = os.path.join(output_dir, f"{base_name}_pointcloud.ply")
    o3d.io.write_point_cloud(pcd_path, pcd)
    print(f"点云已保存: {pcd_path}")


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



def read_csv_model_info(csv_path):
    """
    读取 CSV 文件中的模型信息（球节点和圆管）。
    CSV 结构：
    - 第一列: Object Type (Sphere 或 Tube)
    - 球节点: X, Y, Z, Radius (第2-5列)
    - 圆管: Radius, Start_X, Start_Y, Start_Z, End_X, End_Y, End_Z (第5-11列)
    """
    spheres = []  # (center_x, center_y, center_z, radius)
    tubes = []    # (start_x, start_y, start_z, end_x, end_y, end_z, radius)
    
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


def read_transform_matrix(txt_path):
    """
    读取变换矩阵文件，提取 OBJ 中心化矩阵和 ICP 精配准矩阵。
    """
    print(f"读取变换矩阵文件: {txt_path}")
    
    # 尝试不同的编码
    encodings = ['utf-8', 'gbk', 'gb2312']
    content = None
    
    for encoding in encodings:
        try:
            with open(txt_path, 'r', encoding=encoding) as f:
                content = f.read()
            print(f"成功使用 {encoding} 编码读取文件")
            break
        except UnicodeDecodeError:
            continue
    
    if content is None:
        # 最后尝试使用二进制模式读取并忽略错误
        try:
            with open(txt_path, 'rb') as f:
                content = f.read().decode('utf-8', errors='ignore')
            print("使用二进制模式读取并忽略编码错误")
        except Exception as e:
            raise ValueError(f"无法读取文件: {e}")
    
    # 提取 OBJ 中心化矩阵
    obj_matrix_pattern = r"1\. OBJ Model Translation \(to origin\):.*?Translation Matrix:(.*?)2\. LAS Point Cloud Translation" 
    match = re.search(obj_matrix_pattern, content, re.DOTALL)
    
    if not match:
        raise ValueError("未找到 OBJ 中心化矩阵")
    
    matrix_text = match.group(1)
    
    # 解析矩阵
    matrix = []
    
    # 清理文本，移除多余的空白和括号
    matrix_text = matrix_text.strip()
    # 移除最外层的括号
    if matrix_text.startswith('[[') and matrix_text.endswith(']]'):
        matrix_text = matrix_text[1:-1]
    
    # 按行分割
    matrix_lines = matrix_text.split('\n')
    
    for line in matrix_lines:
        line = line.strip()
        if not line:
            continue
        
        # 移除行首尾的括号
        if line.startswith('['):
            line = line[1:]
        if line.endswith(']'):
            line = line[:-1]
        
        # 分割数字
        numbers = line.split()
        if not numbers:
            continue
        
        # 转换为浮点数
        try:
            row = [float(num) for num in numbers]
            matrix.append(row)
        except ValueError as e:
            print(f"解析行时出错: {line}")
            print(f"错误: {e}")
            continue
    
    if not matrix:
        raise ValueError("无法解析变换矩阵")
    
    obj_transform_matrix = np.array(matrix)
    print(f"成功读取 OBJ 中心化矩阵:")
    print(obj_transform_matrix)
    
    # 提取 ICP 精配准矩阵
    icp_matrix_pattern = r"4\. ICP Refinement Transform.*?ICP Transform Matrix:(.*?)5\. Combined Transform" 
    match = re.search(icp_matrix_pattern, content, re.DOTALL)
    
    icp_transform_matrix = np.eye(4)  # 默认单位矩阵
    if match:
        icp_matrix_text = match.group(1)
        
        # 解析 ICP 矩阵
        icp_matrix = []
        
        # 清理文本
        icp_matrix_text = icp_matrix_text.strip()
        if icp_matrix_text.startswith('[[') and icp_matrix_text.endswith(']]'):
            icp_matrix_text = icp_matrix_text[1:-1]
        
        # 按行分割
        icp_matrix_lines = icp_matrix_text.split('\n')
        
        for line in icp_matrix_lines:
            line = line.strip()
            if not line:
                continue
            
            # 移除行首尾的括号
            if line.startswith('['):
                line = line[1:]
            if line.endswith(']'):
                line = line[:-1]
            
            # 分割数字
            numbers = line.split()
            if not numbers:
                continue
            
            # 转换为浮点数
            try:
                row = [float(num) for num in numbers]
                icp_matrix.append(row)
            except ValueError as e:
                print(f"解析 ICP 矩阵行时出错: {line}")
                print(f"错误: {e}")
                continue
        
        if icp_matrix:
            icp_transform_matrix = np.array(icp_matrix)
            print(f"成功读取 ICP 精配准矩阵:")
            print(icp_transform_matrix)
    
    return obj_transform_matrix, icp_transform_matrix


def apply_transform_to_spheres(spheres, transform_matrix):
    """
    将变换矩阵应用到球节点。
    """
    transformed_spheres = []
    
    # 提取缩放因子（从变换矩阵的对角线元素）
    scale = np.linalg.norm(transform_matrix[:3, 0])
    
    for x, y, z, radius in spheres:
        # 转换为齐次坐标
        point = np.array([x, y, z, 1.0])
        # 应用变换
        transformed_point = transform_matrix @ point
        # 转换回笛卡尔坐标
        tx, ty, tz, _ = transformed_point
        # 应用缩放因子到半径
        scaled_radius = radius * scale
        transformed_spheres.append((tx, ty, tz, scaled_radius))
    
    return transformed_spheres


def apply_transform_to_tubes(tubes, transform_matrix):
    """
    将变换矩阵应用到圆管。
    """
    transformed_tubes = []
    
    # 提取缩放因子（从变换矩阵的对角线元素）
    scale = np.linalg.norm(transform_matrix[:3, 0])
    
    for sx, sy, sz, ex, ey, ez, radius in tubes:
        # 转换起点为齐次坐标
        start_point = np.array([sx, sy, sz, 1.0])
        transformed_start = transform_matrix @ start_point
        tsx, tsy, tsz, _ = transformed_start
        
        # 转换终点为齐次坐标
        end_point = np.array([ex, ey, ez, 1.0])
        transformed_end = transform_matrix @ end_point
        tex, tey, tez, _ = transformed_end
        
        # 应用缩放因子到半径
        scaled_radius = radius * scale
        transformed_tubes.append((tsx, tsy, tsz, tex, tey, tez, scaled_radius))
    
    return transformed_tubes


def read_las_point_cloud(file_path, subsample_factor=10):
    """
    读取点云文件（支持 LAS/LAZ/PLY 格式）。
    """
    # 移除路径中的引号
    file_path = file_path.strip().strip('"').strip("'")
    
    print(f"读取点云文件: {file_path}")
    
    # 检查文件扩展名
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext in ['.las', '.laz']:
        # 使用 laspy 读取 LAS/LAZ 文件
        try:
            import laspy
            las = laspy.read(file_path)
            points = np.vstack([las.x, las.y, las.z]).transpose()
            print(f"原始点云数量: {len(points):,}")
            
            # 下采样
            if subsample_factor > 1:
                indices = np.random.choice(len(points), len(points) // subsample_factor, replace=False)
                points = points[indices]
                print(f"下采样后点云数量: {len(points):,}")
            
        except Exception as e:
            print(f"使用 laspy 读取失败: {e}")
            raise
    
    elif ext in ['.ply', '.pcd', '.xyz']:
        # 使用 Open3D 读取其他格式
        try:
            pcd = o3d.io.read_point_cloud(file_path)
            points = np.asarray(pcd.points)
            print(f"点云数量: {len(points):,}")
            
            # 下采样
            if subsample_factor > 1:
                indices = np.random.choice(len(points), len(points) // subsample_factor, replace=False)
                points = points[indices]
                print(f"下采样后点云数量: {len(points):,}")
                
        except Exception as e:
            print(f"使用 Open3D 读取失败: {e}")
            raise
    
    else:
        raise ValueError(f"不支持的文件格式: {ext}")
    
    return points


def point_to_line_segment_distance(point, seg_start, seg_end):
    """
    计算点到线段的距离。
    """
    # 线段向量
    seg_vec = seg_end - seg_start
    seg_len_sq = np.dot(seg_vec, seg_vec)
    
    # 处理线段退化为点的情况
    if seg_len_sq == 0:
        return np.linalg.norm(point - seg_start)
    
    # 点到线段起点的向量
    pt_vec = point - seg_start
    
    # 计算投影参数 t
    t = np.dot(pt_vec, seg_vec) / seg_len_sq
    t = np.clip(t, 0.0, 1.0)  # 限制在 [0, 1] 范围内
    
    # 计算线段上最近的点
    closest_pt = seg_start + t * seg_vec
    
    # 返回距离
    return np.linalg.norm(point - closest_pt)


def segment_point_cloud(points, spheres, tubes):
    """
    基于模板信息对 LAS 点云进行分割。
    
    Returns:
        numpy.ndarray: 每个点的类别标签
                      0: 未分配
                      1: 球节点
                      2: 圆管
    """
    print(f"开始分割点云，共 {len(points):,} 个点...")
    
    labels = np.zeros(len(points), dtype=int)  # 0: 未分配
    
    # 1. 分割球节点
    print(f"分割球节点...")
    for i, (cx, cy, cz, radius) in enumerate(spheres):
        if i % 5 == 0:
            print(f"处理球节点 {i+1}/{len(spheres)}")
        
        # 计算所有点到球心的距离
        center = np.array([cx, cy, cz])
        distances = np.linalg.norm(points - center, axis=1)
        
        # 标记在球内的点
        sphere_mask = distances <= radius * 1.3  # 允许 30% 的误差
        labels[sphere_mask] = i + 1  # 使用不同的标签区分不同球节点
    
    # 2. 分割圆管（只处理未分配的点）
    unassigned_mask = labels == 0
    unassigned_points = points[unassigned_mask]
    
    print(f"分割圆管...")
    print(f"未分配点数量: {len(unassigned_points):,}")
    print(f"圆管数量: {len(tubes):,}")
    
    if len(unassigned_points) > 0:
        # 预计算未分配点的索引
        unassigned_indices = np.where(unassigned_mask)[0]
        
        # 构建KD树以加速距离计算
        from scipy.spatial import KDTree
        kd_tree = KDTree(unassigned_points)
        
        # 批量处理圆管，使用向量化计算和KD树加速
        for i, (sx, sy, sz, ex, ey, ez, radius) in enumerate(tubes):
            if i % 5 == 0:  # 每5个圆管显示一次进度
                print(f"处理圆管 {i+1}/{len(tubes)}")
            
            start = np.array([sx, sy, sz])
            end = np.array([ex, ey, ez])
            
            # 计算圆管的包围盒，用于提前过滤
            min_x = min(start[0], end[0]) - radius * 1.3
            max_x = max(start[0], end[0]) + radius * 1.3
            min_y = min(start[1], end[1]) - radius * 1.3
            max_y = max(start[1], end[1]) + radius * 1.3
            min_z = min(start[2], end[2]) - radius * 1.3
            max_z = max(start[2], end[2]) + radius * 1.3
            
            # 过滤出在包围盒内的点
            box_mask = (unassigned_points[:, 0] >= min_x) & (unassigned_points[:, 0] <= max_x) & \
                       (unassigned_points[:, 1] >= min_y) & (unassigned_points[:, 1] <= max_y) & \
                       (unassigned_points[:, 2] >= min_z) & (unassigned_points[:, 2] <= max_z)
            
            if not np.any(box_mask):
                continue  # 跳过没有点在包围盒内的圆管
            
            # 只处理包围盒内的点
            box_points = unassigned_points[box_mask]
            box_indices = unassigned_indices[box_mask]
            
            # 向量化计算点到线段的距离
            seg_vec = end - start
            seg_len_sq = np.dot(seg_vec, seg_vec)
            
            if seg_len_sq == 0:
                # 线段退化为点
                distances = np.linalg.norm(box_points - start, axis=1)
            else:
                # 向量化计算点到线段的距离
                pt_vec = box_points - start
                t = np.dot(pt_vec, seg_vec) / seg_len_sq
                t = np.clip(t, 0.0, 1.0)
                closest_pt = start + t[:, np.newaxis] * seg_vec
                distances = np.linalg.norm(box_points - closest_pt, axis=1)
            
            # 标记在线管内的点
            tube_mask = distances <= radius * 1.3  # 允许 30% 的误差
            
            # 更新标签
            if np.any(tube_mask):
                tube_indices = box_indices[tube_mask]
                labels[tube_indices] = len(spheres) + i + 1  # 使用不同的标签区分不同圆管
                print(f"  圆管 {i+1} 分配了 {len(tube_indices):,} 个点")
    else:
        print("没有未分配的点，跳过圆管分割")
    
    # 统计结果
    total_points = len(points)
    assigned_points = np.sum(labels != 0)
    unassigned_points = total_points - assigned_points
    
    print(f"\n分割结果:")
    print(f"总点数: {total_points:,}")
    print(f"已分配点数: {assigned_points:,} ({assigned_points/total_points*100:.2f}%)")
    print(f"未分配点数: {unassigned_points:,} ({unassigned_points/total_points*100:.2f}%)")
    
    return labels


def visualize_segmentation(points, labels):
    """
    可视化分割结果。
    """
    print("\n可视化分割结果...")
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # 为不同类别分配颜色
    colors = np.zeros((len(points), 3))
    
    # 获取所有唯一标签
    unique_labels = np.unique(labels)
    
    # 为每个标签生成唯一颜色
    label_colors = {}
    
    # 未分配点 - 灰色
    label_colors[0] = [0.5, 0.5, 0.5]
    
    # 为其他标签生成随机颜色
    for label in unique_labels:
        if label != 0:
            # 生成随机颜色，确保区分度
            color = np.random.rand(3)
            # 确保颜色不太暗
            color = np.maximum(color, 0.3)
            label_colors[label] = color
    
    # 分配颜色
    for i, label in enumerate(labels):
        colors[i] = label_colors[label]
    
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # 创建可视化器
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="点云分割结果", width=1400, height=900)
    vis.add_geometry(pcd)
    
    # 设置渲染选项
    render_option = vis.get_render_option()
    render_option.point_size = 2.0
    render_option.background_color = np.array([0.1, 0.1, 0.1])
    
    print("可视化窗口已打开，按 'Q' 键关闭")
    print("颜色说明:")
    print("  灰色: 未分配")
    print(f"  其他颜色: 不同的球节点和圆管")
    print(f"  共 {len(unique_labels) - 1} 个不同的组件")
    
    vis.run()
    vis.destroy_window()


def save_segmentation_result(points, labels, output_path):
    """
    保存分割结果。
    """
    print(f"\n保存分割结果到: {output_path}")
    
    # 为不同类别分配颜色
    colors = np.zeros((len(points), 3))
    
    # 获取所有唯一标签
    unique_labels = np.unique(labels)
    
    # 为每个标签生成唯一颜色
    label_colors = {}
    
    # 未分配点 - 灰色
    label_colors[0] = [0.5, 0.5, 0.5]
    
    # 为其他标签生成随机颜色
    for label in unique_labels:
        if label != 0:
            # 生成随机颜色，确保区分度
            color = np.random.rand(3)
            # 确保颜色不太暗
            color = np.maximum(color, 0.3)
            label_colors[label] = color
    
    # 分配颜色
    for i, label in enumerate(labels):
        colors[i] = label_colors[label]
    
    # 保存为 PLY 文件
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(output_path, pcd)
    
    # 保存标签为 CSV
    csv_path = os.path.splitext(output_path)[0] + "_labels.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['X', 'Y', 'Z', 'Label'])
        for i, (x, y, z) in enumerate(points):
            writer.writerow([x, y, z, labels[i]])
    
    print(f"标签已保存到: {csv_path}")


def main():
    """
    主函数：完整的配准和分割流程
    1. 读取 OBJ 模型和 LAS 点云
    2. 交互式选取控制点进行粗配准
    3. 执行 s-ICP 精配准
    4. 读取 CSV 模型信息
    5. 应用变换到模型信息
    6. 分割点云
    7. 可视化结果
    8. 保存结果
    """
    # 文件路径
    default_obj_file = r"D:\1-学术\预拼装\结构模型\泉州项目点云及模型\Final\44品-separated-complete.obj"
    default_las_file = r"D:\1-学术\预拼装\结构模型\泉州项目点云及模型\Final\44-unset-部分去噪.las"
    default_csv_path = r"D:\1-学术\预拼装\结构模型\泉州项目点云及模型\Final\mesh_analysis_results.csv"
    
    print("="*60)
    print("3D 桁架结构点云配准与分割工具")
    print("="*60)
    
    try:
        # ========== 第一步：读取并处理 OBJ 文件 ==========
        print("\n" + "="*60)
        print("第一步：读取 OBJ 文件")
        print("="*60)
        
        obj_file_path = input(f"请输入 OBJ 文件路径 [默认: {default_obj_file}]: ").strip()
        if not obj_file_path:
            obj_file_path = default_obj_file
        
        mesh, vertices = load_obj_file(obj_file_path)
        
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
        
        las_file_path = input(f"请输入 LAS 文件路径 [默认: {default_las_file}]: ").strip()
        if not las_file_path:
            las_file_path = default_las_file
        
        info = get_las_info(las_file_path)
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
        
        las_points = read_las_point_cloud(las_file_path, subsample_factor=subsample_factor)
        
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
        print("\n" + "="*60)
        print("第六步.0：可视化粗配准结果")
        print("="*60)
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
        
        csv_path = input(f"请输入 CSV 模型信息文件路径 [默认: {default_csv_path}]: ").strip()
        if not csv_path:
            csv_path = default_csv_path
        
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
        
        # ========== 第十步：点云分割 ==========
        print("\n" + "="*60)
        print("第十步：点云分割")
        print("="*60)
        
        # 使用原始点云进行分割，因为结构模型已经被配准到点云上
        labels = segment_point_cloud(las_points, transformed_spheres, transformed_tubes)
        
        # ========== 第十一步：可视化分割结果 ==========
        print("\n" + "="*60)
        print("第十一步：可视化分割结果")
        print("="*60)
        
        # 可视化分割结果
        visualize_segmentation(las_points, labels)
        
        # ========== 第十二步：保存结果 ==========
        print("\n" + "="*60)
        print("第十二步：保存结果")
        print("="*60)
        
        save = input("是否保存结果？(y/n) [默认: y]: ").strip().lower()
        
        if save != 'n':
            output_dir = os.path.dirname(las_file_path)
            base_name = os.path.splitext(os.path.basename(las_file_path))[0] + "_registered"
            
            # 1. 保存配准后的结构模型
            mesh_path = os.path.join(output_dir, f"{base_name}_mesh.obj")
            o3d.io.write_triangle_mesh(mesh_path, final_transformed_mesh)
            print(f"配准后的结构模型已保存: {mesh_path}")
            
            # 2. 保存分割结果
            output_path = os.path.join(output_dir, f"{base_name}_segmented.ply")
            save_segmentation_result(las_points, labels, output_path)
            
            # 3. 保存所有变换矩阵
            transform_path = os.path.join(output_dir, f"{base_name}_transform.txt")
            with open(transform_path, 'w') as f:
                f.write("Registration Transformation Parameters\n")
                f.write("="*60 + "\n\n")
                
                f.write("1. OBJ Model Translation (to origin):\n")
                f.write(f"   Centroid: {obj_centroid}\n")
                f.write(f"   Translation Matrix:\n{obj_translation_matrix}\n\n")
                
                f.write("2. LAS Point Cloud Translation (to origin):\n")
                f.write(f"   Centroid: {las_centroid}\n")
                f.write(f"   Translation Matrix:\n{las_translation_matrix}\n\n")
                
                f.write("3. Similarity Transform (OBJ to LAS - 粗配准):\n")
                f.write(f"   Scale: {transform['s']:.10f}\n")
                f.write(f"   Translation:\n{transform['t']}\n")
                f.write(f"   Rotation Matrix:\n{transform['R']}\n")
                f.write(f"   4x4 Transform Matrix:\n{transform['transform_matrix']}\n\n")
                
                f.write("4. ICP Refinement Transform (精配准):\n")
                f.write(f"   RMSE: {icp_result['rmse']:.6f}\n")
                f.write(f"   ICP Transform Matrix:\n{icp_transform}\n\n")
                
                f.write("5. Combined Transform (OBJ original -> LAS space - 粗配准):\n")
                # 计算组合变换：OBJ原始坐标 -> OBJ中心 -> 配准变换 -> LAS坐标
                combined_transform_coarse = transform['transform_matrix'] @ obj_translation_matrix
                f.write(f"   Combined Matrix:\n{combined_transform_coarse}\n\n")
                
                f.write("6. Final Combined Transform (OBJ original -> LAS space - 精配准):\n")
                # 计算最终组合变换：OBJ原始坐标 -> OBJ中心 -> 粗配准变换 -> ICP精配准 -> LAS坐标
                final_combined_transform = icp_transform @ transform['transform_matrix'] @ obj_translation_matrix
                f.write(f"   Final Combined Matrix:\n{final_combined_transform}\n")
            
            print(f"\n所有变换参数已保存: {transform_path}")
        
        print("\n" + "="*60)
        print("配准和分割完成!")
        print("="*60)
        
    except FileNotFoundError as e:
        print(f"\n错误: {e}")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
