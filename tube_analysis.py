import numpy as np
import open3d as o3d
import os
import csv
import laspy
from scipy.spatial import KDTree


# ==================== 文件路径配置 ====================
DEFAULT_LAS_FILE = r"D:\1-学术\预拼装\结构模型\泉州项目点云及模型\Final\44-unset-部分去噪.las"
DEFAULT_OBJ_FILE = r"D:\1-学术\预拼装\结构模型\泉州项目点云及模型\Final\44品-separated-complete.obj"
DEFAULT_CSV_PATH = r"D:\1-学术\预拼装\结构模型\泉州项目点云及模型\Final\mesh_analysis_results.csv"


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


def read_obj_mesh(file_path):
    """
    读取 OBJ 网格文件。
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


def pick_points_on_point_cloud(points, title="Select Control Points", num_points=3):
    """
    在点云上交互式选取控制点。
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
    """
    print(f"\n请在可视化窗口中选取 {num_points} 个控制点")
    print("操作说明:")
    print("  - 按住 Shift + 左键点击: 选取点")
    print("  - 按住 Shift + 右键点击: 取消选取")
    print("  - 选取完成后按 'Q' 或 'ESC' 键关闭窗口")
    print(f"  - 请确保选取恰好 {num_points} 个点")
    
    vertices = np.asarray(mesh.vertices)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)
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
    
    picked_points = vertices[picked_indices]
    
    print(f"\n已选取 {len(picked_points)} 个控制点:")
    for i, pt in enumerate(picked_points):
        print(f"  点 {i+1}: ({pt[0]:.3f}, {pt[1]:.3f}, {pt[2]:.3f})")
    
    return picked_points


def compute_similarity_transform(source_points, target_points):
    """
    计算从源点集到目标点集的相似变换（旋转、平移、缩放）。
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
    """
    vertices = np.asarray(mesh.vertices)
    transformed_vertices = s * (R @ vertices.T).T + t
    
    transformed_mesh = o3d.geometry.TriangleMesh()
    transformed_mesh.vertices = o3d.utility.Vector3dVector(transformed_vertices)
    transformed_mesh.triangles = mesh.triangles
    transformed_mesh.compute_vertex_normals()
    
    return transformed_mesh


def similarity_icp(source, target, max_iterations=20, threshold=1e-4, 
                   initial_correspondence_distance=10.0, final_correspondence_distance=0.5, 
                   voxel_size=0.2):
    """
    相似变换 ICP（Similarity ICP）
    支持旋转、平移和缩放
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
        
        # 计算对应点
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
        print(f"迭代 {i+1}/{max_iterations}: RMSE = {current_error:.6f}, 缩放因子 = {s:.6f}")
        
        # 检查收敛
        if abs(prev_error - current_error) < threshold:
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


def visualize_registration_result(mesh, points, title="Registration Result"):
    """
    可视化配准结果。
    """
    mesh_copy = o3d.geometry.TriangleMesh(mesh)
    mesh_copy.paint_uniform_color([1.0, 0.0, 0.0])
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color([0.0, 1.0, 0.0])
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title, width=1400, height=900)
    vis.add_geometry(mesh_copy)
    vis.add_geometry(pcd)
    
    render_option = vis.get_render_option()
    render_option.point_size = 1.0
    
    vis.run()
    vis.destroy_window()


def main():
    """
    主函数：配准流程
    """
    print("="*60)
    print("结构模型与点云配准工具")
    print("="*60)
    
    # 文件路径
    las_file = DEFAULT_LAS_FILE
    obj_file = DEFAULT_OBJ_FILE
    
    # 读取LAS点云
    print("\n" + "="*60)
    print("第一步：读取LAS点云")
    print("="*60)
    las_points = read_las_point_cloud(las_file, subsample_factor=10)
    
    # 读取OBJ模型
    print("\n" + "="*60)
    print("第二步：读取OBJ结构模型")
    print("="*60)
    mesh, mesh_vertices = read_obj_mesh(obj_file)
    
    # 平移到形心
    print("\n" + "="*60)
    print("第三步：平移到形心")
    print("="*60)
    
    # 计算点云质心
    las_centroid = np.mean(las_points, axis=0)
    print(f"LAS点云质心: {las_centroid}")
    las_points_centered = las_points - las_centroid
    
    # 计算模型质心
    mesh_centroid = np.mean(mesh_vertices, axis=0)
    print(f"OBJ模型质心: {mesh_centroid}")
    mesh_vertices_centered = mesh_vertices - mesh_centroid
    
    print(f"\n平移后:")
    print(f"  LAS点云范围:")
    print(f"    X: [{las_points_centered[:, 0].min():.3f}, {las_points_centered[:, 0].max():.3f}]")
    print(f"    Y: [{las_points_centered[:, 1].min():.3f}, {las_points_centered[:, 1].max():.3f}]")
    print(f"    Z: [{las_points_centered[:, 2].min():.3f}, {las_points_centered[:, 2].max():.3f}]")
    print(f"  OBJ模型范围:")
    print(f"    X: [{mesh_vertices_centered[:, 0].min():.3f}, {mesh_vertices_centered[:, 0].max():.3f}]")
    print(f"    Y: [{mesh_vertices_centered[:, 1].min():.3f}, {mesh_vertices_centered[:, 1].max():.3f}]")
    print(f"    Z: [{mesh_vertices_centered[:, 2].min():.3f}, {mesh_vertices_centered[:, 2].max():.3f}]")
    
    # 创建平移后的点云和网格
    las_pcd_centered = o3d.geometry.PointCloud()
    las_pcd_centered.points = o3d.utility.Vector3dVector(las_points_centered)
    
    mesh_centered = o3d.geometry.TriangleMesh(mesh)
    mesh_centered.vertices = o3d.utility.Vector3dVector(mesh_vertices_centered)
    mesh_centered.compute_vertex_normals()
    
    # 控制点粗配准
    print("\n" + "="*60)
    print("第四步：控制点粗配准")
    print("="*60)
    
    print("\n请在点云上选取3个控制点...")
    las_control_points = pick_points_on_point_cloud(
        las_points_centered, 
        "LAS Point Cloud - Select 3 Control Points", 
        3
    )
    
    if las_control_points is None:
        print("控制点选取失败，程序退出")
        return
    
    print("\n请在模型上选取对应的3个控制点...")
    mesh_control_points = pick_points_on_mesh(
        mesh_centered,
        "OBJ Mesh - Select 3 Corresponding Control Points",
        3
    )
    
    if mesh_control_points is None:
        print("控制点选取失败，程序退出")
        return
    
    # 计算粗配准变换
    print("\n计算粗配准变换...")
    coarse_transform = compute_similarity_transform(mesh_control_points, las_control_points)
    
    print(f"粗配准结果:")
    print(f"  缩放因子 s: {coarse_transform['s']:.6f}")
    print(f"  平移向量 t: ({coarse_transform['t'][0]:.3f}, {coarse_transform['t'][1]:.3f}, {coarse_transform['t'][2]:.3f})")
    
    # 应用粗配准变换
    mesh_coarse = apply_transform_to_mesh(
        mesh_centered, 
        coarse_transform['R'], 
        coarse_transform['t'], 
        coarse_transform['s']
    )
    
    # s-ICP精配准
    print("\n" + "="*60)
    print("第五步：s-ICP精配准")
    print("="*60)
    
    # 将粗配准后的网格转换为点云用于ICP
    mesh_coarse_pcd = mesh_coarse.sample_points_uniformly(number_of_points=50000)
    
    icp_result = similarity_icp(
        mesh_coarse_pcd,
        las_pcd_centered,
        max_iterations=20,
        threshold=1e-4,
        initial_correspondence_distance=10.0,
        final_correspondence_distance=0.5,
        voxel_size=0.2
    )
    
    print(f"\ns-ICP精配准结果:")
    print(f"  最终RMSE: {icp_result['rmse']:.6f}")
    print(f"  缩放因子: {icp_result['s']:.6f}")
    
    # 应用精配准变换
    mesh_fine = apply_transform_to_mesh(
        mesh_coarse,
        icp_result['R'],
        icp_result['t'],
        icp_result['s']
    )
    
    # 可视化结果
    print("\n" + "="*60)
    print("第六步：可视化配准结果")
    print("="*60)
    
    print("显示最终配准结果...")
    visualize_registration_result(mesh_fine, las_points_centered, "Final Registration Result")
    
    # 计算组合变换矩阵（从OBJ原始空间到LAS平移后空间）
    combined_R = icp_result['R'] @ coarse_transform['R']
    combined_s = icp_result['s'] * coarse_transform['s']
    combined_t = icp_result['s'] * (icp_result['R'] @ coarse_transform['t']) + icp_result['t']
    
    combined_transform = np.eye(4)
    combined_transform[:3, :3] = combined_s * combined_R
    combined_transform[:3, 3] = combined_t
    
    # 读取CSV模型信息
    print("\n" + "="*60)
    print("第七步：读取模型信息")
    print("="*60)
    spheres, tubes = read_csv_model_info(DEFAULT_CSV_PATH)
    
    print(f"  球节点数量: {len(spheres)}")
    print(f"  圆管数量: {len(tubes)}")
    
    # 过滤掉半径为4的异常杆件
    print("\n过滤异常杆件（模板半径=4）...")
    filtered_tubes = []
    for i, tube in enumerate(tubes):
        radius, start_x, start_y, start_z, end_x, end_y, end_z = tube
        if radius == 4.0:
            print(f"  跳过圆管 {i+1}: 半径={radius:.4f}m（异常）")
        else:
            filtered_tubes.append(tube)
    
    print(f"过滤后剩余 {len(filtered_tubes)} 个圆管")
    
    # 应用变换到圆管端点和半径
    print("\n" + "="*60)
    print("第八步：应用变换到圆管端点和半径")
    print("="*60)
    
    transformed_tubes = []
    for tube in filtered_tubes:
        radius, start_x, start_y, start_z, end_x, end_y, end_z = tube
        template_start = np.array([start_x, start_y, start_z])
        template_end = np.array([end_x, end_y, end_z])
        
        # 先平移到形心（减去OBJ模型质心）
        template_start_centered = template_start - mesh_centroid
        template_end_centered = template_end - mesh_centroid
        
        # 应用组合变换到端点
        transformed_start = apply_transform_to_point(template_start_centered, combined_R, combined_t, combined_s)
        transformed_end = apply_transform_to_point(template_end_centered, combined_R, combined_t, combined_s)
        
        # 应用缩放变换到半径
        transformed_radius = radius * combined_s
        
        transformed_tubes.append({
            'radius': radius,
            'template_start': template_start,
            'template_end': template_end,
            'transformed_start': transformed_start,
            'transformed_end': transformed_end,
            'transformed_radius': transformed_radius
        })
    
    print(f"已变换 {len(transformed_tubes)} 个圆管")
    
    # 提取圆管杆件（只处理前15个）
    print("\n" + "="*60)
    print("第九步：截面圆环法圆管识别（全部）")
    print("="*60)
    
    max_tubes = len(transformed_tubes)
    print(f"将处理全部 {max_tubes} 个圆管")
    
    tube_results = []
    tube_indices = []
    
    for i in range(max_tubes):
        tube_info = transformed_tubes[i]
        print(f"\n处理圆管 {i+1}/{max_tubes}:")
        print(f"  模板起点: {tube_info['template_start']}")
        print(f"  模板终点: {tube_info['template_end']}")
        print(f"  变换后起点: {tube_info['transformed_start']}")
        print(f"  变换后终点: {tube_info['transformed_end']}")
        print(f"  模板半径: {tube_info['radius']:.4f}m")
        print(f"  变换后半径: {tube_info['transformed_radius']:.4f}m")
        
        # 提取圆管邻域点（1.5倍半径）
        region_points, indices = extract_tube_region_points(
            las_points_centered,
            tube_info['transformed_start'],
            tube_info['transformed_end'],
            tube_info['transformed_radius'],
            search_radius_factor=1.5
        )
        
        if region_points is None or len(region_points) < 10:
            print(f"  圆管 {i+1}: 邻域点不足，跳过")
            tube_results.append({'status': 'insufficient_points'})
            tube_indices.append(None)
            continue
        
        print(f"  邻域点数: {len(region_points)}")
        
        # 提取截面
        print("  提取截面...")
        sections = extract_cross_sections(
            region_points,
            tube_info['transformed_start'],
            tube_info['transformed_end'],
            tube_info['transformed_radius'],
            step_size=0.1
        )
        
        if len(sections) < 2:
            print(f"  圆管 {i+1}: 截面数量不足，跳过")
            tube_results.append({'status': 'insufficient_sections'})
            tube_indices.append(None)
            continue
        
        print(f"  提取到 {len(sections)} 个截面")
        
        # 对每个截面进行圆环拟合
        print("  对截面进行圆环拟合...")
        circle_results = []
        for j, section in enumerate(sections):
            circle_result = ransac_circle_fitting(
                section['2d_points'],
                tube_info['transformed_radius'],
                max_iterations=200,
                distance_threshold=0.02
            )
            if circle_result['status'] == 'success':
                # 还原到3D空间
                circle_3d = reconstruct_3d_circle(section, circle_result)
                circle_results.append(circle_3d)
        
        if len(circle_results) < 2:
            print(f"  圆管 {i+1}: 拟合成功的截面不足，跳过")
            tube_results.append({'status': 'insufficient_circles'})
            tube_indices.append(None)
            continue
        
        print(f"  成功拟合 {len(circle_results)} 个截面圆环")
        
        # 提取3D圆心点
        center_points = [c['center_3d'] for c in circle_results]
        
        # 平滑轴线
        print("  生成平滑轴线...")
        smooth_axis = smooth_centerline(center_points)
        
        # 计算平均半径
        average_radius = np.mean([c['radius'] for c in circle_results])
        
        # 计算整体质量
        average_inlier_ratio = np.mean([c['inlier_ratio'] for c in circle_results])
        average_rmse = np.mean([c['rmse'] for c in circle_results])
        
        # 构建结果
        tube_result = {
            'status': 'success',
            'start': smooth_axis[0],
            'end': smooth_axis[-1],
            'radius': average_radius,
            'center_points': center_points,
            'smooth_axis': smooth_axis,
            'inlier_ratio': average_inlier_ratio,
            'rmse': average_rmse,
            'section_count': len(circle_results)
        }
        
        print(f"  圆管识别成功:")
        print(f"    拟合起点: {tube_result['start']}")
        print(f"    拟合终点: {tube_result['end']}")
        print(f"    平均半径: {tube_result['radius']:.4f}m")
        print(f"    平均内点比例: {tube_result['inlier_ratio']:.3f}")
        print(f"    平均RMSE: {tube_result['rmse']:.4f}m")
        print(f"    有效截面数: {tube_result['section_count']}")
        
        tube_results.append(tube_result)
        tube_indices.append(indices)
    
    # 统计结果
    print("\n" + "="*60)
    print("识别结果统计")
    print("="*60)
    
    success_count = sum(1 for r in tube_results if r['status'] == 'success')
    high_quality_count = sum(1 for r in tube_results if r.get('status') == 'success' and r['inlier_ratio'] > 0.6)
    
    print(f"  处理圆管数: {max_tubes}")
    print(f"  成功识别: {success_count}")
    print(f"  高质量识别（内点比例>0.6）: {high_quality_count}")
    print(f"  失败: {max_tubes - success_count}")
    
    # 可视化结果
    print("\n" + "="*60)
    print("第十步：可视化圆管识别结果")
    print("="*60)
    visualize_tube_extraction(las_points_centered, tube_results, tube_indices)
    
    print("\n程序执行完成!")


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
                    radius = float(row['Radius'])
                    start_x = float(row['Start_X'])
                    start_y = float(row['Start_Y'])
                    start_z = float(row['Start_Z'])
                    end_x = float(row['End_X'])
                    end_y = float(row['End_Y'])
                    end_z = float(row['End_Z'])
                    tubes.append((radius, start_x, start_y, start_z, end_x, end_y, end_z))
                    
            except Exception as e:
                print(f"解析行时出错: {e}")
                continue
    
    print(f"成功读取: {len(spheres)} 个球节点, {len(tubes)} 个圆管")
    return spheres, tubes


def extract_tube_region_points(points, tube_start, tube_end, tube_radius, search_radius_factor=1.5):
    """
    提取圆管邻域内的点。
    
    Args:
        points: 点云坐标 (N, 3)
        tube_start: 圆管起点
        tube_end: 圆管终点
        tube_radius: 圆管半径
        search_radius_factor: 搜索半径因子（相对于半径）
        
    Returns:
        region_points: 邻域点云
        indices: 邻域点在原始点云中的索引
    """
    # 计算圆管参数
    tube_center = (tube_start + tube_end) / 2
    tube_length = np.linalg.norm(tube_end - tube_start)
    tube_direction = (tube_end - tube_start) / (tube_length + 1e-6)
    
    # 搜索半径：1.5倍半径（只考虑垂直方向）
    search_radius = tube_radius * search_radius_factor
    
    # 使用KD树搜索邻域点（只在中心附近搜索，不延长轴线）
    kd_tree = KDTree(points)
    
    # 在圆管中心附近搜索，搜索半径为1.5倍半径
    indices = kd_tree.query_ball_point(tube_center, search_radius)
    
    if len(indices) < 10:
        # 尝试在起点和终点附近搜索
        start_indices = kd_tree.query_ball_point(tube_start, search_radius)
        end_indices = kd_tree.query_ball_point(tube_end, search_radius)
        indices = list(set(start_indices + end_indices))
    
    if len(indices) < 10:
        # 再次扩大搜索范围
        indices = kd_tree.query_ball_point(tube_center, search_radius * 2)
    
    if len(indices) < 10:
        return None, None
    
    region_points = points[indices]
    
    # 计算点到轴线的距离（垂直方向）
    distances_to_axis = np.linalg.norm(
        np.cross(region_points - tube_start, tube_direction), axis=1
    )
    
    # 计算点在轴线上的投影位置
    vectors = region_points - tube_start
    projections = np.dot(vectors, tube_direction)
    
    # 只保留距离轴线1.5倍半径以内的点，并且投影在轴线范围内（不延长轴线长度）
    inlier_mask = (distances_to_axis < search_radius) & (projections >= 0) & (projections <= tube_length)
    final_indices = [indices[i] for i in range(len(indices)) if inlier_mask[i]]
    final_points = region_points[inlier_mask]
    
    if len(final_points) < 10:
        return None, None
    
    return final_points, final_indices


def extract_cross_sections(points, tube_start, tube_end, tube_radius, step_size=0.1):
    """
    沿圆管轴线提取截面点云。确保起点和终点处的截面被提取。

    Args:
        points: 圆管邻域点云 (N, 3)
        tube_start: 圆管起点
        tube_end: 圆管终点
        tube_radius: 圆管半径
        step_size: 截面间隔步长

    Returns:
        list: 包含每个截面信息的列表，每个元素为 {'section_point': 截面中心点, '2d_points': 二维截面点}
    """
    # 计算圆管参数
    tube_axis = tube_end - tube_start
    tube_length = np.linalg.norm(tube_axis)
    tube_direction = tube_axis / (tube_length + 1e-6) # 防止除零

    # --- 核心修改：确保端点和中间点 ---
    # 1. 初始化截面位置列表，加入起点和终点
    section_positions = [0.0, tube_length]

    # 2. 计算中间的截面位置
    if tube_length > step_size:
        # 在 (0, tube_length) 区间内，按步长生成中间点
        # np.arange(start, stop, step) 不包含stop，所以不用担心重复终点
        intermediate_positions = np.arange(step_size, tube_length, step_size)
        section_positions.extend(intermediate_positions)

    # 3. 去重并排序，以应对可能的数值误差导致的重复点
    unique_positions = sorted(list(set(section_positions)))
    # --- 核心修改结束 ---

    # 生成截面点
    section_points = []
    for position in unique_positions:
        # 计算截面位置
        section_point = tube_start + position * tube_direction

        # 提取截面附近的点
        # 计算点到截面的距离（沿轴线方向）
        vectors = points - section_point
        projections = np.dot(vectors, tube_direction)
        distances_to_section = np.abs(projections)

        # 计算点到轴线的距离（垂直于轴线）
        distances_to_axis = np.linalg.norm(np.cross(vectors, tube_direction), axis=1)

        # 提取在垂直于轴线的平面内，以截面中心为圆心，1.5倍半径为半径的圆内的点
        # 这里只限制点到轴线的垂直距离，不限制轴线方向的距离
        section_mask = (distances_to_axis < tube_radius * 1.5)
        section_points_3d = points[section_mask]

        if len(section_points_3d) < 5:
            continue # 如果截面点太少，跳过

        # 将3D点投影到截面平面
        # 构建局部坐标系：u, v 轴垂直于轴线
        # 选择一个垂直于轴线的向量作为u轴
        if abs(tube_direction[0]) < 0.9:
            u_axis = np.cross(tube_direction, [1, 0, 0])
        else:
            u_axis = np.cross(tube_direction, [0, 1, 0])
        u_axis = u_axis / np.linalg.norm(u_axis)
        v_axis = np.cross(tube_direction, u_axis)
        v_axis = v_axis / np.linalg.norm(v_axis)

        # 计算二维坐标
        relative_points = section_points_3d - section_point
        u_coords = np.dot(relative_points, u_axis)
        v_coords = np.dot(relative_points, v_axis)
        points_2d = np.column_stack((u_coords, v_coords))

        section_points.append({
            'section_point': section_point,
            '2d_points': points_2d,
            'u_axis': u_axis,
            'v_axis': v_axis
        })

    return section_points


def ransac_circle_fitting(points_2d, template_radius, max_iterations=200, distance_threshold=0.02):
    """
    RANSAC二维圆环拟合。
    
    Args:
        points_2d: 二维截面点云 (N, 2)
        template_radius: 模板半径（用于约束）
        max_iterations: 最大迭代次数
        distance_threshold: 距离阈值
        
    Returns:
        dict: 包含圆心、半径、内点比例等信息
    """
    if len(points_2d) < 5:
        return {'status': 'insufficient_points'}
    
    best_center = None
    best_radius = None
    best_inliers = []
    best_rmse = float('inf')
    best_inlier_ratio = 0
    
    for iteration in range(max_iterations):
        # 随机选3点
        idx = np.random.choice(len(points_2d), 3, replace=False)
        p1, p2, p3 = points_2d[idx]
        
        # 计算圆心和半径
        # 解方程组: (x - a)^2 + (y - b)^2 = r^2
        A = 2 * (p2[0] - p1[0])
        B = 2 * (p2[1] - p1[1])
        C = 2 * (p3[0] - p1[0])
        D = 2 * (p3[1] - p1[1])
        E = p2[0]**2 + p2[1]**2 - p1[0]**2 - p1[1]**2
        F = p3[0]**2 + p3[1]**2 - p1[0]**2 - p1[1]**2
        
        denominator = A * D - B * C
        if abs(denominator) < 1e-6:
            continue
        
        center_x = (D * E - B * F) / denominator
        center_y = (A * F - C * E) / denominator
        center = np.array([center_x, center_y])
        
        # 计算半径
        radius = np.linalg.norm(p1 - center)
        
        # 使用模板半径作为约束
        if radius < template_radius * 0.8 or radius > template_radius * 1.2:
            continue
        
        # 计算所有点到圆的距离
        distances = np.abs(np.linalg.norm(points_2d - center, axis=1) - radius)
        
        # 找出内点
        inliers_mask = distances < distance_threshold
        inlier_count = np.sum(inliers_mask)
        
        if inlier_count > len(best_inliers):
            # 用内点重新拟合
            if inlier_count >= 5:
                inliers = points_2d[inliers_mask]
                
                # 最小二乘拟合圆
                # 优化圆心
                def circle_error(params):
                    cx, cy, r = params
                    return np.mean((np.linalg.norm(inliers - [cx, cy], axis=1) - r)**2)
                
                from scipy.optimize import minimize
                initial_guess = [center[0], center[1], radius]
                result = minimize(circle_error, initial_guess)
                optimized_center = result.x[:2]
                optimized_radius = result.x[2]
                
                # 重新计算内点
                optimized_distances = np.abs(np.linalg.norm(points_2d - optimized_center, axis=1) - optimized_radius)
                optimized_inliers_mask = optimized_distances < distance_threshold
                optimized_inlier_count = np.sum(optimized_inliers_mask)
                
                if optimized_inlier_count > len(best_inliers):
                    best_center = optimized_center
                    best_radius = optimized_radius
                    best_inliers = points_2d[optimized_inliers_mask]
                    best_rmse = np.mean(optimized_distances[optimized_inliers_mask])
                    best_inlier_ratio = optimized_inlier_count / len(points_2d)
    
    if best_center is None:
        return {'status': 'fitting_failed'}
    
    return {
        'status': 'success',
        'center': best_center,
        'radius': best_radius,
        'inliers': best_inliers,
        'inlier_count': len(best_inliers),
        'inlier_ratio': best_inlier_ratio,
        'rmse': best_rmse
    }


def reconstruct_3d_circle(section_info, circle_result):
    """
    将二维圆环信息还原到3D空间。
    
    Args:
        section_info: 截面信息，包含截面中心点、u轴、v轴
        circle_result: 圆环拟合结果，包含二维圆心
        
    Returns:
        dict: 包含3D圆心位置的信息
    """
    if circle_result['status'] != 'success':
        return {'status': 'fitting_failed'}
    
    # 提取截面信息
    section_point = section_info['section_point']
    u_axis = section_info['u_axis']
    v_axis = section_info['v_axis']
    
    # 提取二维圆心
    center_2d = circle_result['center']
    
    # 计算3D圆心位置
    center_3d = section_point + center_2d[0] * u_axis + center_2d[1] * v_axis
    
    return {
        'status': 'success',
        'center_3d': center_3d,
        'radius': circle_result['radius'],
        'inlier_ratio': circle_result['inlier_ratio'],
        'rmse': circle_result['rmse']
    }


def smooth_centerline(center_points, num_points=100):
    """
    对截面圆心点进行平滑，生成平滑的轴线。
    
    Args:
        center_points: 3D圆心点列表
        num_points: 生成的平滑曲线上的点数量
        
    Returns:
        numpy.ndarray: 平滑后的轴线点云
    """
    if len(center_points) < 2:
        return np.array(center_points)
    
    # 计算参数t
    t = np.linspace(0, 1, len(center_points))
    
    # 使用样条插值
    from scipy.interpolate import CubicSpline
    
    # 分别对x, y, z坐标进行插值
    cs_x = CubicSpline(t, [p[0] for p in center_points])
    cs_y = CubicSpline(t, [p[1] for p in center_points])
    cs_z = CubicSpline(t, [p[2] for p in center_points])
    
    # 生成平滑曲线上的点
    t_new = np.linspace(0, 1, num_points)
    x_new = cs_x(t_new)
    y_new = cs_y(t_new)
    z_new = cs_z(t_new)
    
    # 组合成3D点
    smooth_points = np.column_stack((x_new, y_new, z_new))
    
    return smooth_points


def visualize_tube_extraction(points, tube_results, tube_indices):
    """
    可视化圆管识别结果。
    针对识别完平滑曲线后的圆管，以其两端点以及半径画圆管，
    若点云在这个圆管内部，就识别为杆件上的内点，其他就不管，
    不同的杆件用不同的颜色可视化出来。
    """
    # 为每根圆管分配不同颜色
    colors = [
        [1.0, 0.0, 0.0],  # 红色
        [0.0, 1.0, 0.0],  # 绿色
        [0.0, 0.0, 1.0],  # 蓝色
        [1.0, 1.0, 0.0],  # 黄色
        [1.0, 0.0, 1.0],  # 洋红色
        [0.0, 1.0, 1.0],  # 青色
    ]
    
    # 初始化点云颜色数组（默认灰色）
    point_colors = np.full((len(points), 3), [0.7, 0.7, 0.7])
    
    # 计算点云密度Δ（平均最近邻距离）
    print("计算点云密度Δ...")
    from scipy.spatial import KDTree
    kdtree = KDTree(points)
    # 计算每个点到其最近邻的距离
    distances, _ = kdtree.query(points, k=2)  # k=2因为第一个点是自己
    delta = np.mean(distances[:, 1])  # 取第二列（最近邻距离）的平均值
    print(f"点云密度Δ = {delta:.6f}m")
    
    # 对每个成功识别的圆管，判断点是否在圆柱体内部
    for i, result in enumerate(tube_results):
        if result['status'] != 'success':
            continue
        
        # 获取圆管参数
        tube_start = result['start']
        tube_end = result['end']
        tube_radius = result['radius']
        color = colors[i % len(colors)]
        
        # 计算圆管轴线和长度
        tube_axis = tube_end - tube_start
        tube_length = np.linalg.norm(tube_axis)
        tube_direction = tube_axis / (tube_length + 1e-6)
        
        # 计算每个点到圆管轴线的距离
        vectors = points - tube_start
        projections = np.dot(vectors, tube_direction)
        distances_to_axis = np.linalg.norm(np.cross(vectors, tube_direction), axis=1)
        
        # 判断点是否在圆柱体内部（考虑点云密度Δ的冗余度）
        # 条件1：点到轴线的垂直距离小于半径+Δ
        # 条件2：点在轴线方向上的投影在[0, tube_length]范围内
        in_cylinder_mask = (distances_to_axis < tube_radius + delta) & (projections >= 0) & (projections <= tube_length)
        
        # 将圆柱体内的点染成对应颜色
        point_colors[in_cylinder_mask] = color
        
        # 打印每个圆管的内点数量
        inlier_count = np.sum(in_cylinder_mask)
        print(f"圆管 {i+1}: 内点数 = {inlier_count}")
    
    # 创建点云并设置颜色
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(point_colors)
    
    # 显示点云
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="圆管识别结果", width=1400, height=900)
    vis.add_geometry(pcd)
    render_option = vis.get_render_option()
    render_option.point_size = 2.0  # 设置点的大小（默认是5.0，调小一点）
    vis.run()
    vis.destroy_window()



def apply_transform_to_point(point, R, t, s=1.0):
    """
    应用变换到单个点。
    
    Args:
        point: 点坐标 (3,)
        R: 旋转矩阵 (3, 3)
        t: 平移向量 (3,)
        s: 缩放因子
        
    Returns:
        numpy.ndarray: 变换后的点坐标
    """
    return s * (R @ point) + t


if __name__ == "__main__":
    main()
