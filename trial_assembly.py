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
    
    print("\n" + "="*60)
    print("配准完成")
    print("="*60)
    print(f"组合变换矩阵:")
    print(combined_transform)
    print(f"\nOBJ模型质心: {mesh_centroid}")
    print(f"LAS点云质心: {las_centroid}")
    print(f"\n配准后的OBJ模型顶点数: {len(mesh_fine.vertices)}")
    print(f"配准后的OBJ模型三角面数: {len(mesh_fine.triangles)}")
    
    print("\n程序执行完成!")


if __name__ == "__main__":
    main()
