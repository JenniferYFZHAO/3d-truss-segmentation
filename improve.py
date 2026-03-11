import numpy as np
import open3d as o3d
import os
import csv
import re
from scipy.spatial import KDTree


def read_ply_point_cloud(file_path):
    """
    读取 PLY 点云文件。
    """
    print(f"正在读取 PLY 文件: {file_path}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    pcd = o3d.io.read_point_cloud(file_path)
    
    if pcd.is_empty():
        raise ValueError(f"PLY 文件为空或无法读取: {file_path}")
    
    points = np.asarray(pcd.points)
    
    print(f"读取完成!")
    print(f"  点数: {len(points):,}")
    print(f"  范围:")
    print(f"    X: [{points[:, 0].min():.3f}, {points[:, 0].max():.3f}]")
    print(f"    Y: [{points[:, 1].min():.3f}, {points[:, 1].max():.3f}]")
    print(f"    Z: [{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]")
    
    return points, pcd


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
                    # 处理 f v1/vt1/vn1 格式
                    idx = part.split('/')[0]
                    face_indices.append(int(idx) - 1)  # OBJ 索引从 1 开始
                
                # 将多边形三角化
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


def read_transform_matrix(txt_path):
    """
    读取变换矩阵文件，提取最终组合变换矩阵。
    """
    print(f"读取变换矩阵文件: {txt_path}")
    
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
        try:
            with open(txt_path, 'rb') as f:
                content = f.read().decode('utf-8', errors='ignore')
            print("使用二进制模式读取并忽略编码错误")
        except Exception as e:
            raise ValueError(f"无法读取文件: {e}")
    
    final_matrix_pattern = r"6\. Final Combined Transform.*?Final Combined Matrix:(.*?)(?:\n\n|\Z)"
    match = re.search(final_matrix_pattern, content, re.DOTALL)
    
    if match:
        matrix_text = match.group(1).strip()
        print("找到最终组合变换矩阵")
    else:
        combined_pattern = r"5\. Combined Transform.*?Combined Matrix:(.*?)(?:\n\n|\Z)"
        match = re.search(combined_pattern, content, re.DOTALL)
        if match:
            matrix_text = match.group(1).strip()
            print("找到组合变换矩阵")
        else:
            raise ValueError("未找到变换矩阵")
    
    matrix = []
    lines = matrix_text.strip().split('\n')
    for line in lines:
        line = line.strip()
        if line and not line.startswith('=') and not line.startswith('-'):
            line = re.sub(r'[\[\]]', '', line)
            values = [float(x) for x in line.split()]
            if len(values) == 4:
                matrix.append(values)
    
    if len(matrix) != 4:
        raise ValueError(f"变换矩阵格式错误，应为4x4矩阵，实际为 {len(matrix)} 行")
    
    transform_matrix = np.array(matrix)
    print(f"变换矩阵:\n{transform_matrix}")
    
    return transform_matrix


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


def ransac_sphere_fitting(points, max_iterations=100, distance_threshold=0.01, min_inliers=100):
    """
    使用 RANSAC 方法拟合球面。
    
    Args:
        points: 点云坐标 (N, 3)
        max_iterations: 最大迭代次数
        distance_threshold: 内点距离阈值
        min_inliers: 最小内点数量
        
    Returns:
        center: 球心坐标
        radius: 球半径
        inlier_indices: 内点索引
    """
    if len(points) < 4:
        return None, None, None
    
    best_center = None
    best_radius = None
    best_inliers = []
    
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
        
        if len(inlier_indices) > len(best_inliers):
            best_inliers = inlier_indices
            best_center = center
            best_radius = radius
    
    if len(best_inliers) < min_inliers:
        return None, None, None
    
    inlier_points = points[best_inliers]
    best_center, best_radius = refine_sphere_fit(inlier_points)
    
    return best_center, best_radius, best_inliers


def fit_sphere_from_4_points(points):
    """
    从4个点拟合球面。
    """
    try:
        p1, p2, p3, p4 = points
        
        A = np.array([
            [p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]],
            [p3[0] - p1[0], p3[1] - p1[1], p3[2] - p1[2]],
            [p4[0] - p1[0], p4[1] - p1[1], p4[2] - p1[2]]
        ])
        
        B = np.array([
            0.5 * (np.dot(p2, p2) - np.dot(p1, p1)),
            0.5 * (np.dot(p3, p3) - np.dot(p1, p1)),
            0.5 * (np.dot(p4, p4) - np.dot(p1, p1))
        ])
        
        center = np.linalg.solve(A, B)
        radius = np.linalg.norm(p1 - center)
        
        return center, radius
    except:
        return None, None


def refine_sphere_fit(points):
    """
    使用最小二乘法精化球面拟合。
    """
    if len(points) < 4:
        return None, None
    
    centroid = np.mean(points, axis=0)
    
    def sphere_residuals(params, points):
        center = params[:3]
        radius = params[3]
        distances = np.linalg.norm(points - center, axis=1)
        return distances - radius
    
    from scipy.optimize import least_squares
    
    initial_params = np.concatenate([centroid, [np.mean(np.linalg.norm(points - centroid, axis=1))]])
    
    result = least_squares(sphere_residuals, initial_params, args=(points,))
    
    center = result.x[:3]
    radius = abs(result.x[3])
    
    return center, radius


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
        
        cloud_center, cloud_radius, inlier_indices = ransac_sphere_fitting(
            region_points,
            max_iterations=100,
            distance_threshold=distance_threshold,
            min_inliers=30
        )
        
        if cloud_center is None:
            print(f"  球节点 {i+1}: RANSAC 拟合失败")
            results.append({
                'index': i,
                'model_center': model_center,
                'model_radius': radius,
                'cloud_center': None,
                'cloud_radius': None,
                'center_error': None,
                'radius_error': None,
                'inlier_count': len(indices),
                'status': 'fitting_failed'
            })
            continue
        
        center_error = np.linalg.norm(cloud_center - model_center)
        radius_error = abs(cloud_radius - radius)
        
        print(f"  球节点 {i+1}: 球心误差 = {center_error:.4f}m, 半径误差 = {radius_error:.4f}m")
        
        results.append({
            'index': i,
            'model_center': model_center,
            'model_radius': radius,
            'cloud_center': cloud_center,
            'cloud_radius': cloud_radius,
            'center_error': center_error,
            'radius_error': radius_error,
            'inlier_count': len(inlier_indices) if inlier_indices is not None else 0,
            'status': 'success'
        })
    
    return results


def analyze_tube_errors(points, tubes, search_radius_factor=2.0, distance_threshold=0.02):
    """
    分析圆管的装配误差。
    
    Args:
        points: 点云坐标 (N, 3)
        tubes: 圆管列表 [(sx, sy, sz, ex, ey, ez, radius), ...]
        search_radius_factor: 搜索半径因子
        distance_threshold: 距离阈值
        
    Returns:
        results: 误差分析结果列表
    """
    print("\n分析圆管装配误差...")
    
    kd_tree = KDTree(points)
    
    results = []
    
    for i, (sx, sy, sz, ex, ey, ez, radius) in enumerate(tubes):
        start = np.array([sx, sy, sz])
        end = np.array([ex, ey, ez])
        
        seg_vec = end - start
        seg_length = np.linalg.norm(seg_vec)
        
        if seg_length < 0.001:
            print(f"  圆管 {i+1}: 线段长度为零，跳过")
            results.append({
                'index': i,
                'model_start': start,
                'model_end': end,
                'model_radius': radius,
                'status': 'zero_length'
            })
            continue
        
        seg_dir = seg_vec / seg_length
        
        search_radius = radius * search_radius_factor
        
        num_samples = max(int(seg_length / 0.1), 10)
        t_values = np.linspace(0, 1, num_samples)
        
        all_indices = set()
        for t in t_values:
            sample_point = start + t * seg_vec
            indices = kd_tree.query_ball_point(sample_point, search_radius)
            all_indices.update(indices)
        
        indices = list(all_indices)
        
        if len(indices) < 10:
            print(f"  圆管 {i+1}: 搜索区域内点数不足 ({len(indices)} 个)")
            results.append({
                'index': i,
                'model_start': start,
                'model_end': end,
                'model_radius': radius,
                'status': 'insufficient_points'
            })
            continue
        
        region_points = points[indices]
        
        cylinder_params = fit_cylinder_ransac(region_points, distance_threshold)
        
        if cylinder_params is None:
            print(f"  圆管 {i+1}: 圆柱拟合失败")
            results.append({
                'index': i,
                'model_start': start,
                'model_end': end,
                'model_radius': radius,
                'status': 'fitting_failed'
            })
            continue
        
        cloud_axis_point = cylinder_params['axis_point']
        cloud_axis_dir = cylinder_params['axis_dir']
        cloud_radius = cylinder_params['radius']
        
        axis_error = compute_axis_error(start, end, cloud_axis_point, cloud_axis_dir)
        radius_error = abs(cloud_radius - radius)
        
        print(f"  圆管 {i+1}: 轴线误差 = {axis_error:.4f}m, 半径误差 = {radius_error:.4f}m")
        
        results.append({
            'index': i,
            'model_start': start,
            'model_end': end,
            'model_radius': radius,
            'cloud_axis_point': cloud_axis_point,
            'cloud_axis_dir': cloud_axis_dir,
            'cloud_radius': cloud_radius,
            'axis_error': axis_error,
            'radius_error': radius_error,
            'inlier_count': cylinder_params['inlier_count'],
            'status': 'success'
        })
    
    return results


def fit_cylinder_ransac(points, distance_threshold=0.02, max_iterations=100, min_inliers=50):
    """
    使用 RANSAC 方法拟合圆柱。
    """
    if len(points) < 10:
        return None
    
    best_params = None
    best_inlier_count = 0
    
    for _ in range(max_iterations):
        indices = np.random.choice(len(points), min(10, len(points)), replace=False)
        sample_points = points[indices]
        
        params = fit_cylinder_from_points(sample_points)
        
        if params is None:
            continue
        
        axis_point = params['axis_point']
        axis_dir = params['axis_dir']
        radius = params['radius']
        
        if radius <= 0 or radius > 10:
            continue
        
        distances = point_to_axis_distance(points, axis_point, axis_dir)
        inlier_mask = np.abs(distances - radius) < distance_threshold
        inlier_count = np.sum(inlier_mask)
        
        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_params = params.copy()
            best_params['inlier_count'] = inlier_count
    
    if best_inlier_count < min_inliers:
        return None
    
    return best_params


def fit_cylinder_from_points(points):
    """
    从点集拟合圆柱参数。
    """
    try:
        centroid = np.mean(points, axis=0)
        centered = points - centroid
        
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        
        axis_dir = eigenvectors[:, np.argmax(eigenvalues)]
        
        projected = centered - np.outer(np.dot(centered, axis_dir), axis_dir)
        
        radius = np.mean(np.linalg.norm(projected, axis=1))
        
        return {
            'axis_point': centroid,
            'axis_dir': axis_dir,
            'radius': radius
        }
    except:
        return None


def point_to_axis_distance(points, axis_point, axis_dir):
    """
    计算点到轴线的距离。
    """
    vec = points - axis_point
    cross = np.cross(vec, axis_dir)
    distances = np.linalg.norm(cross, axis=1) / np.linalg.norm(axis_dir)
    return distances


def compute_axis_error(model_start, model_end, cloud_axis_point, cloud_axis_dir):
    """
    计算两轴线间的最短距离（公垂线长度）作为轴线误差。
    
    Args:
        model_start: 模型轴线起点
        model_end: 模型轴线终点
        cloud_axis_point: 实际轴线点
        cloud_axis_dir: 实际轴线方向
        
    Returns:
        float: 轴线误差（公垂线长度）
    """
    model_dir = model_end - model_start
    model_dir = model_dir / np.linalg.norm(model_dir)
    
    cloud_axis_dir = cloud_axis_dir / np.linalg.norm(cloud_axis_dir)
    
    # 计算两轴线间的最短距离（公垂线长度）
    cross_product = np.cross(model_dir, cloud_axis_dir)
    cross_norm = np.linalg.norm(cross_product)
    
    # 如果两轴线平行
    if cross_norm < 1e-6:
        # 计算点到直线的距离
        vec = cloud_axis_point - model_start
        dist = np.linalg.norm(np.cross(vec, model_dir)) / np.linalg.norm(model_dir)
        return dist
    
    # 两轴线不平行，计算公垂线长度
    vec = cloud_axis_point - model_start
    dist = abs(np.dot(vec, cross_product)) / cross_norm
    
    return dist


def visualize_errors(points, sphere_results, error_scale=100.0, show_error_vectors=True):
    """
    可视化装配误差。
    
    Args:
        points: 点云坐标
        sphere_results: 球节点误差结果
        error_scale: 误差显示缩放因子
        show_error_vectors: 是否显示误差矢量箭头
    """
    print("\n可视化装配误差...")
    
    geometries = []
    
    # 球心误差范围设置
    min_error = 0.004
    max_error = 0.05
    
    for result in sphere_results:
        if result['status'] != 'success':
            continue
        
        model_center = result['model_center']
        cloud_center = result['cloud_center']
        
        if cloud_center is None:
            continue
        
        model_sphere = o3d.geometry.TriangleMesh.create_sphere(
            radius=result['model_radius'], resolution=20
        )
        model_sphere.translate(model_center)
        model_sphere.paint_uniform_color([0.2, 0.6, 1.0])
        model_sphere.compute_vertex_normals()
        geometries.append(model_sphere)
        
        cloud_sphere = o3d.geometry.TriangleMesh.create_sphere(
            radius=result['cloud_radius'], resolution=20
        )
        cloud_sphere.translate(cloud_center)
        
        # 计算球心误差的颜色
        center_error = result['center_error']
        if center_error > max_error:
            # 超过最大误差，使用特别红的颜色
            color = [1.0, 0.0, 0.0]  # 纯红色
        else:
            # 在 0.004-0.05 之间，使用绿色到红色的渐变
            if center_error < min_error:
                error_ratio = 0.0
            else:
                error_ratio = (center_error - min_error) / (max_error - min_error)
                error_ratio = max(0.0, min(1.0, error_ratio))
            color = [error_ratio, 1 - error_ratio, 0.0]  # 绿色到红色
        
        cloud_sphere.paint_uniform_color(color)
        cloud_sphere.compute_vertex_normals()
        geometries.append(cloud_sphere)
        
        if show_error_vectors and center_error > 0.001:
            error_vec = cloud_center - model_center
            error_vec_normalized = error_vec / np.linalg.norm(error_vec) if np.linalg.norm(error_vec) > 0 else np.array([0, 0, 1])
            
            arrow = o3d.geometry.TriangleMesh.create_arrow(
                cylinder_radius=0.005,
                cone_radius=0.02,
                cylinder_height=center_error * error_scale,
                cone_height=0.03,
                resolution=20,
                cylinder_split=4,
                cone_split=1
            )
            
            rotation = compute_rotation_between_vectors(np.array([0, 0, 1]), error_vec_normalized)
            arrow.rotate(rotation, center=np.array([0, 0, 0]))
            arrow.translate(model_center)
            arrow.paint_uniform_color([1, 0, 0])
            geometries.append(arrow)
    
    print("显示说明:")
    print("  蓝色: 模型球节点")
    print("  绿色->红色: 实际球节点（颜色表示误差大小）")
    print(f"  误差范围: {min_error:.3f}m - {max_error:.3f}m")
    print("  超过 {max_error:.3f}m: 纯红色")
    if show_error_vectors:
        print("  红色箭头: 球心偏移方向和大小（放大显示）")
    else:
        print("  误差矢量箭头: 已隐藏")
    
    o3d.visualization.draw_geometries(
        geometries,
        window_name="Assembly Error Visualization",
        width=1400,
        height=900
    )


def create_cylinder_mesh(start, end, radius, resolution=20):
    """
    创建圆柱网格。
    """
    direction = end - start
    length = np.linalg.norm(direction)
    
    if length < 0.001:
        return o3d.geometry.TriangleMesh()
    
    cylinder = o3d.geometry.TriangleMesh.create_cylinder(
        radius=radius,
        height=length,
        resolution=resolution,
        split=4
    )
    
    direction_normalized = direction / length
    z_axis = np.array([0, 0, 1])
    rotation = compute_rotation_between_vectors(z_axis, direction_normalized)
    
    cylinder.rotate(rotation, center=np.array([0, 0, 0]))
    cylinder.translate(start + direction / 2)
    
    return cylinder


def compute_rotation_between_vectors(v1, v2):
    """
    计算从向量 v1 到 v2 的旋转矩阵。
    """
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    
    cross = np.cross(v1, v2)
    dot = np.dot(v1, v2)
    
    if dot > 0.9999:
        return np.eye(3)
    
    if dot < -0.9999:
        perpendicular = np.array([1, 0, 0]) if abs(v1[0]) < 0.9 else np.array([0, 1, 0])
        axis = np.cross(v1, perpendicular)
        axis = axis / np.linalg.norm(axis)
        return o3d.geometry.get_rotation_matrix_from_axis_angle(axis * np.pi)
    
    s = np.linalg.norm(cross)
    c = dot
    
    skew = np.array([
        [0, -cross[2], cross[1]],
        [cross[2], 0, -cross[0]],
        [-cross[1], cross[0], 0]
    ])
    
    rotation = np.eye(3) + skew + (skew @ skew) * (1 - c) / (s * s)
    
    return rotation


def save_error_results(sphere_results, tube_results, output_path):
    """
    保存误差分析结果到 CSV 文件。
    """
    print(f"\n保存误差分析结果到: {output_path}")
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        writer.writerow(['球节点误差分析结果'])
        writer.writerow([
            '序号', '模型球心X', '模型球心Y', '模型球心Z', '模型半径',
            '实际球心X', '实际球心Y', '实际球心Z', '实际半径',
            '球心误差(m)', '半径误差(m)', '内点数量', '状态'
        ])
        
        for result in sphere_results:
            model_center = result['model_center']
            cloud_center = result['cloud_center'] if result['cloud_center'] is not None else [None, None, None]
            
            writer.writerow([
                result['index'] + 1,
                f"{model_center[0]:.6f}", f"{model_center[1]:.6f}", f"{model_center[2]:.6f}",
                f"{result['model_radius']:.6f}",
                f"{cloud_center[0]:.6f}" if cloud_center[0] is not None else 'N/A',
                f"{cloud_center[1]:.6f}" if cloud_center[1] is not None else 'N/A',
                f"{cloud_center[2]:.6f}" if cloud_center[2] is not None else 'N/A',
                f"{result['cloud_radius']:.6f}" if result['cloud_radius'] is not None else 'N/A',
                f"{result['center_error']:.6f}" if result['center_error'] is not None else 'N/A',
                f"{result['radius_error']:.6f}" if result['radius_error'] is not None else 'N/A',
                result['inlier_count'],
                result['status']
            ])
    
    print("结果保存完成!")


def main():
    """
    主函数：装配误差分析流程。
    """
    print("="*60)
    print("装配误差分析")
    print("="*60)
    
    ply_path = r"D:\1-学术\预拼装\结构模型\泉州项目点云及模型\Final\44-unset-部分去噪_registered_segmented.ply"
    obj_path = r"D:\1-学术\预拼装\结构模型\泉州项目点云及模型\Final\44品-separated.obj"
    transform_path = r"D:\1-学术\预拼装\结构模型\泉州项目点云及模型\Final\44-unset-部分去噪_registered_transform.txt"
    csv_path = r"D:\1-学术\预拼装\结构模型\泉州项目点云及模型\Final\mesh_analysis_results.csv"
    
    print("\n第一步：读取数据文件")
    print("-"*60)
    
    points, pcd = read_ply_point_cloud(ply_path)
    mesh, vertices = read_obj_mesh(obj_path)
    transform_matrix = read_transform_matrix(transform_path)
    spheres, tubes = read_csv_model_info(csv_path)
    
    print("\n第二步：将CSV结构信息变换到点云空间")
    print("-"*60)
    
    transformed_spheres = apply_transform_to_spheres(spheres, transform_matrix)
    transformed_tubes = apply_transform_to_tubes(tubes, transform_matrix)
    
    print(f"变换后球节点数量: {len(transformed_spheres)}")
    print(f"变换后圆管数量: {len(transformed_tubes)}")
    
    if transformed_spheres:
        print(f"变换后第一个球节点: {transformed_spheres[0]}")
    if transformed_tubes:
        print(f"变换后第一个圆管: {transformed_tubes[0]}")
    
    print("\n第三步：球节点装配误差分析")
    print("-"*60)
    
    sphere_results = analyze_sphere_errors(
        points,
        transformed_spheres,
        search_radius_factor=2.0,
        distance_threshold=0.02
    )
    
    print("\n第四步：统计误差结果")
    print("-"*60)
    
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
    
    print("\n第五步：可视化误差结果")
    print("-"*60)
    
    # 询问用户是否需要显示误差矢量
    show_vectors = input("是否需要显示误差矢量箭头？(y/n): ").strip().lower()
    show_error_vectors = show_vectors == 'y' or show_vectors == 'yes'
    
    visualize_errors(points, sphere_results, error_scale=100.0, show_error_vectors=show_error_vectors)
    
    print("\n第六步：保存误差结果")
    print("-"*60)
    
    output_dir = os.path.dirname(ply_path)
    output_path = os.path.join(output_dir, "assembly_error_analysis.csv")
    save_error_results(sphere_results, [], output_path)
    
    print("\n" + "="*60)
    print("装配误差分析完成!")
    print("="*60)


if __name__ == "__main__":
    main()
