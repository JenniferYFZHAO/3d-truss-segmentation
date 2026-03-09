import open3d as o3d
import numpy as np
import os
import sys

from read_cloudpoints import read_las_point_cloud, get_las_info


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


def main():
    """
    主函数：交互式粗配准流程
    1. 读取 OBJ 文件，选取控制点
    2. 读取 LAS 文件，选取控制点
    3. 计算变换矩阵并应用
    4. 可视化配准结果
    """
    default_obj_file = r"D:\1-学术\预拼装\结构模型\泉州项目点云及模型\Final\44品_centered.obj"
    default_las_file = r"D:\1-学术\预拼装\结构模型\泉州项目点云及模型\Final\44-unset-部分去噪.las"
    
    print("="*60)
    print("OBJ 与 LAS 点云交互式粗配准工具")
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
        
        print(f"\nOBJ 模型信息:")
        print(f"  顶点范围:")
        print(f"    X: [{vertices[:, 0].min():.3f}, {vertices[:, 0].max():.3f}]")
        print(f"    Y: [{vertices[:, 1].min():.3f}, {vertices[:, 1].max():.3f}]")
        print(f"    Z: [{vertices[:, 2].min():.3f}, {vertices[:, 2].max():.3f}]")
        
        # ========== 第二步：在 OBJ 上选取控制点 ==========
        print("\n" + "="*60)
        print("第二步：在 OBJ 模型上选取控制点")
        print("="*60)
        
        obj_control_points = pick_points_on_mesh(mesh, title="OBJ Model - Select 3 Control Points", num_points=3)
        
        if obj_control_points is None:
            print("控制点选取失败，程序退出")
            return
        
        # ========== 第三步：读取 LAS 文件 ==========
        print("\n" + "="*60)
        print("第三步：读取 LAS 点云文件")
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
        
        las_points = read_las_point_cloud(las_file_path, subsample_factor=subsample_factor, show_progress=True)
        
        print(f"\nLAS 点云原始信息:")
        print(f"  点云范围:")
        print(f"    X: [{las_points[:, 0].min():.3f}, {las_points[:, 0].max():.3f}]")
        print(f"    Y: [{las_points[:, 1].min():.3f}, {las_points[:, 1].max():.3f}]")
        print(f"    Z: [{las_points[:, 2].min():.3f}, {las_points[:, 2].max():.3f}]")
        
        # 计算质心并询问是否平移
        las_centroid = np.mean(las_points, axis=0)
        print(f"\n点云质心坐标: ({las_centroid[0]:.3f}, {las_centroid[1]:.3f}, {las_centroid[2]:.3f})")
        
        translate_las = input("是否将 LAS 点云质心平移到坐标原点？(y/n) [默认: y]: ").strip().lower()
        if translate_las != 'n':
            las_points = las_points - las_centroid
            print(f"已将 LAS 点云平移至原点")
            print(f"  平移后范围:")
            print(f"    X: [{las_points[:, 0].min():.3f}, {las_points[:, 0].max():.3f}]")
            print(f"    Y: [{las_points[:, 1].min():.3f}, {las_points[:, 1].max():.3f}]")
            print(f"    Z: [{las_points[:, 2].min():.3f}, {las_points[:, 2].max():.3f}]")
        else:
            print("保持 LAS 点云原始坐标不变")
        
        # ========== 第四步：在 LAS 点云上选取控制点 ==========
        print("\n" + "="*60)
        print("第四步：在 LAS 点云上选取控制点")
        print("="*60)
        print("\n重要提示: 请按照与 OBJ 模型相同的顺序选取对应的控制点！")
        print("例如: 如果 OBJ 上选的是 A->B->C，这里也要选对应的 A'->B'->C'")
        
        las_control_points = pick_points_on_point_cloud(las_points, title="LAS Point Cloud - Select 3 Corresponding Control Points", num_points=3)
        
        if las_control_points is None:
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
            print(f"  OBJ:  ({obj_control_points[i][0]:.3f}, {obj_control_points[i][1]:.3f}, {obj_control_points[i][2]:.3f})")
            print(f"  LAS:  ({las_control_points[i][0]:.3f}, {las_control_points[i][1]:.3f}, {las_control_points[i][2]:.3f})")
        
        transform = compute_similarity_transform(las_control_points, obj_control_points)
        
        print("\n变换参数:")
        print(f"  缩放因子 s: {transform['s']:.6f}")
        print(f"  平移向量 t: ({transform['t'][0]:.3f}, {transform['t'][1]:.3f}, {transform['t'][2]:.3f})")
        print(f"\n旋转矩阵 R:")
        print(transform['R'])
        
        # ========== 第六步：应用变换 ==========
        print("\n" + "="*60)
        print("第六步：应用变换")
        print("="*60)
        
        transformed_las_points = apply_transform_to_point_cloud(
            las_points, 
            transform['R'], 
            transform['t'], 
            transform['s']
        )
        
        print(f"变换后 LAS 点云范围:")
        print(f"  X: [{transformed_las_points[:, 0].min():.3f}, {transformed_las_points[:, 0].max():.3f}]")
        print(f"  Y: [{transformed_las_points[:, 1].min():.3f}, {transformed_las_points[:, 1].max():.3f}]")
        print(f"  Z: [{transformed_las_points[:, 2].min():.3f}, {transformed_las_points[:, 2].max():.3f}]")
        
        # ========== 第七步：可视化配准结果 ==========
        print("\n" + "="*60)
        print("第七步：可视化配准结果")
        print("="*60)
        
        visualize_registration_result(mesh, transformed_las_points, title="Registration Result: OBJ (Red) + LAS (Green)")
        
        # ========== 第八步：询问是否保存 ==========
        print("\n" + "="*60)
        print("第八步：保存结果")
        print("="*60)
        
        save = input("是否保存配准结果？(y/n) [默认: n]: ").strip().lower()
        
        if save == 'y' or save == 'yes':
            output_dir = os.path.dirname(obj_file_path)
            base_name = os.path.splitext(os.path.basename(obj_file_path))[0] + "_registered"
            save_transformed_data(mesh, transformed_las_points, output_dir, base_name)
            
            transform_path = os.path.join(output_dir, f"{base_name}_transform.txt")
            with open(transform_path, 'w') as f:
                f.write("Similarity Transform Parameters\n")
                f.write("="*40 + "\n\n")
                f.write(f"Scale: {transform['s']:.10f}\n\n")
                f.write(f"Translation:\n{transform['t']}\n\n")
                f.write(f"Rotation Matrix:\n{transform['R']}\n\n")
                f.write(f"4x4 Transform Matrix:\n{transform['transform_matrix']}\n")
            print(f"变换参数已保存: {transform_path}")
        
        print("\n" + "="*60)
        print("配准完成!")
        print("="*60)
        
    except FileNotFoundError as e:
        print(f"\n错误: {e}")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
