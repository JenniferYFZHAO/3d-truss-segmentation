import numpy as np
import open3d as o3d
import os
import csv
import re


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
    读取变换矩阵文件，提取 OBJ 中心化矩阵。
    """
    print(f"读取变换矩阵文件: {txt_path}")
    
    with open(txt_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
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
    
    matrix = np.array(matrix)
    print(f"成功读取 OBJ 中心化矩阵:")
    print(matrix)
    
    return matrix


def apply_transform_to_spheres(spheres, transform_matrix):
    """
    将变换矩阵应用到球节点。
    """
    transformed_spheres = []
    
    for x, y, z, radius in spheres:
        # 转换为齐次坐标
        point = np.array([x, y, z, 1.0])
        # 应用变换
        transformed_point = transform_matrix @ point
        # 转换回笛卡尔坐标
        tx, ty, tz, _ = transformed_point
        transformed_spheres.append((tx, ty, tz, radius))
    
    return transformed_spheres


def apply_transform_to_tubes(tubes, transform_matrix):
    """
    将变换矩阵应用到圆管。
    """
    transformed_tubes = []
    
    for sx, sy, sz, ex, ey, ez, radius in tubes:
        # 转换起点为齐次坐标
        start_point = np.array([sx, sy, sz, 1.0])
        transformed_start = transform_matrix @ start_point
        tsx, tsy, tsz, _ = transformed_start
        
        # 转换终点为齐次坐标
        end_point = np.array([ex, ey, ez, 1.0])
        transformed_end = transform_matrix @ end_point
        tex, tey, tez, _ = transformed_end
        
        transformed_tubes.append((tsx, tsy, tsz, tex, tey, tez, radius))
    
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
        sphere_mask = distances <= radius * 1.3  # 允许 10% 的误差
        labels[sphere_mask] = 1  # 1: 球节点
    
    # 2. 分割圆管（只处理未分配的点）
    print(f"分割圆管...")
    unassigned_mask = labels == 0
    unassigned_points = points[unassigned_mask]
    
    if len(unassigned_points) > 0:
        for i, (sx, sy, sz, ex, ey, ez, radius) in enumerate(tubes):
            if i % 10 == 0:
                print(f"处理圆管 {i+1}/{len(tubes)}")
            
            start = np.array([sx, sy, sz])
            end = np.array([ex, ey, ez])
            
            # 计算未分配点到线段的距离
            distances = []
            for p in unassigned_points:
                dist = point_to_line_segment_distance(p, start, end)
                distances.append(dist)
            distances = np.array(distances)
            
            # 标记在线管内的点
            tube_mask = distances <= radius * 1.3  # 允许 10% 的误差
            
            # 更新标签
            tube_indices = np.where(unassigned_mask)[0][tube_mask]
            labels[tube_indices] = 2  # 2: 圆管
    
    # 统计结果
    total_points = len(points)
    sphere_points = np.sum(labels == 1)
    tube_points = np.sum(labels == 2)
    unassigned_points = np.sum(labels == 0)
    
    print(f"\n分割结果:")
    print(f"总点数: {total_points:,}")
    print(f"球节点点数: {sphere_points:,} ({sphere_points/total_points*100:.2f}%)")
    print(f"圆管点数: {tube_points:,} ({tube_points/total_points*100:.2f}%)")
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
    
    # 0: 未分配 - 灰色
    colors[labels == 0] = [0.5, 0.5, 0.5]
    # 1: 球节点 - 红色
    colors[labels == 1] = [1.0, 0.0, 0.0]
    # 2: 圆管 - 蓝色
    colors[labels == 2] = [0.0, 0.0, 1.0]
    
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
    print("  红色: 球节点")
    print("  蓝色: 圆管")
    print("  灰色: 未分配")
    
    vis.run()
    vis.destroy_window()


def save_segmentation_result(points, labels, output_path):
    """
    保存分割结果。
    """
    print(f"\n保存分割结果到: {output_path}")
    
    # 保存为 PLY 文件
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # 为不同类别分配颜色
    colors = np.zeros((len(points), 3))
    colors[labels == 0] = [0.5, 0.5, 0.5]
    colors[labels == 1] = [1.0, 0.0, 0.0]
    colors[labels == 2] = [0.0, 0.0, 1.0]
    
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
    主函数：基于模板信息进行点云分割。
    """
    # 文件路径
    csv_path = r"D:\1-学术\预拼装\结构模型\泉州项目点云及模型\Final\mesh_analysis_results.csv"
    txt_path = r"D:\1-学术\预拼装\结构模型\泉州项目点云及模型\Final\44品-separated_registered_transform.txt"
    point_cloud_path = r"D:\1-学术\预拼装\结构模型\泉州项目点云及模型\Final\44品-separated_registered_pointcloud.ply"
    
    print("="*60)
    print("基于模板信息的点云分割工具")
    print("="*60)
    print(f"使用的点云文件: {point_cloud_path}")
    
    try:
        # 1. 读取模型信息
        spheres, tubes = read_csv_model_info(csv_path)
        
        # 2. 读取变换矩阵
        obj_transform_matrix = read_transform_matrix(txt_path)
        
        # 3. 应用变换到模型信息
        print("\n应用变换到模型信息...")
        transformed_spheres = apply_transform_to_spheres(spheres, obj_transform_matrix)
        transformed_tubes = apply_transform_to_tubes(tubes, obj_transform_matrix)
        
        # 4. 读取点云
        points = read_las_point_cloud(point_cloud_path, subsample_factor=20)  # 适当下采样
        
        # 5. 进行点云分割
        labels = segment_point_cloud(points, transformed_spheres, transformed_tubes)
        
        # 6. 可视化结果
        visualize_segmentation(points, labels)
        
        # 7. 保存结果
        output_dir = os.path.dirname(point_cloud_path)
        base_name = os.path.splitext(os.path.basename(point_cloud_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}_segmented.ply")
        save_segmentation_result(points, labels, output_path)
        
        print("\n" + "="*60)
        print("分割完成!")
        print("="*60)
        
    except FileNotFoundError as e:
        print(f"\n错误: {e}")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
