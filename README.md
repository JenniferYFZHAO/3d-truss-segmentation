# 3D 桁架结构点云分割

这是一个用于 3D 桁架结构点云分割的项目，包含点云生成、分割、可视化和表面重建等功能。

## 功能特性

- **点云生成**：自动生成模拟的桁架结构点云数据，支持圆管杆件和球节点
- **点云读取**：支持读取 .las 格式的工程点云文件（支持大文件）
- **点云分割**：基于节点坐标和构件连接关系的点云粗分割算法
- **可视化**：支持分割前后点云的 3D 可视化
- **表面重建**：使用 Poisson 重建从点云生成表面模型

## 项目结构

```
3d-truss-segmentation/
├── generation.py        # 点云生成模块
├── segmentation.py      # 点云分割模块
├── visualization.py     # 点云可视化模块
├── surface_extract.py   # 表面重建模块
├── read_cloudpoints.py  # .las 文件读取和可视化模块
├── visualization.py     # 可视化模块
├── segmentation.py      # 分割模块（含分割演示主程序）
├── main.py              # 主程序入口（待添加）
├── setup_env.bat        # Windows 批处理环境配置脚本
├── setup_env.ps1        # PowerShell 环境配置脚本
├── requirements.txt     # Python 依赖包列表
└── README.md            # 项目说明文档
```

## 环境要求

- Python 3.10, 3.11, 或 3.12（推荐使用 Python 3.12）
- Windows / macOS / Linux

## 安装步骤

### 方法一：使用自动化脚本（推荐）

#### Windows 批处理脚本
```cmd
setup_env.bat
```

#### PowerShell 脚本
```powershell
.\setup_env.ps1
```

### 方法二：手动安装

1. 克隆或下载项目到本地：
```bash
git clone <repository-url>
cd 3d-truss-segmentation
```

2. 创建并激活虚拟环境：
```bash
# 创建虚拟环境
python -m venv .venv

# 激活虚拟环境（Windows）
.venv\Scripts\Activate

# 激活虚拟环境（macOS/Linux）
source .venv/bin/activate
```

3. 安装依赖包：
```bash
pip install -r requirements.txt
```

## 使用方法

### 运行主程序

```bash
python main.py
```

这将执行完整的点云生成、分割、验证和可视化流程。

### 运行表面重建

```bash
python surface_extract.py
```

此脚本会自动生成带球节点的桁架点云，然后进行表面重建并可视化。

### 运行分割和可视化演示

```bash
python segmentation.py
```

此脚本会：
1. 生成带球节点的桁架点云（圆管杆件 + 球节点）
2. 运行分割算法
3. 计算分割准确率
4. 可视化分割前后的点云对比

### 读取和可视化 .las 文件

```bash
python read_cloudpoints.py
```

程序会自动使用默认文件路径，或者您也可以指定其他文件：

```bash
python read_cloudpoints.py <your_file.las>
```

**交互式使用流程：**
1. 程序会先显示文件基本信息（点数、坐标范围等）
2. 根据文件大小推荐合适的下采样比例
3. 您可以选择使用推荐值或输入自定义值
4. 显示进度条读取点云
5. 可视化显示点云

**在代码中使用：**

```python
from read_cloudpoints import read_las_point_cloud, visualize_point_cloud, read_and_visualize_las, get_las_info

# 获取文件信息（不读取全部数据）
info = get_las_info("your_file.las")
print(f"点数: {info['point_count']:,}")

# 读取点云（大文件推荐下采样，带进度条）
points = read_las_point_cloud("your_file.las", subsample_factor=50, show_progress=True)

# 可视化点云（可选自动关闭）
visualize_point_cloud(points, point_size=1.0, auto_close_time=None)

# 一键读取并可视化
points = read_and_visualize_las("your_file.las", subsample_factor=50)
```

**大文件处理建议：**
- >5000万点: subsample_factor=100-200
- 1000万-5000万点: subsample_factor=50-100
- 100万-1000万点: subsample_factor=20-50
- <100万点: subsample_factor=10

## 模块说明

### generation.py

包含点云生成功能，支持圆管杆件和球节点：

```python
from generation import generate_truss_point_cloud

# 生成桁架点云（圆管杆件 + 球节点）
point_cloud, ground_truth = generate_truss_point_cloud(
    nodes_coords_dict=nodes_coords,
    member_connectivity=member_connectivity,
    points_per_member=100,      # 沿杆件长度的点数
    radius=0.1,                   # 圆管半径
    points_per_circle=20,          # 每个圆周上的点数
    noise_std=0.05,
    num_noise_points=50,
    node_sphere_radius=0.2,        # 球节点半径
    points_per_sphere=50            # 每个球面上的点数
)
```

### segmentation.py

包含点云分割功能：

```python
from segmentation import rough_segmentation

# 执行点云分割
membership = rough_segmentation(
    point_cloud=point_cloud,
    nodes_coords=nodes_coords,
    member_connectivity=member_connectivity,
    max_member_length=6.0,
    tolerance_distance=0.1
)
```

### visualization.py

包含点云可视化功能：

```python
from visualization import visualize_point_cloud

# 可视化分割前的点云
visualize_point_cloud(point_cloud, title="Pre-segmentation Point Cloud")

# 可视化分割结果
visualize_point_cloud(point_cloud, membership, title="Post-segmentation Point Cloud")
```

### surface_extract.py

包含表面重建功能，使用 Open3D 进行 Poisson 重建。

### read_cloudpoints.py

包含 .las 点云文件读取和可视化功能：

```python
from read_cloudpoints import read_las_point_cloud, visualize_point_cloud, get_las_info, read_and_visualize_las

# 获取文件信息（不读取全部数据）
info = get_las_info("your_file.las")

# 读取点云
points = read_las_point_cloud("your_file.las", subsample_factor=10)

# 可视化
visualize_point_cloud(points)

# 一键读取并可视化
points = read_and_visualize_las("your_file.las", subsample_factor=10)
```

## 算法原理

### 点云分割算法

1. **搜索域确定**：为每个点确定搜索范围
2. **潜在节点识别**：在搜索范围内找出相关节点
3. **潜在构件查找**：找出连接这些节点的所有构件
4. **距离计算**：计算点到每个潜在构件中心线的距离
5. **构件分配**：将点分配给最近的构件（如果距离在容差范围内）

### 表面重建

使用 Poisson 重建算法从点云生成平滑的三角形网格表面。

## 示例输出

运行 `main.py` 后，您将看到：

1. 分割进度信息
2. 验证结果（包括准确率）
3. 三个可视化窗口：
   - 分割前点云（灰色）
   - 分割结果（不同构件不同颜色，未分配点为红色）
   - 地面真值（理想分割结果）

## 依赖包

- numpy >= 2.0
- matplotlib >= 3.0
- scipy >= 1.0
- open3d >= 0.17
- laspy >= 2.0
- tqdm >= 4.0

详细版本信息请查看 `requirements.txt`。

## 许可证

本项目仅供学习和研究使用。

## 贡献

欢迎提交 Issue 和 Pull Request！

## 联系方式

2252893@tongji.edu.cn
thisissammeier@gmail.com
