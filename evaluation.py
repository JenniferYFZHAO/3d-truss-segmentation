"""
工程拼装误差分析报告生成器
基于assembly_error_analysis.csv和interface_misalignment_analysis.csv生成工程误差报告
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os


class AssemblyErrorReport:
    """拼装误差分析报告类"""
    
    def __init__(self, assembly_file, interface_file):
        """
        初始化报告生成器
        
        Args:
            assembly_file: 构件拼装误差分析结果文件路径
            interface_file: 接口错边误差分析结果文件路径
        """
        self.assembly_file = assembly_file
        self.interface_file = interface_file
        self.assembly_data = None
        self.interface_data = None
        self.report_content = []
        
        # 工程标准阈值
        self.thresholds = {
            'center_error_warning': 0.05,    # 球心误差警告阈值 (50mm)
            'center_error_critical': 0.10,   # 球心误差严重阈值 (100mm)
            'radius_error_warning': 0.03,    # 半径误差警告阈值 (30mm)
            'radius_error_critical': 0.05,   # 半径误差严重阈值 (50mm)
            'interface_error_warning': 0.05, # 接口错边警告阈值 (50mm)
            'interface_error_critical': 0.08 # 接口错边严重阈值 (80mm)
        }
        
    def load_data(self):
        """加载数据文件"""
        try:
            # 跳过第一行标题，从第二行读取列名
            self.assembly_data = pd.read_csv(self.assembly_file, encoding='utf-8', skiprows=1)
            print(f"✓ 成功加载构件拼装数据: {len(self.assembly_data)} 条记录")
            print(f"  列名: {list(self.assembly_data.columns)}")
        except Exception as e:
            print(f"✗ 加载构件拼装数据失败: {e}")
            return False
            
        try:
            # 跳过第一行标题，从第二行读取列名
            self.interface_data = pd.read_csv(self.interface_file, encoding='utf-8', skiprows=1)
            print(f"✓ 成功加载接口错边数据: {len(self.interface_data)} 条记录")
            print(f"  列名: {list(self.interface_data.columns)}")
        except Exception as e:
            print(f"✗ 加载接口错边数据失败: {e}")
            return False
            
        return True
    
    def analyze_assembly_errors(self):
        """分析构件拼装误差"""
        if self.assembly_data is None:
            return None
            
        df = self.assembly_data
        
        # 基础统计
        stats = {
            'total_nodes': len(df),
            'success_nodes': len(df[df['状态'] == 'success']),
            'center_error': {
                'mean': df['球心误差(m)'].mean(),
                'std': df['球心误差(m)'].std(),
                'max': df['球心误差(m)'].max(),
                'min': df['球心误差(m)'].min(),
                'median': df['球心误差(m)'].median()
            },
            'radius_error': {
                'mean': df['半径误差(m)'].mean(),
                'std': df['半径误差(m)'].std(),
                'max': df['半径误差(m)'].max(),
                'min': df['半径误差(m)'].min(),
                'median': df['半径误差(m)'].median()
            }
        }
        
        # 分级统计
        stats['center_error_levels'] = {
            'normal': len(df[df['球心误差(m)'] < self.thresholds['center_error_warning']]),
            'warning': len(df[(df['球心误差(m)'] >= self.thresholds['center_error_warning']) & 
                             (df['球心误差(m)'] < self.thresholds['center_error_critical'])]),
            'critical': len(df[df['球心误差(m)'] >= self.thresholds['center_error_critical']])
        }
        
        stats['radius_error_levels'] = {
            'normal': len(df[df['半径误差(m)'] < self.thresholds['radius_error_warning']]),
            'warning': len(df[(df['半径误差(m)'] >= self.thresholds['radius_error_warning']) & 
                             (df['半径误差(m)'] < self.thresholds['radius_error_critical'])]),
            'critical': len(df[df['半径误差(m)'] >= self.thresholds['radius_error_critical']])
        }
        
        # 识别问题节点
        stats['problem_nodes'] = df[
            (df['球心误差(m)'] >= self.thresholds['center_error_warning']) |
            (df['半径误差(m)'] >= self.thresholds['radius_error_warning'])
        ].sort_values('球心误差(m)', ascending=False)
        
        return stats
    
    def analyze_interface_errors(self):
        """分析接口错边误差"""
        if self.interface_data is None:
            return None
            
        df = self.interface_data
        
        # 基础统计
        stats = {
            'total_interfaces': len(df),
            'distance_stats': {
                'mean': df['错边距离(m)'].mean(),
                'std': df['错边距离(m)'].std(),
                'max': df['错边距离(m)'].max(),
                'min': df['错边距离(m)'].min(),
                'median': df['错边距离(m)'].median()
            }
        }
        
        # 分级统计
        stats['error_levels'] = {
            'normal': len(df[df['错边距离(m)'] < self.thresholds['interface_error_warning']]),
            'warning': len(df[(df['错边距离(m)'] >= self.thresholds['interface_error_warning']) & 
                             (df['错边距离(m)'] < self.thresholds['interface_error_critical'])]),
            'critical': len(df[df['错边距离(m)'] >= self.thresholds['interface_error_critical']])
        }
        
        # 识别问题接口
        stats['problem_interfaces'] = df[
            df['错边距离(m)'] >= self.thresholds['interface_error_warning']
        ].sort_values('错边距离(m)', ascending=False)
        
        # 26-45品区域分析（根据节点ID判断）
        # 假设节点ID在26-45范围内的属于该区域
        df_26_45 = df[df['重构节点ID'].str.extract(r'(\d+)')[0].astype(int).between(26, 45)]
        if len(df_26_45) > 0:
            stats['region_26_45'] = {
                'count': len(df_26_45),
                'mean_distance': df_26_45['错边距离(m)'].mean(),
                'max_distance': df_26_45['错边距离(m)'].max(),
                'problem_interfaces': df_26_45[df_26_45['错边距离(m)'] >= self.thresholds['interface_error_warning']]
            }
        
        return stats
    
    def analyze_spatial_distribution(self):
        """分析误差的空间分布特征"""
        if self.assembly_data is None:
            return None
            
        df = self.assembly_data
        
        # 计算各区域的误差分布
        # 按X坐标分区
        x_min, x_max = df['实际球心X'].min(), df['实际球心X'].max()
        x_ranges = np.linspace(x_min, x_max, 5)
        
        spatial_stats = {
            'x_ranges': [],
            'region_stats': []
        }
        
        for i in range(len(x_ranges)-1):
            mask = (df['实际球心X'] >= x_ranges[i]) & (df['实际球心X'] < x_ranges[i+1])
            region_df = df[mask]
            
            if len(region_df) > 0:
                spatial_stats['region_stats'].append({
                    'x_range': f"{x_ranges[i]:.2f} ~ {x_ranges[i+1]:.2f}",
                    'node_count': len(region_df),
                    'avg_center_error': region_df['球心误差(m)'].mean(),
                    'max_center_error': region_df['球心误差(m)'].max(),
                    'avg_radius_error': region_df['半径误差(m)'].mean(),
                    'max_radius_error': region_df['半径误差(m)'].max()
                })
        
        return spatial_stats
    
    def generate_report(self, output_file='工程拼装误差分析报告.txt'):
        """生成工程误差报告"""
        
        if not self.load_data():
            print("数据加载失败，无法生成报告")
            return False
        
        # 分析数据
        assembly_stats = self.analyze_assembly_errors()
        interface_stats = self.analyze_interface_errors()
        spatial_stats = self.analyze_spatial_distribution()
        
        # 生成报告内容
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append(" " * 20 + "工程拼装误差分析报告")
        report_lines.append("=" * 80)
        report_lines.append(f"报告生成时间: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}")
        report_lines.append(f"数据来源: {self.assembly_file}, {self.interface_file}")
        report_lines.append("")
        
        # 1. 总体概况
        report_lines.append("-" * 80)
        report_lines.append("一、总体概况")
        report_lines.append("-" * 80)
        report_lines.append(f"分析构件节点总数: {assembly_stats['total_nodes']} 个")
        report_lines.append(f"成功拟合节点数: {assembly_stats['success_nodes']} 个")
        report_lines.append(f"接口匹配对数: {interface_stats['total_interfaces']} 对")
        report_lines.append("")
        
        # 2. 构件节点拼装误差分析
        report_lines.append("-" * 80)
        report_lines.append("二、构件节点拼装误差分析")
        report_lines.append("-" * 80)
        report_lines.append("")
        report_lines.append("2.1 球心位移误差统计")
        report_lines.append(f"  平均值: {assembly_stats['center_error']['mean']*1000:.2f} mm")
        report_lines.append(f"  标准差: {assembly_stats['center_error']['std']*1000:.2f} mm")
        report_lines.append(f"  最大值: {assembly_stats['center_error']['max']*1000:.2f} mm")
        report_lines.append(f"  最小值: {assembly_stats['center_error']['min']*1000:.2f} mm")
        report_lines.append(f"  中位数: {assembly_stats['center_error']['median']*1000:.2f} mm")
        report_lines.append("")
        report_lines.append("  误差分级统计:")
        report_lines.append(f"    正常 (<50mm): {assembly_stats['center_error_levels']['normal']} 个 ({assembly_stats['center_error_levels']['normal']/assembly_stats['total_nodes']*100:.1f}%)")
        report_lines.append(f"    警告 (50-100mm): {assembly_stats['center_error_levels']['warning']} 个 ({assembly_stats['center_error_levels']['warning']/assembly_stats['total_nodes']*100:.1f}%)")
        report_lines.append(f"    严重 (≥100mm): {assembly_stats['center_error_levels']['critical']} 个 ({assembly_stats['center_error_levels']['critical']/assembly_stats['total_nodes']*100:.1f}%)")
        report_lines.append("")
        
        report_lines.append("2.2 半径误差统计")
        report_lines.append(f"  平均值: {assembly_stats['radius_error']['mean']*1000:.2f} mm")
        report_lines.append(f"  标准差: {assembly_stats['radius_error']['std']*1000:.2f} mm")
        report_lines.append(f"  最大值: {assembly_stats['radius_error']['max']*1000:.2f} mm")
        report_lines.append(f"  最小值: {assembly_stats['radius_error']['min']*1000:.2f} mm")
        report_lines.append(f"  中位数: {assembly_stats['radius_error']['median']*1000:.2f} mm")
        report_lines.append("")
        report_lines.append("  误差分级统计:")
        report_lines.append(f"    正常 (<30mm): {assembly_stats['radius_error_levels']['normal']} 个 ({assembly_stats['radius_error_levels']['normal']/assembly_stats['total_nodes']*100:.1f}%)")
        report_lines.append(f"    警告 (30-50mm): {assembly_stats['radius_error_levels']['warning']} 个 ({assembly_stats['radius_error_levels']['warning']/assembly_stats['total_nodes']*100:.1f}%)")
        report_lines.append(f"    严重 (≥50mm): {assembly_stats['radius_error_levels']['critical']} 个 ({assembly_stats['radius_error_levels']['critical']/assembly_stats['total_nodes']*100:.1f}%)")
        report_lines.append("")
        
        # 3. 接口错边误差分析
        report_lines.append("-" * 80)
        report_lines.append("三、接口错边误差分析")
        report_lines.append("-" * 80)
        report_lines.append("")
        report_lines.append("3.1 总体接口错边统计")
        report_lines.append(f"  平均错边量: {interface_stats['distance_stats']['mean']*1000:.2f} mm")
        report_lines.append(f"  标准差: {interface_stats['distance_stats']['std']*1000:.2f} mm")
        report_lines.append(f"  最大错边量: {interface_stats['distance_stats']['max']*1000:.2f} mm")
        report_lines.append(f"  最小错边量: {interface_stats['distance_stats']['min']*1000:.2f} mm")
        report_lines.append(f"  中位数: {interface_stats['distance_stats']['median']*1000:.2f} mm")
        report_lines.append("")
        report_lines.append("  错边分级统计:")
        report_lines.append(f"    正常 (<50mm): {interface_stats['error_levels']['normal']} 对 ({interface_stats['error_levels']['normal']/interface_stats['total_interfaces']*100:.1f}%)")
        report_lines.append(f"    警告 (50-80mm): {interface_stats['error_levels']['warning']} 对 ({interface_stats['error_levels']['warning']/interface_stats['total_interfaces']*100:.1f}%)")
        report_lines.append(f"    严重 (≥80mm): {interface_stats['error_levels']['critical']} 对 ({interface_stats['error_levels']['critical']/interface_stats['total_interfaces']*100:.1f}%)")
        report_lines.append("")
        
        # 26-45品区域分析
        if 'region_26_45' in interface_stats:
            report_lines.append("3.2 26-45品区域接口错边专项分析")
            region = interface_stats['region_26_45']
            report_lines.append(f"  该区域接口匹配数: {region['count']} 对")
            report_lines.append(f"  平均错边量: {region['mean_distance']*1000:.2f} mm")
            report_lines.append(f"  最大错边量: {region['max_distance']*1000:.2f} mm")
            
            if len(region['problem_interfaces']) > 0:
                report_lines.append(f"  问题接口数: {len(region['problem_interfaces'])} 对")
                report_lines.append("")
                report_lines.append("  问题接口详情:")
                for idx, row in region['problem_interfaces'].head(5).iterrows():
                    report_lines.append(f"    - {row['重构节点ID']} ↔ {row['周边节点ID']}: {row['错边距离(m)']*1000:.2f} mm")
            report_lines.append("")
        
        # 4. 问题节点详细分析
        report_lines.append("-" * 80)
        report_lines.append("四、问题节点详细分析")
        report_lines.append("-" * 80)
        report_lines.append("")
        
        # 球心位移严重超差节点
        critical_center = assembly_stats['problem_nodes'][
            assembly_stats['problem_nodes']['球心误差(m)'] >= self.thresholds['center_error_critical']
        ]
        if len(critical_center) > 0:
            report_lines.append("4.1 球心位移严重超差节点 (≥100mm)")
            for idx, row in critical_center.head(10).iterrows():
                report_lines.append(f"  节点 {int(row['序号'])}:")
                report_lines.append(f"    设计位置: ({row['模型球心X']:.3f}, {row['模型球心Y']:.3f}, {row['模型球心Z']:.3f})")
                report_lines.append(f"    实际位置: ({row['实际球心X']:.3f}, {row['实际球心Y']:.3f}, {row['实际球心Z']:.3f})")
                report_lines.append(f"    位移误差: {row['球心误差(m)']*1000:.2f} mm")
                report_lines.append(f"    半径误差: {row['半径误差(m)']*1000:.2f} mm")
                report_lines.append("")
        
        # 警告级别节点
        warning_nodes = assembly_stats['problem_nodes'][
            (assembly_stats['problem_nodes']['球心误差(m)'] >= self.thresholds['center_error_warning']) &
            (assembly_stats['problem_nodes']['球心误差(m)'] < self.thresholds['center_error_critical'])
        ]
        if len(warning_nodes) > 0:
            report_lines.append(f"4.2 球心位移警告级别节点 (50-100mm): 共 {len(warning_nodes)} 个")
            report_lines.append("  前5个警告节点:")
            for idx, row in warning_nodes.head(5).iterrows():
                report_lines.append(f"    节点 {int(row['序号'])}: 位移 {row['球心误差(m)']*1000:.2f} mm, 半径 {row['半径误差(m)']*1000:.2f} mm")
            report_lines.append("")
        
        # 5. 接口错边问题分析
        report_lines.append("-" * 80)
        report_lines.append("五、接口错边问题分析")
        report_lines.append("-" * 80)
        report_lines.append("")
        
        if len(interface_stats['problem_interfaces']) > 0:
            report_lines.append("5.1 严重接口错边 (≥80mm)")
            critical_interfaces = interface_stats['problem_interfaces'][
                interface_stats['problem_interfaces']['错边距离(m)'] >= self.thresholds['interface_error_critical']
            ]
            for idx, row in critical_interfaces.iterrows():
                report_lines.append(f"  {row['重构节点ID']} ↔ {row['周边节点ID']}:")
                report_lines.append(f"    错边距离: {row['错边距离(m)']*1000:.2f} mm")
                report_lines.append(f"    重构节点位置: ({row['重构节点X']:.3f}, {row['重构节点Y']:.3f}, {row['重构节点Z']:.3f})")
                report_lines.append(f"    周边节点位置: ({row['周边节点X']:.3f}, {row['周边节点Y']:.3f}, {row['周边节点Z']:.3f})")
                report_lines.append("")
            
            report_lines.append("5.2 警告级别接口错边 (50-80mm)")
            warning_interfaces = interface_stats['problem_interfaces'][
                (interface_stats['problem_interfaces']['错边距离(m)'] >= self.thresholds['interface_error_warning']) &
                (interface_stats['problem_interfaces']['错边距离(m)'] < self.thresholds['interface_error_critical'])
            ]
            for idx, row in warning_interfaces.iterrows():
                report_lines.append(f"  {row['重构节点ID']} ↔ {row['周边节点ID']}: {row['错边距离(m)']*1000:.2f} mm")
            report_lines.append("")
        else:
            report_lines.append("✓ 所有接口错边均在正常范围内")
            report_lines.append("")
        
        # 6. 空间分布分析
        report_lines.append("-" * 80)
        report_lines.append("六、误差空间分布分析")
        report_lines.append("-" * 80)
        report_lines.append("")
        report_lines.append("6.1 按X坐标分区统计")
        for region in spatial_stats['region_stats']:
            report_lines.append(f"  X范围 {region['x_range']}:")
            report_lines.append(f"    节点数: {region['node_count']} 个")
            report_lines.append(f"    平均球心误差: {region['avg_center_error']*1000:.2f} mm")
            report_lines.append(f"    最大球心误差: {region['max_center_error']*1000:.2f} mm")
            report_lines.append(f"    平均半径误差: {region['avg_radius_error']*1000:.2f} mm")
            report_lines.append("")
        
        # 7. 工程建议
        report_lines.append("-" * 80)
        report_lines.append("七、工程建议")
        report_lines.append("-" * 80)
        report_lines.append("")
        
        # 根据分析结果给出建议
        critical_count = assembly_stats['center_error_levels']['critical']
        warning_count = assembly_stats['center_error_levels']['warning']
        interface_critical = interface_stats['error_levels']['critical']
        interface_warning = interface_stats['error_levels']['warning']
        
        report_lines.append("7.1 总体评价")
        if critical_count == 0 and interface_critical == 0:
            if warning_count == 0 and interface_warning == 0:
                report_lines.append("  ★★★★★ 优秀: 所有节点和接口均在正常范围内，拼装质量良好")
            else:
                report_lines.append("  ★★★★☆ 良好: 存在少量警告级别问题，建议关注但无需立即处理")
        elif critical_count <= 3 and interface_critical <= 1:
            report_lines.append("  ★★★☆☆ 合格: 存在个别严重问题节点，建议针对性调整")
        else:
            report_lines.append("  ★★☆☆☆ 需改进: 存在多处严重问题，建议全面检查并调整")
        report_lines.append("")
        
        report_lines.append("7.2 具体建议")
        
        if critical_count > 0:
            report_lines.append(f"  1. 球心位移严重超差节点处理 (共{critical_count}个):")
            report_lines.append("     - 建议对这些节点进行重新测量和定位")
            report_lines.append("     - 检查支撑体系和临时固定措施")
            report_lines.append("     - 必要时进行构件调整或更换")
            report_lines.append("")
        
        if warning_count > 0:
            report_lines.append(f"  2. 球心位移警告级别节点监控 (共{warning_count}个):")
            report_lines.append("     - 加强监测频率，跟踪误差发展趋势")
            report_lines.append("     - 检查相邻节点的装配情况")
            report_lines.append("")
        
        if interface_critical > 0:
            report_lines.append(f"  3. 接口错边严重问题处理 (共{interface_critical}对):")
            report_lines.append("     - 优先处理错边量超过80mm的接口")
            report_lines.append("     - 检查接口连接件的安装质量")
            report_lines.append("     - 评估对结构整体受力的影响")
            report_lines.append("")
        
        if 'region_26_45' in interface_stats and len(interface_stats['region_26_45']['problem_interfaces']) > 0:
            report_lines.append("  4. 26-45品区域专项处理:")
            report_lines.append(f"     - 该区域存在 {len(interface_stats['region_26_45']['problem_interfaces'])} 个问题接口")
            report_lines.append("     - 建议对该区域进行重点检查和调整")
            report_lines.append("     - 考虑是否存在系统性偏差")
            report_lines.append("")
        
        # 通用建议
        report_lines.append("  5. 通用建议:")
        report_lines.append(f"     - 当前平均球心误差 {assembly_stats['center_error']['mean']*1000:.1f}mm，建议控制在50mm以内")
        report_lines.append(f"     - 当前平均接口错边 {interface_stats['distance_stats']['mean']*1000:.1f}mm，建议控制在50mm以内")
        report_lines.append("     - 加强施工过程中的质量控制")
        report_lines.append("     - 建立定期复测机制")
        report_lines.append("")
        
        # 8. 结论
        report_lines.append("-" * 80)
        report_lines.append("八、结论")
        report_lines.append("-" * 80)
        report_lines.append("")
        report_lines.append(f"本次分析共检测 {assembly_stats['total_nodes']} 个构件节点，")
        report_lines.append(f"成功匹配 {interface_stats['total_interfaces']} 对周边结构接口。")
        report_lines.append("")
        report_lines.append(f"总体球心位移误差: {assembly_stats['center_error']['mean']*1000:.2f}mm ± {assembly_stats['center_error']['std']*1000:.2f}mm")
        report_lines.append(f"总体接口错边误差: {interface_stats['distance_stats']['mean']*1000:.2f}mm ± {interface_stats['distance_stats']['std']*1000:.2f}mm")
        report_lines.append("")
        
        if critical_count == 0 and interface_critical == 0:
            report_lines.append("拼装质量总体可控，建议按正常程序进行后续施工。")
        else:
            report_lines.append("存在部分超差问题，建议按上述专项建议进行处理后再进行后续施工。")
        
        report_lines.append("")
        report_lines.append("=" * 80)
        report_lines.append("报告结束")
        report_lines.append("=" * 80)
        
        # 保存报告
        report_text = "\n".join(report_lines)
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"\n✓ 报告已生成: {output_file}")
            return True
        except Exception as e:
            print(f"\n✗ 报告保存失败: {e}")
            return False
    
    def print_summary(self):
        """打印分析摘要到控制台"""
        if not self.load_data():
            return
            
        assembly_stats = self.analyze_assembly_errors()
        interface_stats = self.analyze_interface_errors()
        
        print("\n" + "=" * 60)
        print(" " * 15 + "误差分析摘要")
        print("=" * 60)
        
        print(f"\n【构件节点分析】")
        print(f"  总节点数: {assembly_stats['total_nodes']}")
        print(f"  球心误差: {assembly_stats['center_error']['mean']*1000:.2f} ± {assembly_stats['center_error']['std']*1000:.2f} mm")
        print(f"  半径误差: {assembly_stats['radius_error']['mean']*1000:.2f} ± {assembly_stats['radius_error']['std']*1000:.2f} mm")
        print(f"  严重问题: {assembly_stats['center_error_levels']['critical']} 个")
        print(f"  警告级别: {assembly_stats['center_error_levels']['warning']} 个")
        
        print(f"\n【接口错边分析】")
        print(f"  总接口数: {interface_stats['total_interfaces']}")
        print(f"  平均错边: {interface_stats['distance_stats']['mean']*1000:.2f} ± {interface_stats['distance_stats']['std']*1000:.2f} mm")
        print(f"  最大错边: {interface_stats['distance_stats']['max']*1000:.2f} mm")
        print(f"  严重问题: {interface_stats['error_levels']['critical']} 对")
        print(f"  警告级别: {interface_stats['error_levels']['warning']} 对")
        
        if 'region_26_45' in interface_stats:
            print(f"\n【26-45品区域】")
            print(f"  接口数: {interface_stats['region_26_45']['count']}")
            print(f"  平均错边: {interface_stats['region_26_45']['mean_distance']*1000:.2f} mm")
        
        print("\n" + "=" * 60)


def main():
    """主函数"""
    # 文件路径
    assembly_file = "assembly_error_analysis.csv"
    interface_file = "interface_misalignment_analysis.csv"
    
    # 检查文件是否存在
    if not os.path.exists(assembly_file):
        print(f"错误: 找不到文件 {assembly_file}")
        print(f"当前工作目录: {os.getcwd()}")
        return
    
    if not os.path.exists(interface_file):
        print(f"错误: 找不到文件 {interface_file}")
        print(f"当前工作目录: {os.getcwd()}")
        return
    
    # 创建报告生成器
    reporter = AssemblyErrorReport(assembly_file, interface_file)
    
    # 打印摘要
    reporter.print_summary()
    
    # 生成详细报告
    output_file = f"工程拼装误差分析报告_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    reporter.generate_report(output_file)
    
    print(f"\n详细报告已保存至: {output_file}")


if __name__ == "__main__":
    main()
