import matplotlib.pyplot as plt
import numpy as np
import math

# 导入管理单元计算模块
from management_units import determine_management_units, visualize_management_units, line_intersection, extend_line_to_boundary, find_nearest_point_in_quadrant

# 导入点位计算所需函数
def calculate_distance(lon1, lat1, lon2, lat2):
    """计算两点间的实际地理距离（米）"""
    arc = 6371.393 * 1000
    a = ((lon2 - lon1) * (arc * math.cos(lat2) * 2 * math.pi) / 360) ** 2
    b = ((lat2 - lat1) * (arc * 2 * math.pi) / 360) ** 2
    return math.sqrt(a + b)

def calculate_middle_points(A, B, num_divisions_AB):
    """计算两点之间的等分点"""
    div_points_AB = []
    for i in range(1, num_divisions_AB):
        ratio = i / (num_divisions_AB)
        lat = A[0] + (B[0] - A[0]) * ratio
        lon = A[1] + (B[1] - A[1]) * ratio
        div_points_AB.append((lat, lon))
    return div_points_AB

def calculate_representative_points(A, B, C, D):
    """
    在四边形区域内计算4个代表点
    
    参数:
        A, B, C, D: 四边形的四个顶点，顺序为逆时针
                  A --- B
                  |     |
                  D --- C
    
    返回:
        包含4个代表点坐标的列表
    """
    a = calculate_distance(A[0], A[1], B[0], B[1])
    b = calculate_distance(B[0], B[1], C[0], C[1])
    c = calculate_distance(C[0], C[1], D[0], D[1])
    d = calculate_distance(D[0], D[1], A[0], A[1])

    if a >= b:
        num_divisions_MN1 = 9
        num_divisions_MN2 = 9
        num_divisions_AB = 18
        num_divisions_DC = 18
    else:
        num_divisions_MN1 = 18
        num_divisions_MN2 = 18
        num_divisions_AB = 9
        num_divisions_DC = 9

    # 计算各边的等分点
    div_points_AB = calculate_middle_points(A, B, num_divisions_AB)
    div_points_BC = calculate_middle_points(B, C, num_divisions_MN1)
    div_points_DC = calculate_middle_points(D, C, num_divisions_DC)
    div_points_AD = calculate_middle_points(A, D, num_divisions_MN2)

    # 计算代表点
    q = []
    if a < b:
        for i in [(2, 2), (6, 6), (10, 3), (14, 4)]:  # 调整为4个点的索引
            M1 = div_points_BC[i[0]]
            N1 = div_points_AD[i[0]]
            a1 = calculate_middle_points(M1, N1, num_divisions_AB)
            q.append(a1[i[1]])
    else:
        for i in [(2, 2), (6, 6), (3, 10), (4, 14)]:  # 调整为4个点的索引
            M1 = div_points_BC[i[0]]
            N1 = div_points_AD[i[0]]
            a1 = calculate_middle_points(M1, N1, num_divisions_AB)
            q.append(a1[i[1]])

    return q

def compute_management_units_with_points(boundary, water_points, rows=None, cols=None):
    """
    计算管理单元及其内部的代表点
    
    参数:
        boundary: 边界点列表
        water_points: 出水桩坐标列表
        rows, cols: 行列数（可选）
        
    返回:
        包含管理单元及其代表点的字典
    """
    # 计算管理单元
    management_units = determine_management_units(boundary, water_points, rows, cols)
    
    # 对每个管理单元计算代表点
    units_with_points = {}
    for point_idx, vertices in management_units.items():
        # 管理单元顶点按逆时针排序：[nw_point, ne_point, se_point, sw_point]
        # 调整为calculate_representative_points需要的顺序
        A = vertices[0]  # 左上 (nw)
        B = vertices[1]  # 右上 (ne)
        C = vertices[2]  # 右下 (se)
        D = vertices[3]  # 左下 (sw)
        
        # 计算代表点
        representative_points = calculate_representative_points(A, B, C, D)
        
        # 存储结果
        units_with_points[point_idx] = {
            "vertices": vertices,
            "representative_points": representative_points
        }
    
    return units_with_points

def visualize_units_with_points(boundary, water_points, units_with_points):
    """
    可视化管理单元及其代表点
    
    参数:
        boundary: 边界点列表
        water_points: 出水桩坐标列表
        units_with_points: 管理单元及其代表点字典
    """
    plt.figure(figsize=(14, 12))
    
    # 绘制边界
    boundary_x = [point[0] for point in boundary] + [boundary[0][0]]
    boundary_y = [point[1] for point in boundary] + [boundary[0][1]]
    # plt.plot(boundary_x, boundary_y, 'k-', linewidth=2, label='地块边界')
    
    # 生成不同的颜色用于各管理单元
    unit_colors = plt.cm.tab20(np.linspace(0, 1, len(water_points)))
    
    # 绘制出水桩
    for i, point in enumerate(water_points):
        plt.scatter(point[0], point[1], c='blue', s=120, marker='o', edgecolors='black', zorder=20)
        plt.annotate(str(i+1), point, xytext=(5, 5), textcoords='offset points',
                   fontsize=12, fontweight='bold', zorder=25)
    
    # 绘制管理单元及其代表点
    for i, (point_idx, unit_data) in enumerate(units_with_points.items()):
        vertices = unit_data["vertices"]
        rep_points = unit_data["representative_points"]
        
        # 绘制管理单元边界并填充颜色
        vertices_x = [p[0] for p in vertices] + [vertices[0][0]]
        vertices_y = [p[1] for p in vertices] + [vertices[0][1]]
        
        # 填充管理单元
        plt.fill(vertices_x, vertices_y, color=unit_colors[i], alpha=0.3, zorder=5)
        
        # 绘制边界线
        plt.plot(vertices_x, vertices_y, color=unit_colors[i], linewidth=2.5, alpha=0.8, zorder=6)
        
        # 标记区域中心
        center_x = sum(p[0] for p in vertices) / 4
        center_y = sum(p[1] for p in vertices) / 4
        plt.text(center_x, center_y, f'Area{point_idx}', fontsize=12, 
                ha='center', va='center', 
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'),
                zorder=25)
        
        # 绘制代表点
        for j, point in enumerate(rep_points):
            plt.scatter(point[0], point[1], c='red', s=80, marker='*', edgecolors='black', zorder=15)
            plt.annotate(f"{point_idx}-{j+1}", point, xytext=(3, 3), textcoords='offset points',
                       fontsize=9, color='black', weight='bold', zorder=16)
    
    # 添加图例
    plt.scatter([], [], c='blue', s=120, marker='o', edgecolors='black', label='Water Point')
    plt.scatter([], [], c='red', s=80, marker='*', edgecolors='black', label='Sample Point')
    
    # 添加每个管理单元的图例
    # for i in range(min(len(water_points), 5)):  # 只显示前5个单元的图例，避免图例过大
        # plt.plot([], [], '-', color=unit_colors[i], linewidth=2.0, label=f'出水桩{i+1}管理单元')
    
    plt.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=10)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Schematic diagram')
    plt.axis('equal')
    plt.grid(False)
    plt.tight_layout()
    plt.show()

def export_data(units_with_points, filename="irrigation_data.txt"):
    """
    导出管理单元及代表点数据
    
    参数:
        units_with_points: 管理单元及其代表点字典
        filename: 输出文件名
    """
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("出水桩管理单元及代表点数据\n")
        f.write("==========================\n\n")
        
        for point_idx, unit_data in units_with_points.items():
            vertices = unit_data["vertices"]
            rep_points = unit_data["representative_points"]
            
            f.write(f"出水桩 {point_idx} 的管理单元:\n")
            f.write("  顶点坐标:\n")
            for i, vertex in enumerate(vertices):
                f.write(f"    顶点{i+1}: ({vertex[0]:.6f}, {vertex[1]:.6f})\n")
            
            f.write("\n  代表点坐标:\n")
            for i, point in enumerate(rep_points):
                f.write(f"    点{i+1}: ({point[0]:.6f}, {point[1]:.6f})\n")
            
            f.write("\n" + "="*50 + "\n\n")
        
        f.write("文件生成时间: 2025年4月6日")