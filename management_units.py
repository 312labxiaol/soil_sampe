import matplotlib.pyplot as plt
import numpy as np
import math

def line_intersection(line1, line2):
    """
    计算两条线段的交点
    
    参数:
        line1: ((x1, y1), (x2, y2)) 线段1的两个端点
        line2: ((x3, y3), (x4, y4)) 线段2的两个端点
    
    返回:
        交点坐标 (x, y) 或 None（如果不相交）
    """
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        return None  # 线段平行

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    
    # 检查交点是否在两条线段上
    def is_point_on_segment(point, segment):
        x, y = point
        (x1, y1), (x2, y2) = segment
        if min(x1, x2) <= x <= max(x1, x2) and min(y1, y2) <= y <= max(y1, y2):
            return True
        return False
    
    if is_point_on_segment((x, y), line1) and is_point_on_segment((x, y), line2):
        return (x, y)
    else:
        return None


def extend_line_to_boundary(point1, point2, boundary):
    """
    将线段延长直到与边界相交，并返回交点
    
    参数:
        point1, point2: 线段的两个点
        boundary: 边界点列表
    
    返回:
        与边界的交点列表
    """
    closed_boundary = boundary + [boundary[0]]
    intersections = []
    
    dx = point2[0] - point1[0]
    dy = point2[1] - point1[1]
    
    # 如果方向向量几乎为零，无法延长线段
    if abs(dx) < 1e-10:
        dx = 1e-10
    if abs(dy) < 1e-10:
        dy = 1e-10
    
    extension_factor = 100
    extended_p1 = (point1[0] - extension_factor * dx, point1[1] - extension_factor * dy)
    extended_p2 = (point2[0] + extension_factor * dx, point2[1] + extension_factor * dy)
    extended_line = (extended_p1, extended_p2)
    
    for i in range(len(closed_boundary) - 1):
        boundary_segment = (closed_boundary[i], closed_boundary[i + 1])
        intersection = line_intersection(extended_line, boundary_segment)
        if intersection:
            intersections.append(intersection)
    
    return intersections


def find_nearest_point_in_quadrant(reference_point, candidate_points, quadrant):
    """
    在给定象限中找到离参考点最近的点
    
    参数:
        reference_point: 参考点坐标
        candidate_points: 候选点列表
        quadrant: 象限编号 (1-右上, 2-左上, 3-左下, 4-右下)
    
    返回:
        最近的点或None
    """
    nearest_point = None
    min_distance = float('inf')
    
    for point in candidate_points:
        dx = point[0] - reference_point[0]
        dy = point[1] - reference_point[1]
        
        # 检查点是否在指定象限
        if ((quadrant == 1 and dx >= 0 and dy >= 0) or  # 右上
            (quadrant == 2 and dx <= 0 and dy >= 0) or  # 左上
            (quadrant == 3 and dx <= 0 and dy <= 0) or  # 左下
            (quadrant == 4 and dx >= 0 and dy <= 0)):   # 右下
            
            distance = dx*dx + dy*dy
            if distance < min_distance:
                min_distance = distance
                nearest_point = point
                
    return nearest_point


def calculate_critical_points(boundary, water_points, rows=None, cols=None):
    """
    计算管理单元所需的所有关键点（交点、中点、边界交点等）
    
    参数:
        boundary: 边界点列表
        water_points: 出水桩坐标列表
        rows, cols: 行列数（可选，如果未指定则自动确定）
    
    返回:
        包含所有关键点的列表
    """
    # 自动确定行列数（如果未指定）
    if rows is None or cols is None:
        y_coords = [p[1] for p in water_points]
        unique_y = []
        threshold = 0.001
        
        for y in y_coords:
            found = False
            for i, known_y in enumerate(unique_y):
                if abs(y - known_y) < threshold:
                    found = True
                    break
            if not found:
                unique_y.append(y)
        
        detected_rows = len(unique_y)
        detected_cols = math.ceil(len(water_points) / detected_rows)
        
        rows = rows or detected_rows
        cols = cols or detected_cols
    
    # 对水点进行排序和分组
    sorted_points = sorted(water_points, key=lambda p: p[1], reverse=True)
    
    water_rows = []
    for i in range(0, len(sorted_points), cols):
        row = sorted_points[i:i+cols]
        row.sort(key=lambda p: p[0])
        water_rows.append(row)
    
    # 计算行中点
    row_midpoints = []
    for row in water_rows:
        current_row_mids = []
        for j in range(len(row) - 1):
            mid_x = (row[j][0] + row[j+1][0]) / 2
            mid_y = (row[j][1] + row[j+1][1]) / 2
            current_row_mids.append((mid_x, mid_y, j))
        row_midpoints.append(current_row_mids)
    
    # 计算列中点
    col_midpoints_by_row = [[] for _ in range(len(water_rows)-1)]
    max_cols = max(len(row) for row in water_rows)
    for col_idx in range(max_cols):
        col_points = []
        for row_idx, row in enumerate(water_rows):
            if col_idx < len(row):
                col_points.append(row[col_idx])
        
        if len(col_points) > 1:
            for j in range(len(col_points) - 1):
                mid_x = (col_points[j][0] + col_points[j+1][0]) / 2
                mid_y = (col_points[j][1] + col_points[j+1][1]) / 2
                col_midpoints_by_row[j].append((mid_x, mid_y))
    
    # 收集关键点
    all_critical_points = list(boundary)  # 首先添加边界点
    
    purple_line_segments = []
    orange_line_segments = []
    purple_lines = []
    orange_lines = []
    
    # 计算紫色线段（列连线）
    if len(row_midpoints) >= 2:
        max_col_idx = max(len(row) for row in row_midpoints) - 1
        for col_idx in range(max_col_idx + 1):
            col_mids = []
            for row_idx, row_mids in enumerate(row_midpoints):
                for mid_x, mid_y, mid_col_idx in row_mids:
                    if mid_col_idx == col_idx:
                        col_mids.append((mid_x, mid_y))
            
            if len(col_mids) >= 2:
                for i in range(len(col_mids) - 1):
                    purple_line_segments.append((col_mids[i], col_mids[i+1]))
                
                if len(col_mids) >= 2:
                    purple_lines.append((col_mids[0], col_mids[-1]))
    
    # 计算橙色线段（行连线）
    for row_idx, row_mids in enumerate(col_midpoints_by_row):
        if len(row_mids) >= 2:
            for i in range(len(row_mids) - 1):
                orange_line_segments.append((row_mids[i], row_mids[i+1]))
            
            if len(row_mids) >= 2:
                orange_lines.append((row_mids[0], row_mids[-1]))
    
    # 计算交点
    intersection_points = []
    for purple_segment in purple_line_segments:
        for orange_segment in orange_line_segments:
            intersection = line_intersection(purple_segment, orange_segment)
            if intersection:
                intersection_points.append(intersection)
                all_critical_points.append(intersection)
    
    # 计算线与边界的交点
    boundary_intersections = []
    for line in purple_lines:
        intersections = extend_line_to_boundary(line[0], line[-1], boundary)
        for intersection in intersections:
            all_critical_points.append(intersection)
            boundary_intersections.append(("紫色线", intersection))
    
    for line in orange_lines:
        intersections = extend_line_to_boundary(line[0], line[-1], boundary)
        for intersection in intersections:
            all_critical_points.append(intersection)
            boundary_intersections.append(("橙色线", intersection))
    
    return all_critical_points


def determine_management_units(boundary, water_points, rows=None, cols=None):
    """
    确定每个出水桩的管理单元顶点坐标
    
    参数:
        boundary: 边界点列表
        water_points: 出水桩坐标列表
        rows, cols: 行列数（可选，如果未指定则自动确定）
    
    返回:
        管理单元字典，键为出水桩索引，值为顶点列表
    """
    all_critical_points = calculate_critical_points(boundary, water_points, rows, cols)
    management_units = {}
    
    for i, point in enumerate(water_points):
        point_idx = i + 1
        
        # 在四个象限中找出离出水桩最近的点
        ne_point = find_nearest_point_in_quadrant(point, all_critical_points, 1)  # 右上
        nw_point = find_nearest_point_in_quadrant(point, all_critical_points, 2)  # 左上
        sw_point = find_nearest_point_in_quadrant(point, all_critical_points, 3)  # 左下
        se_point = find_nearest_point_in_quadrant(point, all_critical_points, 4)  # 右下
        
        # 四个象限都找到点才构成有效的管理单元
        if ne_point and nw_point and sw_point and se_point:
            # 按逆时针顺序排列顶点
            vertices = [nw_point, ne_point, se_point, sw_point]
            management_units[point_idx] = vertices
    
    return management_units


def visualize_management_units(boundary, water_points, management_units, title="Schematic diagram", en_labels=True):
    """
    可视化管理单元
    
    参数:
        boundary: 边界点列表
        water_points: 出水桩坐标列表
        management_units: 管理单元字典
        title: 图表标题
        en_labels: 使用英文标签
    """
    plt.figure(figsize=(4, 3))
    
    # 绘制边界
    boundary_x = [point[0] for point in boundary] + [boundary[0][0]]
    boundary_y = [point[1] for point in boundary] + [boundary[0][1]]
    plt.plot(boundary_x, boundary_y, 'k-', linewidth=2)
    
    # 绘制出水桩
    for i, point in enumerate(water_points):
        plt.scatter(point[0], point[1], c='blue', s=100, marker='o', zorder=20)
        plt.annotate(str(i+1), point, xytext=(5, 5), textcoords='offset points',
                   fontsize=12, fontweight='bold', zorder=25)
    
    # 生成不同的颜色用于各管理单元
    unit_colors = plt.cm.tab20(np.linspace(0, 1, len(water_points)))
    
    # 绘制管理单元
    for i, (point_idx, vertices) in enumerate(management_units.items()):
        # 绘制管理单元边界并填充颜色
        vertices_x = [p[0] for p in vertices] + [vertices[0][0]]  # 闭合多边形
        vertices_y = [p[1] for p in vertices] + [vertices[0][1]]
        
        # 先用半透明颜色填充区域
        plt.fill(vertices_x, vertices_y, color=unit_colors[i], alpha=0.3, zorder=5)
        
        # 绘制边界线
        plt.plot(vertices_x, vertices_y, color=unit_colors[i], linewidth=2.5, alpha=0.8, zorder=6)
        
        # 标记区域中心
        center_x = sum(p[0] for p in vertices) / 4
        center_y = sum(p[1] for p in vertices) / 4
        
        label = f'Area{point_idx}' if en_labels else f'区域{point_idx}'
        plt.text(center_x, center_y, label, fontsize=12, 
                ha='center', va='center', 
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'),
                zorder=25)
    
    # 添加坐标轴标签
    plt.xlabel('Longitude' if en_labels else '经度')
    plt.ylabel('Latitude' if en_labels else '纬度')
    plt.title(title)
    plt.axis('equal')
    plt.grid(True)
    plt.tight_layout()
    

def print_management_units(management_units):
    """打印每个出水桩管理单元的顶点坐标"""
    print("\n每个出水桩管理单元的顶点坐标:")
    for point_idx, vertices in management_units.items():
        print(f"出水桩 {point_idx} 的管理单元顶点:")
        vertices_str = "["
        for i, vertex in enumerate(vertices):
            vertices_str += f"({vertex[0]:.6f}, {vertex[1]:.6f})"
            if i < len(vertices) - 1:
                vertices_str += ", "
        vertices_str += "]"
        print(vertices_str)