from irrigation_system import compute_management_units_with_points, visualize_units_with_points, export_data

def main():
    # 定义边界和出水桩位置
    boundary = [
        (85.559037, 44.861717),
        (85.578052, 44.861417),
        (85.577580, 44.851986),
        (85.558651, 44.852013)
    ]

    water_points = [
        (85.560321, 44.860046),
        (85.565587, 44.860307),
        (85.571283, 44.860543),
        (85.576680, 44.860536),

        (85.576295, 44.857190),
        (85.571325, 44.858109),
        (85.565244, 44.857661),
        (85.560150, 44.857821),

        (85.560321, 44.85546),
        (85.565587, 44.855307),
        (85.571283, 44.85343),
        (85.576680, 44.85436)

    ]

    # 计算管理单元及其代表点
    print("正在计算管理单元和代表点...")
    units_with_points = compute_management_units_with_points(boundary, water_points, 3, 4)
    
    # 输出统计信息
    print(f"计算完成! 共有 {len(units_with_points)} 个出水桩管理单元")
    total_points = sum(len(unit_data['representative_points']) for unit_data in units_with_points.values())
    print(f"共计 {total_points} 个代表点")
    
    # 导出数据
    export_filename = "irrigation_data_result.txt"
    print(f"正在导出数据到 {export_filename}...")
    export_data(units_with_points, export_filename)
    print(f"数据导出完成!")
    
    # 可视化结果
    print("正在生成可视化图表...")
    visualize_units_with_points(boundary, water_points, units_with_points)
    print("可视化完成!")

if __name__ == "__main__":
    main()