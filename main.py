import position_and_distance as pd
import coverage_calculate as cc
import random as rd

# 获取无人机初始位置
print("请设置两架无人机的初始位置：")
print("设置无人机1初始位置:")
drone1_pos = pd.position_set()
print("设置无人机2初始位置:")
drone2_pos = pd.position_set()

# 创建覆盖率计算类
calculator = cc.DroneCoverage(drone1_pos, drone2_pos, scan_range=rd.randint(1,5))

# 显示无人机信息
print(f"\n无人机信息：")
print(f"无人机1速度: {calculator.drone1_speed}")
print(f"无人机2速度: {calculator.drone2_speed}")
print(f"扫描范围: {calculator.scan_range}")

# 获取模拟步数
print("请输入模拟步数:")
steps = int(input())

# 开始模拟
calculator.simulate(steps)

# 计算并显示结果
coverage_percent, scanned_points = calculator.calculate_coverage()
print(f"总飞行步数: {steps}")
print(f"扫描点数量: {scanned_points}")
print(f"覆盖率: {coverage_percent*100}%")