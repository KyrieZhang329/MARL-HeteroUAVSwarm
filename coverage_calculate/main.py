import position_and_distance as pd
import coverage_calculate as cc
import time
from matplotlib import pyplot as plt

# 设置两架异构无人机初始信息
print("请设置两架无人机的初始信息：")
print("设置无人机1初始信息:")
drone1_pos,drone1_speed,drone1_range = pd.drone_initialize()
print("设置无人机2初始信息:")
drone2_pos,drone2_speed,drone2_range = pd.drone_initialize()

# 创建覆盖率计算类
calculator = cc.DroneCoverage(drone1_pos, drone2_pos, drone1_range,drone2_range,drone1_speed=drone1_speed, drone2_speed=drone2_speed)

# 初始化总探测点数
total_points=int(input("请初始化设置总探测点数：")) # 测试“10000000

# 获取模拟步数
print("请输入模拟步数:") # 测试：300
steps = int(input())

# 显示无人机信息
print(f"\n模拟测试信息：")
print(f"无人机1速度: {calculator.drone1_speed}\n无人机1扫描半径:{calculator.drone1_scan_range}")
print(f"无人机2速度: {calculator.drone2_speed}\n无人机2扫描半径:{calculator.drone2_scan_range}")
print(f"模拟总探测点数:{total_points}")
print(f"模拟步数:{steps}")
time.sleep(2)


# 开始模拟
calculator.simulate(steps,total_points)

# 计算并显示结果
coverage_percent, scanned_points = calculator.calculate_coverage(total_points)
print(f"总飞行步数: {steps}")
print(f"扫描点数量: {scanned_points}")
print(f"覆盖率: {coverage_percent*100}%")