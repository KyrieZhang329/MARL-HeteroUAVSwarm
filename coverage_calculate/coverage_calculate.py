import numpy as np
import time

# 定义扫描覆盖率计算类
class DroneCoverage:
    def __init__(self, drone1_pos, drone2_pos, drone1_scan_range,drone2_scan_range,drone1_speed, drone2_speed):

        # 保存无人机位置信息
        self.drone1_pos = np.array(drone1_pos)  # 无人机1当前位置
        self.drone2_pos = np.array(drone2_pos)  # 无人机2当前位置
        self.drone1_scan_range = drone1_scan_range # 无人机1扫描范围
        self.drone2_scan_range = drone2_scan_range # 无人机2扫描范围
        self.scanned_points = set()  # 已扫描的点集合
        
        # 添加初始位置到已扫描点
        self.add_scanned_points(self.drone1_scan_range, self.drone2_scan_range, 
                                self.drone1_pos, self.drone2_pos)
        
        # 两架异构无人机的速度
        self.drone1_speed = drone1_speed  # 无人机1
        self.drone2_speed = drone2_speed  # 无人机2
    
    def add_scanned_points(self, drone1_scan_range, drone2_scan_range, drone1_pos, drone2_pos):
        # 添加无人机1扫描到的所有点
        points1 = self.scan_points(drone1_scan_range, drone1_pos)
        self.scanned_points.update(points1)
        
        # 添加无人机2扫描到的所有点
        points2 = self.scan_points(drone2_scan_range, drone2_pos)
        self.scanned_points.update(points2)

    def scan_points(self, scan_range, pos):
        x, y, z = pos
        scanned_points_set = set()
        
        for dx in range(int(-scan_range), int(scan_range) + 1):
            for dy in range(int(-scan_range), int(scan_range) + 1):
                for dz in range(int(-scan_range), int(scan_range) + 1):
                    # 计算点到无人机位置的距离,判断点是否在扫描范围内
                    distance = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
                    if distance <= scan_range:
                        # 将扫描到的点添加到集合中
                        scanning_point = (int(x + dx), int(y + dy), int(z + dz))
                        scanned_points_set.add(scanning_point)
        
        return scanned_points_set

    def orient(self):  # 确定运动的方向向量
        direction = np.random.uniform(-1, 1, 3)
        norm = np.linalg.norm(direction)
        if norm != 0:
            direction /= norm
        return direction

    def move_drones(self):
        # 无人机移动并添加新位置扫描到的点
        direction1 = self.orient()
        self.drone1_pos = self.drone1_pos + direction1 * self.drone1_speed
        direction2 = self.orient()
        self.drone2_pos = self.drone2_pos + direction2 * self.drone2_speed
        self.add_scanned_points(self.drone1_scan_range, self.drone2_scan_range, self.drone1_pos, self.drone2_pos)

    def simulate(self, steps, total_count):  # 飞行模拟主函数
        print(f"开始模拟飞行 {steps} 步")
        print(f"每10步后停顿2秒")
        for i in range(steps):  # 实时更新扫描覆盖率情况
            self.move_drones() # 移动无人机
            coverage_percent, scanned_points = self.calculate_coverage(total_count)
            print(f"第 {i+1} 步飞行，已扫描{scanned_points}个点，覆盖率更新为{coverage_percent*100}%")
            pause_duration = 0.5
            time.sleep(pause_duration)
        
        print("模拟完成！")
    
    def calculate_coverage(self, total_count):  # 扫描率计算
        scanned_count = len(self.scanned_points)  # 扫描点数量
        coverage_percent = round(min(scanned_count / total_count, 1.00), 4)  # 计算覆盖率
        return coverage_percent, scanned_count
