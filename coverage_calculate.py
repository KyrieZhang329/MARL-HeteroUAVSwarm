import numpy as np

# 定义扫描覆盖率计算类
class DroneCoverage:
    def __init__(self, drone1_pos, drone2_pos, scan_range):

        # 保存无人机位置信息
        self.drone1_pos = np.array(drone1_pos)  # 无人机1当前位置
        self.drone2_pos = np.array(drone2_pos)  # 无人机2当前位置
        self.scan_range = scan_range  # 扫描范围
        self.scanned_points = set()  # 已扫描的点集合
        
        # 添加初始位置到已扫描点
        self.add_scanned_points(self.drone1_pos)
        self.add_scanned_points(self.drone2_pos)
        
        # 两架异构无人机的速度
        self.drone1_speed = 1.0  # 无人机1
        self.drone2_speed = 1.5  # 无人机2
    
    def add_scanned_points(self, pos):
        x, y, z = pos
        for dx in range(-int(self.scan_range), int(self.scan_range)+1):
            for dy in range(-int(self.scan_range), int(self.scan_range)+1):
                for dz in range(-int(self.scan_range), int(self.scan_range)+1):
                    # 计算点到无人机位置的距离,判断点是否在扫描范围内
                    distance = np.sqrt(dx**2 + dy**2 + dz**2)
                    if distance <= self.scan_range:
                        # 将扫描到的点添加到集合中
                        scanning_point = (int(x+dx), int(y+dy), int(z+dz))
                        self.scanned_points.add(scanning_point)

    def orient(self):  # 确定运动的方向向量
        direction=np.random.uniform(-1,1,3)
        norm=np.linalg.norm(direction)
        if norm!=0:
            direction/=norm
        return direction

    def move_drones(self):
        # 无人机1移动
        direction1=self.orient()
        self.drone1_pos += direction1*self.drone1_speed
        self.add_scanned_points(self.drone1_pos)
        
        # 无人机2移动
        direction2=self.orient()
        self.drone2_pos += direction2*self.drone2_speed
        self.add_scanned_points(self.drone2_pos)
    
    def simulate(self, steps): # 指定模拟飞行的步数
        print(f"开始模拟飞行 {steps} 步...")
        for i in range(steps):
            self.move_drones()
            # 每10个步长显示一次进度
            if (i+1) % 10 == 0:
                print(f"已完成 {i+1} 步飞行")
        print("模拟完成！")
    
    def calculate_coverage(self): # 扫描率计算
        # 扫描点数量
        scanned_count = len(self.scanned_points)
        total_count = 10000  # 假设总共有10000个点
        # 计算覆盖率
        coverage_percent = round(min(scanned_count / total_count, 1.00),2)
        return coverage_percent, scanned_count
