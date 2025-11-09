import numpy as np

# 定义函数设置无人机坐标
def drone_initialize():
    name=input("请设置无人机代号：")
    x=float(input("请设置x坐标:"))
    y=float(input("请设置y坐标:"))
    z=float(input("请设置z坐标:"))
    speed=float(input(f"请设置无人机{name}速度:"))
    r=float(input(f"请设置无人机{name}扫描半径:"))
    print(f"无人机'{name}'坐标已设置为:({x},{y},{z}),速度已设置为{speed}，扫描半径已设置为{r}")
    return np.array([x,y,z]),speed,r

# 分别设置三架无人机坐标
# dro_pos1=drone_initialize()
# dro_pos2=drone_initialize()
# dro_pos3=drone_initialize()

# 得到一个含有三架无人机坐标的二维数组并求三维平均轴坐标测试

# pos=np.array([dro_pos1,dro_pos2,dro_pos3])
# ava_x=round(np.mean(pos[:,0]),2)
# ava_y=round(np.mean(pos[:,1]),2)
# ava_z=round(np.mean(pos[:,2]),2)
# print(f"三架无人机平均坐标为{ava_x},{ava_y},{ava_z}")

# 定义函数求两架无人机之间的距离 后续在无人机协同时可以使用
def calculate_distance(np1,np2):
    difference=np1-np2
    distance=np.sqrt(np.sum(difference**2))
    return round(distance,2)

# 分别计算三架无人机之间的距离进行计算
# dis_1_2=calculate_distance(dro_pos1,dro_pos2)
# dis_2_3=calculate_distance(dro_pos2,dro_pos3)
# dis_1_3=calculate_distance(dro_pos3,dro_pos1)
# print(dis_1_2)
# print(dis_2_3)
# print(dis_1_3)
