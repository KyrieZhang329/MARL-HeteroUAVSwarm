import numpy as np

# 定义函数设置无人机坐标
def position_set():
    index=input("请设置无人机编号：")
    a=float(input("请设置x坐标:"))
    b=float(input("请设置y坐标:"))
    c=float(input("请设置z坐标:"))
    print(f"无人机{index}坐标已设置为:({a},{b},{c})")
    return np.array([a,b,c])

# 分别设置三架无人机坐标
# dro_pos1=position_set()
# dro_pos2=position_set()
# dro_pos3=position_set()

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
