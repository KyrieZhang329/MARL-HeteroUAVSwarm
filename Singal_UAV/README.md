



## 环境搭建

1. 创建并激活conda环境：

   ```shell
   conda create -n marl_uav python=3.9
   conda activate marl_uav
   ```

2. 克隆项目：

    ```shell
    git clone https://github.com/KyrieZhang329/MARL-UAVCDP.git
    cd MARL-UAVCDP
    ```

3. 安装依赖库：

   ```shell
   pip install -r requirements.txt
   ```

   或者：

   ```shell
   pip install gymnasium numpy pygame matplotlib
   ```

   运行**main.py**即可可视化仿真环境