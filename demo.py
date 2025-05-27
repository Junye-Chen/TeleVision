
#!/usr/bin/env python3
# -*-coding:utf8-*-
# 注意demo无法直接运行，需要pip安装sdk后才能运行
# 读取机械臂消息并打印,需要先安装piper_sdk
from typing import (
    Optional,
)
from piper_sdk import *
from pytransform3d import rotations


# 测试代码
if __name__ == "__main__":
    # piper = C_PiperInterface()
    # piper.ConnectPort()
    # while True:
    #     import time
    #     print(piper.GetArmStatus())
    #     # piper.JointCtrl()
    #     # piper.JointConfig()
    #     time.sleep(0.005)
    #     pass

    import numpy as np
    import math

    print(math.sqrt(2)/2)

    # 定义一个旋转矩阵
    # R = np.array([
    #     [0.7071, -0.7071, 0],
    #     [0.7071, 0.7071, 0],
    #     [0, 0, 1]
    # ])
    R = np.array([
        [math.sqrt(2)/2, -math.sqrt(2)/2, 0],
        [math.sqrt(2)/2, math.sqrt(2)/2, 0],
        [0, 0, 1]
    ])

    # 计算欧拉角
    euler_angles = rotations.euler_from_matrix(R,0,1,2, extrinsic=True)
    print("欧拉角（以弧度表示）:", euler_angles)

    # 如果你需要将欧拉角转换为角度
    euler_angles_degrees = np.degrees(euler_angles)
    print("欧拉角（以度表示）:", euler_angles_degrees)


