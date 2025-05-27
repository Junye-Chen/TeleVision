import math
import numpy as np

np.set_printoptions(precision=2, suppress=True)
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from pytransform3d import rotations

import time
import cv2
from constants_vuer import *
from TeleVision import OpenTeleVision
# import pyzed.sl as sl                
# TODO: import 新的相机

from piper.active_cam import PiperAgent
from multiprocessing import Array, Process, shared_memory, Queue, Manager, Event, Semaphore

from Preprocessor import VuerPreprocessor
from constants_vuer import grd_yup2grd_zup, hand2inspire
from motion_utils import mat_update, fast_mat_inv

"""
主文件：
1. 读取机械臂相机传回来的图片（需要机械臂的数据接口）
2. 通过VR的摄像头解算人手的姿态和位置（读取VR设备返回的手的矩阵）
3. 用这个位姿求解机械臂关节角度，设置夹爪
4. 关节角回传给机械臂执行


! 注意： 这里的坐标系需要明确，最好是转到基座的坐标上。如果有任何机械臂和操作对不上的都首先检查坐标系
"""


resolution = (720, 1280)
crop_size_w = 1
crop_size_h = 0
resolution_cropped = (resolution[0] - crop_size_h, resolution[1] - 2 * crop_size_w)

# 机器人初始化:
agent = PiperAgent(port="/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT8IT033-if00-port0")
agent._robot._driver.piper.EnableArm(7)
# enable_fun(piper=piper)
# agent._robot.set_torque_mode(True)  # 扭矩模式为开启


# # Create a Camera object
# zed = sl.Camera()

# # Create a InitParameters object and set configuration parameters
# init_params = sl.InitParameters()
# init_params.camera_resolution = sl.RESOLUTION.HD720  # Use HD720 opr HD1200 video mode, depending on camera type.
# init_params.camera_fps = 60  # Set fps at 60

# # Open the camera
# err = zed.open(init_params)
# if err != sl.ERROR_CODE.SUCCESS:
#     print("Camera Open : " + repr(err) + ". Exit program.")
#     exit()

# # Capture 50 frames and stop
# i = 0
# image_left = sl.Mat()
# image_right = sl.Mat()
# runtime_parameters = sl.RuntimeParameters()


# 共享内存和进程间通信设置
img_shape = (resolution_cropped[0], 2 * resolution_cropped[1], 3)
img_height, img_width = resolution_cropped[:2]
shm = shared_memory.SharedMemory(create=True, size=np.prod(img_shape) * np.uint8().itemsize)
img_array = np.ndarray((img_shape[0], img_shape[1], 3), dtype=np.uint8, buffer=shm.buf)
image_queue = Queue()
toggle_streaming = Event()
# tv = OpenTeleVision(resolution_cropped, shm.name, image_queue, toggle_streaming)
tv = OpenTeleVision(resolution_cropped, shm.name, image_queue, toggle_streaming, ngrok=True)
processor = VuerPreprocessor()


while True:
    start = time.time()

    # 从cam获取手腕位置和手指姿态，只需关注单右手的姿态
    # TODO 坐标系转换需要明确，最好是转到基座的坐标上
    head_mat, _, right_wrist_mat, _, right_hand_mat = processor.process(tv, origin=False)  # origin=False表示手的位姿相对于人头

    # 前3维相对于头部的位姿，后3维是欧拉角    
    right_rot = rotations.euler_from_matrix(right_wrist_mat[0:3, 0:3], 0,1,2, extrinsic=True)  # extrinsic 旋转内外旋
    print('right_rot: ', right_rot)
    right_pose = np.concatenate([right_wrist_mat[:3, 3] + np.array([-0.6, 0, 1.6]),  # TODO 后面这个offset需要重新测量
                                 right_rot])  # (6,)
    # TODO 处理夹爪的开闭
    

    # !这里是否需要retargeting呢？
    # right_qpos = right_retargeting.retarget(right_hand_mat[tip_indices])[[4, 5, 6, 7, 10, 11, 8, 9, 0, 1, 2, 3]]


    factor = 1000
    try:
        # agent._robot.command_joint_state(joint_angles)

        X = round(right_pose[0] * factor)
        Y = round(right_pose[1] * factor)
        Z = round(right_pose[2] * factor)
        RX = round(right_pose[3] * factor)
        RY = round(right_pose[4] * factor)
        RZ = round(right_pose[5] * factor)
        gripper = round(right_pose[6] * factor)
        agent._robot._driver.piper.MotionCtrl_2(0x01, 0x00, 100, 0x00)
        agent._robot._driver.piper.EndPoseCtrl(X, Y, Z, RX, RY, RZ)  # !不知道这个输入的旋转是外旋还是内旋，需要尝试
        agent._robot._driver.piper.GripperCtrl(abs(gripper), 1000, 0x01, 0)        

        print("success")
    except:
        # TODO 可能遇到了奇异点，需要清除错误代码 
        pass


    # 图像采集和处理
    # TODO：换成我们的相机接口
    # if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
    #     zed.retrieve_image(image_left, sl.VIEW.LEFT)
    #     zed.retrieve_image(image_right, sl.VIEW.RIGHT)
    #     timestamp = zed.get_timestamp(sl.TIME_REFERENCE.CURRENT)  # Get the timestamp at the time the image was captured
        # print("Image resolution: {0} x {1} || Image timestamp: {2}\n".format(image.get_width(), image.get_height(),
        #         timestamp.get_milliseconds()))
    image: np.ndarray   # 读取相机传回的图像

    bgr = np.hstack((image.numpy()[crop_size_h:, crop_size_w:-crop_size_w],
                     image.numpy()[crop_size_h:, crop_size_w:-crop_size_w]))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGRA2RGB)

    np.copyto(img_array, rgb)

    end = time.time()

    # time.sleep(0.01)  # 控制更新频率
    print(1/(end-start))
    
zed.close()