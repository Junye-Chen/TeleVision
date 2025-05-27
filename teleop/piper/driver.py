import time
from threading import Event, Lock, Thread
from typing import Protocol, Sequence

import numpy as np

# TODO: 这里用到了机器人的sdk
# from dynamixel_sdk.group_sync_read import GroupSyncRead  # 多电机的同步读取功能
# from dynamixel_sdk.group_sync_write import GroupSyncWrite  # 多电机的同步写入功能
# from dynamixel_sdk.packet_handler import PacketHandler   # 数据包处理器
# from dynamixel_sdk.port_handler import PortHandler  # 串口通信处理器
# from dynamixel_sdk.robotis_def import (
#     COMM_SUCCESS,  # 通信成功状态码
#     DXL_HIBYTE,
#     DXL_HIWORD,
#     DXL_LOBYTE,
#     DXL_LOWORD,
# )

#! piper_sdk
from piper_sdk import *


# Constants
ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_POSITION = 116
LEN_GOAL_POSITION = 4
#! rewrite 
ADDR_PRESENT_POSITION = 132
ADDR_PRESENT_POSITION = 140 #Position Trajectory(140)
LEN_PRESENT_POSITION = 4

TORQUE_ENABLE = 1
TORQUE_DISABLE = 0


class PiperDriverProtocol(Protocol):
    def set_joints(self, joint_angles: Sequence[float]):
        """Set the joint angles for the Dynamixel servos.

        Args:
            joint_angles (Sequence[float]): A list of joint angles.
        """
        ...

    def torque_enabled(self) -> bool:
        """Check if torque is enabled for the Dynamixel servos.

        Returns:
            bool: True if torque is enabled, False if it is disabled.
        """
        ...

    def set_torque_mode(self, enable: bool):
        """Set the torque mode for the Dynamixel servos.

        Args:
            enable (bool): True to enable torque, False to disable.
        """
        ...

    def get_joints(self) -> np.ndarray:
        """Get the current joint angles in radians.

        Returns:
            np.ndarray: An array of joint angles.
        """
        ...

    def close(self):
        """Close the driver."""


# TODO: 还没改完
class FakePiperDriver(PiperDriverProtocol):
    def __init__(self, ids: Sequence[int]):
        self._ids = ids
        self._joint_angles = np.zeros(len(ids), dtype=int)
        self._torque_enabled = False

    def set_joints(self, joint_angles: Sequence[float]):
        if len(joint_angles) != len(self._ids):
            raise ValueError(
                "The length of joint_angles must match the number of servos"
            )
        if not self._torque_enabled:
            raise RuntimeError("Torque must be enabled to set joint angles")
        self._joint_angles = np.array(joint_angles)

    def torque_enabled(self) -> bool:
        return self._torque_enabled

    def set_torque_mode(self, enable: bool):
        self._torque_enabled = enable

    def get_joints(self) -> np.ndarray:
        return self._joint_angles.copy()

    def close(self):
        pass


class PiperDriver(PiperDriverProtocol):
    def __init__(
        self, ids: Sequence[int], port: str = "/dev/ttyUSB0", baudrate: int = 57600
    ):
        """Initialize the DynamixelDriver class.
        通过多线程的方式实现了对多个 Dynamixel 伺服电机的控制通信，包括设置目标角度、读取当前角度以及管理扭矩状态。

        Args:
            ids (Sequence[int]): A list of IDs for the Dynamixel servos.
            port (str): The USB port to connect to the arm.
            baudrate (int): The baudrate for communication.
        """
        self._ids = ids
        self._joint_angles = None
        #! 
        self._lock = Lock()

        self.piper = C_PiperInterface()
        self.piper.ConnectPort() # 连接串口


    def set_joints(self, joint_angles: Sequence[float]):
        """
            发送前需要切换机械臂模式为关节控制模式
        """

        if len(joint_angles) != 6:
            raise ValueError(
                "The length of joint_angles must match the number of servos"
            )
        
        # if not self._torque_enabled:
        #     raise RuntimeError("Torque must be enabled to set joint angles")
        
        # 关节控制模式
        self.piper.MotionCtrl_2(move_mode=0x01, move_spd_rate_ctrl=50)  # 关节运动模式，速度百分比控制

        angle_list = []
        for angle in joint_angles:
            angle_list.append(int(angle * 180 / np.pi * 1000))    # 转成角度

        self.piper.JointCtrl(angle_list[0], angle_list[1], angle_list[2], angle_list[3], angle_list[4], angle_list[5]) # 发送指令


    def set_gripper(self, gripper_angle: float, gripper_effort: float):
        """
            发送前需要切换机械臂模式为关节控制模式
        """
        
        # 关节控制模式
        self.piper.MotionCtrl_2(move_mode=0x01, move_spd_rate_ctrl=50)  # 关节运动模式，速度百分比控制
        gripper_angle = int(gripper_angle * 180 / np.pi * 1000)

        if gripper_angle<=0. or gripper_angle>5.:
            raise ValueError("Gripper angle should be between 0 and 5 degrees")
        
        gripper_effort = int(gripper_effort * 1000)

        self.piper.GripperCtrl(gripper_angle, gripper_effort, gripper_code=0x01)


    # def torque_enabled(self) -> bool:
    #     return self._torque_enabled


    # def set_torque_mode(self, enable: bool):
    #     torque_value = TORQUE_ENABLE if enable else TORQUE_DISABLE
    #     with self._lock:
    #         for dxl_id in self._ids:
    #             dxl_comm_result, dxl_error = self._packetHandler.write1ByteTxRx(
    #                 self._portHandler, dxl_id, ADDR_TORQUE_ENABLE, torque_value
    #             )
    #             if dxl_comm_result != COMM_SUCCESS or dxl_error != 0:
    #                 print(dxl_comm_result)
    #                 print(dxl_error)
    #                 raise RuntimeError(
    #                     f"Failed to set torque mode for Dynamixel with ID {dxl_id}"
    #                 )

    #     self._torque_enabled = enable


    # def _start_reading_thread(self):
    #     self._reading_thread = Thread(target=self._read_joint_angles)



    # def _read_joint_angles(self):
    #     # 连续读取关节角度并更新joint_angles数组
    #     while not self._stop_thread.is_set():
    #         time.sleep(0.001)
    #         with self._lock:
    #             _joint_angles = np.zeros(len(self._ids), dtype=int)
    #             dxl_comm_result = self._groupSyncRead.txRxPacket()
    #             if dxl_comm_result != COMM_SUCCESS:
    #                 print(f"warning, comm failed: {dxl_comm_result}")
    #                 continue
    #             for i, dxl_id in enumerate(self._ids):
    #                 if self._groupSyncRead.isAvailable(
    #                     dxl_id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION
    #                 ):
    #                     angle = self._groupSyncRead.getData(
    #                         dxl_id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION
    #                     )
    #                     angle = np.int32(np.uint32(angle))
    #                     _joint_angles[i] = angle
    #                 else:
    #                     raise RuntimeError(
    #                         f"Failed to get joint angles for Dynamixel with ID {dxl_id}"
    #                     )
    #             self._joint_angles = _joint_angles
    #         # self._groupSyncRead.clearParam() # TODO what does this do? should i add it


    def get_joints(self) -> np.ndarray:
        # TODO: 应该返回一个弧度角 np.ndarray，最后一维是夹爪的角度

        joints = self.piper.GetArmJointMsgs()  # TODO: 这个具体返回什么我还没搞清楚
        gripper = self.piper.GetArmGripperMsgs()  # TODO: 这个具体返回什么我还没搞清楚
        joints = np.append(joints, gripper)

        return joints


    def close(self):
        pass
        # self._stop_thread.set()
        # self._reading_thread.join()
        # self._portHandler.closePort()



if __name__ == "__main__":
    # Set the port, baudrate, and servo IDs
    ids = [1,2]
    # Create a DynamixelDriver instance
    try:
        driver = PiperDriver(ids)
    except FileNotFoundError:
        driver = PiperDriver(ids, port="/dev/cu.usbserial-FT7WBMUB")

    driver.set_torque_mode(True)
    driver.set_torque_mode(False)

    # Print the joint angles
    try:
        while True:
            joint_angles = driver.get_joints()
            print(f"Joint angles for IDs {ids}: {joint_angles}")
            # print(f"Joint angles for IDs {ids[1]}: {joint_angles[1]}")
    except KeyboardInterrupt:
        driver.close()
