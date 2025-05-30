# dabai_image_viewer.py
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import os
import threading

"""
    读取dabai_dc1相机的图像并显示
"""

class AstraImageViewer:
    def __init__(self, image_type="rgb"):
        self.bridge = CvBridge()
        self.image_type = image_type.lower()
        self.window_name = f"Astra {image_type.upper()} Viewer"

        self.latest_cv_image = None  # 用于存储最新的图像
        self.image_lock = threading.Lock() # 用于线程安全的锁
        
        # 根据类型设置话题前缀
        self.topic_prefix = rospy.get_param("~camera_name", "camera")
        self.topic_map = {
            "rgb": f"{self.topic_prefix}/color/image_raw",
            "depth": f"{self.topic_prefix}/depth/image_raw",
            "ir": f"{self.topic_prefix}/ir/image_raw"
        }
        
        # 订阅话题
        rospy.Subscriber(
            self.topic_map[self.image_type],
            Image,
            self.image_callback,
            queue_size=10
        )
        rospy.loginfo(f"Subscribed to: {self.topic_map[self.image_type]}")

    def image_callback(self, msg):
        # rospy.loginfo("Image callback triggered.") # [诊断] 确认回调是否被调用
        rospy.loginfo(f"Message encoding: {msg.encoding}, Height: {msg.height}, Width: {msg.width}") # [诊断] 打印消息编码和尺寸

        try:
            cv_image_temp = None # 初始化 cv_image
            if self.image_type == "rgb":
                rospy.loginfo("Attempting to convert to bgr8.")
                cv_image_temp = self.bridge.imgmsg_to_cv2(msg, "bgr8")
                rospy.loginfo(f"Image type: {cv_image_temp.dtype}, shape: {cv_image_temp.shape}")

            elif self.image_type == "depth":
                try:
                    # 尝试获取16位深度图
                    cv_image_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough") # 或者 "16UC1"
                    if cv_image_depth.dtype == 'uint16':
                        # 示例：归一化到0-255以便显示 (需要根据实际深度范围调整)
                        image_min = np.min(cv_image_depth)
                        image_max = np.max(cv_image_depth)       
                        cv_image_display = ((cv_image_depth.astype(float) - image_min) / (image_max - image_min) * 255.0).astype('uint8')
                    
                    else: # 如果不是预期的深度格式，退回到 mono8
                        rospy.logwarn_once(f"Depth image format is {cv_image_depth.dtype}, falling back to mono8 conversion.")
                        cv_image_display = self.bridge.imgmsg_to_cv2(msg, "mono8")

                    cv_image_temp = cv_image_display # 用于显示的图像
                    rospy.loginfo(f"Image type: {cv_image_temp.dtype}, shape: {cv_image_temp.shape}")
                except CvBridgeError as e:
                    rospy.logerr(f"CV Bridge Error for depth: {e}")
                    return # 出现错误则不继续处理
            else: # ir
                rospy.loginfo(f"Attempting to convert to mono8 for type: {self.image_type}.")
                cv_image_temp = self.bridge.imgmsg_to_cv2(msg, "mono8")

            # 如果图像成功处理，则更新latest_cv_image
            if cv_image_temp is not None:
                with self.image_lock: # 获取锁以安全地更新共享数据
                    self.latest_cv_image = cv_image_temp.copy() # 使用copy()以避免后续修改影响存储的图像
                rospy.loginfo("Processed image updated.")
            
        except CvBridgeError as e:
            rospy.logerr(f"CV Bridge Error in image_callback: {e}")
        except Exception as e: # 捕获其他任何可能的异常
            rospy.logerr(f"Unexpected error in image_callback: {e}")

        # return cv_image # 返回图像

    def get_latest_image(self):
        """
        获取最新处理的图像帧。
        返回:
            numpy.ndarray: 最新的图像帧，如果还没有图像则返回None。
        """
        with self.image_lock: # 获取锁以安全地读取共享数据
            if self.latest_cv_image is not None:
                return self.latest_cv_image.copy() # 返回副本以防止外部修改
            return None

    def run(self):
        rospy.spin()  # 阻塞等待消息到达


if __name__ == "__main__":
    import cv2

    rospy.init_node("astra_image_viewer", anonymous=True)
    viewer = AstraImageViewer(
        image_type=rospy.get_param("~image_type", "rgb")
        # image_type=rospy.get_param("~image_type", "depth")
        # image_type=rospy.get_param("~image_type", "ir")
    )

    spin_thread = threading.Thread(target=viewer.run) # 或者直接 target=rospy.spin
    spin_thread.daemon = True # 设置为守护线程，主线程退出时它也退出
    spin_thread.start()

    rate = rospy.Rate(30) # 设置30Hz的速率
    try:
        while not rospy.is_shutdown():
            latest_frame = viewer.get_latest_image()            

            if latest_frame is not None:
                rospy.loginfo(f"Main loop: Got an image of shape {latest_frame.shape} and type {latest_frame.dtype}")
                # cv2.imwrite(f'/home/eigindustry/workspace/TeleVision/saveimg/{i}_rgb.png', latest_frame)
            else:
                rospy.loginfo("Main loop: No image available yet.")
            
            rate.sleep()
    except rospy.ROSInterruptException:
        rospy.loginfo("ROS shutdown request received.")

    # viewer.run()