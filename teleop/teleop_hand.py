import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # 添加这行来帮助调试 CUDA 错误

from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch

import math
import numpy as np
import torch

from TeleVision import OpenTeleVision
from Preprocessor import VuerPreprocessor
from constants_vuer import tip_indices
from dex_retargeting.retargeting_config import RetargetingConfig
from pytransform3d import rotations

from pathlib import Path
import argparse
import time
import yaml
from multiprocessing import Array, Process, shared_memory, Queue, Manager, Event, Semaphore

import atexit

def cleanup_shared_memory(shm):
    if shm is not None:
        shm.close()
        shm.unlink()

# atexit.register(cleanup_shared_memory, self.shm)


class VuerTeleop:
    def __init__(self, config_file_path, resolution = (720, 1280)):
        self.resolution = resolution
        self.crop_size_w = 0
        self.crop_size_h = 0
        self.resolution_cropped = (self.resolution[0]-self.crop_size_h, self.resolution[1]-2*self.crop_size_w)
        print(self.resolution_cropped)

        self.img_shape = (self.resolution_cropped[0], 2 * self.resolution_cropped[1], 3)
        self.img_height, self.img_width = self.resolution_cropped[:2]

        self.shm = shared_memory.SharedMemory(create=True, size=np.prod(self.img_shape) * np.uint8().itemsize)
        self.img_array = np.ndarray((self.img_shape[0], self.img_shape[1], 3), dtype=np.uint8, buffer=self.shm.buf)
        
        image_queue = Queue()
        toggle_streaming = Event()

        # self.tv = OpenTeleVision(self.resolution_cropped, self.shm.name, image_queue, toggle_streaming)

        self.tv = OpenTeleVision(self.resolution_cropped, self.shm.name, image_queue, toggle_streaming, ngrok=True)  # 从事件中获取图像和姿态位置
        # self.tv = OpenTeleVision(self.resolution_cropped, self.shm.name, image_queue, toggle_streaming,
        #                          cert_file="/home/sense/.local/share/mkcert/web.crt", 
        #                          key_file="/home/sense/.local/share/mkcert/web.key")  # port 8012
        # print("Server started successfully")

        self.processor = VuerPreprocessor()  # 用来求解不同的部位坐标

        # 加载重建模型
        RetargetingConfig.set_default_urdf_dir('../assets')

        with Path(config_file_path).open('r') as f:
            cfg = yaml.safe_load(f)

        left_retargeting_config = RetargetingConfig.from_dict(cfg['left'])
        right_retargeting_config = RetargetingConfig.from_dict(cfg['right'])

        self.left_retargeting = left_retargeting_config.build()
        self.right_retargeting = right_retargeting_config.build()

    def step(self):
        """
        return:
            head_rmat: 头部旋转矩阵
            left_pose: 左手位姿（位置+方向）
            right_pose: 右手位姿（位置+方向）
            left_qpos: 左手关节位置
            right_qpos: 右手关节位置    
        """
        head_mat, left_wrist_mat, right_wrist_mat, left_hand_mat, right_hand_mat = self.processor.process(self.tv)

        head_rmat = head_mat[:3, :3]  # 提取头部姿态

        left_pose = np.concatenate([left_wrist_mat[:3, 3] + np.array([-0.6, 0, 1.6]),
                                    rotations.quaternion_from_matrix(left_wrist_mat[:3, :3])[[1, 2, 3, 0]]])
        right_pose = np.concatenate([right_wrist_mat[:3, 3] + np.array([-0.6, 0, 1.6]),
                                     rotations.quaternion_from_matrix(right_wrist_mat[:3, :3])[[1, 2, 3, 0]]])
        
        left_qpos = self.left_retargeting.retarget(left_hand_mat[tip_indices])[[4, 5, 6, 7, 10, 11, 8, 9, 0, 1, 2, 3]]
        right_qpos = self.right_retargeting.retarget(right_hand_mat[tip_indices])[[4, 5, 6, 7, 10, 11, 8, 9, 0, 1, 2, 3]]

        return head_rmat, left_pose, right_pose, left_qpos, right_qpos
    
    def cleanup(self):
        atexit.register(cleanup_shared_memory, self.shm)
    

class Sim:
    def __init__(self, resolution = (720, 1280),
                 print_freq=False):
        self.print_freq = print_freq
        
        # initialize gym
        self.gym = gymapi.acquire_gym()

        # configure sim
        sim_params = gymapi.SimParams()
        sim_params.dt = 1 / 60
        sim_params.substeps = 2
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.max_gpu_contact_pairs = 8388608
        sim_params.physx.contact_offset = 0.002
        sim_params.physx.friction_offset_threshold = 0.001
        sim_params.physx.friction_correlation_distance = 0.0005
        sim_params.physx.rest_offset = 0.0
        sim_params.physx.use_gpu = True
        sim_params.use_gpu_pipeline = False  # 启用 GPU 管道

        self.device = torch.device("cuda" if sim_params.use_gpu_pipeline else "cpu")
        print(self.device, torch.cuda.is_available())

        self.sim = self.gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
        if self.sim is None:
            raise RuntimeError("Failed to create sim")

        plane_params = gymapi.PlaneParams()
        plane_params.distance = 0.0
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

        # load table asset
        table_asset_options = gymapi.AssetOptions()
        table_asset_options.disable_gravity = True
        table_asset_options.fix_base_link = True
        table_asset = self.gym.create_box(self.sim, 0.8, 0.8, 0.1, table_asset_options)

        # load cube asset
        cube_asset_options = gymapi.AssetOptions()
        cube_asset_options.density = 10
        cube_asset = self.gym.create_box(self.sim, 0.05, 0.05, 0.05, cube_asset_options)

        asset_root = "../assets"
        left_asset_path = "inspire_hand/inspire_hand_left.urdf"
        right_asset_path = "inspire_hand/inspire_hand_right.urdf"
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        left_asset = self.gym.load_asset(self.sim, asset_root, left_asset_path, asset_options)
        right_asset = self.gym.load_asset(self.sim, asset_root, right_asset_path, asset_options)
        self.dof = self.gym.get_asset_dof_count(left_asset)

        # set up the env grid
        num_envs = 1
        num_per_row = int(math.sqrt(num_envs))
        env_spacing = 1.25
        env_lower = gymapi.Vec3(-env_spacing, 0.0, -env_spacing)
        env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)
        np.random.seed(0)
        self.env = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)

        # table
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0, 0, 1.2)
        pose.r = gymapi.Quat(0, 0, 0, 1)
        table_handle = self.gym.create_actor(self.env, table_asset, pose, 'table', 0)
        color = gymapi.Vec3(0.5, 0.5, 0.5)
        self.gym.set_rigid_body_color(self.env, table_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)

        # cube
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0, 0, 1.25)
        pose.r = gymapi.Quat(0, 0, 0, 1)
        cube_handle = self.gym.create_actor(self.env, cube_asset, pose, 'cube', 0)
        color = gymapi.Vec3(1, 0.5, 0.5)
        self.gym.set_rigid_body_color(self.env, cube_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)

        # left_hand
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(-0.6, 0, 1.6)
        pose.r = gymapi.Quat(0, 0, 0, 1)
        self.left_handle = self.gym.create_actor(self.env, left_asset, pose, 'left', 1, 1)
        self.gym.set_actor_dof_states(self.env, self.left_handle, np.zeros(self.dof, gymapi.DofState.dtype),
                                      gymapi.STATE_ALL)
        left_idx = self.gym.get_actor_index(self.env, self.left_handle, gymapi.DOMAIN_SIM)

        # right_hand
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(-0.6, 0, 1.6)
        pose.r = gymapi.Quat(0, 0, 0, 1)
        self.right_handle = self.gym.create_actor(self.env, right_asset, pose, 'right', 1, 1)
        self.gym.set_actor_dof_states(self.env, self.right_handle, np.zeros(self.dof, gymapi.DofState.dtype),
                                      gymapi.STATE_ALL)
        right_idx = self.gym.get_actor_index(self.env, self.right_handle, gymapi.DOMAIN_SIM)


        self.root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.root_states = gymtorch.wrap_tensor(self.root_state_tensor)
        self.left_root_states = self.root_states[left_idx]
        self.right_root_states = self.root_states[right_idx]

        # 确保张量在正确的设备上
        if sim_params.use_gpu_pipeline:
            self.root_states = self.root_states.cuda()
            self.left_root_states = self.left_root_states.cuda()
            self.right_root_states = self.right_root_states.cuda()

        # create default viewer
        self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        if self.viewer is None:
            print("*** Failed to create viewer")
            quit()
        cam_pos = gymapi.Vec3(1, 1, 2)
        cam_target = gymapi.Vec3(0, 0, 1)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        self.cam_lookat_offset = np.array([1, 0, 0])
        self.left_cam_offset = np.array([0, 0.033, 0])
        self.right_cam_offset = np.array([0, -0.033, 0])
        self.cam_pos = np.array([-0.6, 0, 1.6])

        # create left 1st preson viewer
        camera_props = gymapi.CameraProperties()
        camera_props.width = resolution[1]
        camera_props.height = resolution[0]
        self.left_camera_handle = self.gym.create_camera_sensor(self.env, camera_props)
        self.gym.set_camera_location(self.left_camera_handle,
                                     self.env,
                                     gymapi.Vec3(*(self.cam_pos + self.left_cam_offset)),
                                     gymapi.Vec3(*(self.cam_pos + self.left_cam_offset + self.cam_lookat_offset)))

        # create right 1st preson viewer
        camera_props = gymapi.CameraProperties()
        camera_props.width = resolution[1]
        camera_props.height = resolution[0]
        self.right_camera_handle = self.gym.create_camera_sensor(self.env, camera_props)
        self.gym.set_camera_location(self.right_camera_handle,
                                     self.env,
                                     gymapi.Vec3(*(self.cam_pos + self.right_cam_offset)),
                                     gymapi.Vec3(*(self.cam_pos + self.right_cam_offset + self.cam_lookat_offset)))

    def step(self, head_rmat, left_pose, right_pose, left_qpos, right_qpos):
        if self.print_freq:
            start = time.time()

        try:
            # 1. 更新手部状态
            left_pose_tensor = torch.tensor(left_pose, dtype=torch.float32, device=self.device)
            right_pose_tensor = torch.tensor(right_pose, dtype=torch.float32, device=self.device)
            
            self.left_root_states[0:7] = left_pose_tensor
            self.right_root_states[0:7] = right_pose_tensor
            
            # 确保数据同步
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

            # 2. 更新关节角度
            left_states = np.zeros(self.dof, dtype=gymapi.DofState.dtype)
            left_states['pos'] = left_qpos
            self.gym.set_actor_dof_states(self.env, self.left_handle, left_states, gymapi.STATE_POS)

            right_states = np.zeros(self.dof, dtype=gymapi.DofState.dtype)
            right_states['pos'] = right_qpos
            self.gym.set_actor_dof_states(self.env, self.right_handle, right_states, gymapi.STATE_POS)

            # 3. 物理模拟
            self.gym.simulate(self.sim)
            
            # 4. 等待物理模拟完成
            self.gym.fetch_results(self.sim, True)
            
            # 5. 更新图形
            self.gym.step_graphics(self.sim)
            
            # 6. 渲染相机
            self.gym.render_all_camera_sensors(self.sim)

            curr_lookat_offset = self.cam_lookat_offset @ head_rmat.T
            curr_left_offset = self.left_cam_offset @ head_rmat.T
            curr_right_offset = self.right_cam_offset @ head_rmat.T

            self.gym.set_camera_location(self.left_camera_handle,
                                         self.env,
                                         gymapi.Vec3(*(self.cam_pos + curr_left_offset)),
                                         gymapi.Vec3(*(self.cam_pos + curr_left_offset + curr_lookat_offset)))
            self.gym.set_camera_location(self.right_camera_handle,
                                         self.env,
                                         gymapi.Vec3(*(self.cam_pos + curr_right_offset)),
                                         gymapi.Vec3(*(self.cam_pos + curr_right_offset + curr_lookat_offset)))

            # 8. 获取相机图像
            left_image = self.gym.get_camera_image(self.sim, self.env, self.left_camera_handle, gymapi.IMAGE_COLOR)
            right_image = self.gym.get_camera_image(self.sim, self.env, self.right_camera_handle, gymapi.IMAGE_COLOR)
            
            # 9. 重塑图像
            left_image = left_image.reshape(left_image.shape[0], -1, 4)[..., :3]
            right_image = right_image.reshape(right_image.shape[0], -1, 4)[..., :3]

            # 10. 更新视图
            self.gym.draw_viewer(self.viewer, self.sim, True)
            
            # 11. 同步帧时间
            self.gym.sync_frame_time(self.sim)
            
            # 12. 刷新状态张量
            self.gym.refresh_actor_root_state_tensor(self.sim)

            if self.print_freq:
                end = time.time()
                print('Frequency:', 1 / (end - start))

            return left_image, right_image

        except Exception as e:
            print(f"Error in step: {str(e)}")
            raise

    def cleanup(self):
        try:
            if hasattr(self, 'viewer') and self.viewer is not None:
                self.gym.destroy_viewer(self.viewer)
                self.viewer = None

            if hasattr(self, 'sim') and self.sim is not None:
                self.gym.destroy_sim(self.sim)
                self.sim = None

            if hasattr(self, 'gym'):
                self.gym = None

            # 清理 CUDA 缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"Error during cleanup: {str(e)}")

    def __del__(self):
        self.cleanup()

    def end(self):
        self.cleanup()


if __name__ == '__main__':
    # resolution=(640, 960)
    resolution=(720, 1280)
    teleoperator = None
    simulator = None

    try:
        # 清理 CUDA 缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        teleoperator = VuerTeleop('inspire_hand.yml', resolution=resolution)
        simulator = Sim(resolution=resolution, print_freq=False)

        f=0
        while True:
            start_step = time.time()
            head_rmat, left_pose, right_pose, left_qpos, right_qpos = teleoperator.step()  # 从事件中获取头部姿态和手部姿态
            left_img, right_img = simulator.step(head_rmat, left_pose, right_pose, left_qpos, right_qpos)  # 更新模拟器状态并渲染图像
            np.copyto(teleoperator.img_array, np.hstack((left_img, right_img)))

            if f == 0:
                print(left_img.shape)
            if f%10 == 0:
                print('step fre:', 1 / (time.time() - start_step))
            f=f+1

    except KeyboardInterrupt:
        print("\nReceived keyboard interrupt, cleaning up...")
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
    finally:
        if simulator is not None:
            simulator.cleanup()
        if teleoperator is not None:
            teleoperator.cleanup()
        # 最后清理 CUDA 缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        exit(0)
