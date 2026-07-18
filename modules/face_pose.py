# 人脸朝向识别模块 / Face pose (head orientation) estimation
# 模型部分移植自亚博官方例程 07.Face/03.face_pose.py 的 FacePoseApp，逻辑保持一致
# Model part ported from Yahboom example 07.Face/03.face_pose.py (FacePoseApp).
# 与官方差异：检测由全工程共享，本模块只做"检测框 → 欧拉角 → 方向分类"
# Difference: detection is shared project-wide; this module only does box -> euler -> direction.

from libs.PipeLine import ScopedTiming
from libs.AIBase import AIBase
from libs.AI2D import Ai2d
from media.media import *
import nncase_runtime as nn
import ulab.numpy as np

import config


class FacePoseApp(AIBase):
    """
    人脸姿态估计应用类，继承自AIBase基类
    (Face pose estimation application class, inherits from AIBase)
    """
    def __init__(self, kmodel_path, model_input_size, rgb888p_size=[640, 480], display_size=[640, 480], debug_mode=0):
        """
        初始化人脸姿态估计应用类
        (Initialize the face pose estimation application class)

        参数:
        kmodel_path: 模型文件路径 / Path to the kmodel file
        model_input_size: 模型输入尺寸 [width, height] / Model input dimensions
        rgb888p_size: 传感器输入图像尺寸 / Sensor input image dimensions
        display_size: 显示输出尺寸 / Display output dimensions
        debug_mode: 调试模式级别 / Debug mode level
        """
        super().__init__(kmodel_path, model_input_size, rgb888p_size, debug_mode)
        self.kmodel_path = kmodel_path
        # 人脸姿态模型输入分辨率 / Face pose model input resolution
        self.model_input_size = model_input_size
        # sensor给到AI的图像分辨率，宽16字节对齐
        # (Image resolution from sensor to AI, width aligned to 16 bytes)
        self.rgb888p_size = [ALIGN_UP(rgb888p_size[0], 16), rgb888p_size[1]]
        # 视频输出VO分辨率，宽16字节对齐
        # (Video output resolution, width aligned to 16 bytes)
        self.display_size = [ALIGN_UP(display_size[0], 16), display_size[1]]
        self.debug_mode = debug_mode
        # 实例化Ai2d，用于实现模型预处理
        # (Instantiate Ai2d for model preprocessing)
        self.ai2d = Ai2d(debug_mode)
        self.ai2d.set_ai2d_dtype(nn.ai2d_format.NCHW_FMT, nn.ai2d_format.NCHW_FMT, np.uint8, np.uint8)

    # 配置预处理操作，这里使用了affine，按共享检测结果的人脸框取图
    # (Configure preprocessing with affine, cropping the face from the shared detection box)
    def config_preprocess(self, det, input_image_size=None):
        with ScopedTiming("set preprocess config", self.debug_mode > 0):
            ai2d_input_size = input_image_size if input_image_size else self.rgb888p_size
            # 计算affine矩阵并设置affine预处理 / Calculate affine matrix and set affine
            matrix_dst = self.get_affine_matrix(det)
            self.ai2d.affine(nn.interp_method.cv2_bilinear, 0, 0, 127, 1, matrix_dst)
            self.ai2d.build([1, 3, ai2d_input_size[1], ai2d_input_size[0]], [1, 3, self.model_input_size[1], self.model_input_size[0]])

    # 自定义后处理，计算旋转矩阵和欧拉角
    # (Custom post-processing: rotation matrix and Euler angles)
    def postprocess(self, results):
        with ScopedTiming("postprocess", self.debug_mode > 0):
            R, eular = self.get_euler(results[0][0])
            return R, eular

    def get_affine_matrix(self, bbox):
        """
        获取仿射变换矩阵，用于将人脸区域变换到模型输入空间
        (Get the affine transformation matrix to transform the face region to model input space)
        """
        with ScopedTiming("get_affine_matrix", self.debug_mode > 1):
            factor = 2.7
            x1, y1, w, h = map(lambda x: int(round(x, 0)), bbox[:4])
            edge_size = self.model_input_size[1]
            trans_distance = edge_size / 2.0
            center_x = x1 + w / 2.0
            center_y = y1 + h / 2.0
            maximum_edge = factor * (h if h > w else w)
            scale = edge_size * 2.0 / maximum_edge
            cx = trans_distance - scale * center_x
            cy = trans_distance - scale * center_y
            affine_matrix = [scale, 0, cx, 0, scale, cy]
            return affine_matrix

    def rotation_matrix_to_euler_angles(self, R):
        """
        将旋转矩阵转换为欧拉角 [pitch, yaw, roll]，单位为度
        (Convert rotation matrix to Euler angles [pitch, yaw, roll] in degrees)
        """
        sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
        if sy < 1e-6:
            pitch = np.arctan2(-R[1, 2], R[1, 1]) * 180 / np.pi
            yaw = np.arctan2(-R[2, 0], sy) * 180 / np.pi
            roll = 0
        else:
            pitch = np.arctan2(R[2, 1], R[2, 2]) * 180 / np.pi
            yaw = np.arctan2(-R[2, 0], sy) * 180 / np.pi
            roll = np.arctan2(R[1, 0], R[0, 0]) * 180 / np.pi
        return [pitch, yaw, roll]

    def get_euler(self, data):
        """从模型输出数据中获取旋转矩阵和欧拉角 / Rotation matrix + Euler angles from model output."""
        R = data[:3, :3].copy()
        eular = self.rotation_matrix_to_euler_angles(R)
        return R, eular


class PoseDirection:
    """欧拉角 → 方向分类（center/up/down/left/right），双阈值迟滞防抖
    Euler angles -> direction label with dual-threshold hysteresis.

    角度符号约定（按官方模型输出，若真机方向反了在 config 里调阈值或交换标签即可）：
    yaw   > 0 → 头向右 / head right；yaw < 0 → 头向左 / head left
    pitch > 0 → 头向下 / head down；pitch < 0 → 头向上 / head up
    """

    def __init__(self):
        self.dir = "center"

    def reset(self):
        self.dir = "center"

    def update(self, pitch, yaw):
        enter = config.POSE_DIR_ENTER
        exit_ = config.POSE_DIR_EXIT
        if self.dir == "center":
            # 出死区需要越过进入阈值 / Leaving the dead zone requires the enter threshold
            if abs(yaw) >= enter or abs(pitch) >= enter:
                self.dir = self._dominant(pitch, yaw)
        else:
            # 回死区只需低于退出阈值（形成迟滞）/ Returning to center uses the exit threshold
            if abs(yaw) < exit_ and abs(pitch) < exit_:
                self.dir = "center"
            else:
                self.dir = self._dominant(pitch, yaw)
        return self.dir

    def _dominant(self, pitch, yaw):
        """按绝对值较大的轴定主方向 / Dominant axis decides the direction."""
        if abs(yaw) >= abs(pitch):
            return "right" if yaw > 0 else "left"
        return "down" if pitch > 0 else "up"
