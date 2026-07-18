# 共享人脸检测模块 / Shared face detection module
# 移植自亚博官方例程 07.Face/03.face_pose.py 中的 FaceDetApp，类结构与逻辑保持一致
# Ported from Yahboom example 07.Face/03.face_pose.py (FaceDetApp), structure unchanged.
# 全工程只实例化一次，检测结果分发给朝向/表情/远近等所有模块
# Instantiated once for the whole project; results are shared by pose/expression/proximity modules.

from libs.PipeLine import ScopedTiming
from libs.AIBase import AIBase
from libs.AI2D import Ai2d
from media.media import *
import nncase_runtime as nn
import ulab.numpy as np
import aidemo


class FaceDetApp(AIBase):
    """
    人脸检测应用类，继承自AIBase基类
    (Face detection application class, inherits from AIBase)
    """
    def __init__(self, kmodel_path, model_input_size, anchors, confidence_threshold=0.5, nms_threshold=0.2, rgb888p_size=[640,480], display_size=[640,480], debug_mode=0):
        """
        初始化人脸检测应用类
        (Initialize the face detection application class)

        参数:
        kmodel_path: 模型文件路径 / Path to the kmodel file
        model_input_size: 模型输入尺寸 [width, height] / Model input dimensions
        anchors: 预定义的锚框 / Predefined anchor boxes
        confidence_threshold: 置信度阈值 / Confidence threshold
        nms_threshold: 非极大值抑制阈值 / NMS threshold
        rgb888p_size: 传感器输入图像尺寸 / Sensor input image dimensions
        display_size: 显示输出尺寸 / Display output dimensions
        debug_mode: 调试模式级别 / Debug mode level
        """
        super().__init__(kmodel_path, model_input_size, rgb888p_size, debug_mode)
        self.kmodel_path = kmodel_path
        self.model_input_size = model_input_size
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.anchors = anchors
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

    # 配置预处理操作，这里使用了pad和resize
    # (Configure preprocessing operations, using pad and resize here)
    def config_preprocess(self, input_image_size=None):
        with ScopedTiming("set preprocess config", self.debug_mode > 0):
            ai2d_input_size = input_image_size if input_image_size else self.rgb888p_size
            # 计算padding参数，并设置padding预处理 / Calculate and set padding
            self.ai2d.pad(self.get_pad_param(), 0, [104, 117, 123])
            self.ai2d.resize(nn.interp_method.tf_bilinear, nn.interp_mode.half_pixel)
            self.ai2d.build([1, 3, ai2d_input_size[1], ai2d_input_size[0]], [1, 3, self.model_input_size[1], self.model_input_size[0]])

    # 自定义后处理，results是模型输出的array列表，这里使用了aidemo库的face_det_post_process接口
    # (Custom post-processing using aidemo.face_det_post_process)
    def postprocess(self, results):
        with ScopedTiming("postprocess", self.debug_mode > 0):
            res = aidemo.face_det_post_process(self.confidence_threshold, self.nms_threshold, self.model_input_size[0], self.anchors, self.rgb888p_size, results)
            if len(res) == 0:
                return res
            else:
                return res[0]

    # 计算padding参数 / Calculate padding parameters
    def get_pad_param(self):
        dst_w = self.model_input_size[0]
        dst_h = self.model_input_size[1]
        # 计算最小的缩放比例，等比例缩放 / Minimum ratio for proportional scaling
        ratio_w = dst_w / self.rgb888p_size[0]
        ratio_h = dst_h / self.rgb888p_size[1]
        if ratio_w < ratio_h:
            ratio = ratio_w
        else:
            ratio = ratio_h
        new_w = (int)(ratio * self.rgb888p_size[0])
        new_h = (int)(ratio * self.rgb888p_size[1])
        dw = (dst_w - new_w) / 2
        dh = (dst_h - new_h) / 2
        top = (int)(round(0))
        bottom = (int)(round(dh * 2 + 0.1))
        left = (int)(round(0))
        right = (int)(round(dw * 2 - 0.1))
        return [0, 0, 0, 0, top, bottom, left, right]
