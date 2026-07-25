# 手势识别模块 / Hand gesture recognition
# 移植自亚博官方例程 08.Body/07.hand_keypoint_class.py，类结构与判定逻辑保持一致
# Ported from Yahboom example 08.Body/07.hand_keypoint_class.py.
# 链路：hand_det 检测手掌 → handkp_det 出 21 关键点 → 指关节角度几何分类（无额外分类模型）
# Pipeline: hand_det -> handkp_det (21 keypoints) -> finger-angle geometric classification.

from libs.PipeLine import ScopedTiming
from libs.AIBase import AIBase
from libs.AI2D import Ai2d
from media.media import *
import nncase_runtime as nn
import ulab.numpy as np
import aicube

import config


class HandDetApp(AIBase):
    """手掌检测应用类 / Hand detection application class."""

    def __init__(self, kmodel_path, labels, model_input_size, anchors, confidence_threshold=0.2, nms_threshold=0.5, nms_option=False, strides=[8, 16, 32], rgb888p_size=[640, 480], display_size=[640, 480], debug_mode=0):
        super().__init__(kmodel_path, model_input_size, rgb888p_size, debug_mode)
        self.kmodel_path = kmodel_path
        self.labels = labels
        self.model_input_size = model_input_size
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.anchors = anchors
        self.strides = strides
        self.nms_option = nms_option
        self.rgb888p_size = [ALIGN_UP(rgb888p_size[0], 16), rgb888p_size[1]]
        self.display_size = [ALIGN_UP(display_size[0], 16), display_size[1]]
        self.debug_mode = debug_mode
        self.ai2d = Ai2d(debug_mode)
        self.ai2d.set_ai2d_dtype(nn.ai2d_format.NCHW_FMT, nn.ai2d_format.NCHW_FMT, np.uint8, np.uint8)

    # 配置预处理：pad + resize / Configure preprocessing: pad + resize
    def config_preprocess(self, input_image_size=None):
        with ScopedTiming("set preprocess config", self.debug_mode > 0):
            ai2d_input_size = input_image_size if input_image_size else self.rgb888p_size
            top, bottom, left, right = self.get_padding_param()
            self.ai2d.pad([0, 0, 0, 0, top, bottom, left, right], 0, [114, 114, 114])
            self.ai2d.resize(nn.interp_method.tf_bilinear, nn.interp_mode.half_pixel)
            self.ai2d.build([1, 3, ai2d_input_size[1], ai2d_input_size[0]], [1, 3, self.model_input_size[1], self.model_input_size[0]])

    # 后处理：aicube anchorbasedet / Postprocess via aicube anchor-based detection
    def postprocess(self, results):
        with ScopedTiming("postprocess", self.debug_mode > 0):
            dets = aicube.anchorbasedet_post_process(results[0], results[1], results[2], self.model_input_size, self.rgb888p_size, self.strides, len(self.labels), self.confidence_threshold, self.nms_threshold, self.anchors, self.nms_option)
            return dets

    def get_padding_param(self):
        dst_w = self.model_input_size[0]
        dst_h = self.model_input_size[1]
        ratio_w = dst_w / self.rgb888p_size[0]
        ratio_h = dst_h / self.rgb888p_size[1]
        if ratio_w < ratio_h:
            ratio = ratio_w
        else:
            ratio = ratio_h
        new_w = int(ratio * self.rgb888p_size[0])
        new_h = int(ratio * self.rgb888p_size[1])
        dw = (dst_w - new_w) / 2
        dh = (dst_h - new_h) / 2
        top = int(round(dh - 0.1))
        bottom = int(round(dh + 0.1))
        left = int(round(dw - 0.1))
        right = int(round(dw + 0.1))
        return top, bottom, left, right


class HandKPClassApp(AIBase):
    """手掌关键点 + 几何分类应用类 / Hand keypoint + geometric classification."""

    def __init__(self, kmodel_path, model_input_size, rgb888p_size=[640, 480], display_size=[640, 480], debug_mode=0):
        super().__init__(kmodel_path, model_input_size, rgb888p_size, debug_mode)
        self.kmodel_path = kmodel_path
        self.model_input_size = model_input_size
        self.rgb888p_size = [ALIGN_UP(rgb888p_size[0], 16), rgb888p_size[1]]
        self.display_size = [ALIGN_UP(display_size[0], 16), display_size[1]]
        self.crop_params = []
        self.debug_mode = debug_mode
        self.ai2d = Ai2d(debug_mode)
        self.ai2d.set_ai2d_dtype(nn.ai2d_format.NCHW_FMT, nn.ai2d_format.NCHW_FMT, np.uint8, np.uint8)

    # 配置预处理：按手掌框 crop + resize / Configure preprocessing: crop by hand box + resize
    def config_preprocess(self, det, input_image_size=None):
        with ScopedTiming("set preprocess config", self.debug_mode > 0):
            ai2d_input_size = input_image_size if input_image_size else self.rgb888p_size
            self.crop_params = self.get_crop_param(det)
            self.ai2d.crop(self.crop_params[0], self.crop_params[1], self.crop_params[2], self.crop_params[3])
            self.ai2d.resize(nn.interp_method.tf_bilinear, nn.interp_mode.half_pixel)
            self.ai2d.build([1, 3, ai2d_input_size[1], ai2d_input_size[0]], [1, 3, self.model_input_size[1], self.model_input_size[0]])

    # 后处理：关键点映射回原图并做几何分类 / Map keypoints back and classify
    def postprocess(self, results):
        with ScopedTiming("postprocess", self.debug_mode > 0):
            results = results[0].reshape(results[0].shape[0] * results[0].shape[1])
            results_show = np.zeros(results.shape, dtype=np.int16)
            results_show[0::2] = results[0::2] * self.crop_params[3] + self.crop_params[0]
            results_show[1::2] = results[1::2] * self.crop_params[2] + self.crop_params[1]
            gesture = self.hk_gesture(results_show)
            return results_show, gesture

    def get_crop_param(self, det_box):
        x1, y1, x2, y2 = det_box[2], det_box[3], det_box[4], det_box[5]
        w, h = int(x2 - x1), int(y2 - y1)
        length = max(w, h) / 2
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        ratio_num = 1.26 * length
        x1_kp = int(max(0, cx - ratio_num))
        y1_kp = int(max(0, cy - ratio_num))
        x2_kp = int(min(self.rgb888p_size[0] - 1, cx + ratio_num))
        y2_kp = int(min(self.rgb888p_size[1] - 1, cy + ratio_num))
        w_kp = int(x2_kp - x1_kp + 1)
        h_kp = int(y2_kp - y1_kp + 1)
        return [x1_kp, y1_kp, w_kp, h_kp]

    # 求两个vector之间的夹角 / Angle between two 2D vectors
    def hk_vector_2d_angle(self, v1, v2):
        with ScopedTiming("hk_vector_2d_angle", self.debug_mode > 0):
            try:
                v1_x, v1_y, v2_x, v2_y = v1[0], v1[1], v2[0], v2[1]
                v1_norm = np.sqrt(v1_x * v1_x + v1_y * v1_y)
                v2_norm = np.sqrt(v2_x * v2_x + v2_y * v2_y)
                dot_product = v1_x * v2_x + v1_y * v2_y
                cos_angle = dot_product / (v1_norm * v2_norm)
                angle = np.acos(cos_angle) * 180 / np.pi
                return angle
            except Exception as e:
                return 0

    # 根据关键点判断手势类别（指关节角度规则）
    # Gesture classification by finger joint angles
    def hk_gesture(self, results):
        with ScopedTiming("hk_gesture", self.debug_mode > 0):
            angle_list = []
            for i in range(5):
                angle = self.hk_vector_2d_angle([(results[0] - results[i * 8 + 4]), (results[1] - results[i * 8 + 5])], [(results[i * 8 + 6] - results[i * 8 + 8]), (results[i * 8 + 7] - results[i * 8 + 9])])
                angle_list.append(angle)
            thr_angle, thr_angle_thumb, thr_angle_s, gesture_str = 65., 53., 49., None
            if 65535. not in angle_list:
                # 拳头 fist
                if (angle_list[0] > thr_angle_thumb) and (angle_list[1] > thr_angle) and (angle_list[2] > thr_angle) and (angle_list[3] > thr_angle) and (angle_list[4] > thr_angle):
                    gesture_str = "fist"
                # 五指张开 five
                elif (angle_list[0] < thr_angle_s) and (angle_list[1] < thr_angle_s) and (angle_list[2] < thr_angle_s) and (angle_list[3] < thr_angle_s) and (angle_list[4] < thr_angle_s):
                    gesture_str = "five"
                # 手枪 gun
                elif (angle_list[0] < thr_angle_s) and (angle_list[1] < thr_angle_s) and (angle_list[2] > thr_angle) and (angle_list[3] > thr_angle) and (angle_list[4] > thr_angle):
                    gesture_str = "gun"
                # 爱心 love
                elif (angle_list[0] < thr_angle_s) and (angle_list[1] < thr_angle_s) and (angle_list[2] > thr_angle) and (angle_list[3] > thr_angle) and (angle_list[4] < thr_angle_s):
                    gesture_str = "love"
                # 数字一 one
                elif (angle_list[0] > 5) and (angle_list[1] < thr_angle_s) and (angle_list[2] > thr_angle) and (angle_list[3] > thr_angle) and (angle_list[4] > thr_angle):
                    gesture_str = "one"
                # 数字六 six
                elif (angle_list[0] < thr_angle_s) and (angle_list[1] > thr_angle) and (angle_list[2] > thr_angle) and (angle_list[3] > thr_angle) and (angle_list[4] < thr_angle_s):
                    gesture_str = "six"
                # 数字三 three
                elif (angle_list[0] > thr_angle_thumb) and (angle_list[1] < thr_angle_s) and (angle_list[2] < thr_angle_s) and (angle_list[3] < thr_angle_s) and (angle_list[4] > thr_angle):
                    gesture_str = "three"
                # 竖大拇指 thumbUp
                elif (angle_list[0] < thr_angle_s) and (angle_list[1] > thr_angle) and (angle_list[2] > thr_angle) and (angle_list[3] > thr_angle) and (angle_list[4] > thr_angle):
                    gesture_str = "thumbUp"
                # 耶 yeah
                elif (angle_list[0] > thr_angle_thumb) and (angle_list[1] < thr_angle_s) and (angle_list[2] < thr_angle_s) and (angle_list[3] > thr_angle) and (angle_list[4] > thr_angle):
                    gesture_str = "yeah"
            return gesture_str


def _to_int(v):
    try:
        return int(round(float(v)))
    except Exception:
        return 0


def _hand_box_json(det_box):
    """检测框 [?, ?, x1, y1, x2, y2, ...] 转 {x,y,w,h} / Convert det box to json dict."""
    x1, y1, x2, y2 = det_box[2], det_box[3], det_box[4], det_box[5]
    return {
        "x": _to_int(x1),
        "y": _to_int(y1),
        "w": _to_int(float(x2) - float(x1)),
        "h": _to_int(float(y2) - float(y1)),
    }


class GestureModule:
    """手势识别模块：检测 + 关键点分类 + N 次确认防抖
    Gesture module: detection + keypoint classification + N-run confirmation.

    每次 run 返回已确认结果 {"label":..., "box":{...}} 或 None；
    同一手势连续 GESTURE_CONFIRM_FRAMES 次出现才切换输出，防抖动。
    """

    def __init__(self, rgb888p_size, display_size, debug_mode=0):
        self.rgb888p_size = [ALIGN_UP(rgb888p_size[0], 16), rgb888p_size[1]]
        self.display_size = [ALIGN_UP(display_size[0], 16), display_size[1]]
        self.hand_det = HandDetApp(config.HAND_DET_KMODEL, config.HAND_LABELS,
                                   model_input_size=config.HAND_DET_INPUT_SIZE,
                                   anchors=config.HAND_ANCHORS,
                                   confidence_threshold=config.HAND_CONF_THRESHOLD,
                                   nms_threshold=config.HAND_NMS_THRESHOLD,
                                   nms_option=False, strides=[8, 16, 32],
                                   rgb888p_size=self.rgb888p_size,
                                   display_size=self.display_size,
                                   debug_mode=debug_mode)
        self.hand_kp = HandKPClassApp(config.HAND_KP_KMODEL,
                                      model_input_size=config.HAND_KP_INPUT_SIZE,
                                      rgb888p_size=self.rgb888p_size,
                                      display_size=self.display_size,
                                      debug_mode=debug_mode)
        self.hand_det.config_preprocess()
        # 确认状态机 / Confirmation state
        self.confirmed = None          # 当前已确认输出 {"label","box"} 或 None
        self.candidate_label = None    # 候选手势
        self.candidate_count = 0       # 候选连续出现次数

    def run(self, input_np):
        """执行一次识别（按调度调用，不是每帧）/ One recognition pass (scheduled)."""
        det_boxes = self.hand_det.run(input_np)
        raw_label = None
        raw_box = None
        for det_box in det_boxes:
            x1, y1, x2, y2 = det_box[2], det_box[3], det_box[4], det_box[5]
            w, h = int(x2 - x1), int(y2 - y1)
            # 过滤太小或贴边的检测框 / Filter tiny or edge boxes
            if (h < (0.1 * self.rgb888p_size[1])):
                continue
            if (w < (0.25 * self.rgb888p_size[0]) and ((x1 < (0.03 * self.rgb888p_size[0])) or (x2 > (0.97 * self.rgb888p_size[0])))):
                continue
            if (w < (0.15 * self.rgb888p_size[0]) and ((x1 < (0.01 * self.rgb888p_size[0])) or (x2 > (0.99 * self.rgb888p_size[0])))):
                continue
            self.hand_kp.config_preprocess(det_box)
            results_show, gesture = self.hand_kp.run(input_np)
            if gesture is not None:
                raw_label = gesture
                raw_box = det_box
                break
        self._confirm(raw_label, raw_box)
        return self.confirmed

    def _confirm(self, label, det_box):
        """N 次确认状态机：与当前输出一致则清零候选；候选连续达标才切换
        N-run confirmation: same as current output resets candidate;
        candidate must persist N runs to take over."""
        if label == (self.confirmed["label"] if self.confirmed else None):
            self.candidate_label = None
            self.candidate_count = 0
            if self.confirmed is not None and det_box is not None:
                self.confirmed["box"] = _hand_box_json(det_box)   # 同手势时跟踪框位置 / track box
            return
        if label == self.candidate_label:
            self.candidate_count += 1
        else:
            self.candidate_label = label
            self.candidate_count = 1
        if self.candidate_count >= config.GESTURE_CONFIRM_FRAMES:
            if label is None:
                self.confirmed = None
            else:
                self.confirmed = {"label": label, "box": _hand_box_json(det_box)}
            self.candidate_label = None
            self.candidate_count = 0

    def deinit(self):
        try:
            self.hand_det.deinit()
        except Exception:
            pass
        try:
            self.hand_kp.deinit()
        except Exception:
            pass
