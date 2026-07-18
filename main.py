# MyVisionHub 入口：共享人脸检测 + 串口 JSON 输出骨架
# Entry: shared face detection + JSON-over-UART output skeleton
#
# 真机运行方式 / Run on device:
#   1. 用 deploy.bat 把本工程同步到 /sdcard/app/MyVisionHub/
#   2. CanMV IDE 打开卡上（或本文件）main.py，点运行
#   3. IDE 终端可见每秒约 4 行 JSON；LCD 显示人脸框（HEADLESS=True 时关闭）

import gc
import sys
import time

# 兼容 TF 卡部署：把工程目录加入模块搜索路径（卡上 /sdcard 本身已在 sys.path）
# Add project dir to module search path for on-card deployment
if "/sdcard/app/MyVisionHub" not in sys.path:
    sys.path.append("/sdcard/app/MyVisionHub")

import ulab.numpy as np

import config
from output import JsonOutput
from modules.face_det import FaceDetApp
from modules.face_pose import FacePoseApp, PoseDirection
from modules.proximity import Proximity
from libs.PipeLine import PipeLine, ScopedTiming


def ticks_ms():
    """毫秒时间戳 / Millisecond timestamp."""
    try:
        return time.ticks_ms()
    except Exception:
        return int(time.time() * 1000)


def ticks_diff(a, b):
    """两个时间戳的差值，兼容 ticks 溢出 / Tick difference with overflow handling."""
    try:
        return time.ticks_diff(a, b)
    except Exception:
        return a - b


def det_count(det_boxes):
    """检测框数量：后处理无人脸时返回空 list，有人脸时为 ndarray
    Number of detections: empty list when no face, ndarray otherwise."""
    if det_boxes is None:
        return 0
    try:
        return det_boxes.shape[0]
    except Exception:
        try:
            return len(det_boxes)
        except Exception:
            return 0


def pick_main_face(det_boxes, count):
    """取面积最大的人脸框作为主目标 / Pick the largest face box as the main target."""
    if count <= 0:
        return None
    best = None
    best_area = -1
    for i in range(count):
        det = det_boxes[i]
        w = float(det[2])
        h = float(det[3])
        area = w * h
        if area > best_area:
            best_area = area
            best = det
    return best


def to_int(v):
    """ulab 数值安全转 int / Safe numeric-to-int conversion."""
    try:
        return int(round(float(v)))
    except Exception:
        return 0


def face_box_json(det):
    """检测框转 JSON dict：[x, y, w, h, ...] 取前四维
    Convert detection row [x, y, w, h, ...] to JSON dict."""
    if det is None:
        return None
    return {
        "x": to_int(det[0]),
        "y": to_int(det[1]),
        "w": to_int(det[2]),
        "h": to_int(det[3]),
    }


def load_anchors():
    """加载人脸检测锚框 / Load face detection anchors."""
    anchors = np.fromfile(config.ANCHORS_PATH, dtype=np.float)
    return anchors.reshape((config.ANCHOR_LEN, config.DET_DIM))


def build_snapshot(frame_id, count, main_face, prox, pose):
    """组装一帧的输出快照 / Build the output snapshot for one frame."""
    return {
        "type": "vision",
        "version": 1,
        "frame_id": frame_id,
        "ts_ms": ticks_ms(),
        "face": {
            "present": count > 0,
            "count": count,
            "box": face_box_json(main_face),
        },
        "proximity": prox,
        "pose": pose,
    }


def draw_osd(pl, main_face, prox, pose):
    """OSD 绘制主目标人脸框与各状态（display 与 sensor 分辨率不同则按比例换算）
    Draw main face box and status texts on OSD (scaled if resolutions differ)."""
    pl.osd_img.clear()
    if main_face is None:
        return
    sx = pl.display_size[0] / pl.rgb888p_size[0]
    sy = pl.display_size[1] / pl.rgb888p_size[1]
    x = int(float(main_face[0]) * sx)
    y = int(float(main_face[1]) * sy)
    w = int(float(main_face[2]) * sx)
    h = int(float(main_face[3]) * sy)
    pl.osd_img.draw_rectangle(x, y, w, h, color=(255, 0, 255, 0), thickness=2)
    if prox is not None:
        text = "%s %s %.2f" % (prox["state"], prox["trend"], prox["ratio"])
        pl.osd_img.draw_string_advanced(x, max(0, y - 40), 32, text, color=(255, 0, 255, 0))
    if pose is not None:
        text = "%s y%.0f p%.0f" % (pose["dir"], pose["yaw"], pose["pitch"])
        pl.osd_img.draw_string_advanced(x, max(0, y - 80), 32, text, color=(255, 255, 0, 0))


def main():
    pl = None
    face_det = None
    face_pose = None
    try:
        # 创建图像处理管线 / Create image processing pipeline
        pl = PipeLine(rgb888p_size=config.RGB888P_SIZE,
                      display_size=config.DISPLAY_SIZE,
                      display_mode=config.DISPLAY_MODE)
        pl.create()

        # 加载共享人脸检测 / Load shared face detection
        face_det = FaceDetApp(config.FACE_DET_KMODEL,
                              model_input_size=config.FACE_DET_INPUT_SIZE,
                              anchors=load_anchors(),
                              confidence_threshold=config.CONFIDENCE_THRESHOLD,
                              nms_threshold=config.NMS_THRESHOLD,
                              rgb888p_size=config.RGB888P_SIZE,
                              display_size=config.DISPLAY_SIZE,
                              debug_mode=0)
        face_det.config_preprocess()

        # 加载人脸姿态模型（复用共享检测框，不再单独检测）/ Face pose model (uses shared box)
        face_pose = FacePoseApp(config.FACE_POSE_KMODEL,
                                model_input_size=config.FACE_POSE_INPUT_SIZE,
                                rgb888p_size=config.RGB888P_SIZE,
                                display_size=config.DISPLAY_SIZE,
                                debug_mode=0)

        out = JsonOutput()
        prox = Proximity()
        pose_dir = PoseDirection()
        last_pose = None      # 最近一次姿态结果，帧间保持输出 / Latest pose result
        frame_id = 0
        last_output = 0
        print("MyVisionHub skeleton started")

        while True:
            with ScopedTiming("total", config.DEBUG_TIMING):
                frame_id += 1
                img = pl.get_frame()                     # 获取当前帧 / Get current frame
                det_boxes = face_det.run(img)            # 共享人脸检测 / Shared face detection
                count = det_count(det_boxes)
                main_face = pick_main_face(det_boxes, count)

                now = ticks_ms()
                # 人脸远近判断（纯几何，每帧）/ Proximity (geometry only, every frame)
                prox_res = prox.update(main_face, config.RGB888P_SIZE[1], now)

                # 人脸朝向：按帧调度，用共享主目标框 / Face pose: scheduled, shared main box
                if main_face is None:
                    last_pose = None
                    pose_dir.reset()
                elif frame_id % config.POSE_RUN_EVERY == 0:
                    face_pose.config_preprocess(main_face)
                    R, eular = face_pose.run(img)
                    pitch, yaw, roll = eular[0], eular[1], eular[2]
                    last_pose = {
                        "dir": pose_dir.update(pitch, yaw),
                        "pitch": round(float(pitch), 1),
                        "yaw": round(float(yaw), 1),
                        "roll": round(float(roll), 1),
                    }

                if not config.HEADLESS:
                    draw_osd(pl, main_face, prox_res, last_pose)  # OSD 画框与状态 / Draw box + status
                    try:
                        pl.show_image()                  # 送显（拆屏异常时静默降级）
                    except Exception:
                        pass

                if ticks_diff(now, last_output) >= config.OUTPUT_INTERVAL_MS:
                    out.send(build_snapshot(frame_id, count, main_face, prox_res, last_pose))
                    last_output = now

                gc.collect()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print("MyVisionHub skeleton exit:", e)
    finally:
        if face_det is not None:
            try:
                face_det.deinit()
            except Exception:
                pass
        if face_pose is not None:
            try:
                face_pose.deinit()
            except Exception:
                pass
        if pl is not None:
            try:
                pl.destroy()
            except Exception:
                pass


if __name__ == "__main__":
    main()
