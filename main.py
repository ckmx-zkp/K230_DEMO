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
APP_DIR = "/sdcard/app/MyVisionHub"
while APP_DIR in sys.path:
    sys.path.remove(APP_DIR)
sys.path.insert(0, APP_DIR)

# IDE runs can retain imports from a previous deployment. Refresh our modules.
for _name in ("config", "output", "camera_pipeline", "modules.face_det",
              "modules.face_pose", "modules.gesture", "modules.proximity",
              "modules.face_target", "modules.face_landmark", "modules.expression",
              "modules"):
    if _name in sys.modules:
        del sys.modules[_name]

import ulab.numpy as np

import config
if getattr(config, "APP_CONFIG_VERSION", None) != "2026-09-13.1":
    raise RuntimeError("Outdated config: run the latest dist/bootstrap_write.py, reset board, then run main.py")
print("MyVisionHub config:", getattr(config, "__file__", APP_DIR + "/config.py"),
      config.APP_CONFIG_VERSION)
from output import JsonOutput
from modules.face_det import FaceDetApp
from modules.face_pose import FacePoseApp, PoseDirection
from modules.gesture import GestureModule
from modules.proximity import Proximity
from libs.PipeLine import PipeLine, ScopedTiming
from camera_pipeline import HeadlessPipeline
from modules.face_target import FaceTarget
from modules.expression import Expression


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


def build_snapshot(frame_id, count, main_face, prox, pose, gesture, target=None, expression=None):
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
        "gesture": gesture,
        "tracking": target,
        "expression": expression,
    }


def save_overrides():
    """把白名单配置项的当前值持久化到 TF 卡 / Persist whitelisted config values."""
    try:
        import ujson
        data = {}
        for k in config.CONFIGURABLE_KEYS:
            data[k] = getattr(config, k)
        with open(config.OVERRIDE_PATH, "w") as f:
            f.write(ujson.dumps(data))
        return True
    except Exception as e:
        print("save overrides failed:", e)
        return False


def handle_command(cmd, out):
    """处理 ESP32 下发的配置命令并回 ack
    Handle config commands from ESP32 and reply with an ack.

    支持 / Supported:
      {"cmd":"set","key":"POSE_DIR_ENTER","value":25}
      {"cmd":"get","key":"POSE_DIR_ENTER"}
      {"cmd":"list"}
      {"cmd":"save"}
    """
    if not isinstance(cmd, dict):
        return
    c = cmd.get("cmd")
    if c == "set":
        key = cmd.get("key")
        ok = key in config.CONFIGURABLE_KEYS
        if ok:
            try:
                setattr(config, key, cmd.get("value"))
            except Exception:
                ok = False
        out.send({"type": "ack", "cmd": "set", "key": key, "ok": ok,
                  "value": getattr(config, key, None) if ok else None})
    elif c == "get":
        key = cmd.get("key")
        ok = key in config.CONFIGURABLE_KEYS
        out.send({"type": "ack", "cmd": "get", "key": key, "ok": ok,
                  "value": getattr(config, key, None) if ok else None})
    elif c == "list":
        items = {}
        for k in config.CONFIGURABLE_KEYS:
            items[k] = getattr(config, k, None)
        out.send({"type": "ack", "cmd": "list", "ok": True, "items": items})
    elif c == "save":
        out.send({"type": "ack", "cmd": "save", "ok": save_overrides()})


def draw_osd(pl, main_face, prox, pose, gesture):
    """OSD 绘制检测框与各状态（display 与 sensor 分辨率不同则按比例换算）
    Draw boxes and status texts on OSD (scaled if resolutions differ)."""
    if config.IDE_PREVIEW:
        pl.begin_overlay()
    else:
        pl.osd_img.clear()
    green = (0, 255, 0) if config.IDE_PREVIEW else (255, 0, 255, 0)
    red = (255, 0, 0) if config.IDE_PREVIEW else (255, 255, 0, 0)
    blue = (0, 0, 255) if config.IDE_PREVIEW else (255, 0, 0, 255)
    sx = pl.display_size[0] / pl.rgb888p_size[0]
    sy = pl.display_size[1] / pl.rgb888p_size[1]
    if main_face is not None:
        x = int(float(main_face[0]) * sx)
        y = int(float(main_face[1]) * sy)
        w = int(float(main_face[2]) * sx)
        h = int(float(main_face[3]) * sy)
        pl.osd_img.draw_rectangle(x, y, w, h, color=green, thickness=2)
        if prox is not None:
            text = "%s %s %.2f" % (prox["state"], prox["trend"], prox["ratio"])
            pl.osd_img.draw_string_advanced(8, 8, 24, text, color=green)
        if pose is not None:
            text = "%s y%.0f p%.0f" % (pose["dir"], pose["yaw"], pose["pitch"])
            pl.osd_img.draw_string_advanced(8, 38, 24, text, color=red)
    if gesture is not None and gesture.get("box") is not None:
        b = gesture["box"]
        gx = int(b["x"] * sx)
        gy = int(b["y"] * sy)
        gw = int(b["w"] * sx)
        gh = int(b["h"] * sy)
        pl.osd_img.draw_rectangle(gx, gy, gw, gh, color=blue, thickness=2)
        pl.osd_img.draw_string_advanced(8, 68, 24, gesture["label"], color=blue)


def main():
    pl = None
    face_det = None
    face_pose = None
    gesture_mod = None
    landmark = None
    expression = Expression()
    target = FaceTarget()
    points = None
    expression_res = None
    out = None
    img = None
    try:
        # 创建图像处理管线 / Create image processing pipeline
        if config.IDE_PREVIEW:
            pl = HeadlessPipeline(config.SENSOR_ID, config.RGB888P_SIZE, preview=True)
            pl.create()
        elif config.HEADLESS:
            pl = HeadlessPipeline(config.SENSOR_ID, config.RGB888P_SIZE)
            pl.create()
        else:
            from media.sensor import Sensor
            pl = PipeLine(rgb888p_size=config.RGB888P_SIZE,
                          display_size=config.DISPLAY_SIZE,
                          display_mode=config.DISPLAY_MODE)
            pl.create(sensor=Sensor(id=config.SENSOR_ID))
        print("Camera ready: CSI", config.SENSOR_ID)

        # 加载共享人脸检测 / Load shared face detection
        print("Loading face detection:", config.FACE_DET_KMODEL)
        face_det = FaceDetApp(config.FACE_DET_KMODEL,
                              model_input_size=config.FACE_DET_INPUT_SIZE,
                              anchors=load_anchors(),
                              confidence_threshold=config.CONFIDENCE_THRESHOLD,
                              nms_threshold=config.NMS_THRESHOLD,
                              rgb888p_size=config.RGB888P_SIZE,
                              display_size=config.DISPLAY_SIZE,
                              debug_mode=0)
        face_det.config_preprocess()
        print("Face detection ready")

        # 加载人脸姿态模型（复用共享检测框，不再单独检测）/ Face pose model (uses shared box)
        print("Loading face pose:", config.FACE_POSE_KMODEL)
        face_pose = FacePoseApp(config.FACE_POSE_KMODEL,
                                model_input_size=config.FACE_POSE_INPUT_SIZE,
                                rgb888p_size=config.RGB888P_SIZE,
                                display_size=config.DISPLAY_SIZE,
                                debug_mode=0)

        # 加载手势识别模块（hand_det + handkp，自带确认防抖）/ Gesture module
        print("Loading hand detection and keypoints")
        gesture_mod = GestureModule(rgb888p_size=config.RGB888P_SIZE,
                                    display_size=config.DISPLAY_SIZE,
                                    debug_mode=0)

        if config.EXPRESSION_ENABLED:
            import os
            try:
                os.stat(config.FACE_LANDMARK_KMODEL)
            except OSError:
                print('Expression disabled: missing', config.FACE_LANDMARK_KMODEL)
                expression_res = {'label':'unknown', 'reason':'missing_model'}
            else:
                from modules.face_landmark import FaceLandMarkApp
                print('Loading facial landmarks:', config.FACE_LANDMARK_KMODEL)
                landmark = FaceLandMarkApp(config.FACE_LANDMARK_KMODEL, [192,192],
                                           rgb888p_size=config.RGB888P_SIZE,
                                           display_size=config.DISPLAY_SIZE)
                print('Facial landmarks ready; expression requires calibration')

        out = JsonOutput()
        prox = Proximity()
        pose_dir = PoseDirection()
        last_pose = None      # 最近一次姿态结果，帧间保持输出 / Latest pose result
        last_gesture = None   # 最近一次手势确认结果，帧间保持输出 / Latest confirmed gesture
        frame_id = 0
        last_output = 0
        print("MyVisionHub skeleton started")

        while True:
            with ScopedTiming("total", config.DEBUG_TIMING):
                frame_id += 1
                img = pl.get_frame()                     # 获取当前帧 / Get current frame
                det_boxes = face_det.run(img)            # 共享人脸检测 / Shared face detection
                count = det_count(det_boxes)
                main_face, changed = target.update(det_boxes, *config.RGB888P_SIZE)
                if changed or main_face is None:
                    prox = Proximity()
                    pose_dir.reset()
                    last_pose = None
                    points = None
                    expression.reset()
                    expression_res = None

                now = ticks_ms()

                # ESP32 命令通道：非阻塞轮询 / UART command channel: non-blocking poll
                cmd = out.read_command()
                if cmd:
                    handle_command(cmd, out)

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

                # 手势识别：按帧调度（与姿态错开，摊薄 KPU 负载）/ Gesture: scheduled (offset)
                if frame_id % config.GESTURE_RUN_EVERY == (1 % config.GESTURE_RUN_EVERY):
                    last_gesture = gesture_mod.run(img)

                if landmark is not None and frame_id % config.EXPRESSION_RUN_EVERY == 0:
                    points = None
                    if main_face is not None:
                        landmark.config_preprocess(main_face)
                        points = landmark.run(img)
                    expression_res = expression.update(points, main_face, last_pose, count)
                if landmark is None and config.EXPRESSION_ENABLED:
                    expression_res = {'label':'unknown', 'reason':'missing_model'}

                if config.IDE_PREVIEW or not config.HEADLESS:
                    draw_osd(pl, main_face, prox_res, last_pose, last_gesture)  # OSD / Draw boxes + status
                    yellow = (255,255,0) if config.IDE_PREVIEW else (255,255,255,0)
                    if points is not None:
                        for i in range(0, len(points), 2):
                            px = int(float(points[i])*pl.display_size[0]/config.RGB888P_SIZE[0])
                            py = int(float(points[i+1])*pl.display_size[1]/config.RGB888P_SIZE[1])
                            if 0 <= px < pl.display_size[0] and 0 <= py < pl.display_size[1]:
                                pl.osd_img.draw_circle(px, py, 1, color=yellow)
                    if expression_res is not None:
                        text = 'expr: ' + expression_res['label'] + ' ' + expression_res.get('reason','')
                        pl.osd_img.draw_string_advanced(8, 98, 20, text, color=yellow)
                    try:
                        pl.show_image()                  # 送显（拆屏异常时静默降级）
                    except Exception as e:
                        print("Display ERROR:", e)
                        raise

                if ticks_diff(now, last_output) >= config.OUTPUT_INTERVAL_MS:
                    out.send(build_snapshot(frame_id, count, main_face, prox_res, last_pose, last_gesture,
                                            target.snapshot(*config.RGB888P_SIZE), expression_res))
                    last_output = now

                gc.collect()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print("MyVisionHub skeleton exit:", e)
        sys.print_exception(e)
    finally:
        img = None
        points = None
        if landmark is not None:
            try:
                landmark.deinit()
            except Exception:
                pass
        if out is not None:
            out.close()
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
        if gesture_mod is not None:
            try:
                gesture_mod.deinit()
            except Exception:
                pass
        if pl is not None:
            try:
                pl.destroy()
            except Exception:
                pass


if __name__ == "__main__":
    main()
