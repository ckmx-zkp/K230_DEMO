# MyVisionHub 全局配置 / Global configuration
# 所有可调参数集中在这里，改阈值、路径、开关只动这一个文件
# All tunable parameters live here; change thresholds/paths/switches in this file only.

# ---- 图像与显示 / Image & display ----
RGB888P_SIZE = [640, 480]      # sensor 给 AI 的图像分辨率 / Sensor-to-AI image resolution
DISPLAY_SIZE = [640, 480]      # 显示分辨率 / Display resolution
DISPLAY_MODE = "lcd"           # "lcd" 或 "hdmi"（PipeLine 仅支持这两种）/ "lcd" or "hdmi" only
HEADLESS = False               # True = 跳过全部 OSD 绘制与显示（拆屏后独立运行用）
                               # True = skip all OSD drawing & display (for headless standalone run)

# ---- 串口输出 / UART output ----
UART_BAUDRATE = 115200         # YbUart 波特率（物理串口引脚）/ Physical UART baudrate
OUTPUT_INTERVAL_MS = 100       # JSON 输出节流间隔（10Hz，兼顾舵机跟随与带宽）
                               # JSON output throttle interval (10Hz)
PRINT_MIRROR = True            # JSON 同步 print 到 IDE 终端（调试免 USB-TTL）
                               # Also print JSON to IDE terminal (no USB-TTL needed while debugging)

# ---- 运行时阈值可配（ESP32 命令通道）/ Runtime-configurable via UART commands ----
# 白名单：只有这些键允许 ESP32 远程 set；模型路径等关键配置不开放
# Whitelist: only these keys may be set remotely; critical paths stay closed
CONFIGURABLE_KEYS = [
    "OUTPUT_INTERVAL_MS",
    "PROX_WINDOW", "PROX_NEAR_ENTER", "PROX_NEAR_EXIT", "PROX_FAR_ENTER", "PROX_FAR_EXIT",
    "PROX_TREND_WINDOW_MS", "PROX_TREND_MIN_SPAN_MS", "PROX_TREND_CHANGE",
    "POSE_RUN_EVERY", "POSE_DIR_ENTER", "POSE_DIR_EXIT",
    "GESTURE_RUN_EVERY", "GESTURE_CONFIRM_FRAMES", "GESTURE_DEBUG",
]
OVERRIDE_PATH = "/sdcard/app/MyVisionHub/config_override.json"  # save 命令的持久化文件


def _load_override():
    """启动时加载 ESP32 保存过的阈值覆盖 / Apply persisted overrides at boot."""
    try:
        import ujson
        with open(OVERRIDE_PATH, "r") as f:
            data = ujson.loads(f.read())
        g = globals()
        for k, v in data.items():
            if k in CONFIGURABLE_KEYS:
                g[k] = v
    except Exception:
        pass


# ---- 调试 / Debug ----
DEBUG_TIMING = False           # True = 打印每帧耗时（会刷屏，排查性能时再开）
                               # True = print per-frame timing (noisy; enable only for profiling)

# ---- 人脸检测模型（全工程共享一次检测）/ Shared face detection model ----
FACE_DET_KMODEL = "/sdcard/kmodel/face_detection_320.kmodel"
FACE_DET_INPUT_SIZE = [320, 320]
ANCHORS_PATH = "/sdcard/utils/prior_data_320.bin"
ANCHOR_LEN = 4200
DET_DIM = 4
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.2

# ---- 人脸远近判断 / Proximity（ratio = 人脸框高 / 帧高）----
PROX_WINDOW = 5              # 滑动平均窗口（帧数）/ Moving-average window in frames
PROX_NEAR_ENTER = 0.45       # ratio ≥ 此值进入 near / enter "near"
PROX_NEAR_EXIT = 0.40        # ratio < 此值退出 near（迟滞）/ exit "near" (hysteresis)
PROX_FAR_ENTER = 0.25        # ratio ≤ 此值进入 far / enter "far"
PROX_FAR_EXIT = 0.30         # ratio > 此值退出 far（迟滞）/ exit "far" (hysteresis)
PROX_TREND_WINDOW_MS = 2000  # 趋势窗口长度（毫秒），对半分成前后两段比较 / Trend window (ms)
PROX_TREND_MIN_SPAN_MS = 800 # 数据跨度不足此不判趋势 / Min data span before judging trend
PROX_TREND_CHANGE = 0.10     # 趋势判定相对变化率 / Relative change to trigger trend

# ---- 人脸朝向 / Face pose ----
FACE_POSE_KMODEL = "/sdcard/kmodel/face_pose.kmodel"
FACE_POSE_INPUT_SIZE = [120, 120]
POSE_RUN_EVERY = 2           # 每 N 帧跑一次姿态估计（帧调度）/ Run pose every N frames
POSE_DIR_ENTER = 20.0        # 偏离中心的方向进入阈值（度）/ Enter-threshold leaving center (deg)
POSE_DIR_EXIT = 15.0         # 回到中心的方向退出阈值（度）/ Exit-threshold back to center (deg)

# ---- 手势识别 / Hand gesture ----
HAND_DET_KMODEL = "/sdcard/kmodel/hand_det.kmodel"
HAND_KP_KMODEL = "/sdcard/kmodel/handkp_det.kmodel"
HAND_DET_INPUT_SIZE = [512, 512]
HAND_KP_INPUT_SIZE = [256, 256]
HAND_CONF_THRESHOLD = 0.2
HAND_NMS_THRESHOLD = 0.5
HAND_LABELS = ["hand"]
HAND_ANCHORS = [26, 27, 53, 52, 75, 71, 80, 99, 106, 82, 99, 134, 140, 113, 161, 172, 245, 276]
GESTURE_RUN_EVERY = 4        # 每 N 帧跑一次手势（与姿态错开）/ Run gesture every N frames (offset from pose)
GESTURE_CONFIRM_FRAMES = 3   # 连续 N 次识别一致才切换输出 / N consecutive runs to confirm a switch
GESTURE_DEBUG = True         # True = 打印每次识别的原始标签与五指角度（调阈值用，约7行/秒）
                             # True = print raw label + finger angles each run (for tuning)

# 所有默认值定义完毕，最后应用 ESP32 保存的阈值覆盖（必须放文件末尾，
# 否则 override 会被下方的默认值重新覆盖）
# Apply persisted overrides LAST, after all defaults are defined.
_load_override()
