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
OUTPUT_INTERVAL_MS = 250       # JSON 输出节流间隔 / JSON output throttle interval
PRINT_MIRROR = True            # JSON 同步 print 到 IDE 终端（调试免 USB-TTL）
                               # Also print JSON to IDE terminal (no USB-TTL needed while debugging)

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
