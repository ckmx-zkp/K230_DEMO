# 串口 JSON 输出封装 / JSON output over UART
# 每行一个 JSON 对象；UART 初始化失败时自动降级为仅 print，保证骨架可独立验证
# One JSON object per line; falls back to print-only if UART init fails.

import ujson

import config


class JsonOutput:
    """JSON 行输出：YbUart（物理串口）+ 可选 IDE 终端镜像
    JSON line output: YbUart (physical UART) + optional IDE terminal mirror."""

    def __init__(self):
        try:
            from ybUtils.YbUart import YbUart
            self.uart = YbUart(baudrate=config.UART_BAUDRATE)
        except Exception as e:
            print("uart init failed, print-only mode:", e)
            self.uart = None

    def send(self, obj):
        """发送一个 dict：序列化为单行 JSON 发出 / Send a dict as one JSON line."""
        try:
            line = ujson.dumps(obj)
        except Exception as e:
            print("json encode failed:", e)
            return
        if self.uart is not None:
            try:
                self.uart.send(line + "\n")
            except Exception as e:
                print("uart send failed:", e)
        if config.PRINT_MIRROR:
            print(line)
