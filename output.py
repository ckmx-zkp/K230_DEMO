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
        self.rx_buf = ""

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

    def read_command(self):
        """非阻塞读取一条命令（按行缓冲，整行为一个 JSON 对象）
        Non-blocking read of one command (line-buffered, one JSON object per line)."""
        if self.uart is None:
            return None
        try:
            data = self.uart.read()
        except Exception:
            return None
        if not data:
            return None
        if not isinstance(data, str):
            try:
                data = data.decode()
            except Exception:
                return None
        self.rx_buf += data
        # 缓冲限长，防异常数据撑爆内存 / Bound the buffer
        if len(self.rx_buf) > 1024:
            self.rx_buf = self.rx_buf[-512:]
        nl = self.rx_buf.find("\n")
        if nl < 0:
            return None
        line = self.rx_buf[:nl].strip()
        self.rx_buf = self.rx_buf[nl + 1:]
        if not line:
            return None
        try:
            cmd = ujson.loads(line)
            if isinstance(cmd, dict):
                return cmd
        except Exception:
            pass
        return None
