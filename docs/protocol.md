# MyVisionHub 串口 JSON 协议

版本：v1（随功能模块逐步扩充，接收端应忽略未知字段以保证前向兼容）

## 传输层

- 通道：K230 物理 UART 引脚（`YbUart`），与 USB-CDC 调试通道无关
- 参数：**115200 8N1**（`config.py` 中 `UART_BAUDRATE` 可改，如 921600）
- 帧格式：**每行一个 JSON 对象**，以 `\n` 结尾（NDJSON / JSON Lines）
- 输出频率：默认 250ms 一帧（`OUTPUT_INTERVAL_MS`），无事件驱动
- 调试镜像：`PRINT_MIRROR=True` 时每行同时打印到 CanMV IDE 终端（USB-CDC），方便免 USB-TTL 观察

## 顶层信封

| 字段 | 类型 | 说明 |
|---|---|---|
| `type` | string | 固定 `"vision"` |
| `version` | int | 协议版本，当前 `1` |
| `frame_id` | int | 帧序号，每处理一帧 +1 |
| `ts_ms` | int | 板端毫秒时间戳（`time.ticks_ms()`） |

## `face` —— 人脸检测（已实现）

| 字段 | 类型 | 说明 |
|---|---|---|
| `present` | bool | 是否检测到人脸 |
| `count` | int | 当帧人脸数量 |
| `box` | object/null | 主目标（面积最大）人脸框，无人脸为 `null` |

`box` 子字段：`x` `y` `w` `h`（int，像素），坐标系为 sensor 分辨率 **640x480**，原点在左上角。

## `proximity` —— 人脸远近判断（已实现）

无人脸时整个字段为 `null`；有人脸时：

| 字段 | 类型 | 说明 |
|---|---|---|
| `state` | string | `"near"` / `"mid"` / `"far"`，双阈值迟滞：`ratio≥0.45` 进 near、`<0.40` 出；`ratio≤0.25` 进 far、`>0.30` 出（见 `config.py` `PROX_*`） |
| `trend` | string | `"approaching"` / `"receding"` / `"stable"`。双半窗算法：趋势窗口（2s）对半分，后半均值与前半均值相对变化超 `PROX_TREND_CHANGE`(0.10) 才输出前两者，否则一律 `stable`（不做判断） |
| `ratio` | float | 人脸框高 / 帧高（0~1），5 帧滑动平均，远近的度量值 |
| `change` | float | 趋势窗口内的相对变化量（正=变大/靠近，负=远离），趋势的量化依据 |

## 规划中字段（未实现，占位说明）

| 字段 | 来源模块 | 内容 |
|---|---|---|
| `pose` | face_pose.kmodel | 人脸朝向：`dir`(up/down/left/right/center) + `pitch`/`yaw`/`roll` 欧拉角 |
| `expression` | face_landmark.kmodel | 夸张表情分类：`label`(happy/angry/sad/surprise/neutral) |
| `gesture` | hand_det + handkp_det | 手势：`label`(fist/five/gun/yeah/love/ok/pinch) + `box` |

无人脸/无手时对应字段为 `null`；未实现的字段**不出现在输出中**（不是 `null`，是直接省略）。

## 输出示例

```json
{"type":"vision","version":1,"frame_id":498,"ts_ms":39923,"face":{"present":true,"count":1,"box":{"x":345,"y":142,"w":96,"h":125}},"proximity":{"state":"mid","trend":"approaching","ratio":0.26,"change":0.14}}
{"type":"vision","version":1,"frame_id":506,"ts_ms":40207,"face":{"present":false,"count":0,"box":null},"proximity":null}
```

## 接收端解析要点

1. 按行分割（`\n`），整行交给 JSON 解析器；半行/粘包需自行缓冲拼接
2. 忽略不认识的字段——协议只增不改，新增字段不应影响旧接收端
3. 字段值判空用 `is null`，不要假定字段一定存在
4. `ts_ms` 会回绕（MicroPython ticks），如需测时间差用差值而非绝对值
