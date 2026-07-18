# MyVisionHub

基于亚博 K230 视觉模块的多模型视觉集成工程：同时运行人脸朝向、表情分类、手势识别、人脸远近判断，结果通过串口以 JSON 文本协议（每行一个对象，115200 波特率）输出，LCD 保留 OSD 可视化用于调试。

## 功能清单

- **人脸朝向识别**：输出方向（up/down/left/right/center）与 pitch/yaw/roll 欧拉角
- **夸张表情分类**：基于人脸关键点几何规则，识别 喜(happy)/怒(angry)/哀(sad)/惊(surprise)/平静(neutral)
- **手势识别**：手掌检测 + 手部关键点几何分类（fist/five/gun/yeah/love/ok/pinch）
- **人脸远近判断**：靠近(near)/适中(mid)/远离(far) 三态 + 接近/远离趋势

## 技术前提

- 纯 MicroPython 应用层开发：固件（RT-Smart+驱动+nncase）、系统库（`libs/PipeLine`、`libs/AIBase`、`libs/AI2D`、`ybUtils/YbUart`）、模型文件（`/sdcard/kmodel/*.kmodel`）均为现成品，不碰底层、不训练模型。
- 本板无 eMMC/Flash，固件与代码全部运行在 TF 卡上；本仓库在 PC 上管理版本，TF 卡仅为部署目标。
- 复用并改造亚博官方例程（`程序源码/07.Face`、`08.Body`），集成架构参考 `程序源码/AI_Toy_Vision_Hub`，但修正其各模块重复跑人脸检测的问题——全工程共享一次人脸检测。

## 目录结构

```
MyVisionHub/
├── main.py            # 入口：PipeLine + 帧调度 + 模块编排
├── config.py          # 全部可调参数（模型路径、阈值、调度周期、串口）
├── output.py          # YbUart 封装：JSON 行发送 + 输出节流
├── modules/
│   ├── face_det.py    # 共享人脸检测（单例，结果分发给各模块）
│   ├── face_pose.py   # 人脸朝向：姿态欧拉角 → 方向分类
│   ├── expression.py  # 表情分类：landmark 几何特征规则法
│   ├── gesture.py     # 手势识别：handkp 指角几何分类
│   └── proximity.py   # 远近判断：纯几何，无额外模型
└── docs/
    └── protocol.md    # 串口 JSON 协议字段说明
```

## 部署与运行

1. K230 板已烧录亚博固件（TF 卡内有 `/sdcard/kmodel`、`/sdcard/app/libs` 等）。
2. 将本仓库全部 `.py` 文件（保持目录结构）拷到 TF 卡，如 `/sdcard/app/MyVisionHub/`。
3. 用 CanMV IDE 打开 `main.py` 运行；串口终端（115200）可见 JSON 输出，LCD 显示可视化画面。
4. 调好后可参考官方资料 `01.快速入门/4.离线运行例程` 设置离线自启。

## 串口协议摘要

每行一个 JSON 对象，字段含 `face`/`pose`/`expression`/`gesture`/`proximity`，无人脸或无手时对应字段为 `null`。详见 `docs/protocol.md`。

## 开发路线图

- [x] 1. 建仓（目录 + git + README）
- [ ] 2. 骨架：共享人脸检测 + JSON 输出链路
- [ ] 3. 功能 d：人脸远近判断
- [ ] 4. 功能 a：人脸朝向识别
- [ ] 5. 功能 c：手势识别
- [ ] 6. 功能 b：表情分类（含真机阈值标定）
- [ ] 7. 整合调优 + 协议文档
