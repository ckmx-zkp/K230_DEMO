# 开发进度记录

## 2026-07-18（第 1 天）

### 完成

- **方案与建仓**：确定四功能（人脸朝向/夸张表情/手势/远近）多模型集成方案；创建 `MyVisionHub` 工程并初始化 git，同步 GitHub。
- **开发环境**：CanMV IDE 安装连接（COM4，固件 CanMV v1.4.3 / rt-smart）；helloworld 验证链路；踩坑并解决 COM 口占用（IDE 进程半连接占端口，需退出 IDE + 拔插 USB 恢复）。
- **部署链路**：证实本板 USB-CDC 为 IDE 私有协议、无标准 REPL，**mpremote 不可用**；落地"引导脚本"方案（`tools/pack_bootstrap.py` 打包 → IDE 运行 `dist/bootstrap_write.py` 写卡），全程免读卡器。
- **骨架**（commit `d63a3cf`）：共享人脸检测 + JSON 行输出（250ms 节流 + IDE 终端镜像）+ LCD OSD + HEADLESS 开关。真机验证：全链路约 30fps。
- **功能 d 远近判断**（commit `c9fd538` / `834dd73`）：人脸框高占比 + 5 帧滑动平均 + 三态双阈值迟滞（near/mid/far）；趋势采用双半窗均值比较（2s 窗口，±10% 阈值），静止一律 stable。真机验证通过。
- **功能 a 人脸朝向**（commit `0fbe6a6`）：移植 FacePoseApp 复用共享检测框（不重复检测），欧拉角 → 方向迟滞分类（center/up/down/left/right，20°/15°）；每 2 帧调度；JSON 新增 `pose` 字段。真机验证通过。
- **文档**：`docs/protocol.md` 串口协议（已实现字段 + 规划字段占位）；仓库外另写 4 篇个人技术笔记（资料包根目录 `docs/`）。

### 下一步

- 第 5 步：手势识别（hand_det + handkp_det，指角几何分类，3 帧确认防抖）
- 第 6 步：表情分类（face_landmark 106 关键点几何规则，需真机标定阈值）
- 第 7 步：整合调优（帧率实测调度、协议文档补全、README 使用说明）

### 关键结论备忘

- 本板无 eMMC/Flash，系统与应用全部跑在 TF 卡；git 在 PC 管理，卡只做部署目标
- 串口分工：结果 JSON 走物理 UART（115200，带宽占用约 5%）；图像回传走 USB-CDC（IDE 帧缓冲区）
- 帧调度模式：检测每帧、二阶段模型按帧号取模、输出 250ms 节流
- 出厂 GUI 即 `/sdcard/main.py`，将来离线自启覆盖前先备份

### Git 记录（7 commits）

```
437b566 docs: 标记人脸朝向识别真机验证通过
0fbe6a6 feat: 集成人脸朝向识别（姿态欧拉角 + 方向迟滞分类）
834dd73 feat: 增强远近趋势判断并新增串口协议文档
c9fd538 feat: 新增人脸远近判断模块（proximity）
e6de202 docs: 更新README，标记骨架真机验证通过
d63a3cf feat: add skeleton with shared face detection and JSON output
c194dff chore: init MyVisionHub project structure
```
