# 人脸与表情原型验收

当前目标是人脸位置/朝向与外观表情分类，不是身份识别。此版本仍须板端验证，不能视为喜怒哀乐准确率已经达标。

## 部署

1. 上传 `dist/models/face_landmark.kmodel` 到 `/sdcard/kmodel/face_landmark.kmodel`。文件从本地原始亚博固件镜像只读提取，大小 1409104 字节，SHA256：`0f501e8a11a1f7bbed4dfd0e7c94bd9c03c986ccae8a9cb7b3cd640407cb2196`。
2. 在 IDE 重新打开并运行最新 `dist/bootstrap_write.py`，应安装 12 个 Python 文件。安装脚本不包含模型，不会覆盖标定文件。
3. 复位后在线运行最新 `main.py`。保留 CSI0 和 IDE 预览。多了一个关键点模型，内存/帧率须真机验证。

## 先验收人脸，后标定表情

必须先看框是否包住实际人脸、黄色关键点是否落在眉眼鼻口。历史日志 y 恒为 0 的原因仍未实测确认；当前检测预处理与官方例程一致，没有凭猜测加坐标补偿。如果框或关键点偏移，停止标定，提供截图和日志。

新 `tracking` 字段为 `{id,cx,cy,dx,dy,clipped}`；dx/dy 按半画幅归一化，左/上为负，右/下为正。使用与上一框 IoU 关联目标，关联失败选择最大脸并重置姿态/远近/表情缓存。人脸丢失立即输出 tracking=null；重新出现重新分配 id。它不是身份 ID，也不是完整遮挡重识别。尚未控制 ESP32 眼睛或舵机。

`expression` 为 null 或分类对象。未标定输出 unknown/needs_calibration，出画输出 clipped_face，多人输出 need_single_face，侧脸输出 face_camera。distance 为几何特征距离，margin 为前两类距离差，均不是概率。表情使用关键点模型 + 五类个人样本最近原型分类，不是训练好的通用情绪分类模型，不表示真实心情。

## 五类采样

推荐在 IDE 直接打开电脑上的 `tools/calibrate_standalone.py` 在线运行，先停止 `main.py`。此独立程序复用板上已安装的模块和模型，不需要重新安装 bootstrap。画面只显示底部白色采样提示，不运行 UART、手势和远近输出，也不绘制关键点。

默认自动依次采集五类，每类先准备 5 秒，再保持表情等待 15 次有效采样，看到 saved 后自动进入下一类。保持单人、正脸、不贴边；无效帧会重置连续样本，超时自动重试当前类别。每类立即保存，全部结束后自动退出，再运行 `main.py` 测试。再次运行会重采五类；只重采一类时修改脚本顶部 `LABELS = ('happy',)`。以下请求脚本方式仍可使用。

依次采样 neutral（中性）、happy（笑脸）、angry（皱眉生气外观）、sad（难过外观）、surprise（惊讶）。喜和乐合并为 happy。

每一类：停止主程序 → IDE 打开 `tools/calibrate_expression.py`，修改 LABEL → 在线运行一次 → 重新运行 main.py。保持完整正脸、照明稳定，3 秒准备后保持该表情，收集 15 个有效样本。终端出现 `Expression calibration saved: 标签` 才表示保存成功。无效姿态或丢脸会清空正在收集的连续样本，30 秒超时后需重新发起。保存至 `/sdcard/app/MyVisionHub/expression_profiles.json`；重复某类会覆盖该类样本。

全部五类收齐才开始分类。明确分类需连续三次一致；距离过大或类别相近立即 unknown。采样时不要让五类都是同一张中性脸；样本区分不足应重新采样，不要通过降低拒识门槛强制输出。此方案需同一用户/类似距离光线，跨人泛化未验证。

## 验收记录

- 同一人在左、中、右及上、下移动：框、关键点和中心偏移一致；多人时避免频繁抢目标。
- 转头和移动分别测试；框贴边、遮脸、无人脸时不能继续报告旧表情。
- 五类每类重复 10 次，记录正确、错误、unknown 数量；换距离和光线复测。
- 同时举手，检查第五模型加入后的帧率、手势和内存稳定性。
- IDE 的 RGB565 预览与 AI 通道分开抓帧，快速运动会有时序差，不能据此直接断言坐标公式错误。

参考：官方人脸检测及 106 点索引：
https://github.com/kendryte/canmv_k230/blob/canmv_k230/resources/examples/05-AI-Demo/face_detection.py
https://github.com/kendryte/canmv_k230/blob/canmv_k230/resources/examples/05-AI-Demo/face_landmark.py
