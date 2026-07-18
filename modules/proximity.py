# 人脸远近判断模块 / Face proximity estimation
# 纯几何方法，无额外模型：人脸框高 / 帧高 = 远近比例 ratio
# Pure geometry, no extra model: ratio = face_box_height / frame_height
#
# 三态 + 双阈值迟滞防抖，另输出接近/远离趋势
# Three states with dual-threshold hysteresis, plus approaching/receding trend.

import config


class Proximity:
    """人脸远近判断：滑动平均 + 迟滞状态机 + 趋势
    Face proximity: moving average + hysteresis state machine + trend."""

    def __init__(self):
        self.ratios = []      # 滑动平均窗口（最近 PROX_WINDOW 帧）/ Moving-average window
        self.history = []     # (ts_ms, avg) 趋势参考历史 / History for trend reference
        self.state = None     # "near" / "mid" / "far" / None
        self.trend = "stable"

    def update(self, det, frame_h, ts_ms):
        """
        每帧更新 / Per-frame update.

        参数:
        det: 主目标人脸框 [x, y, w, h, ...]，无人脸传 None / Main face box or None
        frame_h: 帧高度（sensor 坐标系）/ Frame height in sensor coords
        ts_ms: 当前毫秒时间戳 / Current millisecond timestamp

        返回: {"state":..., "trend":..., "ratio":...} 或 None（无人脸）
        """
        if det is None:
            # 人脸丢失即清空，输出 null，不保留旧状态 / Reset on face loss
            self.ratios = []
            self.history = []
            self.state = None
            self.trend = "stable"
            return None

        # 滑动平均 / Moving average
        r = float(det[3]) / frame_h
        self.ratios.append(r)
        if len(self.ratios) > config.PROX_WINDOW:
            self.ratios.pop(0)
        avg = sum(self.ratios) / len(self.ratios)

        self.state = self._update_state(avg)
        self.trend = self._update_trend(ts_ms, avg)
        return {"state": self.state, "trend": self.trend, "ratio": round(avg, 3)}

    def _update_state(self, avg):
        """双阈值迟滞：进入阈值与退出阈值分离，防止边界抖动
        Dual-threshold hysteresis: separate enter/exit thresholds to avoid flapping."""
        if self.state == "near":
            if avg < config.PROX_NEAR_EXIT:
                return "mid"
            return "near"
        if self.state == "far":
            if avg > config.PROX_FAR_EXIT:
                return "mid"
            return "far"
        # 当前为 mid 或初始 / Currently mid or initial
        if avg >= config.PROX_NEAR_ENTER:
            return "near"
        if avg <= config.PROX_FAR_ENTER:
            return "far"
        return "mid"

    def _update_trend(self, ts_ms, avg):
        """与 PROX_TREND_MS 前的均值比较，相对变化超阈值判定接近/远离
        Compare with the average PROX_TREND_MS ago; large relative change => trend."""
        # 记录历史并裁剪到 2 倍趋势窗口 / Record history, trim to 2x trend window
        self.history.append((ts_ms, avg))
        while self.history and ts_ms - self.history[0][0] > config.PROX_TREND_MS * 2:
            self.history.pop(0)

        # 窗口未集满时不判趋势，避免起步误报 / Skip trend until window is filled
        if len(self.ratios) < config.PROX_WINDOW:
            return "stable"

        # 找最接近"1 个趋势间隔前"的参考样本 / Find sample closest to one trend-interval ago
        target = ts_ms - config.PROX_TREND_MS
        ref = None
        for ts, val in self.history:
            if ts <= target:
                ref = val
            else:
                break
        if ref is None or ref <= 0:
            return "stable"

        change = (avg - ref) / ref
        if change > config.PROX_TREND_CHANGE:
            return "approaching"
        if change < -config.PROX_TREND_CHANGE:
            return "receding"
        return "stable"
