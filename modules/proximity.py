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
        self.change = 0.0     # 趋势窗口内的相对变化量 / Relative change over trend window

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
            self.change = 0.0
            return None

        # 滑动平均 / Moving average
        r = float(det[3]) / frame_h
        self.ratios.append(r)
        if len(self.ratios) > config.PROX_WINDOW:
            self.ratios.pop(0)
        avg = sum(self.ratios) / len(self.ratios)

        self.state = self._update_state(avg)
        self.trend = self._update_trend(ts_ms, avg)
        return {"state": self.state, "trend": self.trend,
                "ratio": round(avg, 3), "change": round(self.change, 3)}

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
        """双半窗均值比较：把趋势窗口对半分，比较后半均值与前半均值的相对变化。
        有持续变化才输出 approaching/receding，静止或小幅波动一律 stable（不做判断）。
        Two-half-window comparison: split the trend window in half, compare the
        relative change between the latter-half mean and the former-half mean.
        Only sustained change yields approaching/receding; stillness stays stable."""
        # 记录历史并裁剪到趋势窗口 / Record history, trim to trend window
        self.history.append((ts_ms, avg))
        while self.history and ts_ms - self.history[0][0] > config.PROX_TREND_WINDOW_MS:
            self.history.pop(0)

        # 数据跨度不足时不判趋势 / Not enough time span => no judgment
        if len(self.history) < 4 or (ts_ms - self.history[0][0]) < config.PROX_TREND_MIN_SPAN_MS:
            self.change = 0.0
            return "stable"

        # 按时间中点分成前后两半 / Split into former/latter halves by time midpoint
        mid_ts = (self.history[0][0] + ts_ms) / 2
        old_vals = []
        new_vals = []
        for ts, val in self.history:
            if ts <= mid_ts:
                old_vals.append(val)
            else:
                new_vals.append(val)
        if not old_vals or not new_vals:
            self.change = 0.0
            return "stable"

        old_mean = sum(old_vals) / len(old_vals)
        new_mean = sum(new_vals) / len(new_vals)
        if old_mean <= 0:
            self.change = 0.0
            return "stable"

        self.change = (new_mean - old_mean) / old_mean
        if self.change > config.PROX_TREND_CHANGE:
            return "approaching"
        if self.change < -config.PROX_TREND_CHANGE:
            return "receding"
        return "stable"
