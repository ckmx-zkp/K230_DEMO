"""可标定的表情几何原型。只描述外观，不推断真实情绪。"""
import math
import os
import time
import ujson
import config

LABELS = ('neutral', 'happy', 'angry', 'sad', 'surprise')
GROUPS = ((35,36,33,37,39,42,40,41), (89,90,87,91,93,96,94,95),
          (43,44,45,47,46,50,51,49,48), (97,98,99,100,101,105,104,103,102),
          (52,55,56,53,59,58,61,68,67,71,63,64), (65,54,60,57,69,70,62,66))


def features(points):
    if len(points) != 212:
        raise ValueError('Expected 106 landmark points')
    p = [(float(points[i]), float(points[i+1])) for i in range(0, 212, 2)]
    if not all(math.isfinite(v) for xy in p for v in xy):
        raise ValueError('Invalid landmarks')
    a, b = p[34], p[88]
    dx, dy = b[0]-a[0], b[1]-a[1]
    scale = math.sqrt(dx*dx+dy*dy)
    if scale < 15:
        raise ValueError('Face too small')
    ux, uy = dx/scale, dy/scale
    q = [(((x-a[0])*ux+(y-a[1])*uy)/scale,
          (-(x-a[0])*uy+(y-a[1])*ux)/scale) for x,y in p]
    result = []
    for group in GROUPS:
        xs = [q[i][0] for i in group]
        ys = [q[i][1] for i in group]
        result.extend((max(xs)-min(xs), max(ys)-min(ys), sum(ys)/len(ys)))
    return result


def _exists(path):
    try:
        os.stat(path)
        return True
    except OSError as error:
        if error.args[0] != 2:
            raise
        return False


def _recover_profile(path):
    # Recover an interrupted replacement before reading or writing profiles.
    if not _exists(path) and _exists(path + '.bak'):
        os.rename(path + '.bak', path)


def save_profiles(profiles):
    path = config.EXPRESSION_PROFILE_PATH
    temp, backup = path + '.tmp', path + '.bak'
    _recover_profile(path)
    payload = ujson.dumps({'version': 1, 'profiles': profiles})
    with open(temp, 'w') as f:
        f.write(payload)
    with open(temp) as f:
        if f.read() != payload:
            raise OSError('Profile readback failed')
    # FAT on the board rejects rename(source, existing_destination).
    # Keep the previous file as a backup until the new file is installed.
    if _exists(backup):
        os.remove(backup)
    had_previous = _exists(path)
    if had_previous:
        os.rename(path, backup)
    try:
        os.rename(temp, path)
    except OSError:
        if had_previous:
            os.rename(backup, path)
        raise


class Expression:
    def __init__(self):
        self.profiles = {}
        self.pending = None
        self.samples = []
        self.candidate = None
        self.hits = 0
        self.confirmed = 'unknown'
        try:
            _recover_profile(config.EXPRESSION_PROFILE_PATH)
            with open(config.EXPRESSION_PROFILE_PATH) as f:
                data = ujson.loads(f.read())
            if data.get('version') == 1:
                for label, vector in data['profiles'].items():
                    if label in LABELS and len(vector) == 18 and all(math.isfinite(float(v)) for v in vector):
                        self.profiles[label] = vector
        except Exception:
            pass

    def reset(self):
        self.candidate = None
        self.hits = 0
        self.confirmed = 'unknown'
        self.samples = []

    def poll_request(self):
        try:
            with open(config.EXPRESSION_REQUEST_PATH) as f:
                cmd = ujson.loads(f.read())
            os.remove(config.EXPRESSION_REQUEST_PATH)
        except OSError:
            return
        except Exception as e:
            print('Expression request invalid:', e)
            return
        label = cmd.get('label')
        if label not in LABELS:
            print('Expression invalid label:', label)
            return
        self.reset()
        self.pending = label
        self.start_ms = time.ticks_ms()
        print('Expression calibration: hold', label, 'after 3 seconds')

    def update(self, points, box, pose, count):
        self.poll_request()
        if self.pending and time.ticks_diff(time.ticks_ms(), self.start_ms) > 30000:
            self.pending = None
            self.reset()
            return {'label':'unknown', 'reason':'calibration_timeout'}
        w,h = config.RGB888P_SIZE
        reason = None
        if count != 1 or box is None:
            reason = 'need_single_face'
        elif box[0] <= 1 or box[1] <= 1 or box[0]+box[2] >= w-1 or box[1]+box[3] >= h-1:
            reason = 'clipped_face'
        elif pose is None or abs(pose['yaw']) > 25 or abs(pose['pitch']) > 25:
            reason = 'face_camera'
        if reason:
            self.reset()
            return {'label':'unknown', 'reason':reason, 'method':'landmark_calibrated'}
        try:
            vector = features(points)
            if any(points[i] < 0 or points[i] >= w or points[i+1] < 0 or points[i+1] >= h
                   for i in range(0, len(points), 2)):
                raise ValueError('landmarks_outside_frame')
        except Exception as e:
            self.reset()
            return {'label':'unknown', 'reason':str(e), 'method':'landmark_calibrated'}
        if self.pending:
            elapsed = time.ticks_diff(time.ticks_ms(), self.start_ms)
            if elapsed > 30000:
                self.pending = None
                self.samples = []
                return {'label':'unknown', 'reason':'calibration_timeout'}
            if elapsed >= 3000:
                self.samples.append(vector)
            n = len(self.samples)
            if n >= 15:
                updated = dict(self.profiles)
                updated[self.pending] = [sum(v[i] for v in self.samples)/n for i in range(18)]
                save_profiles(updated)
                self.profiles = updated
                print('Expression calibration saved:', self.pending)
                self.pending = None
                self.reset()
            return {'label':'unknown', 'reason':'calibrating', 'samples':n}
        if len(self.profiles) != len(LABELS):
            return {'label':'unknown', 'reason':'needs_calibration', 'calibrated':list(self.profiles.keys())}
        scores = sorted((math.sqrt(sum((a-b)**2 for a,b in zip(vector, ref))/18), label)
                        for label,ref in self.profiles.items())
        distance, label = scores[0]
        margin = scores[1][0] - distance
        if distance > config.EXPRESSION_MAX_DISTANCE or margin < config.EXPRESSION_MIN_MARGIN:
            label = 'unknown'
        if label == self.candidate:
            self.hits += 1
        else:
            self.candidate, self.hits = label, 1
        # 不确定时立即清除旧标签，明确的表情连续确认后再输出。
        if label == 'unknown' or self.hits >= 3:
            self.confirmed = label
        return {'label':self.confirmed, 'distance':round(distance,4),
                'margin':round(margin,4), 'method':'landmark_calibrated'}
