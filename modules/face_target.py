"""按框重叠保持目标连续性；不是身份识别。"""
import math


def overlap(a, b):
    area = max(0, min(a[0]+a[2], b[0]+b[2])-max(a[0], b[0])) * max(0, min(a[1]+a[3], b[1]+b[3])-max(a[1], b[1]))
    return area / max(1, a[2]*a[3] + b[2]*b[3] - area)


class FaceTarget:
    def __init__(self):
        self.box = None
        self.target_id = 0

    def update(self, boxes, width, height):
        valid = []
        for row in boxes:
            b = [float(v) for v in row[:4]]
            if not all(math.isfinite(v) for v in b) or b[2] <= 0 or b[3] <= 0:
                continue
            x, y = max(0, b[0]), max(0, b[1])
            w, h = min(width, b[0]+b[2])-x, min(height, b[1]+b[3])-y
            if w > 0 and h > 0:
                valid.append([x, y, w, h])
        if not valid:
            self.box = None
            return None, False
        best = max(valid, key=lambda b: overlap(self.box, b)) if self.box else None
        changed = best is None or overlap(self.box, best) < 0.2
        if changed:
            best = max(valid, key=lambda b: b[2]*b[3])
            self.target_id += 1
        self.box = best
        return best, changed

    def snapshot(self, width, height):
        if self.box is None:
            return None
        x, y, w, h = self.box
        cx, cy = x+w/2, y+h/2
        return {'id': self.target_id, 'cx': round(cx, 1), 'cy': round(cy, 1),
                'dx': round((cx-width/2)/(width/2), 3),
                'dy': round((cy-height/2)/(height/2), 3),
                'clipped': x <= 1 or y <= 1 or x+w >= width-1 or y+h >= height-1}
