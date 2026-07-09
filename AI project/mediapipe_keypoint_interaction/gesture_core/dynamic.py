import math
import time
from collections import deque
from typing import Optional, Tuple

import numpy as np


class DynamicGestureDetector:
    """Very lightweight sequence recognizer for wave/circle.

    It only uses keypoint-derived hand center, not image CNN features.
    """

    def __init__(self, maxlen=24):
        self.history = deque(maxlen=maxlen)
        self.last_trigger = {}
        self.cooldown = 1.0

    def reset(self):
        self.history.clear()

    def update(self, multi10: Optional[np.ndarray]) -> Tuple[Optional[str], float]:
        now = time.time()
        if multi10 is None:
            self.history.clear()
            return None, 0.0

        cx = float(multi10[6])
        cy = float(multi10[7])
        self.history.append((now, cx, cy))

        if len(self.history) < 10:
            return None, 0.0

        label, conf = self._detect_wave(now)
        if label:
            return label, conf

        label, conf = self._detect_circle(now)
        if label:
            return label, conf

        return None, 0.0

    def _cooldown_ok(self, label, now):
        if now - self.last_trigger.get(label, 0) >= self.cooldown:
            self.last_trigger[label] = now
            return True
        return False

    def _recent_points(self, seconds=1.2):
        now = time.time()
        pts = [(t, x, y) for t, x, y in self.history if now - t <= seconds]
        return pts

    def _detect_wave(self, now) -> Tuple[Optional[str], float]:
        pts = self._recent_points(1.2)
        if len(pts) < 10:
            return None, 0.0
        xs = np.array([p[1] for p in pts], dtype=np.float32)
        ys = np.array([p[2] for p in pts], dtype=np.float32)
        amp_x = float(xs.max() - xs.min())
        amp_y = float(ys.max() - ys.min())

        vx = np.diff(xs)
        signs = np.sign(vx)
        signs = signs[np.abs(vx) > 0.006]
        changes = int(np.sum(signs[1:] * signs[:-1] < 0)) if len(signs) >= 2 else 0

        if amp_x > 0.18 and amp_y < 0.18 and changes >= 2:
            if self._cooldown_ok("wave", now):
                return "wave", min(0.95, 0.55 + amp_x + changes * 0.08)
        return None, 0.0

    def _detect_circle(self, now) -> Tuple[Optional[str], float]:
        pts = self._recent_points(1.6)
        if len(pts) < 14:
            return None, 0.0
        arr = np.array([[p[1], p[2]] for p in pts], dtype=np.float32)
        center = arr.mean(axis=0)
        vec = arr - center
        radii = np.linalg.norm(vec, axis=1)
        if float(radii.mean()) < 0.06:
            return None, 0.0

        angles = np.unwrap(np.arctan2(vec[:, 1], vec[:, 0]))
        coverage = abs(float(angles[-1] - angles[0]))
        path = float(np.sum(np.linalg.norm(np.diff(arr, axis=0), axis=1)))
        straight = float(np.linalg.norm(arr[-1] - arr[0])) + 1e-6
        circularity = path / straight

        if coverage > math.pi * 1.55 and circularity > 3.0:
            if self._cooldown_ok("circle", now):
                return "circle", min(0.95, 0.55 + coverage / (2 * math.pi) * 0.35)
        return None, 0.0
