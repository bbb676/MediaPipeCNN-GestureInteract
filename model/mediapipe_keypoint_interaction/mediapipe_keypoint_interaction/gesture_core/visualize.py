import time
from typing import Dict, Optional

import cv2
import numpy as np

from .features import MULTI10_NAMES
from .gesture_rules import display_name


class FpsMeter:
    def __init__(self):
        self.t0 = time.time()
        self.frames = 0
        self.fps = 0.0

    def update(self):
        self.frames += 1
        now = time.time()
        if now - self.t0 >= 0.5:
            self.fps = self.frames / (now - self.t0)
            self.frames = 0
            self.t0 = now
        return self.fps


def draw_bbox_label(frame, hand_info, label: str, conf: float, source: str):
    if hand_info is None:
        return
    x1, y1, x2, y2 = hand_info.bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    text = f"{label}/{display_name(label)} {conf:.2f} [{source}]"
    cv2.putText(frame, text, (x1, max(25, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 0), 2)


def draw_top_info(frame, fps, label, conf, source, mode, action_enabled, face_enabled, action_text=""):
    y = 28
    rows = [
        f"FPS: {fps:.1f}",
        f"Gesture: {label} / {display_name(label)}  conf={conf:.2f}  source={source}",
        f"Mode: {mode}   Actions: {'ON' if action_enabled else 'OFF'}   Face10D: {'ON' if face_enabled else 'OFF'}",
    ]
    if action_text:
        rows.append(f"Action: {action_text}")

    for row in rows:
        cv2.putText(frame, row, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
        y += 28


def draw_multi10(frame, multi10: Optional[np.ndarray], x=10, y=205):
    if multi10 is None:
        cv2.putText(frame, "10D: no hand", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 180, 180), 1)
        return

    cv2.putText(frame, "10D multimodal vector:", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 0), 1)
    y += 22
    for i, (name, value) in enumerate(zip(MULTI10_NAMES, multi10)):
        text = f"{i}:{name}={float(value):.2f}"
        cv2.putText(frame, text, (x, y + i * 18), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (220, 220, 220), 1)


def draw_help(frame, show=True):
    if not show:
        return
    h, w = frame.shape[:2]
    lines = [
        "Keys: q quit | s screenshot | a actions on/off | d debug text | f FaceMesh on/off | h help",
        "Built-in gestures: fist palm point ok thumbs_up victory | dynamic: wave circle",
        "Record custom: python main.py --record your_label --samples 80 --no-action",
    ]
    y = h - 68
    for line in lines:
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (230, 230, 230), 1)
        y += 20
