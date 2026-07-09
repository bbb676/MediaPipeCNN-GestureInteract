import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .features import index_tip_pixel


class WhiteboardApp:
    """Virtual whiteboard driven by MediaPipe keypoint gestures."""

    def __init__(self, save_dir="captures"):
        self.canvas = None
        self.prev_point = None
        self.colors = [
            (0, 0, 255),
            (0, 255, 0),
            (255, 0, 0),
            (0, 255, 255),
            (255, 0, 255),
            (255, 255, 255),
        ]
        self.color_idx = 0
        self.last_switch = 0.0
        self.last_save = 0.0
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    @property
    def current_color(self):
        return self.colors[self.color_idx % len(self.colors)]

    def update(self, frame, gesture: str, hand_info) -> str:
        h, w = frame.shape[:2]
        if self.canvas is None or self.canvas.shape[:2] != (h, w):
            self.canvas = np.zeros_like(frame)

        action_text = ""

        if hand_info is None:
            self.prev_point = None
            return self.render(frame, action_text)

        pt = index_tip_pixel(hand_info.landmarks, w, h)

        if gesture == "point":
            cv2.circle(frame, pt, 8, self.current_color, -1)
            if self.prev_point is not None:
                cv2.line(self.canvas, self.prev_point, pt, self.current_color, 8, cv2.LINE_AA)
            self.prev_point = pt
            action_text = "Drawing"

        elif gesture == "fist":
            cv2.circle(self.canvas, pt, 32, (0, 0, 0), -1)
            cv2.circle(frame, pt, 32, (80, 80, 80), 2)
            self.prev_point = None
            action_text = "Eraser"

        elif gesture == "ok":
            self.canvas[:] = 0
            self.prev_point = None
            action_text = "Clear canvas"

        elif gesture == "victory":
            self.prev_point = None
            now = time.time()
            if now - self.last_switch > 0.8:
                self.color_idx = (self.color_idx + 1) % len(self.colors)
                self.last_switch = now
            action_text = "Switch color"

        elif gesture == "thumbs_up":
            self.prev_point = None
            now = time.time()
            if now - self.last_save > 1.2:
                out = self.save_dir / f"whiteboard_{int(now)}.png"
                cv2.imwrite(str(out), self.render(frame.copy(), "Saved"))
                self.last_save = now
                action_text = f"Saved {out.name}"

        else:
            self.prev_point = None

        return self.render(frame, action_text)

    def render(self, frame, action_text=""):
        if self.canvas is None:
            return frame

        mask = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        mask = mask > 0
        out = frame.copy()
        out[mask] = cv2.addWeighted(frame, 0.35, self.canvas, 0.95, 0)[mask]

        cv2.putText(out, f"Whiteboard: {action_text}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.rectangle(out, (10, 135), (60, 165), self.current_color, -1)
        cv2.putText(out, "point=draw fist=erase ok=clear victory=color thumbs_up=save",
                    (70, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (230, 230, 230), 1)
        return out
