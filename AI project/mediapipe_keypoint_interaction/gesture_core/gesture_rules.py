from typing import Dict, Tuple

import numpy as np

from .features import get_finger_states, dist


GESTURE_CN = {
    "unknown": "未知",
    "fist": "拳头",
    "palm": "手掌",
    "point": "食指",
    "ok": "OK",
    "thumbs_up": "点赞",
    "victory": "剪刀",
    "wave": "挥手",
    "circle": "画圈",
}


def _state_tuple(states: Dict[str, int]):
    return (
        states["thumb"],
        states["index"],
        states["middle"],
        states["ring"],
        states["pinky"],
    )


def _folded_non_thumb(landmarks: np.ndarray) -> Dict[str, bool]:
    """Judge whether non-thumb fingers are folded.

    Thumbs-up is hard for MediaPipe rule recognition because the folded fingers
    are often occluded.  This helper uses looser voting from both y-position and
    distance to wrist, instead of relying only on get_finger_states().
    """
    lm = landmarks
    wrist = lm[0]
    specs = {
        "index": (5, 6, 7, 8),
        "middle": (9, 10, 11, 12),
        "ring": (13, 14, 15, 16),
        "pinky": (17, 18, 19, 20),
    }
    folded = {}
    for name, (mcp, pip, dip, tip) in specs.items():
        folded_by_y = lm[tip, 1] > lm[pip, 1] - 0.015
        folded_by_dist = dist(lm[tip], wrist) < dist(lm[pip], wrist) * 1.10
        folded[name] = bool(folded_by_y or folded_by_dist)
    return folded


def _looks_like_thumbs_up(landmarks: np.ndarray, palm_size: float) -> bool:
    """A more tolerant thumbs-up detector from MediaPipe hand keypoints only."""
    lm = landmarks
    folded = _folded_non_thumb(lm)
    folded_count = sum(int(v) for v in folded.values())

    thumb_tip = lm[4]
    thumb_ip = lm[3]
    index_mcp = lm[5]
    middle_mcp = lm[9]

    thumb_is_high = (
        thumb_tip[1] < thumb_ip[1] - 0.018
        and thumb_tip[1] < index_mcp[1] - 0.035
    )

    thumb_extended = (
        dist(lm[4], lm[2]) / palm_size > 0.30
        or dist(lm[4], lm[0]) > dist(lm[3], lm[0]) * 1.04
    )

    vertical_motion = abs(float(thumb_tip[1] - middle_mcp[1]))
    horizontal_motion = abs(float(thumb_tip[0] - middle_mcp[0]))
    thumb_more_vertical = vertical_motion > horizontal_motion * 0.45

    return folded_count >= 3 and thumb_is_high and thumb_extended and thumb_more_vertical


def rule_based_gesture(landmarks: np.ndarray, handedness: str, multi10: np.ndarray) -> Tuple[str, float, Dict[str, int]]:
    """Static gesture recognition from MediaPipe keypoints.

    Returns (label, confidence, finger_states).
    """
    states = get_finger_states(landmarks, handedness)
    thumb, index, middle, ring, pinky = _state_tuple(states)
    open_count = thumb + index + middle + ring + pinky

    palm_size = dist(landmarks[0], landmarks[9]) + 1e-6
    pinch_raw = dist(landmarks[4], landmarks[8]) / palm_size

    # OK: thumb tip close to index tip, other fingers are mostly open.
    if pinch_raw < 0.38 and (middle + ring + pinky) >= 2:
        return "ok", 0.88, states

    # Check thumbs-up before fist.  get_finger_states() may mark the thumb as
    # closed when the thumb is vertical to the camera.
    if _looks_like_thumbs_up(landmarks, palm_size):
        return "thumbs_up", 0.86, states

    # Fist: all or almost all fingers folded.
    if open_count == 0 or (open_count == 1 and thumb == 1):
        return "fist", 0.86, states

    # Palm: four or five fingers open.
    if open_count >= 4:
        return "palm", 0.84, states

    # Victory / scissors: index and middle open, ring/pinky folded.
    if index and middle and not ring and not pinky:
        return "victory", 0.86, states

    # Point: only index is open.
    if index and not middle and not ring and not pinky:
        return "point", 0.86, states

    return "unknown", 0.20, states


def display_name(label: str) -> str:
    return GESTURE_CN.get(label, label)
