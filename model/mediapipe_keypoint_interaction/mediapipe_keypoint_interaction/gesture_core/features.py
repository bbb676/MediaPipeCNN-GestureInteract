import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


MULTI10_NAMES = [
    "thumb_open",
    "index_open",
    "middle_open",
    "ring_open",
    "pinky_open",
    "pinch",
    "hand_cx",
    "hand_cy",
    "face_yaw",
    "face_pitch",
]


@dataclass
class HandInfo:
    landmarks: np.ndarray       # shape: (21, 3), normalized by image size from MediaPipe
    handedness: str             # "Left" or "Right"
    score: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2


def landmarks_to_np(hand_landmarks) -> np.ndarray:
    return np.array(
        [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
        dtype=np.float32,
    )


def hand_bbox(landmarks: np.ndarray, frame_w: int, frame_h: int, margin: int = 20):
    xs = landmarks[:, 0] * frame_w
    ys = landmarks[:, 1] * frame_h
    x1 = max(0, int(xs.min()) - margin)
    y1 = max(0, int(ys.min()) - margin)
    x2 = min(frame_w - 1, int(xs.max()) + margin)
    y2 = min(frame_h - 1, int(ys.max()) + margin)
    return x1, y1, x2, y2


def parse_hands(hand_results, frame_w: int, frame_h: int) -> List[HandInfo]:
    hands = []
    if not hand_results or not hand_results.multi_hand_landmarks:
        return hands

    handedness_list = hand_results.multi_handedness or []
    for i, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
        label = "Right"
        score = 1.0
        if i < len(handedness_list):
            cls = handedness_list[i].classification[0]
            label = cls.label
            score = float(cls.score)

        lm = landmarks_to_np(hand_landmarks)
        bbox = hand_bbox(lm, frame_w, frame_h)
        hands.append(HandInfo(lm, label, score, bbox))
    return hands


def select_primary_hand(hands: List[HandInfo]) -> Optional[HandInfo]:
    """Choose the largest detected hand as the active one."""
    if not hands:
        return None

    def area(h: HandInfo):
        x1, y1, x2, y2 = h.bbox
        return max(1, x2 - x1) * max(1, y2 - y1)

    return max(hands, key=area)


def normalize_keypoints(landmarks: np.ndarray, handedness: str) -> np.ndarray:
    """Normalize 21 keypoints to a translation/scale invariant 63D vector.

    Left hand is mirrored so that left/right hands are compared in the same
    coordinate convention.
    """
    pts = landmarks.astype(np.float32).copy()
    pts = pts - pts[0:1]  # wrist as origin

    if handedness.lower() == "left":
        pts[:, 0] *= -1.0

    scale = np.max(np.linalg.norm(pts[:, :2], axis=1))
    if scale < 1e-6:
        scale = 1.0
    pts = pts / scale
    return pts.reshape(-1).astype(np.float32)


def dist(a, b) -> float:
    return float(np.linalg.norm(a - b))


def angle_deg(a, b, c) -> float:
    """Angle ABC in degrees."""
    ba = a - b
    bc = c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-8
    cosv = float(np.dot(ba, bc) / denom)
    cosv = max(-1.0, min(1.0, cosv))
    return math.degrees(math.acos(cosv))


def get_finger_states(landmarks: np.ndarray, handedness: str) -> Dict[str, int]:
    """Return five binary finger states.

    It combines angle and distance rules. This is more robust than comparing
    only x/y coordinates.
    """
    lm = landmarks
    wrist = lm[0]

    # Thumb: CMC(1)-MCP(2)-IP(3)-TIP(4)
    thumb_angle = angle_deg(lm[2], lm[3], lm[4])
    thumb_len = dist(lm[4], wrist) / (dist(lm[2], wrist) + 1e-6)
    thumb_open = int(thumb_angle > 145 and thumb_len > 1.03)

    states = {"thumb": thumb_open}

    specs = {
        "index": (5, 6, 7, 8),
        "middle": (9, 10, 11, 12),
        "ring": (13, 14, 15, 16),
        "pinky": (17, 18, 19, 20),
    }
    for name, (mcp, pip, dip, tip) in specs.items():
        a1 = angle_deg(lm[mcp], lm[pip], lm[dip])
        a2 = angle_deg(lm[pip], lm[dip], lm[tip])
        far = dist(lm[tip], wrist) > dist(lm[pip], wrist) * 1.03
        # y condition helps distinguish bent fingers when the hand is roughly upright.
        y_ok = lm[tip, 1] < lm[pip, 1] + 0.025
        states[name] = int(a1 > 145 and a2 > 145 and far and y_ok)

    return states


def face_yaw_pitch(face_results) -> Tuple[float, float]:
    """Return lightweight yaw/pitch proxy from FaceMesh.

    Values are clipped into [-1, 1]. No solvePnP is used, so it is fast and
    has no extra dependency.
    """
    if not face_results or not face_results.multi_face_landmarks:
        return 0.0, 0.0

    lm = face_results.multi_face_landmarks[0].landmark

    # FaceMesh indices:
    # 1 nose tip, 33 left eye outer, 263 right eye outer, 10 forehead, 152 chin.
    nose = np.array([lm[1].x, lm[1].y], dtype=np.float32)
    left_eye = np.array([lm[33].x, lm[33].y], dtype=np.float32)
    right_eye = np.array([lm[263].x, lm[263].y], dtype=np.float32)
    forehead = np.array([lm[10].x, lm[10].y], dtype=np.float32)
    chin = np.array([lm[152].x, lm[152].y], dtype=np.float32)

    eye_center = (left_eye + right_eye) / 2.0
    eye_dist = max(1e-6, float(np.linalg.norm(left_eye - right_eye)))
    face_height = max(1e-6, float(np.linalg.norm(chin - forehead)))

    # Positive yaw means face/nose moves to user's right side in the image.
    yaw = float((nose[0] - eye_center[0]) / eye_dist)
    pitch = float((nose[1] - eye_center[1]) / face_height - 0.18)

    return float(np.clip(yaw * 2.2, -1.0, 1.0)), float(np.clip(pitch * 3.0, -1.0, 1.0))


def extract_multi10(landmarks: np.ndarray, handedness: str, face_results=None) -> np.ndarray:
    """Build a 10D multimodal vector: hand state + hand position + face pose."""
    states = get_finger_states(landmarks, handedness)
    palm_size = dist(landmarks[0], landmarks[9]) + 1e-6
    pinch = dist(landmarks[4], landmarks[8]) / palm_size
    pinch = float(np.clip(pinch, 0.0, 2.0) / 2.0)

    center = landmarks[:, :2].mean(axis=0)
    yaw, pitch = face_yaw_pitch(face_results)

    return np.array(
        [
            states["thumb"],
            states["index"],
            states["middle"],
            states["ring"],
            states["pinky"],
            pinch,
            float(np.clip(center[0], 0.0, 1.0)),
            float(np.clip(center[1], 0.0, 1.0)),
            yaw,
            pitch,
        ],
        dtype=np.float32,
    )


def build_feature_vector(landmarks: np.ndarray, handedness: str, face_results=None) -> np.ndarray:
    """73D feature = 63D normalized keypoints + 10D multimodal vector."""
    return np.concatenate(
        [normalize_keypoints(landmarks, handedness), extract_multi10(landmarks, handedness, face_results)],
        axis=0,
    ).astype(np.float32)


def index_tip_pixel(landmarks: np.ndarray, frame_w: int, frame_h: int) -> Tuple[int, int]:
    return int(landmarks[8, 0] * frame_w), int(landmarks[8, 1] * frame_h)
