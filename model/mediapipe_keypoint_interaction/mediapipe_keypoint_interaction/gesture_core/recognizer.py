from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from .custom_gesture import CustomGestureDB
from .dynamic import DynamicGestureDetector
from .features import build_feature_vector, extract_multi10
from .gesture_rules import rule_based_gesture


@dataclass
class Recognition:
    label: str
    confidence: float
    source: str
    multi10: Optional[np.ndarray]
    feature: Optional[np.ndarray]
    finger_states: Optional[Dict[str, int]]
    custom_distance: float = 999.0


class GestureRecognizer:
    """MediaPipe keypoint gesture recognizer.

    Priority:
    dynamic gesture > custom keypoint gesture > built-in rule gesture.
    """

    def __init__(
        self,
        custom_db_path="data/custom_gestures.jsonl",
        custom_threshold=0.56,
        enable_dynamic=True,
    ):
        self.custom_db = CustomGestureDB(custom_db_path)
        self.custom_threshold = custom_threshold
        self.dynamic = DynamicGestureDetector()
        self.enable_dynamic = enable_dynamic

    def reload_custom(self):
        self.custom_db.load()

    def recognize(self, hand_info, face_results=None) -> Recognition:
        if hand_info is None:
            self.dynamic.update(None)
            return Recognition(
                label="unknown",
                confidence=0.0,
                source="none",
                multi10=None,
                feature=None,
                finger_states=None,
            )

        multi10 = extract_multi10(hand_info.landmarks, hand_info.handedness, face_results)
        feature = build_feature_vector(hand_info.landmarks, hand_info.handedness, face_results)

        rule_label, rule_conf, states = rule_based_gesture(
            hand_info.landmarks,
            hand_info.handedness,
            multi10,
        )

        if self.enable_dynamic:
            dyn_label, dyn_conf = self.dynamic.update(multi10)
            if dyn_label and dyn_conf >= 0.70:
                return Recognition(dyn_label, dyn_conf, "dynamic", multi10, feature, states)

        custom_label, custom_conf, custom_dist = self.custom_db.predict(
            feature,
            threshold=self.custom_threshold,
        )
        if custom_label is not None and custom_conf >= max(self.custom_threshold, rule_conf + 0.03):
            return Recognition(custom_label, custom_conf, "custom", multi10, feature, states, custom_dist)

        return Recognition(rule_label, rule_conf, "rule", multi10, feature, states)
