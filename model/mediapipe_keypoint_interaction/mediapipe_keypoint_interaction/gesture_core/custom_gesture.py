import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


class CustomGestureDB:
    """Simple keypoint custom gesture recognizer.

    Data format: JSON Lines:
    {"label": "my_gesture", "feature": [73 floats]}

    It uses KNN-style weighted voting. No CNN and no sklearn needed.
    """

    def __init__(self, path="data/custom_gestures.jsonl", k=5):
        self.path = Path(path)
        self.k = k
        self.samples: List[Tuple[str, np.ndarray]] = []
        self.labels: List[str] = []
        self.load()

    def load(self):
        self.samples.clear()
        self.labels.clear()
        if not self.path.exists():
            return
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                label = str(obj["label"])
                feat = np.asarray(obj["feature"], dtype=np.float32)
                if feat.ndim == 1 and feat.size >= 10:
                    self.samples.append((label, feat))
            except Exception:
                continue
        self.labels = sorted(set(label for label, _ in self.samples))

    def add_sample(self, label: str, feature: np.ndarray):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        label = str(label).strip()
        obj = {"label": label, "feature": [float(x) for x in feature.astype(float).tolist()]}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        self.samples.append((label, feature.astype(np.float32)))
        if label not in self.labels:
            self.labels.append(label)
            self.labels.sort()

    def count_by_label(self) -> Dict[str, int]:
        d = defaultdict(int)
        for label, _ in self.samples:
            d[label] += 1
        return dict(d)

    def predict(self, feature: np.ndarray, threshold: float = 0.56) -> Tuple[Optional[str], float, float]:
        """Return (label or None, confidence, normalized_distance)."""
        if not self.samples:
            return None, 0.0, 999.0

        x = feature.astype(np.float32)
        dists = []
        dim = max(1, x.size)
        for label, feat in self.samples:
            n = min(dim, feat.size)
            d = float(np.linalg.norm(x[:n] - feat[:n]) / math.sqrt(n))
            dists.append((d, label))

        dists.sort(key=lambda t: t[0])
        nearest = dists[: max(1, min(self.k, len(dists)))]

        votes = defaultdict(float)
        best_dist = nearest[0][0]
        for d, label in nearest:
            votes[label] += 1.0 / (d + 1e-4)

        best_label, best_vote = max(votes.items(), key=lambda kv: kv[1])
        total_vote = sum(votes.values()) + 1e-8

        # Confidence mixes local distance and voting purity.
        purity = best_vote / total_vote
        distance_score = 1.0 / (1.0 + best_dist * 4.0)
        confidence = float(0.65 * purity + 0.35 * distance_score)

        if confidence >= threshold:
            return best_label, confidence, best_dist
        return None, confidence, best_dist
