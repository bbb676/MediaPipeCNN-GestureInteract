import time
from collections import Counter, deque
from typing import Dict, Optional, Tuple


# Face-mode PPT interaction:
# Face pose switches the current mode. Hand gesture only executes actions inside
# the active mode.  This avoids the old problem where face turning directly
# flips slides and feels insensitive/unstable.
#
# multi10 format from features.py:
# 0 thumb_open, 1 index_open, 2 middle_open, 3 ring_open, 4 pinky_open,
# 5 pinch, 6 hand_cx, 7 hand_cy, 8 face_yaw, 9 face_pitch

# In PowerPoint/WPS slideshow mode, these keys are usually supported.
# If one shortcut does not work in WPS, only change the key below; the face/hand
# mode logic does not need to change.
PPT_MODE_ACTIONS: Dict[str, Dict[str, Tuple[str, str, str]]] = {
    # Default mode: normal page control.
    # Face: look straight / centered.
    "nav": {
        "point": ("press", "right", "Navigation: next slide"),
        "palm": ("press", "right", "Navigation: next slide"),
        "victory": ("press", "left", "Navigation: previous slide"),
        "fist": ("press", "space", "Navigation: pause / resume"),
        "ok": ("press", "esc", "Navigation: exit slideshow"),
        # thumbs_up is kept, but point/palm are safer backups because thumbs-up
        # is less stable under MediaPipe Hands when the thumb faces the camera.
        "thumbs_up": ("press", "f5", "Navigation: start slideshow"),
        "wave": ("press", "right", "Navigation: next slide by wave"),
        "circle": ("press", "left", "Navigation: previous slide by circle"),
    },

    # Annotation mode: use hand to switch pen/arrow, erase, blank screen.
    # Face: turn head to image-left side. If it feels reversed, see
    # FaceModeSwitcher._raw_mode_from_multi10() below and swap left/right modes.
    "annotate": {
        "point": ("hotkey", "ctrl+p", "Annotation: pen tool"),
        "ok": ("hotkey", "ctrl+a", "Annotation: arrow pointer"),
        "fist": ("press", "e", "Annotation: erase ink"),
        "palm": ("press", "b", "Annotation: black screen toggle"),
        "victory": ("press", "w", "Annotation: white screen toggle"),
        "thumbs_up": ("press", "f5", "Annotation: start slideshow"),
    },

    # Screen/focus mode: use hand to control attention effects.
    # Face: turn head to image-right side.
    "screen": {
        "palm": ("press", "b", "Screen: black screen toggle"),
        "fist": ("press", "w", "Screen: white screen toggle"),
        "point": ("press", "right", "Screen: next slide"),
        "victory": ("press", "left", "Screen: previous slide"),
        "ok": ("press", "esc", "Screen: exit slideshow"),
        "thumbs_up": ("press", "f5", "Screen: start slideshow"),
    },

    # Lock mode: ignore all hand gestures. Useful when explaining with hands.
    # Face: look down / large positive pitch.
    "lock": {},
}

MODE_CN = {
    "nav": "翻页模式",
    "annotate": "标注模式",
    "screen": "聚焦模式",
    "lock": "锁定模式",
}

MODE_HINT = {
    "nav": "正脸：point/palm下一页，victory上一页，fist暂停，ok退出",
    "annotate": "左转：point画笔，ok鼠标，fist擦除，palm黑屏，victory白屏",
    "screen": "右转：palm黑屏，fist白屏，point下一页，victory上一页",
    "lock": "低头：锁定，所有手势不触发",
}


class FaceModeSwitcher:
    """Switch interaction mode from face yaw/pitch in the 10D multimodal vector.

    This class only decides the mode; it never presses keys.  The actual PPT
    operation is still decided by the hand gesture in ActionController.
    """

    def __init__(
        self,
        yaw_threshold: float = 0.09,
        pitch_lock_threshold: float = 0.18,
        history_size: int = 3,
    ):
        # Smaller yaw_threshold = more sensitive. Suggested range: 0.07 - 0.14.
        self.yaw_threshold = yaw_threshold
        # pitch in current features.py is positive when the nose is lower in the
        # image, usually close to looking down.
        self.pitch_lock_threshold = pitch_lock_threshold
        # Smaller history_size makes mode switching faster and more sensitive.
        self.history = deque(maxlen=history_size)
        self.current_mode = "nav"
        self.last_mode = "nav"
        self.last_yaw = 0.0
        self.last_pitch = 0.0

    def update(self, multi10=None) -> Tuple[str, bool]:
        """Return (mode, changed)."""
        if multi10 is None:
            # Keep the previous mode if no hand/10D vector exists.
            return self.current_mode, False

        raw_mode = self._raw_mode_from_multi10(multi10)
        self.history.append(raw_mode)
        stable_mode = Counter(self.history).most_common(1)[0][0]

        changed = stable_mode != self.current_mode
        self.last_mode = self.current_mode
        self.current_mode = stable_mode
        return self.current_mode, changed

    def _raw_mode_from_multi10(self, multi10) -> str:
        yaw = float(multi10[8])
        pitch = float(multi10[9])
        self.last_yaw = yaw
        self.last_pitch = pitch

        # Priority: lock first, then yaw modes, then default nav.
        if pitch > self.pitch_lock_threshold:
            return "lock"
        if yaw < -self.yaw_threshold:
            return "annotate"
        if yaw > self.yaw_threshold:
            return "screen"
        return "nav"

    def mode_text(self) -> str:
        return f"FaceMode: {self.current_mode}/{MODE_CN.get(self.current_mode, self.current_mode)}"

    def debug_text(self) -> str:
        return (
            f"{self.mode_text()} | yaw={self.last_yaw:.2f} "
            f"pitch={self.last_pitch:.2f} | {MODE_HINT.get(self.current_mode, '')}"
        )


class ActionController:
    """Map face mode + recognized keypoint gestures to real PPT actions.

    Main rule:
        face = switch mode
        hand = execute action inside the selected mode
    """

    def __init__(self, mode="debug", enabled=True, cooldown=0.85):
        self.mode = mode
        self.enabled = enabled
        self.cooldown = cooldown
        self.last_time = 0.0
        self.last_action = ""
        self.face_mode = FaceModeSwitcher()

        # Prevent one held gesture from repeatedly firing too quickly.
        self.last_signature = None
        self.repeat_same_after = 2.4

    def toggle_enabled(self):
        self.enabled = not self.enabled
        return self.enabled

    def handle(self, gesture: str, multi10=None) -> str:
        if self.mode != "ppt":
            return ""

        active_mode, mode_changed = self.face_mode.update(multi10)

        # Unknown means no deliberate hand action.  It also resets repeat block,
        # so the next real gesture can trigger again.
        if gesture in (None, "", "unknown"):
            self.last_signature = None
            return self.face_mode.debug_text() if mode_changed else ""

        action = self._ppt_action(active_mode, gesture)
        if not action:
            # In lock mode or unsupported gesture, only show mode information.
            return self.face_mode.debug_text() if mode_changed or active_mode == "lock" else ""

        op, key, name = action
        now = time.time()
        signature = (active_mode, gesture, op, key)

        if now - self.last_time < self.cooldown:
            return self.last_action

        # Same gesture in the same mode only repeats after a longer delay.
        if signature == self.last_signature and now - self.last_time < self.repeat_same_after:
            return self.last_action

        self.last_time = now
        self.last_signature = signature
        self.last_action = f"{self.face_mode.mode_text()} | {name}"

        if self.enabled:
            self._perform(op, key)
        return self.last_action

    def _ppt_action(self, mode: str, gesture: str) -> Optional[Tuple[str, str, str]]:
        return PPT_MODE_ACTIONS.get(mode, {}).get(gesture)

    def _perform(self, op: str, key: str):
        try:
            import pyautogui

            pyautogui.PAUSE = 0.02
            if op == "press":
                pyautogui.press(key)
            elif op == "hotkey":
                keys = key.split("+")
                pyautogui.hotkey(*keys)
        except Exception as exc:
            # Keep the vision demo running even if PyAutoGUI is not available.
            print(f"[Action warning] Could not perform action {op}:{key}: {exc}")
