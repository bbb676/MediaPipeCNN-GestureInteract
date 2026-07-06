import cv2
import mediapipe as mp


class MPDetector:
    """MediaPipe Hands + optional FaceMesh detector.

    The interaction system uses MediaPipe keypoints only.
    No traditional CNN image classifier is used.
    """

    def __init__(
        self,
        max_num_hands=2,
        enable_face=True,
        min_detection_confidence=0.55,
        min_tracking_confidence=0.55,
    ):
        self.enable_face = enable_face
        self.mp_hands = mp.solutions.hands
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            model_complexity=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        self.face_mesh = None
        if enable_face:
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=False,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )

    def set_face_enabled(self, enabled: bool):
        """Turn FaceMesh on/off at runtime."""
        if enabled == self.enable_face:
            return
        self.enable_face = enabled
        if enabled and self.face_mesh is None:
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=False,
                min_detection_confidence=0.55,
                min_tracking_confidence=0.55,
            )
        if not enabled and self.face_mesh is not None:
            self.face_mesh.close()
            self.face_mesh = None

    def process(self, frame_bgr):
        """Return (hand_results, face_results) for one BGR frame."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        hand_results = self.hands.process(rgb)
        face_results = None
        if self.enable_face and self.face_mesh is not None:
            face_results = self.face_mesh.process(rgb)
        rgb.flags.writeable = True
        return hand_results, face_results

    def draw_hands(self, frame_bgr, hand_results):
        if not hand_results or not hand_results.multi_hand_landmarks:
            return frame_bgr

        for hand_landmarks in hand_results.multi_hand_landmarks:
            self.mp_draw.draw_landmarks(
                frame_bgr,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_styles.get_default_hand_landmarks_style(),
                self.mp_styles.get_default_hand_connections_style(),
            )
        return frame_bgr

    def close(self):
        self.hands.close()
        if self.face_mesh is not None:
            self.face_mesh.close()
