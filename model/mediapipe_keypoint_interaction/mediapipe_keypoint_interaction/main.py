import argparse
import time
from pathlib import Path

import cv2

from gesture_core.actions import ActionController
from gesture_core.features import parse_hands, select_primary_hand
from gesture_core.game import RPSGame
from gesture_core.gesture_rules import display_name
from gesture_core.mediapipe_engine import MPDetector
from gesture_core.recognizer import GestureRecognizer
from gesture_core.visualize import (
    FpsMeter,
    draw_bbox_label,
    draw_help,
    draw_multi10,
    draw_top_info,
)
from gesture_core.whiteboard import WhiteboardApp


def parse_args():
    p = argparse.ArgumentParser(
        description="MediaPipe keypoint gesture interaction system, no CNN."
    )
    p.add_argument("--camera", type=int, default=0, help="camera index, default 0")
    p.add_argument("--width", type=int, default=960, help="camera width")
    p.add_argument("--height", type=int, default=720, help="camera height")
    p.add_argument("--mode", choices=["debug", "ppt", "whiteboard", "game"], default="debug")
    p.add_argument("--no-action", action="store_true", help="disable real PyAutoGUI actions")
    p.add_argument("--no-face", action="store_true", help="disable FaceMesh 10D face pose")
    p.add_argument("--record", type=str, default="", help="record custom gesture label")
    p.add_argument("--samples", type=int, default=80, help="number of custom gesture samples")
    p.add_argument("--custom-db", type=str, default="data/custom_gestures.jsonl")
    p.add_argument("--custom-threshold", type=float, default=0.56)
    p.add_argument("--mirror", action="store_true", default=True, help="mirror camera frame")
    p.add_argument("--no-mirror", dest="mirror", action="store_false")
    return p.parse_args()


def open_camera(index, width, height):
    # CAP_DSHOW helps on Windows; ignored on many non-Windows systems.
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def save_frame(frame, prefix="capture"):
    out_dir = Path("captures")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{prefix}_{int(time.time())}.png"
    cv2.imwrite(str(path), frame)
    print(f"[Saved] {path}")


def main():
    args = parse_args()

    cap = open_camera(args.camera, args.width, args.height)
    if not cap.isOpened():
        print(f"Could not open camera {args.camera}. Try --camera 1.")
        return

    detector = MPDetector(enable_face=not args.no_face)
    recognizer = GestureRecognizer(
        custom_db_path=args.custom_db,
        custom_threshold=args.custom_threshold,
        enable_dynamic=True,
    )
    action_controller = ActionController(
        mode=args.mode,
        enabled=not args.no_action,
        cooldown=0.85,
    )
    whiteboard = WhiteboardApp()
    game = RPSGame()
    fps_meter = FpsMeter()

    show_debug = True
    show_help = True
    face_enabled = not args.no_face

    recording = bool(args.record)
    recorded = 0
    last_record_time = 0.0

    print("=" * 70)
    print("MediaPipe Keypoint Gesture Interaction System")
    print("No traditional CNN is used.")
    print(f"Mode: {args.mode}, real actions: {'OFF' if args.no_action else 'ON'}")
    if recording:
        print(f"Recording custom gesture: {args.record}, target samples: {args.samples}")
        print("Hold the gesture steadily in front of the camera.")
    print("Press q to quit.")
    print("=" * 70)

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Camera frame read failed.")
            break

        if args.mirror:
            frame = cv2.flip(frame, 1)

        h, w = frame.shape[:2]
        hand_results, face_results = detector.process(frame)
        hands = parse_hands(hand_results, w, h)
        primary = select_primary_hand(hands)
        rec = recognizer.recognize(primary, face_results if face_enabled else None)

        detector.draw_hands(frame, hand_results)
        draw_bbox_label(frame, primary, rec.label, rec.confidence, rec.source)

        action_text = ""

        if recording:
            now = time.time()
            if primary is not None and rec.feature is not None and now - last_record_time > 0.06:
                recognizer.custom_db.add_sample(args.record, rec.feature)
                recorded += 1
                last_record_time = now
                action_text = f"Recording {args.record}: {recorded}/{args.samples}"
                print(action_text)

            if recorded >= args.samples:
                print(f"[Done] Saved {recorded} samples to {args.custom_db}")
                recording = False
                recognizer.reload_custom()

        elif args.mode == "ppt":
            action_text = action_controller.handle(rec.label, rec.multi10)

        elif args.mode == "whiteboard":
            frame = whiteboard.update(frame, rec.label, primary)

        elif args.mode == "game":
            action_text = game.update(rec.label)
            cv2.putText(frame, action_text, (10, 125),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)

        if show_debug:
            draw_top_info(
                frame,
                fps_meter.update(),
                rec.label,
                rec.confidence,
                rec.source,
                args.mode,
                action_controller.enabled,
                face_enabled,
                action_text,
            )
            draw_multi10(frame, rec.multi10)

            if rec.finger_states:
                st = " ".join(f"{k}:{v}" for k, v in rec.finger_states.items())
                cv2.putText(frame, f"Fingers: {st}", (10, 185),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.50, (230, 230, 230), 1)

            if recording:
                cv2.putText(frame, f"Recording custom: {args.record} {recorded}/{args.samples}",
                            (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 255, 255), 2)

        draw_help(frame, show_help)

        cv2.imshow("MediaPipe Keypoint Gesture Interaction", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        if key == ord("s"):
            save_frame(frame, prefix="screen")
        if key == ord("a"):
            state = action_controller.toggle_enabled()
            print(f"Actions {'ON' if state else 'OFF'}")
        if key == ord("d"):
            show_debug = not show_debug
        if key == ord("h"):
            show_help = not show_help
        if key == ord("f"):
            face_enabled = not face_enabled
            detector.set_face_enabled(face_enabled)
            print(f"FaceMesh {'ON' if face_enabled else 'OFF'}")

    detector.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
