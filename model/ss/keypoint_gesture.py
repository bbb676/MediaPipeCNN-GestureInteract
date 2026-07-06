import cv2
import mediapipe as mp
import math
from collections import deque

# 初始化 MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# 手势历史
left_hand_history = deque(maxlen=5)
right_hand_history = deque(maxlen=5)


def get_finger_states(landmarks, handedness):
    """获取手指状态（根据左右手调整判断方向）"""
    fingers = []

    # 拇指：根据左右手判断方向
    if handedness == "Right":
        # 右手：拇指伸直时 x 坐标更小
        thumb_extended = landmarks[4].x < landmarks[3].x
    else:
        # 左手：拇指伸直时 x 坐标更大
        thumb_extended = landmarks[4].x > landmarks[3].x

    fingers.append(1 if thumb_extended else 0)

    # 其他四指：比较 y 坐标（左右手相同）
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]

    for tip, pip in zip(tips, pips):
        if landmarks[tip].y < landmarks[pip].y - 0.02:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers


def get_finger_tip_positions(landmarks, w, h):
    """获取指尖的像素坐标"""
    tips = [4, 8, 12, 16, 20]
    positions = []
    for tip in tips:
        x = int(landmarks[tip].x * w)
        y = int(landmarks[tip].y * h)
        positions.append((x, y))
    return positions


def is_thumbs_up(landmarks, handedness, w, h):
    """检测点赞手势（区分左右手）"""
    thumb_tip = landmarks[4]
    thumb_ip = landmarks[3]

    # 拇指伸直判断
    if handedness == "Right":
        thumb_straight = thumb_tip.x < thumb_ip.x - 0.03
    else:
        thumb_straight = thumb_tip.x > thumb_ip.x + 0.03

    other_fingers_bent = True
    for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]:
        if landmarks[tip].y < landmarks[pip].y - 0.01:
            other_fingers_bent = False
            break

    return thumb_straight and other_fingers_bent


def is_fist(landmarks, handedness, w, h):
    """检测拳头"""
    if handedness == "Right":
        # 右手：所有指尖 x 坐标 > 对应指根
        for tip, pip in [(4, 3), (8, 6), (12, 10), (16, 14), (20, 18)]:
            if landmarks[tip].x < landmarks[pip].x:
                return False
    else:
        # 左手：所有指尖 x 坐标 < 对应指根
        for tip, pip in [(4, 3), (8, 6), (12, 10), (16, 14), (20, 18)]:
            if landmarks[tip].x > landmarks[pip].x:
                return False
    return True


def is_palm(landmarks, handedness, w, h):
    """检测手掌"""
    if handedness == "Right":
        for tip, pip in [(4, 3), (8, 6), (12, 10), (16, 14), (20, 18)]:
            if landmarks[tip].x > landmarks[pip].x:
                return False
    else:
        for tip, pip in [(4, 3), (8, 6), (12, 10), (16, 14), (20, 18)]:
            if landmarks[tip].x < landmarks[pip].x:
                return False
    return True


def recognize_gesture(fingers, landmarks, handedness, w, h):
    """识别手势（根据左右手）"""
    count = sum(fingers)
    tip_positions = get_finger_tip_positions(landmarks, w, h)

    # 1. 石头
    if is_fist(landmarks, handedness, w, h):
        return "1", "石头", 0.95

    # 2. 点赞
    if is_thumbs_up(landmarks, handedness, w, h):
        return "8", "点赞", 0.95

    # 3. 手掌
    if count >= 4 and is_palm(landmarks, handedness, w, h):
        return "9", "手掌", 0.95

    # 4. 食指指向
    if (fingers[1] == 1 and
            fingers[0] == 0 and fingers[2] == 0 and fingers[3] == 0 and fingers[4] == 0):
        return "2", "食指", 0.9

    # 5. 剪刀
    if (fingers[1] == 1 and fingers[2] == 1 and
            fingers[0] == 0 and fingers[3] == 0 and fingers[4] == 0):
        index_tip = tip_positions[1]
        middle_tip = tip_positions[2]
        distance = math.sqrt((index_tip[0] - middle_tip[0]) ** 2 + (index_tip[1] - middle_tip[1]) ** 2)
        if distance > 30:
            return "6", "剪刀", 0.9

    # 6. OK 手势
    thumb_tip = tip_positions[0]
    index_tip = tip_positions[1]
    ok_distance = math.sqrt((thumb_tip[0] - index_tip[0]) ** 2 + (thumb_tip[1] - index_tip[1]) ** 2)

    if ok_distance < 30:
        if fingers[2] == 1 and fingers[3] == 1 and fingers[4] == 1:
            return "7", "OK", 0.9

    # 默认根据手指数量返回
    if count == 0:
        return "1", "石头", 0.7
    elif count == 1:
        return "2", "食指", 0.6
    elif count == 2:
        return "6", "两指", 0.6
    elif count == 3:
        return "3", "三指", 0.6
    elif count == 4:
        return "4", "四指", 0.6
    elif count == 5:
        return "9", "手掌", 0.7
    else:
        return "?", "未知", 0.4


# 主程序
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("\n" + "=" * 50)
print("双手手势识别系统（已优化左手）")
print("=" * 50)
print("手势对应:")
print("  1: 石头    2: 食指    3: 三指    4: 四指")
print("  6: 剪刀    7: OK      8: 点赞    9: 手掌")
print("-" * 50)
print("按 'q' 退出")
print("=" * 50 + "\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # 获取左右手信息
            handedness = results.multi_handedness[idx].classification[0].label

            # 绘制关键点
            mp_draw.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2),
                mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2)
            )

            # 获取手指状态（传入左右手）
            fingers = get_finger_states(hand_landmarks.landmark, handedness)

            # 识别手势
            gesture_num, gesture_name, confidence = recognize_gesture(
                fingers, hand_landmarks.landmark, handedness, w, h
            )

            # 手势平滑
            if handedness == "Right":
                gesture_history = right_hand_history
            else:
                gesture_history = left_hand_history

            gesture_history.append((gesture_num, gesture_name))
            if len(gesture_history) >= 3:
                from collections import Counter

                most_common = Counter(gesture_history).most_common(1)[0][0]
                gesture_num, gesture_name = most_common

            # 获取手部边界框
            x_min, x_max = w, 0
            y_min, y_max = h, 0
            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                x_min, x_max = min(x_min, x), max(x_max, x)
                y_min, y_max = min(y_min, y), max(y_max, y)

            # 绘制边界框
            margin = 10
            x_min = max(0, x_min - margin)
            x_max = min(w, x_max + margin)
            y_min = max(0, y_min - margin)
            y_max = min(h, y_max + margin)

            # 边框颜色：右手绿色，左手蓝色
            color = (0, 255, 0) if handedness == "Right" else (255, 200, 0)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)

            # 显示手势结果
            label = f"{handedness}: {gesture_num}-{gesture_name}"
            cv2.putText(frame, label, (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # 显示手指状态
            finger_text = f"[{''.join(map(str, fingers))}]"
            cv2.putText(frame, finger_text, (x_min, y_max + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

            # 绘制指尖点
            tip_positions = get_finger_tip_positions(hand_landmarks.landmark, w, h)
            for i, (x, y) in enumerate(tip_positions):
                cv2.circle(frame, (x, y), 6, (0, 255, 255), -1)

        # 显示双手数量
        hand_count = len(results.multi_hand_landmarks)
        cv2.putText(frame, f"Hands: {hand_count}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "No hand detected", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imshow('Two-Hand Gesture Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()