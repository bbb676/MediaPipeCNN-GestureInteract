import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import mediapipe as mp


# ==================== 加载你的 CNN 模型 ====================
class GestureCNN(nn.Module):
    def __init__(self, num_classes=9):
        super(GestureCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 8 * 8, 512), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GestureCNN(num_classes=9)
model.load_state_dict(torch.load(r'E:\人工智能作业\ss\best_gesture_model.pth', map_location=device))
model.to(device)
model.eval()
print("✓ CNN 模型加载成功")

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 手势映射
IDX_TO_GESTURE = {0: '1', 1: '2', 2: '3', 3: '4', 4: '6', 5: '7', 6: '8', 7: '9', 8: '10'}
GESTURE_NAME = {
    '1': '石头', '2': '食指', '3': '小指', '4': '枪',
    '6': '剪刀', '7': 'OK', '8': '点赞', '9': '手掌', '10': '观音'
}

# ==================== MediaPipe 手部检测 ====================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# ==================== 主程序 ====================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("\n" + "=" * 50)
print("MediaPipe 手部检测 + CNN 手势识别")
print("=" * 50)
print("按 'q' 退出")
print("=" * 50 + "\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # 绘制关键点
            mp_draw.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2),
                mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2)
            )

            # 获取手部边界框
            h, w, _ = frame.shape
            x_min, x_max = w, 0
            y_min, y_max = h, 0

            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                x_min, x_max = min(x_min, x), max(x_max, x)
                y_min, y_max = min(y_min, y), max(y_max, y)

            # 添加边距并裁剪
            margin = 30
            x_min = max(0, x_min - margin)
            x_max = min(w, x_max + margin)
            y_min = max(0, y_min - margin)
            y_max = min(h, y_max + margin)

            if x_max > x_min and y_max > y_min:
                hand_roi = frame[y_min:y_max, x_min:x_max]

                if hand_roi.size > 0:
                    # 用 CNN 模型识别
                    hand_rgb = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(hand_rgb)
                    image_tensor = transform(pil_img).unsqueeze(0).to(device)

                    with torch.no_grad():
                        output = model(image_tensor)
                        probs = torch.nn.functional.softmax(output, dim=1)
                        confidence, predicted = torch.max(probs, 1)

                    idx = predicted.item()
                    gesture_num = IDX_TO_GESTURE[idx]
                    gesture_name = GESTURE_NAME.get(gesture_num, gesture_num)
                    confidence_score = confidence.item()

                    # 显示结果
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    label = f"{gesture_num} - {gesture_name} ({confidence_score:.2%})"
                    cv2.putText(frame, label, (x_min, y_min - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    # 全局显示
                    cv2.putText(frame, f"CNN: {gesture_num} - {gesture_name}", (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.putText(frame, f"Confidence: {confidence_score:.2%}", (10, 85),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    else:
        cv2.putText(frame, "No hand detected", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imshow('CNN + MediaPipe', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()