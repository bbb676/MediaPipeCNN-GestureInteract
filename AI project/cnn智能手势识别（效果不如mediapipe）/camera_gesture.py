import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import time


# ============ 模型定义 ============
class GrayScaleGestureCNN(nn.Module):
    def __init__(self, num_classes=20):
        super(GrayScaleGestureCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(256 * 8 * 8, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# ============ 加载模型 ============
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

model_path = r"E:\人工智能作业\ss\best_gesture_model.pth"
model = GrayScaleGestureCNN(num_classes=20)
checkpoint = torch.load(model_path, map_location=device)

if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint)

model.to(device)
model.eval()
print("✓ 模型加载成功！")

# ============ 预处理 ============
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

class_names = [str(i) for i in range(20)]


# ============ 肤色检测（屏蔽上半部分中间）============
def extract_hand_binary(frame):
    """
    提取手部，只屏蔽上半部分的中间（脸和脖子位置）
    """
    h, w = frame.shape[:2]

    # 创建全黑掩码
    mask_full = np.zeros((h, w), dtype=np.uint8)

    # 上半部分高度
    upper_height = int(h * 0.5)

    # 肤色范围
    lower_skin = np.array([0, 30, 60], dtype=np.uint8)
    upper_skin = np.array([25, 180, 255], dtype=np.uint8)
    kernel = np.ones((5, 5), np.uint8)

    # 1. 左上区域
    left_upper = frame[0:upper_height, 0:int(w * 0.35)]
    if left_upper.size > 0:
        hsv_left = cv2.cvtColor(left_upper, cv2.COLOR_BGR2HSV)
        mask_left = cv2.inRange(hsv_left, lower_skin, upper_skin)
        mask_left = cv2.erode(mask_left, kernel, iterations=1)
        mask_left = cv2.dilate(mask_left, kernel, iterations=2)
        mask_full[0:upper_height, 0:int(w * 0.35)] = mask_left

    # 2. 右上区域
    right_upper = frame[0:upper_height, int(w * 0.65):w]
    if right_upper.size > 0:
        hsv_right = cv2.cvtColor(right_upper, cv2.COLOR_BGR2HSV)
        mask_right = cv2.inRange(hsv_right, lower_skin, upper_skin)
        mask_right = cv2.erode(mask_right, kernel, iterations=1)
        mask_right = cv2.dilate(mask_right, kernel, iterations=2)
        mask_full[0:upper_height, int(w * 0.65):w] = mask_right

    # 3. 下方区域（整个下半部分）
    lower = frame[int(h * 0.45):h, 0:w]
    if lower.size > 0:
        hsv_lower = cv2.cvtColor(lower, cv2.COLOR_BGR2HSV)
        mask_lower = cv2.inRange(hsv_lower, lower_skin, upper_skin)
        mask_lower = cv2.erode(mask_lower, kernel, iterations=1)
        mask_lower = cv2.dilate(mask_lower, kernel, iterations=2)
        mask_full[int(h * 0.45):h, 0:w] = mask_lower

    return mask_full


# ============ 获取手部区域 ============
def get_hand_roi(frame, mask):
    """从掩码中找出手部区域"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, None

    # 按面积排序
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]

    h, w = frame.shape[:2]

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < 800 or area > 30000:
            continue

        x, y, bw, bh = cv2.boundingRect(contour)

        # 添加边距
        margin = 25
        x = max(0, x - margin)
        y = max(0, y - margin)
        bw = min(frame.shape[1] - x, bw + 2 * margin)
        bh = min(frame.shape[0] - y, bh + 2 * margin)

        hand_binary = mask[y:y + bh, x:x + bw]

        if hand_binary.size > 0:
            return (x, y, bw, bh), hand_binary

    return None, None


# ============ 主程序 ============
print("\n" + "=" * 55)
print("实时手势识别系统（屏蔽人脸+脖子）")
print("=" * 55)
print("将手放在摄像头前")
print("按 'q' 退出")
print("按 's' 保存截图")
print("=" * 55 + "\n")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

fps = 0
frame_count = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count >= 30:
        end_time = time.time()
        fps = frame_count / (end_time - start_time)
        frame_count = 0
        start_time = end_time

    # 肤色检测（屏蔽上半部分中间）
    mask = extract_hand_binary(frame)

    # 获取手部区域
    bbox, hand_binary = get_hand_roi(frame, mask)

    gesture = "None"
    confidence_value = 0.0

    if hand_binary is not None and hand_binary.size > 0:
        hand_resized = cv2.resize(hand_binary, (128, 128))
        pil_image = Image.fromarray(hand_resized)
        image_tensor = transform(pil_image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        gesture = class_names[predicted.item()]
        confidence_value = confidence.item()

        if bbox:
            x, y, bw, bh = bbox
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            cv2.putText(frame, f"{gesture}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            bar_width = int(confidence_value * bw)
            cv2.rectangle(frame, (x, y + bh + 5), (x + bar_width, y + bh + 20), (0, 255, 0), -1)
            cv2.rectangle(frame, (x, y + bh + 5), (x + bw, y + bh + 20), (255, 255, 255), 1)

    # 显示信息
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"Gesture: {gesture}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, f"Conf: {confidence_value:.2%}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    # 绘制检测区域（可视化）
    h, w = frame.shape[:2]
    upper_height = int(h * 0.5)

    # 屏蔽区域（上半部分中间）- 画半透明遮罩
    overlay = frame.copy()
    cv2.rectangle(overlay, (int(w * 0.35), 0), (int(w * 0.65), upper_height), (0, 0, 0), -1)
    frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

    # 画检测区域边框
    # 左上区域
    cv2.rectangle(frame, (0, 0), (int(w * 0.35), upper_height), (0, 255, 0), 2)
    # 右上区域
    cv2.rectangle(frame, (int(w * 0.65), 0), (w, upper_height), (0, 255, 0), 2)
    # 下方区域
    cv2.rectangle(frame, (0, int(h * 0.45)), (w, h), (0, 255, 0), 2)

    # 添加文字说明
    cv2.putText(frame, "Hand detection areas", (10, upper_height - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imshow('Camera', frame)

    if hand_binary is not None:
        hand_display = cv2.resize(hand_binary, (200, 200))
        cv2.imshow('Hand Processed', hand_display)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        timestamp = int(time.time())
        cv2.imwrite(f"screenshot_{timestamp}.jpg", frame)
        print(f"✓ 截图已保存")

cap.release()
cv2.destroyAllWindows()
print("\n系统已退出")