import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os


# ==================== 模型定义 ====================
class GestureCNN(nn.Module):
    def __init__(self, num_classes=9):
        super(GestureCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
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
            nn.Dropout(0.5),
            nn.Linear(256 * 8 * 8, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# ==================== 配置 ====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 加载模型
model_path = r"E:\人工智能作业\ss\best_gesture_model.pth"
model = GestureCNN(num_classes=9)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()
print("✓ 模型加载成功！")

# 预处理
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 手势映射
GESTURE_MAP = {0: '1', 1: '2', 2: '3', 3: '4', 4: '6', 5: '7', 6: '8', 7: '9', 8: '10'}
GESTURE_NAME = {
    '1': '石头(拳头)', '2': '食指指向', '3': '小指', '4': '枪',
    '6': '三分(剪刀)', '7': '六六六(OK)', '8': '我爱你(点赞)',
    '9': '布(手掌)', '10': '观音'
}

# 使用实际存在的图片
test_image_path = r"E:\人工智能作业\gesture_processed\val\1\1000_1.jpg"

if os.path.exists(test_image_path):
    image = Image.open(test_image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted = torch.max(probs, 1)

    idx = predicted.item()
    gesture_idx = GESTURE_MAP[idx]
    gesture_name = GESTURE_NAME.get(gesture_idx, gesture_idx)

    print(f"\n测试图片: {test_image_path}")
    print(f"预测结果: 手势 {gesture_name}")
    print(f"置信度: {confidence.item():.2%}")

    # 显示 Top-5 预测
    print("\nTop-5 预测:")
    top5_probs, top5_idx = torch.topk(probs, 5)
    for i in range(5):
        g_idx = top5_idx[0][i].item()
        g_name = GESTURE_NAME.get(GESTURE_MAP[g_idx], GESTURE_MAP[g_idx])
        print(f"  {i + 1}. 手势 {g_name}: {top5_probs[0][i].item():.2%}")
else:
    print(f"图片不存在: {test_image_path}")