import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os


# 模型定义（必须和训练时一致）
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


# 手势映射（索引 -> 手势序号）
GESTURE_MAP = {
    0: '1', 1: '2', 2: '3', 3: '4',
    4: '6', 5: '7', 6: '8', 7: '9', 8: '10'
}

# 手势名称（序号 -> 中文名称）
GESTURE_NAME = {
    '1': '石头(拳头)',
    '2': '食指指向',
    '3': '小指',
    '4': '枪',
    '6': '三分(剪刀)',
    '7': '六六六(OK)',
    '8': '我爱你(点赞)',
    '9': '布(手掌)',
    '10': '观音',
}


def load_model(model_path='models/best_gesture_model.pth', device='cpu'):
    """加载训练好的模型"""
    model = GestureCNN(num_classes=9)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def get_transform():
    """获取图像预处理"""
    return transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def predict_image(image, model, transform, device):
    """预测单张图片（PIL Image）"""
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    idx = predicted.item()
    gesture_idx = GESTURE_MAP[idx]
    gesture_name = GESTURE_NAME.get(gesture_idx, gesture_idx)
    confidence_score = confidence.item()

    return gesture_idx, gesture_name, confidence_score


def predict_image_path(image_path, model, transform, device):
    """预测单张图片（文件路径）"""
    image = Image.open(image_path).convert('RGB')
    return predict_image(image, model, transform, device)