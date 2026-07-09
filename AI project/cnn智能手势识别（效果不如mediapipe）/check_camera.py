import cv2

print("检查摄像头...")

# 尝试打开摄像头0
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("✗ 无法打开摄像头0")
    print("尝试打开摄像头1...")
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("✗ 无法打开摄像头1")
        print("请检查:")
        print("1. 摄像头是否已连接")
        print("2. 摄像头是否被其他程序占用")
        print("3. 是否有摄像头驱动问题")
        exit()
    else:
        print("✓ 摄像头1打开成功")
else:
    print("✓ 摄像头0打开成功")

# 读取一帧测试
ret, frame = cap.read()
if ret:
    print(f"✓ 图像读取成功，尺寸: {frame.shape}")

    # 显示图像
    cv2.imshow('Camera Test', frame)
    print("按任意键关闭窗口...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("✗ 无法读取图像")

cap.release()
print("摄像头已释放")