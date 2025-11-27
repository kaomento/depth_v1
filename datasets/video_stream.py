"""
视频流模拟器 - 用于实时推理测试
"""
import torch
import numpy as np
import cv2
from PIL import Image
import time
from torchvision import transforms

class VideoStreamSimulator:
    """视频流模拟器"""
    def __init__(self, source=0, img_size=224):
        """
        source: 视频源 (0=摄像头, 或视频文件路径)
        img_size: 处理尺寸
        """
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise ValueError("无法打开视频源")
        
        self.img_size = img_size
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 缓存上一帧
        self.prev_frame = None
    
    def __iter__(self):
        return self
    
    def __next__(self):
        """获取下一帧"""
        ret, frame = self.cap.read()
        if not ret:
            raise StopIteration
        
        # 转换为RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 转换为PIL图像
        pil_img = Image.fromarray(frame)
        
        # 应用转换
        tensor = self.transform(pil_img)
        
        # 创建帧对 (当前帧 + 上一帧)
        if self.prev_frame is None:
            self.prev_frame = tensor
            # 再获取一帧
            ret, frame = self.cap.read()
            if not ret:
                raise StopIteration
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame)
            tensor = self.transform(pil_img)
        
        # 组合成(B=1, T=2, C, H, W)
        frames = torch.stack([self.prev_frame, tensor], dim=0).unsqueeze(0)
        
        # 更新上一帧
        self.prev_frame = tensor
        
        return frames, None  # 无标签
    
    def release(self):
        """释放资源"""
        self.cap.release()

# 测试视频流模拟器
if __name__ == "__main__":
    print("启动视频流模拟器 (按'q'退出)...")
    stream = VideoStreamSimulator()
    
    try:
        for frames, _ in stream:
            print(f"获取视频帧: {frames.shape}")
            
            # 模拟处理延迟
            time.sleep(0.1)
            
            # 显示帧 (仅显示第二帧)
            frame = frames[0, 1].permute(1, 2, 0).numpy()
            frame = (frame * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406])  # 反归一化
            frame = np.clip(frame, 0, 1)
            
            cv2.imshow("Video Stream", frame[:, :, ::-1])  # BGR to RGB
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        stream.release()
        cv2.destroyAllWindows()
