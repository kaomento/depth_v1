"""
Cityscapes数据集加载器 - 用于自动驾驶场景测试 (申报书[1])
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import cv2

class CityscapesDataset(Dataset):
    """Cityscapes语义分割数据集"""
    def __init__(self, root, split='train', img_size=224):
        self.root = root
        self.split = split
        self.img_size = img_size
        
        # 城市列表
        self.cities = [d for d in os.listdir(os.path.join(root, 'leftImg8bit', split)) 
                      if os.path.isdir(os.path.join(root, 'leftImg8bit', split, d))]
        
        # 收集图像对 (连续帧)
        self.image_pairs = []
        for city in self.cities:
            img_dir = os.path.join(root, 'leftImg8bit', split, city)
            for img_name in sorted(os.listdir(img_dir)):
                if img_name.endswith('_10.png'):  # 取间隔10帧的图像对
                    base_name = img_name.replace('_10.png', '')
                    img1_path = os.path.join(img_dir, f"{base_name}_10.png")
                    img2_path = os.path.join(img_dir, f"{base_name}_20.png")
                    if os.path.exists(img2_path):
                        self.image_pairs.append((img1_path, img2_path))
        
        # 预处理
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def __len__(self):
        return len(self.image_pairs)
    
    def __getitem__(self, idx):
        img1_path, img2_path = self.image_pairs[idx]
        
        # 加载图像
        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')
        
        # 转换
        img1 = self.transform(img1)
        img2 = self.transform(img2)
        
        # 模拟分割标签 (实际应用中应加载真实标签)
        # 这里用简单边缘检测模拟"关键特征" (申报书[1]危险因素)
        img1_np = np.array(Image.open(img1_path).convert('L'))
        edges = cv2.Canny(img1_np, 100, 200)
        target = torch.from_numpy(edges).float() / 255.0
        target = transforms.Resize((self.img_size, self.img_size))(target.unsqueeze(0))
        
        return torch.stack([img1, img2], dim=0), target

def get_cityscapes_loaders(root, batch_size=4, img_size=224, num_workers=2):
    """获取Cityscapes数据加载器"""
    train_set = CityscapesDataset(root, split='train', img_size=img_size)
    val_set = CityscapesDataset(root, split='val', img_size=img_size)
    
    train_loader = DataLoader(
        train_set, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_set, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader

# 测试数据集 (确保路径正确)
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # 假设Cityscapes数据集路径
    CITYSCAPES_ROOT = "/path/to/cityscapes"  # 替换为实际路径
    
    if os.path.exists(CITYSCAPES_ROOT):
        train_loader, _ = get_cityscapes_loaders(CITYSCAPES_ROOT)
        frames, target = next(iter(train_loader))
        
        print(f"Frames shape: {frames.shape} (B, T=2, C, H, W)")
        print(f"Target shape: {target.shape}")
        
        # 可视化
        plt.figure(figsize=(12, 4))
        plt.subplot(131)
        plt.imshow(frames[0, 0].permute(1, 2, 0).numpy())
        plt.title("Frame 1")
        plt.subplot(132)
        plt.imshow(frames[0, 1].permute(1, 2, 0).numpy())
        plt.title("Frame 2")
        plt.subplot(133)
        plt.imshow(target[0].squeeze(), cmap='gray')
        plt.title("Target (Edge)")
        plt.savefig("cityscapes_sample.png")
        print("Saved sample visualization to cityscapes_sample.png")
    else:
        print("Cityscapes dataset not found. Skipping test.")
        print("To test, set CITYSCAPES_ROOT to your Cityscapes path.")
