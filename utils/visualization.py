"""
结果可视化工具
"""
import matplotlib.pyplot as plt
import numpy as np
import os

def save_results(image, target, output, save_path):
    """
    保存可视化结果
    image: 原始图像 (3, H, W)
    target: 真实标签 (H, W)
    output: 模型输出 (C, H, W)
    save_path: 保存路径
    """
    # 转换为numpy
    image = image.cpu().permute(1, 2, 0).numpy()
    image = (image * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406])  # 反归一化
    target = target.cpu().numpy()
    
    # 获取预测结果
    _, pred = output.cpu().max(0)
    pred = pred.numpy()
    
    # 创建可视化
    plt.figure(figsize=(12, 4))
    
    # 原始图像
    plt.subplot(131)
    plt.imshow(image)
    plt.title("Input Image")
    plt.axis('off')
    
    # 真实标签
    plt.subplot(132)
    plt.imshow(target, cmap='viridis')
    plt.title("Ground Truth")
    plt.axis('off')
    
    # 预测结果
    plt.subplot(133)
    plt.imshow(pred, cmap='viridis')
    plt.title("Prediction")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

# 测试可视化
if __name__ == "__main__":
    import torch
    
    # 模拟数据
    image = torch.rand(3, 224, 224)
    target = torch.randint(0, 19, (224, 224))
    output = torch.randn(19, 224, 224)
    
    # 保存可视化
    save_results(image, target, output, "visualization_test.png")
    print("Saved visualization test to visualization_test.png")
