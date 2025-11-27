"""
评估指标工具 - mIoU计算 (申报书[3]适应度函数)
"""
import torch
import torch.nn as nn

class mIoU:
    """mIoU计算类"""
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        self.intersections = torch.zeros(self.num_classes)
        self.unions = torch.zeros(self.num_classes)
        self.total_samples = 0
    
    def update(self, outputs, targets):
        """
        更新mIoU计算
        outputs: 模型输出 (B, C, H, W)
        targets: 真实标签 (B, H, W)
        """
        # 获取预测类别
        _, preds = torch.max(outputs, dim=1)
        
        # 计算每个类别的IoU
        for cls in range(self.num_classes):
            pred_mask = (preds == cls)
            target_mask = (targets == cls)
            
            intersection = (pred_mask & target_mask).sum().float()
            union = (pred_mask | target_mask).sum().float()
            
            self.intersections[cls] += intersection
            self.unions[cls] += union
        
        self.total_samples += targets.shape[0]
    
    def compute(self):
        """计算mIoU"""
        ious = self.intersections / (self.unions + 1e-6)
        return ious.mean().item()
    
    def __call__(self, outputs, targets):
        self.update(outputs, targets)
        return self.compute()

# 测试mIoU
if __name__ == "__main__":
    num_classes = 19
    metric = mIoU(num_classes)
    
    # 模拟输出和目标
    outputs = torch.randn(4, num_classes, 224, 224)
    targets = torch.randint(0, num_classes, (4, 224, 224))
    
    # 更新指标
    metric.update(outputs, targets)
    
    # 计算mIoU
    miou = metric.compute()
    print(f"mIoU: {miou:.4f}")
