"""
类脑脉冲恢复机制 - 模拟生物神经系统的脉冲触发特性 (申报书[1][3][6])
"""
import torch
import torch.nn as nn
import spikingjelly.activation_based as sj
from spikingjelly.activation_based import functional

class LIFNeuron(sj.LIFNode):
    """LIF神经元模型 (申报书[3]图b)"""
    def __init__(self, tau=2.0, v_threshold=0.5, v_reset=0.0):
        """
        tau: 时间常数
        v_threshold: 脉冲触发阈值 (申报书[3] Vth)
        v_reset: 脉冲重置电位
        """
        super().__init__(tau=tau, v_threshold=v_threshold, v_reset=v_reset)
    
    def forward(self, x):
        """
        x: 输入信号 (申报书[3]图b: Input)
        输出: 脉冲信号 (申报书[3]图b: Spikes)
        """
        # 申报书[3]图b: I_total = I_LO + I_ph1 + I_ph2
        # 简化实现: 直接使用输入作为总电流
        return super().forward(x)

class CriticalFeatureDetector(nn.Module):
    """关键特征检测器 (申报书[1]: "危险因素等")"""
    def __init__(self, input_channels, hidden_dim=64):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """
        x: ViT特征图 (B, C, H, W)
        输出: 关键区域热力图 (B, 1, H, W)
        """
        x = torch.relu(self.conv1(x))
        x = self.sigmoid(self.conv2(x))
        return x

class SpikeRecovery(nn.Module):
    """类脑脉冲恢复机制 (申报书[1][3][6])"""
    def __init__(self, input_size, config):
        """
        input_size: (C, H, W) ViT特征图尺寸
        config: 配置参数
        """
        super().__init__()
        C, H, W = input_size
        self.config = config
        
        # 关键特征检测器
        self.detector = CriticalFeatureDetector(C)
        
        # LIF神经元
        self.lif = LIFNeuron(
            tau=config['SPIKE_CONFIG']['tau'],
            v_threshold=config['SPIKE_CONFIG']['v_threshold']
        )
        
        # 恢复强度参数
        self.recovery_strength = nn.Parameter(torch.tensor(0.8))
    
    def forward(self, x, motion_mask=None):
        """
        x: ViT特征图 (B, C, H, W)
        motion_mask: (B, 1, H, W) 运动掩码 (可选)
        
        申报书[1]: "在模型识别到关键特征（如危险因素等）时激活"
        """
        # 步骤1: 检测关键区域
        critical_map = self.detector(x)  # (B, 1, H, W)
        
        # 步骤2: 脉冲触发机制 (申报书[3] LIF神经元模型)
        spike = self.lif(critical_map)  # (B, 1, H, W)
        
        # 步骤3: 恢复关键特征 (脉冲区域增强)
        recovered = x * (1 + spike * torch.sigmoid(self.recovery_strength))
        
        # 申报书[1]: "作为保护机制，以提升模型对突发状况的响应和适应能力"
        # 如果提供运动掩码，额外增强运动区域
        if motion_mask is not None:
            motion_factor = torch.sigmoid(10 * (motion_mask - self.config['SPIKE_CONFIG']['critical_threshold']))
            recovered = recovered * (1 + motion_factor * 0.5)
        
        return recovered

# 测试脉冲恢复机制
if __name__ == "__main__":
    import sys
    import os
    import matplotlib.pyplot as plt
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import BASE_CONFIG, SPIKE_CONFIG
    
    # 创建配置
    config = {'SPIKE_CONFIG': SPIKE_CONFIG}
    
    # 模拟ViT特征图 (B=1, C=768, H=14, W=14)
    features = torch.rand(1, 768, 14, 14)
    
    # 创建脉冲恢复模块
    spike_recovery = SpikeRecovery(input_size=(768, 14, 14), config=config)
    
    # 模拟运动掩码
    motion_mask = torch.rand(1, 1, 14, 14) > 0.7
    
    # 前向传播
    recovered = spike_recovery(features, motion_mask)
    
    print(f"Input shape: {features.shape}")
    print(f"Recovered shape: {recovered.shape}")
    
    # 可视化
    plt.figure(figsize=(15, 5))
    
    # 1. 关键特征检测
    plt.subplot(141)
    with torch.no_grad():
        critical_map = spike_recovery.detector(features)
    plt.imshow(critical_map[0, 0].detach().numpy(), cmap='hot')
    plt.title("Critical Map")
    plt.colorbar()
    
    # 2. 脉冲输出
    plt.subplot(142)
    with torch.no_grad():
        spike = spike_recovery.lif(critical_map)
    plt.imshow(spike[0, 0].detach().numpy(), cmap='hot')
    plt.title("Spikes")
    plt.colorbar()
    
    # 3. 运动掩码
    plt.subplot(143)
    plt.imshow(motion_mask[0, 0].detach().numpy(), cmap='hot')
    plt.title("Motion Mask")
    plt.colorbar()
    
    # 4. 恢复效果
    plt.subplot(144)
    diff = (recovered - features).mean(1)[0]  # 平均通道差异
    plt.imshow(diff.detach().numpy(), cmap='bwr', vmin=-0.5, vmax=0.5)
    plt.title("Recovery Effect")
    plt.colorbar()
    
    plt.tight_layout()
    plt.savefig("spike_recovery.png")
    print("Saved spike recovery visualization to spike_recovery.png")
