"""
动态剪枝模块 - 实现时空联合动态剪枝策略 (申报书[1][2][3])
"""
import torch
import torch.nn as nn
import numpy as np

class DynamicPruner:
    """时空联合动态剪枝模块 (申报书[2]核心)"""
    def __init__(self, vit_model, config):
        """
        vit_model: Vision Transformer模型
        config: 配置参数 (来自config.py)
        """
        self.vit = vit_model
        self.config = config
        self.rho_head = 0.5  # 默认注意力头剪枝率
        self.rho_token = 0.5  # 默认Token剪枝率
        self.tau = 0.1  # 默认光流阈值 (由FLOWNE设置)
        
        # 申报书[2]图a: 30个编码个体 → 这里存储当前最优参数
        self.best_params = None
    
    def set_params(self, rho_head, rho_token, tau):
        """设置剪枝参数"""
        # 限制在配置范围内
        self.rho_head = np.clip(rho_head, 
                               self.config['PRUNER_CONFIG']['rho_head_min'], 
                               self.config['PRUNER_CONFIG']['rho_head_max'])
        self.rho_token = np.clip(rho_token, 
                                self.config['PRUNER_CONFIG']['rho_token_min'], 
                                self.config['PRUNER_CONFIG']['rho_token_max'])
        self.tau = np.clip(tau, 
                          self.config['FLOWNE_CONFIG']['tau_min'], 
                          self.config['FLOWNE_CONFIG']['tau_max'])
    
    def apply(self, motion_mask):
        """
        根据运动掩码应用动态剪枝
        motion_mask: (B, 1, H, W) 运动区域二值掩码
        
        申报书[2][3]:
        - 基因编码: θ=(ρ_head, ρ_token, τ)
        - 根据块的数量，补丁被分组处理
        """
        B, _, H, W = motion_mask.shape
        N = self.vit.patch_embed.n_patches  # 196 (申报书[2]: 196个补丁)
        
        # 步骤1: 基于运动区域动态调整剪枝强度 (申报书[1]实时性优化)
        motion_ratio = motion_mask.mean().item()  # 当前帧运动区域占比
        
        # 运动越多，剪枝越保守 (保护关键区域)
        adaptive_rho_token = self.rho_token * (1 - 0.5 * motion_ratio)
        
        # 步骤2: Token剪枝 (保留关键区域Token)
        motion_mask_flat = motion_mask.view(B, -1)  # 展平为(B, H*W)
        
        # 计算需要保留的Token数量
        keep_ratio = 1 - adaptive_rho_token
        keep_num = int(N * keep_ratio)
        
        # 优先保留运动区域的Token (申报书[1]: 危险因素保护)
        _, keep_idx = torch.topk(motion_mask_flat, keep_num, dim=1)
        
        # 创建Token保留掩码
        keep_token_mask = torch.zeros(B, N, device=motion_mask.device)
        for i in range(B):
            keep_token_mask[i, keep_idx[i]] = 1
        
        # 步骤3: 注意力头剪枝 (申报书[3]: ρ_head)
        head_importance = self._compute_head_importance()  # 计算头重要性
        
        # 选择最重要的头保留
        num_keep_heads = int(self.vit.num_heads * (1 - self.rho_head))
        _, keep_heads = torch.topk(head_importance, num_keep_heads)
        
        # 创建头保留掩码
        keep_head_mask = torch.zeros(self.vit.num_heads, device=motion_mask.device)
        keep_head_mask[keep_heads] = 1
        
        return keep_token_mask, keep_head_mask
    
    def _compute_head_importance(self):
        """计算注意力头重要性 (申报书[2]进化算法基础)
        简化实现: 基于注意力头参数L2范数
        """
        importance = []
        for blk in self.vit.blocks:
            # 获取QKV权重
            qkv_weight = blk.attn.qkv.weight.data
            head_dim = qkv_weight.shape[0] // 3 // self.vit.num_heads
            
            # 按头分割并计算L2范数
            for i in range(self.vit.num_heads):
                start = i * head_dim
                end = (i+1) * head_dim
                head_weight = qkv_weight[start*3:end*3]  # Q, K, V权重
                importance.append(torch.norm(head_weight).item())
        
        return torch.tensor(importance)

# 测试动态剪枝
if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import VIT_CONFIG, PRUNER_CONFIG
    from vit import VisionTransformer
    
    # 创建配置
    config = {
        'VIT_CONFIG': VIT_CONFIG,
        'PRUNER_CONFIG': PRUNER_CONFIG,
        'FLOWNE_CONFIG': {'tau_min': 0.05, 'tau_max': 0.3}
    }
    
    # 初始化ViT
    vit = VisionTransformer(
        img_size=224,
        patch_size=16,
        num_classes=19,
        **VIT_CONFIG
    )
    
    # 初始化剪枝器
    pruner = DynamicPruner(vit, config)
    
    # 模拟运动掩码 (B=2, 1, H=224, W=224)
    motion_mask = torch.rand(2, 1, 224, 224) > 0.7  # 30%运动区域
    
    # 设置剪枝参数 (申报书[3]基因编码)
    pruner.set_params(rho_head=0.6, rho_token=0.7, tau=0.15)
    
    # 应用动态剪枝
    keep_token_mask, keep_head_mask = pruner.apply(motion_mask)
    
    print(f"Token保留率: {keep_token_mask.mean().item():.2f}")
    print(f"Head保留率: {keep_head_mask.mean().item():.2f}")
    
    # 可视化Token保留掩码
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.imshow(motion_mask[0, 0].cpu().numpy(), cmap='gray')
    plt.title("Motion Mask")
    plt.subplot(122)
    plt.imshow(keep_token_mask[0].view(14, 14).cpu().numpy(), cmap='hot')
    plt.title("Token Keep Mask")
    plt.savefig("pruner_sample.png")
    print("Saved pruner visualization to pruner_sample.png")
