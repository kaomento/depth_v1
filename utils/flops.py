"""
FLOPs计算工具 - 用于评估计算复杂度 (申报书[3])
"""
import torch
import numpy as np

class FLOPsCalculator:
    """FLOPs计算器"""
    def __init__(self, img_size=224, patch_size=16, vit_config=None, base_flops=120e9):
        """
        img_size: 输入图像尺寸
        patch_size: ViT分块尺寸
        vit_config: ViT配置参数
        base_flops: 基础FLOPs (未剪枝)
        """
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.vit_config = vit_config or {}
        self.base_flops = base_flops
    
    def estimate(self, rho_head=0.5, rho_token=0.5, motion_ratio=0.5):
        """
        估算剪枝后的FLOPs
        
        rho_head: 注意力头剪枝率
        rho_token: Token剪枝率
        motion_ratio: 运动区域比例 (用于自适应剪枝)
        
        申报书[3]: FLOPs(θ) = FLOPs_base * (1 - reduction)
        """
        # 计算自适应剪枝率 (运动越多，剪枝越少)
        adaptive_rho_token = rho_token * (1 - 0.5 * motion_ratio)
        
        # 计算总剪枝率
        total_reduction = (1 - rho_head) * (1 - adaptive_rho_token)
        
        # 保守估计 (运动区域保留更多计算)
        flops = self.base_flops * (1 - total_reduction * 0.7)
        
        return flops
    
    def get_base_flops(self):
        """获取基础FLOPs"""
        return self.base_flops

# 测试FLOPs计算器
if __name__ == "__main__":
    calculator = FLOPsCalculator()
    
    # 测试不同参数下的FLOPs
    params = [
        (0.0, 0.0, 0.0),  # 无剪枝
        (0.5, 0.5, 0.3),  # 中等剪枝
        (0.8, 0.7, 0.6)   # 高剪枝率
    ]
    
    print(f"基础FLOPs: {calculator.get_base_flops()/1e9:.2f} GFLOPs")
    for rho_head, rho_token, motion_ratio in params:
        flops = calculator.estimate(rho_head, rho_token, motion_ratio)
        reduction = (1 - flops/calculator.get_base_flops()) * 100
        print(f"ρ_head={rho_head:.1f}, ρ_token={rho_token:.1f}, motion={motion_ratio:.1f} | "
              f"FLOPs={flops/1e9:.2f} GFLOPs | 减少: {reduction:.1f}%")
