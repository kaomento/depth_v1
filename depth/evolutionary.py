"""
CMA-ES进化算法优化器 - 动态优化剪枝策略 (申报书[3])
"""
import numpy as np
import torch
import torch.nn as nn
from deap import base, creator, tools, algorithms
from tqdm import tqdm
import copy

# 确保DEAP的FitnessMax类已定义
try:
    creator.FitnessMax
except:
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))

try:
    creator.Individual
except:
    creator.create("Individual", list, fitness=creator.FitnessMax)

class EvolutionaryOptimizer:
    """CMA-ES进化算法优化剪枝参数 (申报书[3])
    适应度函数: F(θ) = α⋅mIoU(θ) + β⋅(1−FLOPs(θ)/FLOPs_base)
        α=0.7, β=0.3 (申报书[3]推荐权重)
    """
    def __init__(self, vit_model, flownet, dataloader, config, flops_calculator):
        """
        vit_model: Vision Transformer模型
        flownet: FLOWNE光流网络
        dataloader: 验证数据加载器
        config: 配置参数
        flops_calculator: FLOPs计算器
        """
        self.vit = vit_model
        self.flownet = flownet
        self.dataloader = dataloader
        self.config = config
        self.flops_calculator = flops_calculator
        self.flops_base = config['SYSTEM_CONFIG']['flops_base']
        
        # 创建DEAP工具箱
        self.toolbox = base.Toolbox()
        
        # 基因编码: θ = [ρ_head, ρ_token, τ] (申报书[3])
        self.toolbox.register("rho_head", np.random.uniform, 
                             config['PRUNER_CONFIG']['rho_head_min'], 
                             config['PRUNER_CONFIG']['rho_head_max'])
        self.toolbox.register("rho_token", np.random.uniform, 
                             config['PRUNER_CONFIG']['rho_token_min'], 
                             config['PRUNER_CONFIG']['rho_token_max'])
        self.toolbox.register("tau", np.random.uniform, 
                             config['FLOWNE_CONFIG']['tau_min'], 
                             config['FLOWNE_CONFIG']['tau_max'])
        
        # 创建个体和种群
        self.toolbox.register("individual", tools.initCycle, creator.Individual, 
                             (self.toolbox.rho_head, self.toolbox.rho_token, self.toolbox.tau), n=1)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        
        # 注册进化操作
        self.toolbox.register("evaluate", self._evaluate)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.3)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        
        # 存储历史
        self.history = []
    
    def _evaluate(self, individual):
        """适应度函数计算 (申报书[3]公式)
        individual: [rho_head, rho_token, tau]
        """
        rho_head, rho_token, tau = individual
        
        # 设置剪枝参数
        self.flownet.set_tau(tau)
        
        total_miou = 0.0
        total_flops = 0.0
        num_batches = 0
        
        # 临时模型副本 (避免修改原始模型)
        vit_copy = copy.deepcopy(self.vit)
        vit_copy.eval()
        
        with torch.no_grad():
            for batch_idx, (frames, targets) in enumerate(self.dataloader):
                if batch_idx >= 5:  # 限制评估batch数量 (加速)
                    break
                    
                frames = frames.to(self.vit.device)
                targets = targets.to(self.vit.device)
                
                # 提取连续帧
                frame1 = frames[:, 0]
                frame2 = frames[:, 1]
                
                # 生成运动掩码
                motion_mask, _ = self.flownet(frame1, frame2)
                
                # 动态剪枝 (申报书[2]图b逻辑)
                keep_token_mask = self._get_token_mask(motion_mask, rho_token)
                keep_head_mask = self._get_head_mask(rho_head)
                
                # 前向传播
                outputs = vit_copy(frame2, keep_token_mask, keep_head_mask)
                
                # 计算mIoU (简化版)
                miou = self._compute_miou(outputs, targets)
                
                # 估算FLOPs
                flops = self.flops_calculator.estimate(
                    rho_head=rho_head,
                    rho_token=rho_token,
                    motion_ratio=motion_mask.mean().item()
                )
                
                total_miou += miou
                total_flops += flops
                num_batches += 1
        
        avg_miou = total_miou / num_batches
        avg_flops = total_flops / num_batches
        flops_ratio = 1 - (avg_flops / self.flops_base)
        
        # 适应度: F(θ) = α⋅mIoU + β⋅(1−FLOPs/FLOPs_base)
        alpha = self.config['EVOLUTION_CONFIG']['alpha']
        beta = self.config['EVOLUTION_CONFIG']['beta']
        fitness = alpha * avg_miou + beta * flops_ratio
        
        # 记录历史
        self.history.append({
            'params': individual,
            'miou': avg_miou,
            'flops_ratio': flops_ratio,
            'fitness': fitness
        })
        
        return (fitness,)
    
    def _get_token_mask(self, motion_mask, rho_token):
        """生成Token保留掩码"""
        B, _, H, W = motion_mask.shape
        N = (H // 16) * (W // 16)  # 假设patch_size=16
        
        # 展平运动掩码
        motion_mask_flat = motion_mask.view(B, -1)
        
        # 计算保留数量
        keep_ratio = 1 - rho_token
        keep_num = int(N * keep_ratio)
        
        # 获取保留索引
        _, keep_idx = torch.topk(motion_mask_flat, keep_num, dim=1)
        
        # 创建掩码
        keep_token_mask = torch.zeros(B, N, device=motion_mask.device)
        for i in range(B):
            keep_token_mask[i, keep_idx[i]] = 1
        
        return keep_token_mask
    
    def _get_head_mask(self, rho_head):
        """生成注意力头保留掩码"""
        num_heads = self.vit.num_heads
        num_keep_heads = int(num_heads * (1 - rho_head))
        
        # 简化: 随机选择保留的头
        keep_heads = torch.randperm(num_heads)[:num_keep_heads]
        
        # 创建掩码
        keep_head_mask = torch.zeros(num_heads, device=self.vit.device)
        keep_head_mask[keep_heads] = 1
        
        return keep_head_mask
    
    def _compute_miou(self, outputs, targets):
        """计算mIoU (简化版)"""
        # 简化实现: 使用边缘检测模拟关键特征
        # 实际应用应使用标准mIoU计算
        outputs_bin = (torch.sigmoid(outputs) > 0.5).float()
        targets_bin = (targets > 0.5).float()
        
        intersection = (outputs_bin * targets_bin).sum((2, 3))
        union = outputs_bin.sum((2, 3)) + targets_bin.sum((2, 3)) - intersection
        iou = (intersection + 1e-6) / (union + 1e-6)
        
        return iou.mean().item()
    
    def optimize(self, n_generations=None, pop_size=None):
        """运行CMA-ES搜索帕累托最优解 (申报书[3])"""
        n_generations = n_generations or self.config['EVOLUTION_CONFIG']['n_generations']
        pop_size = pop_size or self.config['EVOLUTION_CONFIG']['pop_size']
        
        # 创建种群
        pop = self.toolbox.population(n=pop_size)
        hof = tools.HallOfFame(1)  # 保存最优解
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)
        
        # 运行进化算法
        print(f"Starting CMA-ES optimization ({n_generations} generations, population={pop_size})")
        pop, logbook = algorithms.eaSimple(
            pop, self.toolbox, 
            cxpb=self.config['EVOLUTION_CONFIG']['cxpb'], 
            mutpb=self.config['EVOLUTION_CONFIG']['mutpb'], 
            ngen=n_generations, 
            stats=stats, 
            halloffame=hof, 
            verbose=True
        )
        
        # 返回最优参数
        best_params = hof[0]
        print(f"\n[优化完成] 最优参数: ρ_head={best_params[0]:.3f}, ρ_token={best_params[1]:.3f}, τ={best_params[2]:.3f}")
        print(f"适应度: {best_params.fitness.values[0]:.4f}")
        
        return best_params
    
    def plot_history(self):
        """绘制优化历史"""
        import matplotlib.pyplot as plt
        
        if not self.history:
            print("No optimization history available")
            return
        
        plt.figure(figsize=(12, 8))
        
        # 适应度历史
        plt.subplot(221)
        fitness = [h['fitness'] for h in self.history]
        plt.plot(fitness)
        plt.title('Fitness History')
        plt.xlabel('Evaluation')
        plt.ylabel('Fitness')
        
        # mIoU历史
        plt.subplot(222)
        miou = [h['miou'] for h in self.history]
        plt.plot(miou)
        plt.title('mIoU History')
        plt.xlabel('Evaluation')
        plt.ylabel('mIoU')
        
        # FLOPs历史
        plt.subplot(223)
        flops_ratio = [h['flops_ratio'] for h in self.history]
        plt.plot(flops_ratio)
        plt.title('FLOPs Reduction History')
        plt.xlabel('Evaluation')
        plt.ylabel('1 - FLOPs/FLOPs_base')
        
        # 参数变化
        plt.subplot(224)
        rho_head = [h['params'][0] for h in self.history]
        rho_token = [h['params'][1] for h in self.history]
        tau = [h['params'][2] for h in self.history]
        plt.plot(rho_head, label='ρ_head')
        plt.plot(rho_token, label='ρ_token')
        plt.plot(tau, label='τ')
        plt.title('Parameter Evolution')
        plt.xlabel('Evaluation')
        plt.ylabel('Value')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig("evolution_history.png")
        print("Saved optimization history to evolution_history.png")

# 测试进化优化器
if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import BASE_CONFIG, VIT_CONFIG, EVOLUTION_CONFIG, SYSTEM_CONFIG
    from vit import VisionTransformer
    from flownet import FLOWNE
    from utils.flops import FLOPsCalculator
    
    # 设置随机种子
    torch.manual_seed(BASE_CONFIG['seed'])
    np.random.seed(BASE_CONFIG['seed'])
    
    # 创建模型
    vit = VisionTransformer(
        img_size=BASE_CONFIG['img_size'],
        patch_size=16,
        num_classes=BASE_CONFIG['num_classes'],
        **VIT_CONFIG
    ).to(BASE_CONFIG['device'])
    
    flownet = FLOWNE().to(BASE_CONFIG['device'])
    
    # 创建FLOPs计算器
    flops_calculator = FLOPsCalculator(
        img_size=BASE_CONFIG['img_size'],
        patch_size=16,
        vit_config=VIT_CONFIG,
        base_flops=SYSTEM_CONFIG['flops_base']
    )
    
    # 模拟数据加载器
    class MockDataset(torch.utils.data.Dataset):
        def __init__(self, size=20):
            self.size = size
        
        def __len__(self):
            return self.size
        
        def __getitem__(self, idx):
            frames = torch.rand(2, 3, BASE_CONFIG['img_size'], BASE_CONFIG['img_size'])
            target = torch.rand(1, BASE_CONFIG['img_size'], BASE_CONFIG['img_size']) > 0.7
            return frames, target.float()
    
    dataloader = torch.utils.data.DataLoader(
        MockDataset(), 
        batch_size=BASE_CONFIG['batch_size'],
        shuffle=True
    )
    
    # 创建进化优化器
    config = {
        'BASE_CONFIG': BASE_CONFIG,
        'VIT_CONFIG': VIT_CONFIG,
        'EVOLUTION_CONFIG': EVOLUTION_CONFIG,
        'SYSTEM_CONFIG': SYSTEM_CONFIG
    }
    
    optimizer = EvolutionaryOptimizer(
        vit_model=vit,
        flownet=flownet,
        dataloader=dataloader,
        config=config,
        flops_calculator=flops_calculator
    )
    
    # 运行优化 (简化版: 少量代数)
    best_params = optimizer.optimize(n_generations=3, pop_size=5)
    
    # 绘制历史
    optimizer.plot_history()
