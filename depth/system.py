"""
DEPTH系统集成 - 整合所有组件 (申报书[1][4][5])
"""
import torch
import torch.nn as nn
import pytorch_lightning as pl
from tqdm import tqdm
import os
from utils.metrics import mIoU
from utils.visualization import save_results

class DEPTHSystem(pl.LightningModule):
    """DEPTH系统主模块 (申报书[1]整体框架)"""
    def __init__(self, vit, flownet, pruner, spike_recovery, optimizer, config, 
                 num_classes=19, save_dir="results"):
        """
        vit: Vision Transformer模型
        flownet: FLOWNE光流网络
        pruner: 动态剪枝模块
        spike_recovery: 脉冲恢复机制
        optimizer: 进化优化器
        config: 配置参数
        num_classes: 分割类别数
        save_dir: 结果保存目录
        """
        super().__init__()
        self.vit = vit
        self.flownet = flownet
        self.pruner = pruner
        self.spike_recovery = spike_recovery
        self.optimizer = optimizer
        self.config = config
        self.num_classes = num_classes
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # 系统状态
        self.optimize_params = config['SYSTEM_CONFIG']['optimize_params']
        self.best_params = None
        self.current_epoch = 0
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss(ignore_index=255)
        
        # 指标
        self.train_miou = mIoU(num_classes)
        self.val_miou = mIoU(num_classes)
    
    def on_train_start(self):
        """训练开始时运行进化优化 (申报书[5]阶段四)"""
        if self.optimize_params:
            print("\n[系统启动] 开始进化参数优化...")
            self.best_params = self.optimizer.optimize()
            
            # 应用最优参数
            rho_head, rho_token, tau = self.best_params
            self.pruner.set_params(rho_head, rho_token, tau)
            self.flownet.set_tau(tau)
            print(f"[系统] 应用最优参数: ρ_head={rho_head:.3f}, ρ_token={rho_token:.3f}, τ={tau:.3f}\n")
    
    def training_step(self, batch, batch_idx):
        """训练步骤"""
        frames, targets = batch
        B, T, C, H, W = frames.shape
        
        # 只使用第二帧进行训练 (简化)
        frame2 = frames[:, 1]
        target = targets.squeeze(1).long()
        
        # 生成运动掩码
        with torch.no_grad():
            motion_mask, _ = self.flownet(frames[:, 0], frame2)
        
        # 应用动态剪枝
        keep_token_mask, keep_head_mask = self.pruner.apply(motion_mask)
        
        # 前向传播
        features = self.vit.forward_features(frame2, keep_token_mask, keep_head_mask)
        
        # 脉冲恢复 (申报书[1]保护机制)
        if self._is_critical_event(motion_mask):
            features = self.spike_recovery(features, motion_mask)
        
        # 分割预测
        outputs = self.vit.decoder(features)
        outputs = F.interpolate(outputs, size=(H, W), mode='bilinear', align_corners=True)
        
        # 计算损失
        loss = self.criterion(outputs, target)
        
        # 更新mIoU
        self.train_miou.update(outputs, target)
        
        # 记录
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_miou', self.train_miou, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """验证步骤"""
        frames, targets = batch
        B, T, C, H, W = frames.shape
        
        # 只使用第二帧进行验证
        frame2 = frames[:, 1]
        target = targets.squeeze(1).long()
        
        # 生成运动掩码
        with torch.no_grad():
            motion_mask, _ = self.flownet(frames[:, 0], frame2)
        
        # 应用动态剪枝
        keep_token_mask, keep_head_mask = self.pruner.apply(motion_mask)
        
        # 前向传播
        features = self.vit.forward_features(frame2, keep_token_mask, keep_head_mask)
        
        # 脉冲恢复
        if self._is_critical_event(motion_mask):
            features = self.spike_recovery(features, motion_mask)
        
        # 分割预测
        outputs = self.vit.decoder(features)
        outputs = F.interpolate(outputs, size=(H, W), mode='bilinear', align_corners=True)
        
        # 计算损失
        loss = self.criterion(outputs, target)
        
        # 更新mIoU
        self.val_miou.update(outputs, target)
        
        # 记录
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_miou', self.val_miou, on_step=False, on_epoch=True, prog_bar=True)
        
        # 保存可视化结果 (每5个epoch)
        if batch_idx == 0 and self.current_epoch % 5 == 0:
            save_path = os.path.join(self.save_dir, f"epoch_{self.current_epoch}_batch_{batch_idx}.png")
            save_results(frame2[0], target[0], outputs[0], save_path)
        
        return loss
    
    def on_validation_end(self):
        """验证结束时保存指标"""
        print(f"\nEpoch {self.current_epoch} | Val mIoU: {self.val_miou.compute():.4f}")
        self.current_epoch += 1
    
    def configure_optimizers(self):
        """配置优化器"""
        optimizer = torch.optim.Adam(self.vit.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5, verbose=True
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_miou',
                'interval': 'epoch',
                'frequency': 1
            }
        }
    
    def _is_critical_event(self, motion_mask):
        """检测关键事件 (申报书[1]: 危险因素)"""
        motion_ratio = motion_mask.mean().item()
        return motion_ratio > self.config['SPIKE_CONFIG']['critical_threshold']

# 测试系统集成
if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import BASE_CONFIG, VIT_CONFIG, SYSTEM_CONFIG
    from vit import VisionTransformer
    from flownet import FLOWNE
    from pruner import DynamicPruner
    from spike import SpikeRecovery
    from evolutionary import EvolutionaryOptimizer
    from utils.flops import FLOPsCalculator
    
    # 创建配置
    config = {
        'BASE_CONFIG': BASE_CONFIG,
        'VIT_CONFIG': VIT_CONFIG,
        'SYSTEM_CONFIG': SYSTEM_CONFIG
    }
    
    # 初始化组件
    vit = VisionTransformer(
        img_size=BASE_CONFIG['img_size'],
        patch_size=16,
        num_classes=BASE_CONFIG['num_classes'],
        **VIT_CONFIG
    ).to(BASE_CONFIG['device'])
    
    flownet = FLOWNE().to(BASE_CONFIG['device'])
    
    pruner = DynamicPruner(vit, config)
    
    spike_recovery = SpikeRecovery(
        input_size=(VIT_CONFIG['embed_dim'], 14, 14),
        config=config
    ).to(BASE_CONFIG['device'])
    
    # 创建FLOPs计算器
    flops_calculator = FLOPsCalculator(
        img_size=BASE_CONFIG['img_size'],
        patch_size=16,
        vit_config=VIT_CONFIG,
        base_flops=SYSTEM_CONFIG['flops_base']
    )
    
    # 模拟数据加载器
    class MockDataset(torch.utils.data.Dataset):
        def __init__(self, size=100):
            self.size = size
        
        def __len__(self):
            return self.size
        
        def __getitem__(self, idx):
            frames = torch.rand(2, 3, BASE_CONFIG['img_size'], BASE_CONFIG['img_size'])
            target = torch.randint(0, BASE_CONFIG['num_classes'], 
                                  (1, BASE_CONFIG['img_size'], BASE_CONFIG['img_size']))
            return frames, target.float()
    
    dataloader = torch.utils.data.DataLoader(
        MockDataset(), 
        batch_size=BASE_CONFIG['batch_size'],
        shuffle=True
    )
    
    # 创建进化优化器
    optimizer = EvolutionaryOptimizer(
        vit_model=vit,
        flownet=flownet,
        dataloader=dataloader,
        config=config,
        flops_calculator=flops_calculator
    )
    
    # 创建DEPTH系统
    depth_system = DEPTHSystem(
        vit=vit,
        flownet=flownet,
        pruner=pruner,
        spike_recovery=spike_recovery,
        optimizer=optimizer,
        config=config,
        num_classes=BASE_CONFIG['num_classes']
    )
    
    # 模拟训练
    print("\n[系统测试] 开始模拟训练...")
    for epoch in range(2):  # 仅运行2个epoch测试
        print(f"\nEpoch {epoch+1}/2")
        
        # 训练步骤
        depth_system.train()
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Training")):
            if batch_idx >= 3:  # 仅处理3个batch
                break
            loss = depth_system.training_step(batch, batch_idx)
        
        # 验证步骤
        depth_system.eval()
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Validation")):
            if batch_idx >= 2:  # 仅处理2个batch
                break
            loss = depth_system.validation_step(batch, batch_idx)
        
        depth_system.on_validation_end()
    
    print("\n[系统测试] 完成! 模拟训练成功运行。")
    print("完整实现包含: 动态剪枝 + 进化优化 + 脉冲恢复机制 (申报书[1][3][6])")
