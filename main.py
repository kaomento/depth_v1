"""
DEPTH (Dynamic Evolutionary Pruning Transformer) 主程序
基于申报书[1]-[6]实现的完整系统
"""
import os
import sys
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import warnings

# 忽略警告
warnings.filterwarnings("ignore")

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入项目模块
from config import BASE_CONFIG, VIT_CONFIG, SYSTEM_CONFIG
from datasets.cityscapes import get_cityscapes_loaders
from depth.vit import VisionTransformer
from depth.flownet import FLOWNE
from depth.pruner import DynamicPruner
from depth.spike import SpikeRecovery
from depth.evolutionary import EvolutionaryOptimizer
from depth.system import DEPTHSystem
from utils.flops import FLOPsCalculator

def main():
    """主函数"""
    print("="*80)
    print("DEPTH (Dynamic Evolutionary Pruning Transformer) 系统启动")
    print("基于类脑智能的视觉感知技术 - 天津大学大创项目")
    print("="*80)
    
    # 设置随机种子
    pl.seed_everything(BASE_CONFIG['seed'])
    
    # 1. 准备数据
    print("\n[步骤1/5] 准备数据集...")
    CITYSCAPES_ROOT = "/path/to/cityscapes"  # 替换为实际路径
    
    if not os.path.exists(CITYSCAPES_ROOT):
        print(f"警告: Cityscapes数据集未找到 ({CITYSCAPES_ROOT})")
        print("使用模拟数据集进行测试...")
        
        # 创建模拟数据集
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
        
        train_set = MockDataset(size=200)
        val_set = MockDataset(size=50)
        
        train_loader = torch.utils.data.DataLoader(
            train_set, 
            batch_size=BASE_CONFIG['batch_size'], 
            shuffle=True, 
            num_workers=BASE_CONFIG['num_workers']
        )
        val_loader = torch.utils.data.DataLoader(
            val_set, 
            batch_size=BASE_CONFIG['batch_size'], 
            shuffle=False, 
            num_workers=BASE_CONFIG['num_workers']
        )
    else:
        print(f"加载Cityscapes数据集: {CITYSCAPES_ROOT}")
        train_loader, val_loader = get_cityscapes_loaders(
            CITYSCAPES_ROOT,
            batch_size=BASE_CONFIG['batch_size'],
            img_size=BASE_CONFIG['img_size'],
            num_workers=BASE_CONFIG['num_workers']
        )
    
    print(f"训练集大小: {len(train_loader.dataset)}")
    print(f"验证集大小: {len(val_loader.dataset)}")
    
    # 2. 初始化模型组件
    print("\n[步骤2/5] 初始化模型组件...")
    
    # Vision Transformer
    print("  - 初始化Vision Transformer...")
    vit = VisionTransformer(
        img_size=BASE_CONFIG['img_size'],
        patch_size=16,
        num_classes=BASE_CONFIG['num_classes'],
        **VIT_CONFIG
    ).to(BASE_CONFIG['device'])
    
    # FLOWNE光流网络
    print("  - 初始化FLOWNE光流网络...")
    flownet = FLOWNE().to(BASE_CONFIG['device'])
    
    # 动态剪枝模块
    print("  - 初始化动态剪枝模块...")
    pruner = DynamicPruner(vit, {
        'PRUNER_CONFIG': BASE_CONFIG,
        'FLOWNE_CONFIG': BASE_CONFIG
    })
    
    # 类脑脉冲恢复机制
    print("  - 初始化脉冲恢复机制...")
    spike_recovery = SpikeRecovery(
        input_size=(VIT_CONFIG['embed_dim'], 14, 14),
        config={'SPIKE_CONFIG': BASE_CONFIG}
    ).to(BASE_CONFIG['device'])
    
    # FLOPs计算器
    print("  - 初始化FLOPs计算器...")
    flops_calculator = FLOPsCalculator(
        img_size=BASE_CONFIG['img_size'],
        patch_size=16,
        vit_config=VIT_CONFIG,
        base_flops=SYSTEM_CONFIG['flops_base']
    )
    
    # 进化算法优化器
    print("  - 初始化进化算法优化器...")
    optimizer = EvolutionaryOptimizer(
        vit_model=vit,
        flownet=flownet,
        dataloader=val_loader,
        config={
            'BASE_CONFIG': BASE_CONFIG,
            'VIT_CONFIG': VIT_CONFIG,
            'EVOLUTION_CONFIG': BASE_CONFIG,
            'SYSTEM_CONFIG': SYSTEM_CONFIG
        },
        flops_calculator=flops_calculator
    )
    
    # 3. 创建DEPTH系统
    print("\n[步骤3/5] 创建DEPTH系统...")
    depth_system = DEPTHSystem(
        vit=vit,
        flownet=flownet,
        pruner=pruner,
        spike_recovery=spike_recovery,
        optimizer=optimizer,
        config={
            'BASE_CONFIG': BASE_CONFIG,
            'VIT_CONFIG': VIT_CONFIG,
            'SYSTEM_CONFIG': SYSTEM_CONFIG,
            'SPIKE_CONFIG': BASE_CONFIG
        },
        num_classes=BASE_CONFIG['num_classes'],
        save_dir=SYSTEM_CONFIG['log_dir']
    )
    
    # 4. 配置训练器
    print("\n[步骤4/5] 配置训练器...")
    
    # 回调函数
    checkpoint_callback = ModelCheckpoint(
        dirpath=SYSTEM_CONFIG['checkpoint_dir'],
        filename='depth-{epoch:02d}-{val_miou:.4f}',
        save_top_k=3,
        monitor='val_miou',
        mode='max'
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val_miou',
        patience=10,
        mode='max',
        verbose=True
    )
    
    # 训练器
    trainer = pl.Trainer(
        max_epochs=30,
        devices=1 if BASE_CONFIG['device'] == 'cuda' else None,
        accelerator=BASE_CONFIG['device'],
        callbacks=[checkpoint_callback, early_stop_callback],
        log_every_n_steps=10,
        check_val_every_n_epoch=1
    )
    
    # 5. 训练模型
    print("\n[步骤5/5] 开始训练DEPTH系统...")
    print("="*80)
    
    try:
        trainer.fit(
            depth_system,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader
        )
        
        # 保存最终模型
        final_path = os.path.join(SYSTEM_CONFIG['checkpoint_dir'], 'depth_final.ckpt')
        trainer.save_checkpoint(final_path)
        print(f"\n训练完成! 模型已保存至: {final_path}")
        
        # 打印最优参数
        if depth_system.best_params:
            rho_head, rho_token, tau = depth_system.best_params
            print("\n[系统参数] 最优配置:")
            print(f"  - 注意力头剪枝率 (ρ_head): {rho_head:.3f}")
            print(f"  - Token剪枝率 (ρ_token): {rho_token:.3f}")
            print(f"  - 光流阈值 (τ): {tau:.3f}")
            print(f"  - 系统FLOPs减少: {100*(1 - depth_system.optimizer.history[-1]['flops_ratio']):.1f}%")
            print(f"  - 验证集mIoU: {depth_system.optimizer.history[-1]['miou']:.4f}")
        
    except Exception as e:
        print(f"\n训练过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("\nDEPTH系统执行完毕。")

if __name__ == "__main__":
    main()
