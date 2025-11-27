"""
DEPTH系统全局配置
基于申报书[1][2][3]技术参数设定
"""
import os

# 基础配置
BASE_CONFIG = {
    'seed': 42,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'img_size': 224,
    'patch_size': 16,
    'num_classes': 19,  # Cityscapes类别数 (申报书[4]自动驾驶场景)
    'batch_size': 4,
    'num_workers': 2,
}

# ViT配置 (申报书[4] Vision Transformer)
VIT_CONFIG = {
    'embed_dim': 768,
    'depth': 12,
    'num_heads': 12,
    'mlp_ratio': 4.0,
    'qkv_bias': True,
    'drop_rate': 0.0,
    'attn_drop_rate': 0.0,
    'drop_path_rate': 0.1,
}

# FLOWNE配置 (申报书[3]光流网络)
FLOWNE_CONFIG = {
    'model': 'small',  # 使用简化版RAFT
    'tau_min': 0.05,   # 光流阈值下限
    'tau_max': 0.3,    # 光流阈值上限
}

# 动态剪枝配置 (申报书[2][3])
PRUNER_CONFIG = {
    'rho_head_min': 0.0,
    'rho_head_max': 0.8,
    'rho_token_min': 0.0,
    'rho_token_max': 0.9,
}

# 进化算法配置 (申报书[3] CMA-ES)
EVOLUTION_CONFIG = {
    'pop_size': 15,    # 申报书[2]图a: 30个编码个体 → 简化为15 (资源优化)
    'n_generations': 10,
    'alpha': 0.7,      # 申报书[3]推荐权重
    'beta': 0.3,
    'cxpb': 0.5,
    'mutpb': 0.2,
}

# 脉冲恢复机制配置 (申报书[1][3][6])
SPIKE_CONFIG = {
    'v_threshold': 0.5,  # 申报书[3] Vth
    'tau': 2.0,          # 时间常数
    'critical_threshold': 0.3,  # 关键事件阈值
}

# 系统配置
SYSTEM_CONFIG = {
    'optimize_params': True,  # 训练阶段启用进化优化
    'flops_base': 120e9,      # ViT-B/16基础FLOPs
    'log_dir': 'logs',
    'checkpoint_dir': 'checkpoints',
}

# 创建必要目录
os.makedirs(SYSTEM_CONFIG['log_dir'], exist_ok=True)
os.makedirs(SYSTEM_CONFIG['checkpoint_dir'], exist_ok=True)
