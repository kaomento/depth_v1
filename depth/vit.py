"""
Vision Transformer实现 - 用于语义分割 (申报书[4])
"""
import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class PatchEmbed(nn.Module):
    """图像分块嵌入 (申报书[4] ViT基础)"""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        x = self.proj(x)  # (B, E, H/P, W/P)
        x = x.flatten(2)  # (B, E, N)
        x = x.transpose(1, 2)  # (B, N, E)
        return x

class Attention(nn.Module):
    """多头自注意力 (申报书[4]全局注意力)"""
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, keep_head_mask=None):
        """
        x: (B, N, D)
        keep_head_mask: (num_heads,) 1表示保留，0表示剪枝
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # 应用头剪枝 (申报书[2][3])
        if keep_head_mask is not None:
            attn = attn * keep_head_mask.view(1, self.num_heads, 1, 1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MLP(nn.Module):
    """MLP层"""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class TransformerBlock(nn.Module):
    """Transformer块 (申报书[4] ViT Encoder)"""
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., 
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = nn.Identity()  # 简化版
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, keep_head_mask=None):
        x = x + self.drop_path(self.attn(self.norm1(x), keep_head_mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class VisionTransformer(nn.Module):
    """Vision Transformer主干网络 (申报书[4])"""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, 
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.n_patches
        
        # 位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Transformer编码器
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i],
                norm_layer=nn.LayerNorm
            )
            for i in range(depth)
        ])
        
        # 分割头
        self.decoder = nn.Sequential(
            nn.Conv2d(embed_dim, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )
        
        # 初始化
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)
        
        # 用于动态剪枝的中间特征
        self.tokens = None
        self.keep_token_mask = None
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward_features(self, x, keep_token_mask=None, keep_head_mask=None):
        """前向传播 (支持动态剪枝)"""
        B = x.shape[0]
        x = self.patch_embed(x)  # (B, N, E)
        
        # 保存tokens用于剪枝
        self.tokens = x.clone().detach()
        
        # 应用Token剪枝 (申报书[2][3])
        if keep_token_mask is not None:
            x = x * keep_token_mask.unsqueeze(-1)  # (B, N, E)
        
        # 位置编码
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=B)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embed
        
        # Transformer编码器
        for blk in self.blocks:
            x = blk(x, keep_head_mask)
        
        # 移除cls_token
        x = x[:, 1:, :]
        
        # 重塑为特征图
        H = W = int(x.shape[1] ** 0.5)
        x = rearrange(x, 'b (h w) e -> b e h w', h=H, w=W)
        
        return x
    
    def forward(self, x, keep_token_mask=None, keep_head_mask=None):
        """完整前向传播"""
        x = self.forward_features(x, keep_token_mask, keep_head_mask)
        x = self.decoder(x)
        return x

# 测试ViT
if __name__ == "__main__":
    import torchinfo
    
    vit = VisionTransformer(
        img_size=224,
        patch_size=16,
        num_classes=19,  # Cityscapes类别
        embed_dim=768,
        depth=12,
        num_heads=12
    )
    
    print("Vision Transformer Summary:")
    torchinfo.summary(vit, input_size=(1, 3, 224, 224))
    
    # 测试动态剪枝
    x = torch.randn(2, 3, 224, 224)
    
    # 常规前向传播
    out_normal = vit(x)
    print(f"Normal output shape: {out_normal.shape}")
    
    # 应用Token剪枝 (保留50%)
    keep_token_mask = torch.rand(2, 196) > 0.5  # 196 = (224/16)^2
    out_pruned = vit(x, keep_token_mask=keep_token_mask.float())
    print(f"Pruned output shape: {out_pruned.shape}")
    
    # 应用头剪枝 (保留6个头)
    keep_head_mask = torch.zeros(12)
    keep_head_mask[:6] = 1
    out_head_pruned = vit(x, keep_head_mask=keep_head_mask)
    print(f"Head pruned output shape: {out_head_pruned.shape}")
