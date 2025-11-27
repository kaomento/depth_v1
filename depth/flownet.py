"""
FLOWNE (神经光流网络) 实现 - 用于提取运动区域 (申报书[3])
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class FlowNetC(nn.Module):
    """简化版FlowNetC (申报书[3] FLOWNE基础)"""
    def __init__(self):
        super(FlowNetC, self).__init__()
        
        # 特征提取分支
        self.conv1a = self._conv(3, 16, kernel_size=3, stride=2)
        self.conv1aa = self._conv(16, 16, kernel_size=3, stride=1)
        self.conv1b = self._conv(16, 16, kernel_size=3, stride=2)
        self.conv2a = self._conv(16, 32, kernel_size=3, stride=2)
        self.conv2aa = self._conv(32, 32, kernel_size=3, stride=1)
        self.conv2b = self._conv(32, 32, kernel_size=3, stride=2)
        self.conv3a = self._conv(32, 64, kernel_size=3, stride=2)
        self.conv3aa = self._conv(64, 64, kernel_size=3, stride=1)
        self.conv3b = self._conv(64, 64, kernel_size=3, stride=2)
        self.conv4a = self._conv(64, 96, kernel_size=3, stride=2)
        self.conv4aa = self._conv(96, 96, kernel_size=3, stride=1)
        self.conv4b = self._conv(96, 96, kernel_size=3, stride=2)
        self.conv5a = self._conv(96, 128, kernel_size=3, stride=2)
        self.conv5aa = self._conv(128, 128, kernel_size=3, stride=1)
        self.conv5b = self._conv(128, 128, kernel_size=3, stride=2)
        self.conv6aa = self._conv(128, 192, kernel_size=3, stride=2)
        self.conv6a = self._conv(192, 192, kernel_size=3, stride=1)
        self.conv6b = self._conv(192, 192, kernel_size=3, stride=2)

        # 相关层
        self.corr = Correlation(pad_size=4, kernel_size=1, max_displacement=4, stride1=1, stride2=1, corr_multiply=1)

        # 光流估计
        self.leakyRELU = nn.LeakyReLU(0.1, inplace=True)
        nd = (4*2+1)**2  # 申报书[3]相关层参数
        self.conv_redir = self._conv(192, 32, kernel_size=1, stride=1)
        self.conv3_1 = self._conv(nd + 32 + 128, 256)
        self.conv4_1 = self._conv(256, 256)
        self.conv5_1 = self._conv(256, 256)
        self.conv6_1 = self._conv(256, 256)
        
        self.predict_flow6 = self._conv(256, 2, kernel_size=3, stride=1, padding=1)
        self.predict_flow5 = self._conv(256 + 2 + 2, 2, kernel_size=3, stride=1, padding=1)
        self.predict_flow4 = self._conv(256 + 2 + 2, 2, kernel_size=3, stride=1, padding=1)
        self.predict_flow3 = self._conv(256 + 2 + 2, 2, kernel_size=3, stride=1, padding=1)
        self.predict_flow2 = self._conv(256 + 2 + 2, 2, kernel_size=3, stride=1, padding=1)
        
        self.upsampled_flow6_to_5 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow5_to_4 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow4_to_3 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow3_to_2 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    nn.init.uniform_(m.bias)
                nn.init.xavier_uniform_(m.weight)
                if isinstance(m, nn.Conv2d):
                    if m.stride[0] > 1:
                        nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.LeakyReLU):
                m.inplace = True

    def _conv(self, in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, 
                    padding=padding, dilation=dilation, bias=True),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x1, x2):
        # 特征提取
        x1 = self.conv1b(self.conv1aa(self.conv1a(x1)))
        x2 = self.conv1b(self.conv1aa(self.conv1a(x2)))
        
        x1 = self.conv2b(self.conv2aa(self.conv2a(x1)))
        x2 = self.conv2b(self.conv2aa(self.conv2a(x2)))
        
        x1 = self.conv3b(self.conv3aa(self.conv3a(x1)))
        x2 = self.conv3b(self.conv3aa(self.conv3a(x2)))
        x1_ = x1
        
        x2 = self.conv3b(self.conv3aa(self.conv3a(x2)))
        x1 = self.conv4a(x1)
        x2 = self.conv4a(x2)
        x1 = self.conv4b(self.conv4aa(x1))
        x2 = self.conv4b(self.conv4aa(x2))
        x1_ = x1
        
        # 相关层
        out_corr = self.corr(x1, x2)
        out_corr = self.leakyRELU(out_corr)
        
        # 合并相关特征和x1特征
        x = torch.cat((self.conv_redir(x1), out_corr), 1)
        
        # 光流估计
        x = self.conv3_1(x)
        flow6 = self.predict_flow6(x)
        up_flow6 = self.upsampled_flow6_to_5(flow6)
        up_feat6 = self.upsample1(x)
        
        # 后续层 (简化版)
        flow = flow6
        for i in range(5, 1, -1):
            up_flow = getattr(self, f"upsampled_flow{i}_to_{i-1}")(flow)
            flow = getattr(self, f"predict_flow{i-1}")(torch.cat([up_flow, up_feat6], 1))
        
        # 返回上采样到原始尺寸的光流
        flow_up = F.interpolate(flow, scale_factor=4, mode='bilinear', align_corners=True) * 4.0
        return flow_up

class Correlation(nn.Module):
    """相关层实现 (申报书[3]光流计算基础)"""
    def __init__(self, pad_size=3, kernel_size=3, max_displacement=20, stride1=1, stride2=2, corr_multiply=1):
        super(Correlation, self).__init__()
        self.pad_size = pad_size
        self.kernel_size = kernel_size
        self.max_displacement = max_displacement
        self.stride1 = stride1
        self.stride2 = stride2
        self.corr_multiply = corr_multiply

    def forward(self, input1, input2):
        return correlation_function.apply(input1, input2, self.pad_size, self.kernel_size, 
                                         self.max_displacement, self.stride1, self.stride2, self.corr_multiply)

class correlation_function(torch.autograd.Function):
    """相关层前向计算"""
    @staticmethod
    def forward(ctx, input1, input2, pad_size, kernel_size, max_displacement, stride1, stride2, corr_multiply):
        ctx.save_for_backward(input1, input2)
        ctx.pad_size = pad_size
        ctx.kernel_size = kernel_size
        ctx.max_displacement = max_displacement
        ctx.stride1 = stride1
        ctx.stride2 = stride2
        ctx.corr_multiply = corr_multiply
        
        # 简化版: 使用欧氏距离代替相关计算
        b, c, h, w = input1.shape
        max_disp = (2 * max_displacement + 1)
        
        # 创建输出张量
        out = torch.zeros(b, max_disp * max_disp, h // stride1, w // stride1, device=input1.device)
        
        # 简化计算 (实际应使用完整相关计算)
        for i in range(max_disp):
            for j in range(max_disp):
                dx = (i - max_displacement // 2) * stride2
                dy = (j - max_displacement // 2) * stride2
                shifted = F.pad(input2, (pad_size+dx, pad_size-dx, pad_size+dy, pad_size-dy))[:, :, pad_size:-pad_size, pad_size:-pad_size]
                out[:, i*max_disp+j, :, :] = (input1 * shifted).sum(1)
        
        return out / (c * kernel_size * kernel_size)  # 归一化

class FLOWNE(nn.Module):
    """神经光流网络 (申报书[3] FLOWNE)"""
    def __init__(self, tau=0.1):
        super().__init__()
        self.flownet = FlowNetC()
        self.tau = tau  # 光流运动阈值 (申报书[3] τ)
    
    def set_tau(self, tau):
        """动态设置光流阈值 (供进化算法调用)"""
        self.tau = max(0.05, min(0.3, tau))  # 限制在合理范围
    
    def forward(self, frame1, frame2):
        """计算光流并生成运动掩码"""
        # 标准化输入 (FlowNetC期望[-20, 20]范围)
        frame1_norm = (frame1 * 255.0 - 128.0) / 128.0
        frame2_norm = (frame2 * 255.0 - 128.0) / 128.0
        
        # 计算光流
        flow = self.flownet(frame1_norm, frame2_norm)  # (B, 2, H, W)
        
        # 计算光流幅值
        flow_magnitude = torch.sqrt(flow[:, 0]**2 + flow[:, 1]**2)  # (B, H, W)
        
        # 生成运动掩码 (申报书[3]: τ为运动阈值)
        motion_mask = (flow_magnitude > self.tau).float()  # (B, H, W)
        
        return motion_mask.unsqueeze(1), flow  # (B, 1, H, W), (B, 2, H, W)

# 测试FLOWNE
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    flownet = FLOWNE()
    
    # 模拟输入 (B=2, C=3, H=224, W=224)
    frame1 = torch.randn(2, 3, 224, 224)
    frame2 = torch.randn(2, 3, 224, 224)
    
    # 前向传播
    motion_mask, flow = flownet(frame1, frame2)
    
    print(f"Motion mask shape: {motion_mask.shape}")
    print(f"Flow shape: {flow.shape}")
    
    # 可视化
    plt.figure(figsize=(12, 4))
    plt.subplot(131)
    plt.imshow(frame1[0].permute(1, 2, 0).detach().numpy())
    plt.title("Frame 1")
    plt.subplot(132)
    plt.imshow(frame2[0].permute(1, 2, 0).detach().numpy())
    plt.title("Frame 2")
    plt.subplot(133)
    plt.imshow(motion_mask[0, 0].detach().numpy(), cmap='hot')
    plt.title(f"Motion Mask (tau={flownet.tau})")
    plt.colorbar()
    plt.savefig("flownet_sample.png")
    print("Saved FLOWNE visualization to flownet_sample.png")
