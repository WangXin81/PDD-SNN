#中间层全序列保留
import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import surrogate, functional, base
from wavelet_layers import Haar2DForward, Haar2DInverse
from color_spaces import (RGB2Lab, Lab2RGB)
import os
import numpy as np
from glob import glob
import random
import logging
from neuron import MultiStepNegIFNode

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== 基础组件 ====================

# class TimeAverage(nn.Module):
#     """
#     速率编码读出层：将时间维度 T 上的脉冲求平均。
#     Input: [T, B, C, H, W]
#     Output: [B, C, H, W]
#     """
#     def __init__(self):
#         super().__init__()
#     def forward(self, x_seq):
#         return x_seq.mean(dim=0) 

class TimeAverage(nn.Module):
    def __init__(self, scale=1.0): 
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float32)) 
    
    def forward(self, x_seq):
        return x_seq.mean(0) * self.scale

def time_distributed_conv(layer, x_seq):
    """
    辅助函数：让普通 Conv2d 处理 [T, B, C, H, W] 数据
    """
    T, B, C, H, W = x_seq.shape
    x_flat = x_seq.reshape(T * B, C, H, W)
    out_flat = layer(x_flat)
    _, C_out, H_out, W_out = out_flat.shape
    out_seq = out_flat.reshape(T, B, C_out, H_out, W_out)
    return out_seq

class SeqConv2d(nn.Module):
    """
    包装类：将 nn.Conv2d 包装为支持 [T, B, ...] 输入的层
    用于 FusionModule 和 FeaturePyramid 中保持时序维度
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)

    def forward(self, x_seq):
        return time_distributed_conv(self.conv, x_seq)

class TemporalEncoder(nn.Module):
    """
    将静态输入扩展为时序输入
    """
    def __init__(self, time_steps=5, mode="repeat", noise_scale=0.1):
        super(TemporalEncoder, self).__init__()
        self.time_steps = time_steps
        self.mode = mode
        self.noise_scale = noise_scale

    def forward(self, x):
        if self.mode == "repeat":
            seq = x.unsqueeze(1).repeat(1, self.time_steps, *([1] * (x.ndim - 1)))
        elif self.mode == "noisy":
            seqs = []
            for t in range(self.time_steps):
                noise = torch.randn_like(x) * self.noise_scale
                seqs.append(x + noise)
            seq = torch.stack(seqs, dim=1)
        else:
            raise ValueError(f"未知模式: {self.mode}")
        return seq.permute(1, 0, 2, 3, 4) # [T, B, C, H, W]

# ==================== 神经元定义 (含稳定性修复) ====================

# class MultiStepPmLIFNode(base.MemoryModule):
#     def __init__(self, v_th=0.1, v_reset=0.0, tau=2.0, surrogate_function=surrogate.Sigmoid(alpha=1.0)):
#         super().__init__()
#         self.register_buffer('v_th', torch.tensor(v_th, dtype=torch.float32))
#         self.v_reset = v_reset
#         self.tau = nn.Parameter(torch.tensor(tau, dtype=torch.float32))
#         self.surrogate_function = surrogate_function
#         self.register_memory('v', None)

#     def forward(self, x_seq: torch.Tensor):
        
#         with torch.no_grad():
#             self.tau.data.clamp_(0.5, 5.0)
        
#         T, B = x_seq.shape[0], x_seq.shape[1]
#         if self.v is None:
#             self.v = torch.zeros_like(x_seq[0])    
#         spikes = []
#         # [保险1] 限制 tau 防止除零
#         # effective_tau = self.tau.clamp(min=0.1) 
        
#         for t in range(T):
#             x = x_seq[t]
            
#             self.v = self.v + (1.0 / self.tau) * (-(self.v - self.v_reset) + x)
#             self.v = torch.clamp(self.v, min=-10.0, max=10.0)
#             s = self.surrogate_function(self.v - self.v_th)
#             v_after_fire = self.v - s * self.v_th
#             self.v = torch.clamp(v_after_fire, min=-2.0, max=2.0)
            
#             # 膜电位更新
# #             h = self.v + (1.0 / self.tau) * (-(self.v - self.v_reset) + x)
            
# #             s = self.surrogate_function(h - self.v_th)
# #             # self.v = h * (1 - s) + self.v_reset * s
# #             self.v = h - s * self.v_th
#             spikes.append(s)
            
            
# #         # ================= [开始插入诊断代码] =================
# #         # 1. 先把结果存下来，不要直接 return
# #         out_spikes = torch.stack(spikes, dim=0)

# #         # 2. 限制打印频率：防止刷屏，每层只在第一次运行时打印
# #         if not hasattr(self, 'debug_printed'):
# #             # 计算发放率 (0.0 表示全死，1.0 表示全发)
# #             firing_rate = out_spikes.float().mean().item()
# #             max_val = out_spikes.float().max().item()
            
# #             # 获取层名称
# #             layer_name = getattr(self, 'name', self.__class__.__name__)
# #             # 获取当前阈值
# #             current_th = self.v_th.item() if hasattr(self, 'v_th') else '未知'

# #             # 打印详细“体检报告”
# #             print(f"\n[Spike侦探] Layer: {layer_name} (ID: {id(self)})")
# #             print(f"   ├── 🛑 阈值 (v_th): {current_th}")
# #             print(f"   ├── 📊 发放率 (Firing Rate): {firing_rate:.8f}")
# #             print(f"   └── 📈 脉冲最大值: {max_val:.4f}")

# #             # 智能报警
# #             if firing_rate == 0.0:
# #                 print(f"   ⚠️⚠️⚠️ [严重警告] 此层已死 (Dead Silence)！梯度必为 0！ ⚠️⚠️⚠️")
# #             elif firing_rate < 0.001:
# #                 print(f"   ⚠️ [警告] 极度稀疏 (Rate < 0.1%)，可能有危险")
# #             else:
# #                 print(f"   ✅ [正常] 神经元存活")

# #             # 标记已打印
# #             self.debug_printed = True
        
# #         # 3. 返回刚才存下来的结果
# #         return out_spikes
# #         # ================= [结束插入] =================
        

#         return torch.stack(spikes, dim=0)

    
class MultiStepPmLIFNode(base.MemoryModule):
    def __init__(self, v_th=0.1, v_reset=0.0, tau=2.0, surrogate_function=surrogate.Sigmoid(alpha=4.0), 
                 soft_reset=False):
        super().__init__()
        self.register_buffer('v_th', torch.tensor(v_th, dtype=torch.float32))
        self.v_reset = v_reset
        self.tau = nn.Parameter(torch.tensor(tau, dtype=torch.float32))
        self.surrogate_function = surrogate_function
        self.soft_reset = soft_reset
        self.register_memory('v', None)

    def forward(self, x_seq: torch.Tensor):
        # 1. 限制 Tau，防止除以零
        with torch.no_grad():
            self.tau.data.clamp_(0.5, 5.0)
        
        T, B = x_seq.shape[0], x_seq.shape[1]
        if self.v is None:
            self.v = torch.zeros_like(x_seq[0])    
        spikes = []
#         # =================【新增：直接打印第二阶段信号】=================
#         # 1. 判断是否是训练模式 (self.training)
#         # 2. 判断输入尺寸是否小于 100 (特征图高度 H < 100 说明是 LR 输入，即重建阶段)
#         # 3. 加一点随机概率防止刷屏 (1% 概率)
#         if self.training and x_seq.shape[-2] < 100 and random.random() < 0.01:
#             print(f"\n>>> [Stage 2 (Binary)] Layer {id(self)%1000} Signal Check <<<")
#             print(f"    Shape: {x_seq.shape} (Small input confirmed)")
#             print(f"    Max Input: {x_seq.max().item():.6f}")
#             print(f"    Mean Input: {x_seq.mean().item():.6f}")
#             print(f"    Threshold (v_th): {self.v_th.item():.4f}")
            
#             # 严重警告：如果输入最大值连阈值的 10% 都不到，说明该层大概率“饿死”了
#             if x_seq.max().item() < self.v_th.item() * 0.1:
#                 print("    🚨 警告: 信号强度过低！神经元无法发放脉冲！")
#         # =============================================================
        for t in range(T):
            x = x_seq[t]
            # 膜电位更新
            self.v = self.v + (1.0 / self.tau) * (-(self.v - self.v_reset) + x)
            
            # 发放脉冲
            s = self.surrogate_function(self.v - self.v_th)
            
            
#             # ========== 【插入这段调试代码】 ==========
#             # 只在第0个Batch，第0个时间步打印，避免刷屏
#             if t == 0 and torch.rand(1).item() < 0.01: # 1%的概率打印，防止日志爆炸
#                 v_max = self.v.max().item()
#                 v_mean = self.v.mean().item()
#                 s_sum = s.sum().item()
#                 print(f"!!! 神经元监控 | V_th: {self.v_th.item():.2f} | V_max: {v_max:.4f} | V_mean: {v_mean:.4f} | Spikes: {s_sum} !!!")
#             # =======================================
            
            
            # === [修改这里：重置逻辑] ===
            if self.soft_reset:
                # [软重置]：减去阈值，保留剩余电位
                v_after_fire = self.v - s * self.v_th
            else:
                # [硬重置]：强制归零 (或归为 v_reset)
                v_after_fire = self.v * (1.0 - s) + self.v_reset * s
            # =========================
            
            # # ================= [必须修改这里：硬重置] =================
            # # 你的原代码可能是: v_after_fire = self.v - s * self.v_th (这是软重置 -> 会炸)
            # # 请改为:
            # v_after_fire = self.v * (1.0 - s) + self.v_reset * s
            # # =======================================================
            
            self.v = torch.clamp(v_after_fire, min=-2.0, max=2.0)
            spikes.append(s)
            
        return torch.stack(spikes, dim=0)    
    
class MultiStepTernaryPmLIFNode(base.MemoryModule):
    """三元脉冲神经元 {-1, 0, 1} [已修正为硬重置]"""
    def __init__(self, v_th=0.1, v_reset=0.0, tau=2.0, surrogate_function=surrogate.Sigmoid(alpha=4.0), 
                 soft_reset=False):
        super().__init__()
        self.register_buffer('v_th', torch.tensor(v_th, dtype=torch.float32))
        self.v_reset = v_reset
        self.tau = nn.Parameter(torch.tensor(tau, dtype=torch.float32))
        self.surrogate_function = surrogate_function
        self.register_memory('v', None)
        self.soft_reset = soft_reset

    def forward(self, x_seq: torch.Tensor):
        T, B = x_seq.shape[0], x_seq.shape[1]
        if self.v is None:
            self.v = torch.zeros_like(x_seq[0])
            
        with torch.no_grad():
            self.tau.data.clamp_(0.5, 5.0)

        spikes = []
#        # =================【新增：直接打印第二阶段信号】=================
#         # 1. 判断是否是训练模式 (self.training)
#         # 2. 判断输入尺寸是否小于 100 (特征图高度 H < 100 说明是 LR 输入，即重建阶段)
#         # 3. 加一点随机概率防止刷屏 (1% 概率)
#         if self.training and x_seq.shape[-2] < 100 and random.random() < 0.01:
#             print(f"\n>>> [Stage 2 (Binary)] Layer {id(self)%1000} Signal Check <<<")
#             print(f"    Shape: {x_seq.shape} (Small input confirmed)")
#             print(f"    Max Input: {x_seq.max().item():.6f}")
#             print(f"    Mean Input: {x_seq.mean().item():.6f}")
#             print(f"    Threshold (v_th): {self.v_th.item():.4f}")
            
#             # 严重警告：如果输入最大值连阈值的 10% 都不到，说明该层大概率“饿死”了
#             if x_seq.max().item() < self.v_th.item() * 0.1:
#                 print("    🚨 警告: 信号强度过低！神经元无法发放脉冲！")
#         # =============================================================
        for t in range(T):
            x = x_seq[t]
            # 膜电位更新
            self.v = self.v + (1.0 / self.tau) * (-(self.v - self.v_reset) + x)
            # self.v = torch.clamp(self.v, min=-10.0, max=10.0)
            
            # 计算正负脉冲
            s_pos = self.surrogate_function(self.v - self.v_th)
            s_neg = self.surrogate_function(-self.v - self.v_th)
            s = s_pos - s_neg # 输出 {-1, 0, 1}
            # # ========== 【插入这段调试代码】 ==========
            # # 只在第0个Batch，第0个时间步打印，避免刷屏
            # if t == 0 and torch.rand(1).item() < 0.01: # 1%的概率打印，防止日志爆炸
            #     v_max = self.v.max().item()
            #     v_mean = self.v.mean().item()
            #     s_sum = s.sum().item()
            #     print(f"!!! 神经元监控 | V_th: {self.v_th.item():.2f} | V_max: {v_max:.4f} | V_mean: {v_mean:.4f} | Spikes: {s_sum} !!!")
            # # =======================================
            # === [修改这里] ===
            if self.soft_reset:
                # [软重置]
                v_after_fire = self.v - s * self.v_th
            else:
                # [硬重置] 使用 abs() 判断是否有脉冲
                v_after_fire = self.v * (1.0 - s.abs()) + self.v_reset * s.abs()
            # =================
            
            
            # # ================= [关键修改：三元硬重置] =================
            # # s.abs() 将 {-1, 0, 1} 变成 {1, 0, 1}
            # # 如果 s.abs() 是 1 (发放了)，则 (1 - s.abs()) 是 0，电压被清空。
            # # 如果 s.abs() 是 0 (没发放)，则电压保持不变。
            # v_after_fire = self.v * (1.0 - s.abs()) + self.v_reset * s.abs()
            # # =======================================================
            
            self.v = torch.clamp(v_after_fire, min=-2.0, max=2.0)
            spikes.append(s)
        
        return torch.stack(spikes, dim=0)    
    
# class MultiStepTernaryPmLIFNode(base.MemoryModule):
#     """三元脉冲神经元 {-1, 0, 1}"""
#     def __init__(self, v_th=0.1, v_reset=0.0, tau=2.0, surrogate_function=surrogate.Sigmoid(alpha=1.0)):
#         super().__init__()
#         self.register_buffer('v_th', torch.tensor(v_th, dtype=torch.float32))
#         self.v_reset = v_reset
#         self.tau = nn.Parameter(torch.tensor(tau, dtype=torch.float32))
#         self.surrogate_function = surrogate_function
#         self.register_memory('v', None)

#     def forward(self, x_seq: torch.Tensor):
#         T, B = x_seq.shape[0], x_seq.shape[1]
#         if self.v is None:
#             self.v = torch.zeros_like(x_seq[0])
            
#         with torch.no_grad():
#             self.tau.data.clamp_(0.5, 5.0)

#         spikes = []
#         # [保险1]
#         # effective_tau = self.tau.clamp(min=0.1)
        
#         for t in range(T):
#             x = x_seq[t]
#             self.v = self.v + (1.0 / self.tau) * (-(self.v - self.v_reset) + x)
#             self.v = torch.clamp(self.v, min=-10.0, max=10.0)
#             s_pos = self.surrogate_function(self.v - self.v_th)
#             s_neg = self.surrogate_function(-self.v - self.v_th)
#             s = s_pos - s_neg # 输出 {-1, 0, 1}
#             v_after_fire = self.v - s * self.v_th
#             self.v = torch.clamp(v_after_fire, min=-2.0, max=2.0)
            
            
            
# #             h = self.v + (1.0 / self.tau) * (-(self.v - self.v_reset) + x)


# #             s_pos = self.surrogate_function(h - self.v_th)
# #             s_neg = self.surrogate_function(-h - self.v_th)
# #             s = s_pos - s_neg 

# #             # self.v = h * (1 - (s_pos + s_neg)) + self.v_reset * (s_pos + s_neg)
# #             self.v = h - s * self.v_th
#             spikes.append(s)
            
            
# #         # ================= [开始插入诊断代码] =================
# #         out_spikes = torch.stack(spikes, dim=0) # 先捕获输出

# #         if not hasattr(self, 'debug_printed'):
# #             firing_rate = out_spikes.float().abs().mean().item() # 三元神经元取绝对值算发放率
# #             max_val = out_spikes.float().max().item()
# #             min_val = out_spikes.float().min().item()
            
# #             layer_name = getattr(self, 'name', self.__class__.__name__)
# #             current_th = self.v_th.item() if hasattr(self, 'v_th') else '未知'

# #             print(f"\n[Spike侦探 (三元)] Layer: {layer_name} (ID: {id(self)})")
# #             print(f"   ├── 🛑 阈值 (v_th): {current_th}")
# #             print(f"   ├── 📊 发放率 (Abs Mean): {firing_rate:.8f}")
# #             print(f"   └── 📈 范围: [{min_val:.2f}, {max_val:.2f}]")

# #             if firing_rate == 0.0:
# #                 print(f"   ⚠️⚠️⚠️ [严重警告] 此层已死 (Dead Silence)！ ⚠️⚠️⚠️")
# #             else:
# #                 print(f"   ✅ [正常] 神经元存活")

# #             self.debug_printed = True
        
#         # return out_spikes
#         # ================= [结束插入] =================    
        

#         return torch.stack(spikes, dim=0)


# ==================== 卷积块定义 ====================

class ConvTemporalPmLIFBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, v_th=0.1, v_reset=0.0, tau=2.0, neuron_type=MultiStepPmLIFNode, soft_reset=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.lif = neuron_type(v_th=v_th, v_reset=v_reset, tau=tau, soft_reset=soft_reset)
        # self.lif = MultiStepPmLIFNode(v_th=v_th, v_reset=v_reset, tau=tau)

    def forward(self, x_seq):
        x_seq = time_distributed_conv(self.conv, x_seq)
        return self.lif(x_seq)

class DilatedConvTemporalPmLIFBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=2, dilation=2, v_th=0.1, v_reset=0.0, tau=2.0, soft_reset=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation)
        self.lif = MultiStepPmLIFNode(v_th=v_th, v_reset=v_reset, tau=tau, soft_reset=soft_reset)

    def forward(self, x_seq):
        x_seq = time_distributed_conv(self.conv, x_seq)
        return self.lif(x_seq)

class ConvNeuronBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1, v_th=0.1, v_reset=0.0, tau=2.0, neuron_type=MultiStepPmLIFNode, soft_reset=False):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding)
        self.neu = neuron_type(v_th=v_th, v_reset=v_reset, tau=tau, soft_reset=soft_reset)
        # self.neu = MultiStepPmLIFNode(v_th=v_th, v_reset=v_reset, tau=tau)

    def forward(self, x_seq):
        x_seq = time_distributed_conv(self.conv, x_seq)
        return self.neu(x_seq)

class TwoConvNeuron(nn.Module):
    def __init__(self, channels, v_th=0.1, v_reset=0.0, tau=2.0, neuron_type=MultiStepPmLIFNode, soft_reset=False):
        super().__init__()
        self.block1 = ConvNeuronBlock(channels, channels, v_th=v_th, v_reset=v_reset, tau=tau, neuron_type=neuron_type, soft_reset=soft_reset)
        self.block2 = ConvNeuronBlock(channels, channels, v_th=v_th, v_reset=v_reset, tau=tau, neuron_type=neuron_type, soft_reset=soft_reset)

    def forward(self, x_seq):
        x_seq = self.block1(x_seq)
        x_seq = self.block2(x_seq)
        return x_seq

# class SubPixelConvBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, upscale_factor=2, kernel_size=3, padding=1, activation=nn.ReLU):
#         super(SubPixelConvBlock, self).__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels * (upscale_factor ** 2), kernel_size=kernel_size, padding=padding)
#         self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
#         self.activation = activation() if activation is not None else nn.Identity()

#     def forward(self, x):
#         x = self.conv(x)
#         x = self.pixel_shuffle(x)
#         x = self.activation(x)
#         return x
    
class SubPixelConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, upscale_factor=2, kernel_size=3, padding=1, activation=nn.ReLU):
        super(SubPixelConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * (upscale_factor ** 2), kernel_size=kernel_size, padding=padding)
        
        # [新增] ICNR 初始化：从源头消除棋盘格，无需平滑
        self._icnr_init(self.conv.weight, upscale_factor)
        
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self.activation = activation() if activation is not None else nn.Identity()

    def _icnr_init(self, tensor, upscale_factor=2, initializer=nn.init.kaiming_normal_):
        """ICNR 初始化逻辑"""
        new_shape = [int(tensor.shape[0] / (upscale_factor ** 2))] + list(tensor.shape[1:])
        subkernel = torch.zeros(new_shape)
        subkernel = initializer(subkernel)
        subkernel = subkernel.transpose(0, 1)
        subkernel = subkernel.contiguous().view(subkernel.shape[0], subkernel.shape[1], -1)
        kernel = subkernel.repeat(1, 1, upscale_factor ** 2)
        transposed_shape = [tensor.shape[1]] + [tensor.shape[0]] + list(tensor.shape[2:])
        kernel = kernel.contiguous().view(transposed_shape)
        kernel = kernel.transpose(0, 1)
        tensor.data.copy_(kernel)

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.activation(x)
        return x

# ==================== 功能分支模块 (全序列版本) ====================

class ABSubbandBranch(nn.Module):
    def __init__(self, ch_in, ch_mid=32, v_th=0.1, tau=2.0, neuron_type=MultiStepPmLIFNode, soft_reset=False):
        super().__init__()
        self.tl1 = TwoConvNeuron(ch_in, v_th=v_th, tau=tau, neuron_type=neuron_type, soft_reset=soft_reset)
        self.tl2 = TwoConvNeuron(ch_in, v_th=v_th, tau=tau, neuron_type=neuron_type, soft_reset=soft_reset)
        self.convneu = ConvNeuronBlock(ch_in, ch_mid, v_th=v_th, tau=tau, neuron_type=neuron_type, soft_reset=soft_reset)
        self.final_conv = nn.Conv2d(ch_mid, ch_in, kernel_size=3, padding=1)
        self.residual_adjust = nn.Conv2d(ch_in, ch_in, kernel_size=1)

    def forward(self, x_sub):
        # x_sub: [T, B, C, H, W]
        x = self.tl1(x_sub)
        x = self.tl2(x)
        x = self.convneu(x)
        
        # 保持全序列输出
        out = time_distributed_conv(self.final_conv, x)

        # 残差也做 time_distributed (或者 reshape)
        adjusted_baseline = time_distributed_conv(self.residual_adjust, x_sub)
        return out + adjusted_baseline

class LSubbandBranch(nn.Module):
    def __init__(self, ch_in=1, ch_mid=32, v_th=0.1, tau=2.0, neuron_type=MultiStepPmLIFNode, soft_reset=False):
        super().__init__()
        self.dilated = ConvNeuronBlock(ch_in, ch_mid, v_th=v_th, tau=tau, neuron_type=neuron_type, soft_reset=soft_reset)
        self.two1 = TwoConvNeuron(ch_mid, v_th=v_th, tau=tau, neuron_type=neuron_type, soft_reset=soft_reset)
        self.convneu = ConvNeuronBlock(ch_mid, ch_mid, v_th=v_th, tau=tau, neuron_type=neuron_type, soft_reset=soft_reset)
        self.final_conv = nn.Conv2d(ch_mid, ch_in, kernel_size=3, padding=1)
        self.residual_adjust = nn.Conv2d(ch_in, ch_in, kernel_size=1)

    def forward(self, x_sub):
        x = self.dilated(x_sub)
        x = self.two1(x)
        x = self.convneu(x)
        out = time_distributed_conv(self.final_conv, x)
        adjusted_baseline = time_distributed_conv(self.residual_adjust, x_sub)
        return out + adjusted_baseline

    
    
# ========== 时序L通道处理器 ==========
class TemporalLChannelProcessor(nn.Module):
    def __init__(self, in_channels=1, features=16, time_steps=5,
                 tau=2.0, threshold=0.5, spike_type='binary'):
        super().__init__()
        self.time_steps = time_steps
        # 使用现有的TemporalEncoder
        self.temporal_encoder = TemporalEncoder(time_steps=time_steps, mode="noisy", noise_scale=0.02)

        # 原有的编码器层
        self.encoder1 = EncoderLayer(
            in_channels, features, stride=1,
            tau=tau, threshold=threshold,
            time_steps=time_steps, spike_type=spike_type
        )
        self.encoder2 = EncoderLayer(
            features, features * 2, stride=1,
            tau=tau, threshold=threshold,
            time_steps=time_steps, spike_type=spike_type
        )
        self.encoder3 = EncoderLayer(
            features * 2, features * 4, stride=1,
            tau=tau, threshold=threshold,
            time_steps=time_steps, spike_type=spike_type
        )

        self.latent_space = LatentSpaceLayer(
            features * 4, features * 4, features * 4,
            tau=tau, threshold=threshold,
            time_steps=time_steps, spike_type=spike_type
        )

        self.decoder1 = DecoderLayer(
            features * 4, features * 2,
            kernel_size=3, stride=1, padding=1,
            tau=tau, threshold=threshold,
            time_steps=time_steps, spike_type=spike_type
        )

        self.decoder2 = DecoderLayer(
            features * 2, features,
            kernel_size=3, stride=1, padding=1,
            tau=tau, threshold=threshold,
            time_steps=time_steps, spike_type=spike_type
        )
        self.decoder3 = DecoderLayer(
            features, features,
            kernel_size=3, stride=1, padding=1, tau=tau, threshold=threshold,
            time_steps=time_steps, spike_type=spike_type
        )

        # 修改输出部分：使用现有的脉冲神经元
        self.out_conv1 = nn.Conv2d(features, features // 2, kernel_size=1)
        self.out_lif = MultiStepPmLIFNode(tau=tau, v_th=threshold) if spike_type == 'binary' else MultiStepTernaryPmLIFNode(
            tau=tau, v_th=threshold)
        self.out_conv2 = nn.Conv2d(features // 2, 1, kernel_size=1)

        self.skip_conv1 = nn.Conv2d(features, features, kernel_size=1)
        self.skip_conv2 = nn.Conv2d(features * 2, features, kernel_size=1)
        self.skip_conv3 = nn.Conv2d(features * 4, features * 2, kernel_size=1)

    def forward(self, x):
        # [关键修改]：在网络入口处重置状态
        functional.reset_net(self)

        # 1. 统一时序编码
        if x.dim() == 4:
            x_temporal = self.temporal_encoder(x) # 输出 [B, T, C, H, W]
            # x_temporal = x_temporal.permute(1, 0, 2, 3, 4) # 转换为 SpikingJelly 格式 [T, B, C, H, W]
        else:
            x_temporal = x
            
            if x_temporal.shape[0] != self.time_steps:
                print(f"警告: 输入时间步数 {x_temporal.shape[0]} 与模型时间步数 {self.time_steps} 不匹配")
            else:
                raise ValueError(f"不支持的输入维度: {x.dim()}，期望4D或5D")
                
            # 验证维度
            T, B, C, H, W = x_temporal.shape
    
            # 验证是否为正确的 SpikingJelly 格式 [T, B, C, H, W]
            if T != self.time_steps:
                print(f"警告: 时间维度 {T} 与模型时间步数 {self.time_steps} 不匹配")

            # 验证输入通道（期望L通道）
            if C != 1:
                print(f"警告: TemporalLChannelProcessor 期望单通道L输入，但得到 {C} 通道")
                if C > 1:
                    x_temporal = x_temporal[:, :, 0:1, :, :]
                    print("已取第一个通道作为L通道")
                else:
                    raise ValueError("输入通道数必须至少为1")    

        # [关键修改]：删除 for t in range(time_steps) 循环
        # 整个序列直接流入网络，保持时序关联性
        
        # Encoder Path
        x1 = self.encoder1(x_temporal) # [T, B, ...]
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)

        # Latent Path
        latent = self.latent_space(x3)

        # Decoder Path (Skip connections 处理)
        # 注意：skip conv 也要做 time_distributed 处理
        skip3 = time_distributed_conv(self.skip_conv3, x3)
        d1 = self.decoder1(latent, skip_connection=skip3)
        
        skip2 = time_distributed_conv(self.skip_conv2, x2)
        d2 = self.decoder2(d1, skip_connection=skip2)
        
        skip1 = time_distributed_conv(self.skip_conv1, x1)
        d3 = self.decoder3(d2, skip_connection=skip1)

        # Output Path
        out = time_distributed_conv(self.out_conv1, d3)
        out = self.out_lif(out)
        outputs_temporal = time_distributed_conv(self.out_conv2, out) # [T, B, 1, H, W]

        # [关键修改]：在最后对时间维度求平均，而不是在中间
        # final_output = outputs_temporal.mean(dim=0) # 对 T 维度求平均 -> [B, 1, H, W]
        # final_output = outputs_temporal[-1]
        # [修改点 1]：添加 Tanh 激活函数，限制输出在 [-1, 1]
        # final_output = torch.tanh(outputs_temporal[-1])
        final_output = torch.tanh(outputs_temporal.mean(dim=0))

        return final_output
    
    
# ========== 时序LAB增强器 ==========
class TemporalLABEnhancer(nn.Module):
    def __init__(self, time_steps=5, downsample_factor=2,
                 stats_path="/root/results18/degradation_stats.pth",
                 kernelgan_kernels_path="/root/results18/kernelgan_kernels.npy",
                 noise_patches_dir=None, spike_type='binary', use_temporal=True):
        super().__init__()
        self.resid_scale = nn.Parameter(torch.tensor(0.05))
        self.time_steps = time_steps
        self.downsample_factor = downsample_factor
        self.spike_type = spike_type
        self.use_temporal = use_temporal  # 控制是否使用时序处理

        # 加载退化分布统计量
        if os.path.exists(stats_path):
            self.stats = torch.load(stats_path)
            self.kernel_mean = self.stats["kernel_mean"]
            self.kernel_std = self.stats["kernel_std"]
            self.noise_mean = self.stats["noise_mean"]
            self.noise_std = self.stats["noise_std"]
        else:
            # 默认值
            self.kernel_mean = 1.0
            self.kernel_std = 0.1
            self.noise_mean = 0.0
            self.noise_std = 0.01

        # 加载KernelGAN估计的模糊核
        if os.path.exists(kernelgan_kernels_path):
            self.kernelgan_kernels = np.load(kernelgan_kernels_path)
            logger.info(f"Loaded {len(self.kernelgan_kernels)} kernels from KernelGAN")
        else:
            self.kernelgan_kernels = None
            logger.warning(f"KernelGAN kernels not found at {kernelgan_kernels_path}")

        # 加载真实噪声补丁
        self.noise_patches = []
        if noise_patches_dir and os.path.exists(noise_patches_dir):
            noise_files = glob(os.path.join(noise_patches_dir, "*.pt"))
            for noise_file in noise_files:
                noise_patch = torch.load(noise_file)
                self.noise_patches.append(noise_patch)
            logger.info(f"Loaded {len(self.noise_patches)} real noise patches from {noise_patches_dir}")
        else:
            logger.warning(f"No noise patches found at {noise_patches_dir}, will use Gaussian noise as fallback")

        # 初始化L通道处理器（使用时序版本）
        self.l_channel_processor = TemporalLChannelProcessor(
            in_channels=1,
            features=16,
            time_steps=time_steps,
            tau=1.2,
            threshold=0.5,
            spike_type=spike_type
        )

        # 可学习参数 λ - 初始化为非零值 (0.7) - 保留这个参数！
        self.lambda_param = nn.Parameter(torch.tensor(0.7))

    def forward(self, rgb_input):
        # RGB -> LAB (使用现有的转换)
        lab_tensor = self.rgb_to_lab(rgb_input)
        L_hr = lab_tensor[:, 0:1, :, :]  # HR L通道

        h, w = lab_tensor.shape[2:]
        lr_size = (h // self.downsample_factor, w // self.downsample_factor)

        # --- 分支一：bicubic 下采样 ---
        lab_bicubic = F.interpolate(lab_tensor, size=lr_size, mode="bicubic", align_corners=False)
        L3 = lab_bicubic[:, 0:1, :, :]

        # --- 分支二：基于统计分布退化 ---
        if self.kernelgan_kernels is not None and len(self.kernelgan_kernels) > 0:
            # 从KernelGAN估计的核中随机选择一个
            kernel_idx = np.random.randint(0, len(self.kernelgan_kernels))
            kernel = self.kernelgan_kernels[kernel_idx]
            kernel_tensor = torch.from_numpy(kernel).unsqueeze(0).unsqueeze(0).to(L_hr.device).to(L_hr.dtype)

            # 应用模糊核
            L_blurred = F.conv2d(L_hr, kernel_tensor, padding=kernel.shape[0] // 2)
        else:
            # 模糊核采样 (高斯 σ)
            sigma = np.random.normal(self.kernel_mean, self.kernel_std)
            # 使用高斯模糊
            kernel_size = int(2 * np.ceil(2 * sigma) + 1)
            if kernel_size > 0:
                L_blurred = self.gaussian_blur(L_hr, kernel_size, sigma)
            else:
                L_blurred = L_hr

        # 下采样
        L_down = F.interpolate(L_blurred, size=lr_size, mode="bicubic", align_corners=False)

        # 加噪声 - 使用真实噪声补丁而不是高斯噪声
        if len(self.noise_patches) > 0:
            noise_patch = random.choice(self.noise_patches)
            noise_patch = noise_patch.to(L_down.device)
            if noise_patch.dim() == 2:
                noise_patch = noise_patch.unsqueeze(0).unsqueeze(0)
            if noise_patch.dim() == 3:
                noise_patch = noise_patch.unsqueeze(0)
            noise_patch = F.interpolate(noise_patch, size=L_down.shape[2:], mode='bicubic', align_corners=False)
            if noise_patch.size(0) == 1:
                noise_patch = noise_patch.repeat(L_down.size(0), 1, 1, 1)
            else:
                noise_patch = noise_patch[:L_down.size(0)]
            L_noisy = torch.clamp(L_down + noise_patch, 0, 1)
        else:
            noise = torch.randn_like(L_down) * self.noise_std + self.noise_mean
            L_noisy = torch.clamp(L_down + noise, 0, 1)

        # 使用时序L通道处理器处理加噪后的L通道
        L_processed = self.l_channel_processor(L_noisy)

        # # λ 融合 - 使用sigmoid确保在0-1之间
        # lambda_val = torch.sigmoid(self.lambda_param)
        # L_fused = lambda_val * L_processed[-1] + (1 - lambda_val) * L3
        
        safe_scale = torch.clamp(self.resid_scale, -1.0, 1.0)
        
        # [修改点 3]：残差融合逻辑
        # L_fused = 原始L + (缩放系数 * SNN产生的扰动)
        # 这样即使 SNN 乱输出，乘以 0.05 后也不会导致图像过曝
        L_fused = L3 + safe_scale * L_processed
        # L_fused = L3 + self.resid_scale * L_processed

        # [修改点 4]：必须 Clamp，防止溢出 [0, 1] 变成死白或死黑
        L_fused = torch.clamp(L_fused, 0, 1)

        # 合并 AB
        lab_AB = lab_bicubic[:, 1:, :, :]
        generated_lab_lr = torch.cat([L_fused, lab_AB], dim=1)

        return generated_lab_lr, lab_tensor, L_noisy, L_hr

    def rgb_to_lab(self, rgb):
        """RGB转LAB的简化实现"""
        # 这里使用现有的RGB2Lab类
        rgb2lab = RGB2Lab()
        return rgb2lab(rgb)

    def gaussian_blur(self, x, kernel_size, sigma):
        """高斯模糊实现"""
        # 创建高斯核
        kernel = self._gaussian_kernel(kernel_size, sigma).to(x.device)
        kernel = kernel.repeat(x.size(1), 1, 1, 1)

        # 应用卷积
        padding = kernel_size // 2
        return F.conv2d(x, kernel, padding=padding, groups=x.size(1))

    def _gaussian_kernel(self, size, sigma):
        """创建高斯核"""
        coords = torch.arange(size).float() - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g.unsqueeze(0) * g.unsqueeze(1)
        return g / g.sum()

    
# ==================== 特征提取器 (全序列版本) ====================

class SpatialEncoder(nn.Module):
    def __init__(self, time_steps=5, v_th=0.1, tau=2.0, spike_type='binary', soft_reset=False):
        super().__init__()
        self.temporal_encoder = TemporalEncoder(time_steps, mode="noisy", noise_scale=0.05)
        
        NType = MultiStepTernaryPmLIFNode if spike_type == 'ternary' else MultiStepPmLIFNode

        self.initial = nn.Sequential(
            ConvTemporalPmLIFBlock(3, 32, v_th=v_th, tau=tau, neuron_type=NType, soft_reset=soft_reset),
            ConvTemporalPmLIFBlock(32, 64, v_th=v_th, tau=tau, neuron_type=NType, soft_reset=soft_reset),
        )

        self.ab_branch = nn.Sequential(
            ConvTemporalPmLIFBlock(64, 64, v_th=v_th, tau=tau, neuron_type=NType, soft_reset=soft_reset),
            ConvTemporalPmLIFBlock(64, 64, v_th=v_th, tau=tau, neuron_type=NType, soft_reset=soft_reset),
            ConvTemporalPmLIFBlock(64, 64, v_th=v_th, tau=tau, neuron_type=NType, soft_reset=soft_reset),
            ConvTemporalPmLIFBlock(64, 64, v_th=v_th, tau=tau, neuron_type=NType, soft_reset=soft_reset),
            ConvTemporalPmLIFBlock(64, 32, v_th=v_th, tau=tau, neuron_type=NType, soft_reset=soft_reset),
        )
        self.ab_final_conv = nn.Conv2d(32, 2, kernel_size=3, padding=1)
        self.ab_neuron = NType(v_th=v_th, tau=tau, soft_reset=soft_reset) # [修改] 使用 NType
        # self.ab_neuron = MultiStepTernaryPmLIFNode(v_th=v_th, tau=tau)

        self.l_branch = nn.Sequential(
            # DilatedConvTemporalPmLIFBlock(64, 64, v_th=v_th),
            ConvTemporalPmLIFBlock(64, 64, v_th=v_th, tau=tau, neuron_type=NType, soft_reset=soft_reset),
            ConvTemporalPmLIFBlock(64, 64, v_th=v_th, tau=tau, neuron_type=NType, soft_reset=soft_reset),
            ConvTemporalPmLIFBlock(64, 64, v_th=v_th, tau=tau, neuron_type=NType, soft_reset=soft_reset),
            ConvTemporalPmLIFBlock(64, 64, v_th=v_th, tau=tau, neuron_type=NType, soft_reset=soft_reset),
            ConvTemporalPmLIFBlock(64, 64, v_th=v_th, tau=tau, neuron_type=NType, soft_reset=soft_reset),
            ConvTemporalPmLIFBlock(64, 32, v_th=v_th, tau=tau, neuron_type=NType, soft_reset=soft_reset),
        )
        self.l_final_conv = nn.Conv2d(32, 1, kernel_size=3, padding=1)
        self.l_neuron = NType(v_th=v_th, tau=tau, soft_reset=soft_reset) # [修改] 使用 NType
        # self.l_neuron = MultiStepPmLIFNode(v_th=v_th, tau=tau)

        self.ab_residual_adjust = nn.Conv2d(2, 2, kernel_size=1)
        self.l_residual_adjust = nn.Conv2d(1, 1, kernel_size=1)
        
    def forward(self, lr_lab):
        # [修改] 增加对 5D 输入的判断与处理
        if lr_lab.dim() == 5:
            # 输入已经是序列 [T, B, 3, H, W]
            x_seq = lr_lab
            T = x_seq.shape[0]
            # 5D 数据切片：Dim 2 是 Channel
            l = x_seq[:, :, :1, :, :]
            ab = x_seq[:, :, 1:, :, :]
        else:
            # 4D 输入 [B, 3, H, W]，需要扩展
            l, ab = lr_lab[:, :1], lr_lab[:, 1:]
            x_seq = self.temporal_encoder(lr_lab)
            T = x_seq.shape[0]
        
        feat = self.initial(x_seq)

        # === AB 分支 ===
        ab_seq = self.ab_branch(feat) 
        ab_conv = time_distributed_conv(self.ab_final_conv, ab_seq)
        ab_out = self.ab_neuron(ab_conv)
        
        # [修改] 处理残差：如果是 5D 输入，残差路径也需要 time_distributed
        if ab.dim() == 5:
            ab_res = time_distributed_conv(self.ab_residual_adjust, ab)
        else:
            ab_res = self.ab_residual_adjust(ab).unsqueeze(0).expand(T, -1, -1, -1, -1)
        ab_last = ab_out + ab_res

        # === L 分支 ===
        l_seq = self.l_branch(feat)
        l_conv = time_distributed_conv(self.l_final_conv, l_seq)
        l_out = self.l_neuron(l_conv)
        
        # [修改] 处理 L 分支残差
        if l.dim() == 5:
            l_res = time_distributed_conv(self.l_residual_adjust, l)
        else:
            l_res = self.l_residual_adjust(l).unsqueeze(0).expand(T, -1, -1, -1, -1)
        l_last = l_out + l_res

        lab_out = torch.cat([l_last, ab_last], dim=2) 
        return lab_out

#     def forward(self, lr_lab):
#         l, ab = lr_lab[:, :1], lr_lab[:, 1:]
        
#         # 1. 扩展时间 [T, B, C, H, W]
#         x_seq = self.temporal_encoder(lr_lab)
#         T = x_seq.shape[0]
        
#         feat = self.initial(x_seq)

#         # 2. AB 分支 (全序列)
#         ab_seq = self.ab_branch(feat) 
#         ab_conv = time_distributed_conv(self.ab_final_conv, ab_seq)
#         ab_out = self.ab_neuron(ab_conv) # [T, B, 2, H, W]
        
#         # 扩展残差维度以匹配 T
#         ab_res = self.ab_residual_adjust(ab).unsqueeze(0).expand(T, -1, -1, -1, -1)
#         ab_last = ab_out + ab_res

#         # 3. L 分支 (全序列)
#         l_seq = self.l_branch(feat)
#         l_conv = time_distributed_conv(self.l_final_conv, l_seq)
#         l_out = self.l_neuron(l_conv) # [T, B, 1, H, W]
        
#         l_res = self.l_residual_adjust(l).unsqueeze(0).expand(T, -1, -1, -1, -1)
#         l_last = l_out + l_res

#         # 4. 拼接保持 T
#         lab_out = torch.cat([l_last, ab_last], dim=2)  # [T, B, 3, H, W]
#         return lab_out


class FrequencyFeatureExtractor(nn.Module):
    def __init__(self, time_steps=5, base_ch=64, v_th=0.1, tau=2.0, spike_type='binary', soft_reset=False):
        super().__init__()
        
        NType = MultiStepTernaryPmLIFNode if spike_type == 'ternary' else MultiStepPmLIFNode
        
        self.temporal_encoder = TemporalEncoder(time_steps=time_steps)
        self.init_conv1 = ConvNeuronBlock(3, base_ch, v_th=v_th, tau=tau, neuron_type=NType, soft_reset=soft_reset)
        self.init_conv2 = ConvNeuronBlock(base_ch, base_ch, v_th=v_th, tau=tau, neuron_type=NType, soft_reset=soft_reset)
        self.head_conv = nn.Conv2d(base_ch, 3, kernel_size=1)

        self.haar_forward = Haar2DForward(neuron_type=MultiStepNegIFNode, vth=1.0)
        self.haar_inverse = Haar2DInverse(neuron_type=MultiStepNegIFNode, vth=1.0)
        
        # [修改 1] 手动记录 Haar 构建状态
        self._haar_built = False
        self._current_haar_size = None  # 新增：记录当前构建的尺寸

        self.ab_sub_branches = nn.ModuleList([ABSubbandBranch(ch_in=2, ch_mid=32, v_th=v_th, tau=tau, neuron_type=NType, soft_reset=soft_reset) for _ in range(4)])
        self.l_sub_branches = nn.ModuleList([LSubbandBranch(ch_in=1, ch_mid=32, v_th=v_th, tau=tau, neuron_type=NType, soft_reset=soft_reset) for _ in range(4)])

        self.ab_wavelet_neuron = MultiStepNegIFNode(v_threshold=0.1, neg_v_threshold=0.1, v_reset=None)
        self.l_wavelet_neuron = MultiStepNegIFNode(v_threshold=float(v_th), neg_v_threshold=float(v_th), v_reset=None)

        self.freq_residual_adjust = nn.Conv2d(3, 3, kernel_size=1)

    def _split_quadrants(self, coeffs):
        T, B, C, H, W = coeffs.shape
        H2, W2 = H // 2, W // 2
        return [coeffs[..., :H2, :W2], coeffs[..., :H2, W2:], coeffs[..., H2:, :W2], coeffs[..., H2:, W2:]]

    def _merge_quadrants(self, quads):
        HH, LL, LH, HL = quads
        top = torch.cat([LL, LH], dim=-1)
        bottom = torch.cat([HL, HH], dim=-1)
        return torch.cat([top, bottom], dim=-2)

    def forward(self, lr_lab):
        import math
        import torch.nn.functional as F
        
        # ==========================================
        # 1. 自动填充逻辑 (Auto-Padding)
        # ==========================================
        if lr_lab.dim() == 5: # [T, B, C, H, W]
            _, _, _, H, W = lr_lab.shape
        else:
            _, _, H, W = lr_lab.shape

        def next_power_of_2(x):
            return 1 if x == 0 else 2**(x - 1).bit_length()

        target_H = next_power_of_2(H)
        target_W = next_power_of_2(W)

        pad_h = target_H - H
        pad_w = target_W - W

        if pad_h > 0 or pad_w > 0:
            if lr_lab.dim() == 5:
                T, B, C, h_in, w_in = lr_lab.shape
                x_pad = lr_lab.reshape(T * B, C, h_in, w_in)
                x_pad = F.pad(x_pad, (0, pad_w, 0, pad_h), mode='reflect')
                lr_lab_padded = x_pad.reshape(T, B, C, target_H, target_W)
            else:
                lr_lab_padded = F.pad(lr_lab, (0, pad_w, 0, pad_h), mode='reflect')
        else:
            lr_lab_padded = lr_lab

        # ==========================================
        # 2. 核心处理逻辑
        # ==========================================
        if lr_lab_padded.dim() == 5:
            lab_seq = lr_lab_padded
        else:
            lab_seq = self.temporal_encoder(lr_lab_padded) 
            
        feat = self.init_conv1(lab_seq)
        feat = self.init_conv2(feat)

        feat_3 = time_distributed_conv(self.head_conv, feat) 
        AB_seq = feat_3[:, :, 1:3, :, :]
        L_seq = feat_3[:, :, 0:1, :, :]

        # [修改 2] 使用 self._current_haar_size 判断是否需要重新 build
        if not self._haar_built or self._current_haar_size != target_H:
            self.haar_forward.build(target_H, AB_seq.device)
            self.haar_inverse.build(target_H, AB_seq.device)
            self._haar_built = True
            self._current_haar_size = target_H  # 更新当前尺寸记录
        
        # --- AB Processing ---
        AB_coefs = self.haar_forward(AB_seq)
        AB_subs = self._split_quadrants(AB_coefs)
        processed_AB_subs = []
        for i, sub in enumerate(AB_subs):
            processed = self.ab_sub_branches[i](sub)
            spike = self.ab_wavelet_neuron(processed)
            processed_AB_subs.append(spike + AB_subs[i])
        AB_coefs_processed = self._merge_quadrants(processed_AB_subs)
        ab1_seq = self.haar_inverse(AB_coefs_processed)

        # --- L Processing ---
        L_coefs = self.haar_forward(L_seq)
        L_subs = self._split_quadrants(L_coefs)
        processed_L_subs = []
        for i, sub in enumerate(L_subs):
            sub_in = sub[:, :, 0:1, :, :] if sub.shape[2] != 1 else sub
            processed = self.l_sub_branches[i](sub_in)
            spike = self.l_wavelet_neuron(processed)
            baseline = L_subs[i][:, :, 0:1, :, :] if L_subs[i].shape[2] != 1 else L_subs[i]
            processed_L_subs.append(spike + baseline)
        L_coefs_processed = self._merge_quadrants(processed_L_subs)
        l1_seq = self.haar_inverse(L_coefs_processed)

        # 合并
        lab_out = torch.cat([l1_seq, ab1_seq], dim=2)

        if lab_out.shape[2] != 3:
             if lab_out.shape[2] < 3:
                 zeros = torch.zeros_like(lab_out[:, :, 0:1, :, :])
                 while lab_out.shape[2] < 3:
                     lab_out = torch.cat([lab_out, zeros], dim=2)
             elif lab_out.shape[2] > 3:
                 lab_out = lab_out[:, :, :3, :, :] 

        # ==========================================
        # 3. 残差处理
        # ==========================================
        if lr_lab_padded.dim() == 5:
            residual_adjusted = time_distributed_conv(self.freq_residual_adjust, lr_lab_padded)
        else:
            residual_adjusted = self.freq_residual_adjust(lr_lab_padded)
            if residual_adjusted.shape[-2:] != lab_out.shape[-2:]:
                residual_adjusted = F.interpolate(residual_adjusted, size=lab_out.shape[-2:], mode='bicubic', align_corners=False)
            T = lab_out.shape[0]
            residual_adjusted = residual_adjusted.unsqueeze(0).expand(T, -1, -1, -1, -1)

        output_padded = lab_out + residual_adjusted

        # ==========================================
        # 4. 剪裁回原始尺寸
        # ==========================================
        output_final = output_padded[:, :, :, :H, :W]

        return output_final




# class FrequencyFeatureExtractor(nn.Module):
#     def __init__(self, time_steps=5, base_ch=64, v_th=0.1, tau=2.0, spike_type='binary', soft_reset=False):
#         super().__init__()
        
#         NType = MultiStepTernaryPmLIFNode if spike_type == 'ternary' else MultiStepPmLIFNode
        
#         self.temporal_encoder = TemporalEncoder(time_steps=time_steps)
#         self.init_conv1 = ConvNeuronBlock(3, base_ch, v_th=v_th, tau=tau, neuron_type=NType, soft_reset=soft_reset)
#         self.init_conv2 = ConvNeuronBlock(base_ch, base_ch, v_th=v_th, tau=tau, neuron_type=NType, soft_reset=soft_reset)
#         self.head_conv = nn.Conv2d(base_ch, 3, kernel_size=1)

#         self.haar_forward = Haar2DForward(neuron_type=MultiStepNegIFNode, vth=1.0)
#         self.haar_inverse = Haar2DInverse(neuron_type=MultiStepNegIFNode, vth=1.0)
#         self._haar_built = False

#         self.ab_sub_branches = nn.ModuleList([ABSubbandBranch(ch_in=2, ch_mid=32, v_th=v_th, tau=tau, neuron_type=NType, soft_reset=soft_reset) for _ in range(4)])
#         self.l_sub_branches = nn.ModuleList([LSubbandBranch(ch_in=1, ch_mid=32, v_th=v_th, tau=tau, neuron_type=NType, soft_reset=soft_reset) for _ in range(4)])

#         self.ab_wavelet_neuron = MultiStepNegIFNode(v_threshold=0.1, neg_v_threshold=0.1, v_reset=None)
#         self.l_wavelet_neuron = MultiStepNegIFNode(v_threshold=float(v_th), neg_v_threshold=float(v_th), v_reset=None)

#         self.freq_residual_adjust = nn.Conv2d(3, 3, kernel_size=1)

#     def _check_power2(self, lr_lab):
#         if isinstance(lr_lab, torch.Tensor):
#             H, W = lr_lab.shape[-2], lr_lab.shape[-1]
#         else:
#             H, W = lr_lab
#         def is_pow2(n): return (n & (n - 1)) == 0
#         assert is_pow2(H) and is_pow2(W), f"H/W must be powers of 2 (got H={H}, W={W})"

#     def _split_quadrants(self, coeffs):
#         T, B, C, H, W = coeffs.shape
#         H2, W2 = H // 2, W // 2
#         return [coeffs[..., :H2, :W2], coeffs[..., :H2, W2:], coeffs[..., H2:, :W2], coeffs[..., H2:, W2:]]

#     def _merge_quadrants(self, quads):
#         HH, LL, LH, HL = quads
#         top = torch.cat([LL, LH], dim=-1)
#         bottom = torch.cat([HL, HH], dim=-1)
#         return torch.cat([top, bottom], dim=-2)
    
#     def forward(self, lr_lab):
#         B, C, H, W = lr_lab.shape[-4:] 
#         self._check_power2((H, W))

#         if lr_lab.dim() == 5:
#             lab_seq = lr_lab
#         else:
#             lab_seq = self.temporal_encoder(lr_lab) 
            
#         feat = self.init_conv1(lab_seq)
#         feat = self.init_conv2(feat)

#         feat_3 = time_distributed_conv(self.head_conv, feat) 
#         AB_seq = feat_3[:, :, 1:3, :, :]
#         L_seq = feat_3[:, :, 0:1, :, :]

#         if not self._haar_built:
#             self.haar_forward.build(H, AB_seq.device)
#             self.haar_inverse.build(H, AB_seq.device)
#             self._haar_built = True

#         # AB Processing
#         AB_coefs = self.haar_forward(AB_seq)
#         AB_subs = self._split_quadrants(AB_coefs)
#         processed_AB_subs = []
#         for i, sub in enumerate(AB_subs):
#             processed = self.ab_sub_branches[i](sub)
#             spike = self.ab_wavelet_neuron(processed)
#             processed_AB_subs.append(spike + AB_subs[i])
#         AB_coefs_processed = self._merge_quadrants(processed_AB_subs)
#         ab1_seq = self.haar_inverse(AB_coefs_processed)

#         # L Processing
#         L_coefs = self.haar_forward(L_seq)
#         L_subs = self._split_quadrants(L_coefs)
#         processed_L_subs = []
#         for i, sub in enumerate(L_subs):
#             sub_in = sub[:, :, 0:1, :, :] if sub.shape[2] != 1 else sub
#             processed = self.l_sub_branches[i](sub_in)
#             spike = self.l_wavelet_neuron(processed)
#             baseline = L_subs[i][:, :, 0:1, :, :] if L_subs[i].shape[2] != 1 else L_subs[i]
#             processed_L_subs.append(spike + baseline)
#         L_coefs_processed = self._merge_quadrants(processed_L_subs)
#         l1_seq = self.haar_inverse(L_coefs_processed)

#         # ==================== 【修正核心开始】 ====================
#         lab_out = torch.cat([l1_seq, ab1_seq], dim=2)

#         # 强制通道对齐 (Safety Guard)
#         # 确保 lab_out 的通道数（dim 2）严格等于 3
#         if lab_out.shape[2] != 3:
#              # 情况1：通道不足，补零
#              if lab_out.shape[2] < 3:
#                  zeros = torch.zeros_like(lab_out[:, :, 0:1, :, :])
#                  while lab_out.shape[2] < 3:
#                      lab_out = torch.cat([lab_out, zeros], dim=2)
#              # 情况2：通道过多（导致报错的原因），截取前3个
#              elif lab_out.shape[2] > 3:
#                  lab_out = lab_out[:, :, :3, :, :] 
#         # ==================== 【修正核心结束】 ====================

#         if lr_lab.dim() == 5:
#             residual_adjusted = time_distributed_conv(self.freq_residual_adjust, lr_lab)
#         else:
#             residual_adjusted = self.freq_residual_adjust(lr_lab)
#             if residual_adjusted.shape[-2:] != lab_out.shape[-2:]:
#                 residual_adjusted = F.interpolate(residual_adjusted, size=lab_out.shape[-2:], mode='bicubic', align_corners=False)
#             T = lab_out.shape[0]
#             residual_adjusted = residual_adjusted.unsqueeze(0).expand(T, -1, -1, -1, -1)

#         return lab_out + residual_adjusted

#     def forward(self, lr_lab):
#         B, C, H, W = lr_lab.shape
#         self._check_power2((H, W))

#         lab_seq = self.temporal_encoder(lr_lab) # [T, B, 3, H, W]
#         feat = self.init_conv1(lab_seq)
#         feat = self.init_conv2(feat)

#         feat_3 = time_distributed_conv(self.head_conv, feat) # [T, B, 3, H, W]
#         AB_seq = feat_3[:, :, 1:3, :, :]
#         L_seq = feat_3[:, :, 0:1, :, :]

#         if not self._haar_built:
#             self.haar_forward.build(H, AB_seq.device)
#             self.haar_inverse.build(H, AB_seq.device)
#             self._haar_built = True

#         # AB Processing
#         AB_coefs = self.haar_forward(AB_seq)
#         AB_subs = self._split_quadrants(AB_coefs)
#         processed_AB_subs = []
#         for i, sub in enumerate(AB_subs):
#             processed = self.ab_sub_branches[i](sub)
#             spike = self.ab_wavelet_neuron(processed)
#             processed_AB_subs.append(spike + AB_subs[i])
        
#         AB_coefs_processed = self._merge_quadrants(processed_AB_subs)
#         ab1_seq = self.haar_inverse(AB_coefs_processed) # [T, B, 2, H, W]

#         # L Processing
#         L_coefs = self.haar_forward(L_seq)
#         L_subs = self._split_quadrants(L_coefs)
#         processed_L_subs = []
#         for i, sub in enumerate(L_subs):
#             sub_in = sub[:, :, 0:1, :, :] if sub.shape[2] != 1 else sub
#             processed = self.l_sub_branches[i](sub_in)
#             spike = self.l_wavelet_neuron(processed)
#             baseline = L_subs[i][:, :, 0:1, :, :] if L_subs[i].shape[2] != 1 else L_subs[i]
#             processed_L_subs.append(spike + baseline)

#         L_coefs_processed = self._merge_quadrants(processed_L_subs)
#         l1_seq = self.haar_inverse(L_coefs_processed) # [T, B, 1, H, W]

#         # Concatenate keeping T
#         lab_out = torch.cat([l1_seq, ab1_seq], dim=2) # [T, B, 3, H, W]

#         # Padding if necessary
#         if lab_out.shape[2] != 3:
#              zeros = torch.zeros_like(lab_out[:, :, 0:1, :, :])
#              while lab_out.shape[2] < 3:
#                  lab_out = torch.cat([lab_out, zeros], dim=2)

#         # Residual with T expansion
#         residual_adjusted = self.freq_residual_adjust(lr_lab)
#         if residual_adjusted.shape[-2:] != lab_out.shape[-2:]:
#             residual_adjusted = F.interpolate(residual_adjusted, size=lab_out.shape[-2:], mode='bilinear', align_corners=False)
        
#         T = lab_out.shape[0]
#         residual_adjusted = residual_adjusted.unsqueeze(0).expand(T, -1, -1, -1, -1)

#         return lab_out + residual_adjusted

# ==================== 融合与金字塔 (全序列版本) ====================

class FusionModule(nn.Module):
    def __init__(self, time_steps=5, base_ch=64, v_th=0.1, v_reset=0.0, tau=2.0, spike_type='binary', soft_reset=False):
        super().__init__()
        
        NType = MultiStepTernaryPmLIFNode if spike_type == 'ternary' else MultiStepPmLIFNode
        
        self.spatial_extractor = SpatialEncoder(time_steps=time_steps, v_th=v_th, tau=tau, spike_type=spike_type, soft_reset=soft_reset)
        self.freq_extractor = FrequencyFeatureExtractor(time_steps=time_steps, base_ch=base_ch, v_th=v_th, tau=tau, spike_type=spike_type, soft_reset=soft_reset)
        self.input_adjust_to_3ch = nn.Conv2d(base_ch, 3, 1)
        self.output_adjust_to_base_ch = nn.Conv2d(base_ch, base_ch, 1)

        # [修改] 使用 SeqConv2d 处理 5D 数据
        
        self.feature_fusion = nn.Sequential(
            SeqConv2d(6, base_ch * 2, 3, padding=1), 
            NType(v_th=v_th, v_reset=v_reset, tau=tau, soft_reset=soft_reset),
            SeqConv2d(base_ch * 2, base_ch, 3, padding=1),
            NType(v_th=v_th, v_reset=v_reset, tau=tau, soft_reset=soft_reset)
        )
        
        # self.feature_fusion = nn.Sequential(
        #     SeqConv2d(6, base_ch * 2, 3, padding=1), 
        #     MultiStepPmLIFNode(v_th=v_th, v_reset=v_reset, tau=tau),
        #     SeqConv2d(base_ch * 2, base_ch, 3, padding=1),
        #     MultiStepPmLIFNode(v_th=v_th, v_reset=v_reset, tau=tau)
        # )
        self.residual_adjust = nn.Conv2d(base_ch, base_ch, 1)
        
    def forward(self, x):
        # [修改] 判断输入维度，如果是 5D [T, B, C, H, W]，使用 time_distributed_conv
        if x.dim() == 5:
            x_3ch = time_distributed_conv(self.input_adjust_to_3ch, x)
        else:
            # 4D 输入，直接卷积
            x_3ch = self.input_adjust_to_3ch(x)
        
        spatial_feat = self.spatial_extractor(x_3ch) 
        freq_feat = self.freq_extractor(x_3ch)       

        combined = torch.cat([spatial_feat, freq_feat], dim=2) # Dim 2 is Channel
        fused_features = self.feature_fusion(combined) 

        # [修改] 残差连接适配 5D
        if x.dim() == 5:
            residual = time_distributed_conv(self.residual_adjust, x)
        else:
            residual = self.residual_adjust(x)
            T = fused_features.shape[0]
            residual = residual.unsqueeze(0).expand(T, -1, -1, -1, -1)

        return fused_features + residual

#     def forward(self, x):
#         # x is [B, base_ch, H, W]
#         x_3ch = self.input_adjust_to_3ch(x)
        
#         spatial_feat = self.spatial_extractor(x_3ch) # [T, B, 3, H, W]
#         freq_feat = self.freq_extractor(x_3ch)       # [T, B, 3, H, W]

#         combined = torch.cat([spatial_feat, freq_feat], dim=2) # Concat on C
#         fused_features = self.feature_fusion(combined) # [T, B, base_ch, H, W]

#         residual = self.residual_adjust(x)
#         T = fused_features.shape[0]
#         residual = residual.unsqueeze(0).expand(T, -1, -1, -1, -1)

#         return fused_features + residual

class ProgressiveFeaturePyramid(nn.Module):
    def __init__(self, num_layers=8, base_ch=64, v_th=0.1, tau=2.0, spike_type='binary', soft_reset=False):
        super().__init__()
        NType = MultiStepTernaryPmLIFNode if spike_type == 'ternary' else MultiStepPmLIFNode
        # [修改] 使用 SeqConv2d
        
        self.bottom_up_fusions = nn.ModuleList([
            nn.Sequential(
                SeqConv2d(base_ch * 2, base_ch, 3, padding=1),
                NType(v_th=v_th, tau=tau, soft_reset=soft_reset), # [修改]
                SeqConv2d(base_ch, base_ch, 3, padding=1)
            ) for _ in range(num_layers - 1)
        ])
        
        self.top_down_refinements = nn.ModuleList([
            nn.Sequential(
                SeqConv2d(base_ch * 2, base_ch, 3, padding=1),
                NType(v_th=v_th, tau=tau, soft_reset=soft_reset)   # [修改]
            ) for _ in range(num_layers - 1)
        ])
        
        
        # self.bottom_up_fusions = nn.ModuleList([
        #     nn.Sequential(
        #         SeqConv2d(base_ch * 2, base_ch, 3, padding=1),
        #         MultiStepPmLIFNode(v_th=v_th, tau=tau),
        #         SeqConv2d(base_ch, base_ch, 3, padding=1)
        #     ) for _ in range(num_layers - 1)
        # ])

        # self.top_down_refinements = nn.ModuleList([
        #     nn.Sequential(
        #         SeqConv2d(base_ch * 2, base_ch, 3, padding=1),
        #         MultiStepPmLIFNode(v_th=v_th, tau=tau)
        #     ) for _ in range(num_layers - 1)
        # ])

    def forward(self, features_list):
        if len(features_list) == 1: return features_list[0]

        current = features_list[0]
        for i in range(1, len(features_list)):
            layer = self.bottom_up_fusions[i-1] if i-1 < len(self.bottom_up_fusions) else self.bottom_up_fusions[-1]
            current = layer(torch.cat([features_list[i], current], dim=2)) # dim 2 is C

        final_feature = current
        refined = final_feature
        for i in range(len(features_list) - 2, -1, -1):
            layer = self.top_down_refinements[i] if i < len(self.top_down_refinements) else self.top_down_refinements[-1]
            refined = layer(torch.cat([features_list[i], refined], dim=2))

        return final_feature + refined

# ==================== 主网络 (重建头) ====================

class StackedFusionReconstructionNet(nn.Module):
    def __init__(self, num_fusion_modules=8, time_steps=5, base_ch=64,
                 upscale_factor=2, v_th=0.1, v_reset=0.0, tau=2.0, spike_type='binary', soft_reset=False):
        super().__init__()
        self.num_fusion_modules = num_fusion_modules
        self.base_ch = base_ch
        self.input_adjust = nn.Conv2d(3, base_ch, 1)

        self.fusion_modules = nn.ModuleList([
            FusionModule(time_steps, base_ch, v_th, v_reset, tau, spike_type=spike_type, soft_reset=soft_reset) for _ in range(num_fusion_modules)
        ])

        # 使用 SeqConv2d 处理中间 5D 特征
        self.inter_module_connections = nn.ModuleList([
            SeqConv2d(base_ch, base_ch, 1) for _ in range(num_fusion_modules - 1)
        ])

        self.feature_pyramid = ProgressiveFeaturePyramid(num_fusion_modules, base_ch, v_th, tau=tau, spike_type=spike_type, soft_reset=soft_reset)
        
        
#        # neurons1.py -> StackedFusionReconstructionNet -> __init__
        
#         # [修改] 使用 Bilinear 插值，彻底消除所有网格和锯齿
        
#         if upscale_factor == 4:
#             self.upsampling_path = nn.Sequential(
#                 # 第一级放大 x2 (使用双线性插值 -> 天然平滑)
#                 nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
#                 nn.Conv2d(base_ch, base_ch, kernel_size=3, padding=1),
#                 nn.LeakyReLU(0.1, inplace=True),
                
#                 # 第二级放大 x2
#                 nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
#                 nn.Conv2d(base_ch, base_ch, kernel_size=3, padding=1),
#                 nn.LeakyReLU(0.1, inplace=True),
                
#                 # 输出层
#                 nn.Conv2d(base_ch, 3, kernel_size=3, padding=1)
#             )
#         else:
#             # x2 或 x3
#             self.upsampling_path = nn.Sequential(
#                 # 直接上采样 (Bilinear)
#                 nn.Upsample(scale_factor=upscale_factor, mode='bilinear', align_corners=False),
#                 nn.Conv2d(base_ch, base_ch, kernel_size=3, padding=1),
#                 nn.LeakyReLU(0.1, inplace=True),
                
#                 # 输出层
#                 nn.Conv2d(base_ch, 3, kernel_size=3, padding=1)
#             )
        
        
        
        
        # 修改 StackedFusionReconstructionNet 的 __init__ 方法中的 upsampling_path 部分
        
#         # 1. 确保 activation 是一个类，而不是实例 (因为 SubPixelConvBlock 内部会实例化)
#         act_layer = lambda: nn.LeakyReLU(0.1, inplace=True)

#         if upscale_factor == 4:
#             # 渐进式上采样： x2 -> x2
#             self.upsampling_path = nn.Sequential(
#                 # [修正] 使用 SubPixelConvBlock，它内部包含 _icnr_init
#                 SubPixelConvBlock(base_ch, base_ch, upscale_factor=2, activation=act_layer),
#                 SubPixelConvBlock(base_ch, base_ch, upscale_factor=2, activation=act_layer),
#                 nn.Conv2d(base_ch, 3, kernel_size=3, padding=1)
#             )
#         else:
#             # x2 或 x3
#             self.upsampling_path = nn.Sequential(
#                 # [修正] 使用 SubPixelConvBlock 替代原本的手动 Conv+PixelShuffle
#                 SubPixelConvBlock(base_ch, base_ch, upscale_factor=upscale_factor, activation=act_layer),
#                 nn.Conv2d(base_ch, 3, kernel_size=3, padding=1)
#             )
        
        
        # 修改 StackedFusionReconstructionNet 的 __init__ 方法
        if upscale_factor == 4:
            # 渐进式上采样： x2 -> x2
            self.upsampling_path = nn.Sequential(
                nn.Conv2d(base_ch, base_ch * 4, kernel_size=3, padding=1),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.1, inplace=True), # 激活函数帮助过渡
                nn.Conv2d(base_ch, base_ch * 4, kernel_size=3, padding=1),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(base_ch, 3, kernel_size=3, padding=1)
            )
        else:
            # x2 或 x3 保持原样
            self.upsampling_path = nn.Sequential(
                nn.Conv2d(base_ch, base_ch * (upscale_factor ** 2), kernel_size=3, padding=1),
                nn.PixelShuffle(upscale_factor),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(base_ch, 3, kernel_size=3, padding=1)
            )
        
        
        
#         self.upsampling_path = nn.Sequential(
#             # 1. 删掉了 LIFNode，让模拟信号流过去
#             # TimeAverage(), 

#             # 2. 既然去掉了 LIF，这里的卷积最好加个激活函数
#             nn.Conv2d(base_ch, base_ch, kernel_size=3, padding=1),
#             nn.LeakyReLU(0.1, inplace=True), 
#             nn.Conv2d(base_ch, base_ch * (upscale_factor ** 2), kernel_size=3, padding=1),
#             nn.PixelShuffle(upscale_factor),
#             nn.LeakyReLU(0.1, inplace=True),
#             nn.Conv2d(base_ch, 3, kernel_size=3, padding=1)
#         )
        

        # # [关键] 上采样重建路径
        # # 顺序: 5D特征 -> Rate Coding(Mean) -> 4D特征 -> 4D上采样 -> RGB映射 -> Tanh
        # self.upsampling_path = nn.Sequential(
        #     MultiStepPmLIFNode(v_th=v_th, v_reset=v_reset, tau=tau), # [T, B, ...]
        #     TimeAverage(), # [B, ...] -> 这里压缩时间 T
        #     nn.Conv2d(base_ch, base_ch, kernel_size=3, padding=1),   # 平滑解码
        #     SubPixelConvBlock(base_ch, base_ch // 2, upscale_factor=upscale_factor),
        #     nn.Conv2d(base_ch // 2, 3, 3, padding=1),
        #     # nn.Sigmoid()
        #     # nn.Tanh() # [B, 3, H, W] -> [-1, 1] 
        # )
        
        
#         self.upsampling_path = nn.Sequential(
#             # 1. 删掉了 LIFNode，让模拟信号流过去
#             TimeAverage(), 

#             # 2. 既然去掉了 LIF，这里的卷积最好加个激活函数
#             nn.Conv2d(base_ch, base_ch, kernel_size=3, padding=1),
#             nn.LeakyReLU(0.1, inplace=True), 

#             # 3. 后面的保持不变
#             SubPixelConvBlock(base_ch, base_ch // 2, upscale_factor=upscale_factor),
#             nn.Conv2d(base_ch // 2, 3, 3, padding=1),
#             # nn.Tanh() # 建议保留 Tanh
#         )
        
        self.lab2rgb_layer = Lab2RGB()
        
        
        
        
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 1. Kaiming 初始化：让信号变强，解决梯度为零
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                
                # 2. 正偏置初始化：给神经元一点点“启动资金”，进一步防止死寂
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01) # 设为 0.01 或 0.1
        for m in self.upsampling_path.modules():
            if isinstance(m, nn.Conv2d) and m.out_channels == 3:
                nn.init.constant_(m.weight, 0)
                # nn.init.normal_(m.weight, mean=0.0, std=0.001)
                 
                
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        
    def forward(self, lr_lab):
        functional.reset_net(self)
        
        # 1. 调整输入：将 4D 的 lr_lab 扩展为 5D 序列 [T, B, C, H, W]
        adjusted_input = self.input_adjust(lr_lab) 
        T = self.fusion_modules[0].spatial_extractor.temporal_encoder.time_steps
        current_input = adjusted_input.unsqueeze(0).expand(T, -1, -1, -1, -1)

        # 2. Backbone 运算：逐个流过融合模块，保持 5D 结构
        features = []
        for i, fusion_module in enumerate(self.fusion_modules):
            fused_features = fusion_module(current_input)
            features.append(fused_features)
            if i < len(self.fusion_modules) - 1:
                adjusted_current = self.inter_module_connections[i](current_input)
                current_input = fused_features + adjusted_current

        # 3. 特征金字塔融合
        pyramid_fused = self.feature_pyramid(features) # 输出 size: [T, B, C, H, W]

        # =================【修改点 3：膜电位直读逻辑】=================
        x = pyramid_fused
        T, B, C, H, W = x.shape
        
        # 1. 融合 T 和 B，让普通层也能处理
        x_reshaped = x.reshape(T * B, C, H, W)
        
        # 2. 通过上采样路径 (注意：如果你的 SeqConv2d 会自动处理 reshape，这里直接传 x 也行)
        # 为了保险，建议手动 reshape 确保万无一失
        out_reshaped = self.upsampling_path(x_reshaped) # 输出 [T*B, 3, H_hr, W_hr]
        
        # 3. 恢复 T 维度
        _, C_out, H_hr, W_hr = out_reshaped.shape
        out_seq = out_reshaped.reshape(T, B, C_out, H_hr, W_hr)
        
        # 4. 【关键】对时间维度求平均 (Direct Readout)
        rgb_residual = out_seq.mean(0) # [B, 3, H_hr, W_hr]
        # =============================================================

        # 叠加底图
        upsampled_lr_lab = F.interpolate(lr_lab, size=rgb_residual.shape[-2:], mode='bicubic', align_corners=False)
        upsampled_lr_rgb = self.lab2rgb_layer(upsampled_lr_lab)
        
        # 此时 rgb_residual 初始全是 0 (因为零初始化)，输出完全等于 Bicubic，SSIM 正常
        hr_output_final = upsampled_lr_rgb + rgb_residual 
        return torch.clamp(hr_output_final, 0, 1)    
    
        
        
#     def forward(self, lr_lab):
#         functional.reset_net(self)
        
#         # 1. 静态 -> 扩展输入
#         adjusted_input = self.input_adjust(lr_lab) 
#         T = self.fusion_modules[0].spatial_extractor.temporal_encoder.time_steps
#         current_input = adjusted_input.unsqueeze(0).expand(T, -1, -1, -1, -1)

#         # 2. Backbone (全程保持 5D)
#         features = []
#         for i, fusion_module in enumerate(self.fusion_modules):
#             fused_features = fusion_module(current_input)
#             features.append(fused_features)
#             if i < len(self.fusion_modules) - 1:
#                 adjusted_current = self.inter_module_connections[i](current_input)
#                 current_input = fused_features + adjusted_current

#         pyramid_fused = self.feature_pyramid(features) # [T, B, base_ch, H, W]

#         # 3. 重建头 (Rate Coding + 先转后加)
#         rgb_residual = self.upsampling_path(pyramid_fused) # 输出 RGB 残差 [-1, 1]
        
#         # 1. 准备数据：为了让普通 Conv2d 处理 5D 数据，我们将 T 融合进 Batch
#         x = pyramid_fused
#         T, B, C, H, W = x.shape
#         x_reshaped = x.reshape(T * B, C, H, W) # [T*B, C, H, W]
        
#         # 2. 让数据流过 upsampling_path
#         # 因为我们刚才定义的全是普通层 (Conv, PixelShuffle)，它们可以直接处理 [N, C, H, W]
#         out_reshaped = self.upsampling_path(x_reshaped) # 输出 [T*B, 3, H_hr, W_hr]
        
#         # 3. 恢复时间维度 T
#         _, C_out, H_out, W_out = out_reshaped.shape
#         out_seq = out_reshaped.reshape(T, B, C_out, H_out, W_out) # [T, B, 3, H_hr, W_hr]
        
#         # 4. 【关键一步】膜电位直读
#         # 直接对连续的浮点数值求平均。
#         # 这里的 rgb_residual 包含了 T 个时刻的微小变化积累，精度极高。
#         rgb_residual = out_seq.mean(0) # [B, 3, H_hr, W_hr]
        
#         # -----------------------------------------------------------
#         # 后续处理 (叠加底图)
#         # -----------------------------------------------------------
#         upsampled_lr_lab = F.interpolate(lr_lab, size=rgb_residual.shape[-2:], mode='bicubic', align_corners=False)
#         upsampled_lr_rgb = self.lab2rgb_layer(upsampled_lr_lab)

#         hr_output_final = upsampled_lr_rgb + rgb_residual
        
#         # 最后限制范围到 [0, 1]
#         return torch.clamp(hr_output_final, 0, 1)    
    
    
    

#     def forward(self, lr_lab):
#         functional.reset_net(self)
        
#         # 1. 静态 -> 扩展输入
#         adjusted_input = self.input_adjust(lr_lab) 
#         T = self.fusion_modules[0].spatial_extractor.temporal_encoder.time_steps
#         current_input = adjusted_input.unsqueeze(0).expand(T, -1, -1, -1, -1)

#         # 2. Backbone (全程保持 5D)
#         features = []
#         for i, fusion_module in enumerate(self.fusion_modules):
#             fused_features = fusion_module(current_input)
#             features.append(fused_features)
#             if i < len(self.fusion_modules) - 1:
#                 adjusted_current = self.inter_module_connections[i](current_input)
#                 current_input = fused_features + adjusted_current

#         pyramid_fused = self.feature_pyramid(features) # [T, B, base_ch, H, W]

#         # 3. 重建头 (Rate Coding + 先转后加)
#         rgb_residual = self.upsampling_path(pyramid_fused) # 输出 RGB 残差 [-1, 1]
        
#         # # ==================== 【在这里插入调试代码】 ====================
#         # # 放在 neurons1.py 里！
#         # print(f"Residual Max: {rgb_residual.abs().max().item():.4f}, Mean: {rgb_residual.abs().mean().item():.4f}")
#         # # ==============================================================

#         # 准备底图
#         upsampled_lr_lab = F.interpolate(lr_lab, size=rgb_residual.shape[-2:], mode='bicubic', align_corners=False)
#         upsampled_lr_rgb = self.lab2rgb_layer(upsampled_lr_lab) # 先转 RGB

#         # 叠加
#         hr_output_final = upsampled_lr_rgb + rgb_residual
        
#         return torch.clamp(hr_output_final, 0, 1)

    
# ========== 网络组件 ==========
class EncoderLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 tau=2.0, threshold=0.5, time_steps=5, spike_type='binary'):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)
        self.time_steps = time_steps
        # self.scale = nn.Parameter(torch.tensor(1.0))

        # 神经元初始化保持不变
        if spike_type == 'ternary':
            self.lif1 = MultiStepTernaryPmLIFNode(v_th=threshold, tau=tau)
            self.lif2 = MultiStepTernaryPmLIFNode(v_th=threshold, tau=tau)
        else:
            self.lif1 = MultiStepPmLIFNode(v_th=threshold, tau=tau)
            self.lif2 = MultiStepPmLIFNode(v_th=threshold, tau=tau)

    def forward(self, x_seq):
        # [关键修改]：移除原有的 if x.dim() == 4 check 和 repeat 操作。
        # 强制要求输入必须是 [T, B, C, H, W]，确保上游传递的是真正的时序数据。
        
        # 1. 第一层卷积 + LIF
        x_seq = time_distributed_conv(self.conv1, x_seq)
        # x_seq = x_seq * self.scale
        # MultiStepNode 能够直接处理 [T, B, ...] 数据，无需手动循环
        x_seq = self.lif1(x_seq) 

        # 2. 第二层卷积 + LIF
        x_seq = time_distributed_conv(self.conv2, x_seq)
        # x_seq = x_seq * self.scale
        x_seq = self.lif2(x_seq)

        return x_seq # 返回完整的脉冲序列 [T, B, C, H, W]

class LatentSpaceLayer(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, kernel_size=3, padding=1,
                 tau=2.0, threshold=0.5, time_steps=5, spike_type='binary'):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size, padding=padding)
        # self.scale = nn.Parameter(torch.tensor(1.0))
        
        if spike_type == 'ternary':
            self.lif1 = MultiStepTernaryPmLIFNode(v_th=threshold, tau=tau)
            self.lif2 = MultiStepTernaryPmLIFNode(v_th=threshold, tau=tau)
        else:
            self.lif1 = MultiStepPmLIFNode(v_th=threshold, tau=tau)
            self.lif2 = MultiStepPmLIFNode(v_th=threshold, tau=tau)

    def forward(self, x_seq):
        # [关键修改]：同样使用 T-B 合并策略，不再只处理单帧
        x_seq = time_distributed_conv(self.conv1, x_seq)
        # x_seq = x_seq * self.scale
        x_seq = self.lif1(x_seq)

        x_seq = time_distributed_conv(self.conv2, x_seq)
        # x_seq = x_seq * self.scale
        x_seq = self.lif2(x_seq)
        
        return x_seq

class DecoderLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1,
                 tau=2.0, threshold=0.5, time_steps=5, spike_type='binary', scale_factor=None):
        super().__init__()
        # self.conv_trans = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
        # 如果外部没有指定 scale_factor，则默认使用 stride；否则使用指定的值
        if scale_factor is None:
            self.scale_val = stride
        else:
            self.scale_val = scale_factor
        
        self.up_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.upsample = nn.Upsample(scale_factor=self.scale_val, mode='bicubic', align_corners=False) # scale_factor=2 对应原来的 stride=2
        
        self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        # self.scale = nn.Parameter(torch.tensor(1.0))
        
        if spike_type == 'ternary':
            self.lif1 = MultiStepTernaryPmLIFNode(v_th=threshold, tau=tau)
            self.lif2 = MultiStepTernaryPmLIFNode(v_th=threshold, tau=tau)
        else:
            self.lif1 = MultiStepPmLIFNode(v_th=threshold, tau=tau)
            self.lif2 = MultiStepPmLIFNode(v_th=threshold, tau=tau)

    def forward(self, x_seq, skip_connection=None):
        # [关键修改]：ConvTranspose2d 也要处理 T 维度
        T, B, C, H, W = x_seq.shape
        x_flat = x_seq.reshape(T * B, C, H, W)
        # out_flat = self.conv_trans(x_flat)
        
        out_flat = self.upsample(x_flat)      # [TB, C, 2H, 2W]
        out_flat = self.up_conv(out_flat)     # [TB, OutC, 2H, 2W]
        
        _, C_out, H_out, W_out = out_flat.shape
        x_seq = out_flat.reshape(T, B, C_out, H_out, W_out)

        # 处理跳跃连接 (skip_connection 也是 [T, B, ...])
        if skip_connection is not None:
            # 此时不需要再 repeat，因为上游传递下来的 skip_connection 应该已经具备正确的 T 维度
            if x_seq.shape[-2:] != skip_connection.shape[-2:]:
                 skip_connection = F.interpolate(skip_connection.reshape(-1, *skip_connection.shape[2:]), 
                                                 size=x_seq.shape[-2:], mode='bicubic', align_corners=False).reshape(T, B, -1, H_out, W_out)
            
            
            x_seq = x_seq + skip_connection

        x_seq = time_distributed_conv(self.conv1, x_seq)
        # x_seq = x_seq * self.scale
        x_seq = self.lif1(x_seq)

        x_seq = time_distributed_conv(self.conv2, x_seq)
        # x_seq = x_seq * self.scale
        x_seq = self.lif2(x_seq)

        return x_seq
# ==================== 其他辅助类 (保持原样) ====================

# DegradationModule 等其他不需要动的类请保留在文件中...
# (这里为了简洁省略了 DegradationModule 和 DiscriminatorModule, 
#  因为它们在 trainer_loss.py 中被导入，且不需要修改)
#  如果您原来的 neurons1.py 包含它们，请把它们也加回来。
#  通常 DegradationModule 在 models.py 里，这里只放重建网络。