#退化损失转为rgb通道进行计算
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import vgg16
from glob import glob
from spikingjelly.activation_based import surrogate
from spikingjelly.activation_based import functional

# 导入其他必要的模块
from .neurons import (
    TemporalEncoder, MultiStepPmLIFNode, MultiStepTernaryPmLIFNode,
    SubPixelConvBlock, SpatialEncoder, FrequencyFeatureExtractor,
    FusionModule, ProgressiveFeaturePyramid, StackedFusionReconstructionNet,
)
from .color_spaces import (RGB2Lab, Lab2RGB)


class TemporalSNNDiscriminatorLocal(nn.Module):
    def __init__(self, in_channels=3, base_channels=32, time_steps=8, spike_type='binary', use_temporal=True):
        super().__init__()
        self.time_steps = time_steps
        self.spike_type = spike_type
        self.use_temporal = use_temporal

        if self.use_temporal:
            self.temporal_encoder = TemporalEncoder(time_steps=time_steps, mode="noisy", noise_scale=0.02)
            self.conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=2, padding=1)
            self.conv2 = nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1)
            self.conv3 = nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1)
            if spike_type == 'binary':
                self.lif1 = MultiStepPmLIFNode(tau=1.2, v_th=0.5, surrogate_function=surrogate.Sigmoid(alpha=4.0))
                self.lif2 = MultiStepPmLIFNode(tau=1.2, v_th=0.5, surrogate_function=surrogate.Sigmoid(alpha=4.0))
                self.lif3 = MultiStepPmLIFNode(tau=1.2, v_th=0.5, surrogate_function=surrogate.Sigmoid(alpha=4.0))
            else:
                self.lif1 = MultiStepTernaryPmLIFNode(tau=1.2, v_th=0.5, surrogate_function=surrogate.Sigmoid(alpha=4.0))
                self.lif2 = MultiStepTernaryPmLIFNode(tau=1.2, v_th=0.5, surrogate_function=surrogate.Sigmoid(alpha=4.0))
                self.lif3 = MultiStepTernaryPmLIFNode(tau=1.2, v_th=0.5, surrogate_function=surrogate.Sigmoid(alpha=4.0))
            self.gap = nn.AdaptiveAvgPool2d(1)
        else:
            self.conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=2, padding=1)
            self.conv2 = nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1)
            self.conv3 = nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1)
            if spike_type == 'ternary':
                self.lif1 = MultiStepTernaryPmLIFNode(tau=1.2, v_th=0.5, surrogate_function=surrogate.Sigmoid(alpha=4.0))
                self.lif2 = MultiStepTernaryPmLIFNode(tau=1.2, v_th=0.5, surrogate_function=surrogate.Sigmoid(alpha=4.0))
                self.lif3 = MultiStepTernaryPmLIFNode(tau=1.2, v_th=0.5, surrogate_function=surrogate.Sigmoid(alpha=4.0))
            else:
                self.lif1 = MultiStepPmLIFNode(tau=1.2, v_th=0.5, surrogate_function=surrogate.Sigmoid(alpha=4.0))
                self.lif2 = MultiStepPmLIFNode(tau=1.2, v_th=0.5, surrogate_function=surrogate.Sigmoid(alpha=4.0))
                self.lif3 = MultiStepPmLIFNode(tau=1.2, v_th=0.5, surrogate_function=surrogate.Sigmoid(alpha=4.0))
            self.gap = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Linear(base_channels * 4, 1)

    def forward(self, x, return_features=False):
        functional.reset_net(self)
        assert x.dim() == 4
        batch_size, channels, height, width = x.shape
        # 移除对通道数的断言，现在支持1通道或3通道

        features = []
        if self.use_temporal:
            x_temporal = self.temporal_encoder(x)
            x_temporal = x_temporal.permute(1, 0, 2, 3, 4)
            temporal_outputs = []
            for t in range(self.time_steps):
                x_t = x_temporal[:, t]
                out = self.conv1(x_t)
                out = self.lif1(out.unsqueeze(0)).squeeze(0)
                features.append(out)
                out = self.conv2(out)
                out = self.lif2(out.unsqueeze(0)).squeeze(0)
                features.append(out)
                out = self.conv3(out)
                out = self.lif3(out.unsqueeze(0)).squeeze(0)
                features.append(out)
                temporal_outputs.append(out.unsqueeze(1))
            temporal_outputs = torch.cat(temporal_outputs, dim=1)
            out = temporal_outputs.mean(dim=1)
            out = self.gap(out).view(out.size(0), -1)
        else:
            out = self.conv1(x)
            out = self.lif1(out.unsqueeze(0)).squeeze(0)
            features.append(out)
            out = self.conv2(out)
            out = self.lif2(out.unsqueeze(0)).squeeze(0)
            features.append(out)
            out = self.conv3(out)
            out = self.lif3(out.unsqueeze(0)).squeeze(0)
            features.append(out)
            out = self.gap(out).view(out.size(0), -1)

        logits = self.fc(out)
        if return_features:
            return logits, features
        else:
            return logits

class DegradationModule(nn.Module):
    def __init__(self, degradation_config, device='cuda'):
        super().__init__()
        # self.device = device
        from .neurons import TemporalLABEnhancer
        self.enhancer = TemporalLABEnhancer(
            time_steps=degradation_config['time_steps'],
            downsample_factor=degradation_config['downsample_factor'],
            stats_path=degradation_config['stats_path'],
            kernelgan_kernels_path=degradation_config['kernelgan_kernels_path'],
            noise_patches_dir=degradation_config['noise_patches_dir'],
            spike_type=degradation_config['spike_type'],
            use_temporal=degradation_config['use_temporal']
        )
        # .to(device)
    def forward(self, hr_rgb):
        # self._safe_reset()
        functional.reset_net(self)
        generated_lab_lr, _, _, _ = self.enhancer(hr_rgb)
        return generated_lab_lr
    # def _safe_reset(self):
    #     """安全重置神经元状态"""
    #     for module in self.modules():
    #         if hasattr(module, 'reset'):
    #             try:
    #                 module.reset()
    #             except Exception as e:
    #                 # 如果reset失败，尝试其他方式
    #                 if hasattr(module, 'v'):
    #                     module.v = None
    #                 if hasattr(module, 'v_seq'):
    #                     module.v_seq = None

class DiscriminatorModule(nn.Module):
    def __init__(self, degradation_config, device='cuda'):
        super().__init__()
        # self.device = device
        
        # 修改：支持RGB三通道输入
        self.expected_channels = 3  # 现在期望RGB三通道输入
        
        self.discriminator = TemporalSNNDiscriminatorLocal(
            in_channels=3,  # 修改：改为3通道RGB输入
            base_channels=32,
            time_steps=degradation_config['time_steps'],
            spike_type=degradation_config['spike_type'],
            use_temporal=degradation_config['use_temporal']
        )
        # .to(device)

    def forward(self, input, return_features=False):
        functional.reset_net(self)
        
        # 输入验证 - 确保是RGB三通道
        if input.shape[1] != self.expected_channels:
            raise ValueError(f"判别器期望{self.expected_channels}通道输入(RGB通道)，但得到{input.shape[1]}通道")
        
        # 确保在正确范围内
        input = torch.clamp(input, 0, 1)

        if return_features:
            return self.discriminator(input, return_features=True)
        else:
            return self.discriminator(input)

class ReconstructionModule(nn.Module):
    def __init__(self, num_fusion_modules=3, time_steps=5, base_ch=64, upscale_factor=2,
                 v_th=0.3, v_reset=0.0, tau=2.0,spike_type='binary', soft_reset=False,ablation_mode='dual', device='cuda'):
        super().__init__()
        # self.device = device
        self.reconstruction_net = StackedFusionReconstructionNet(
            num_fusion_modules=num_fusion_modules,
            time_steps=time_steps,
            base_ch=base_ch,
            upscale_factor=upscale_factor,
            v_th=v_th,
            v_reset=v_reset,
            tau=tau,
            spike_type=spike_type,
            soft_reset=soft_reset,
            ablation_mode=ablation_mode
        )
        # .to(device)

    def forward(self, lr_lab):
        functional.reset_net(self)
        # self._safe_reset()
        hr_reconstructed = self.reconstruction_net(lr_lab)
        return hr_reconstructed
        # """安全重置神经元状态"""
        # for module in self.modules():
        #     if hasattr(module, 'reset'):
        #         try:
        #             module.reset()
        #         except Exception as e:
        #             if hasattr(module, 'v'):
        #                 module.v = None
        #             if hasattr(module, 'v_seq'):
        #                 module.v_seq = None
        
        
        
        
import torch
import torch.nn as nn

class ReconstructionDiscriminator(nn.Module):
    """
    专门用于超分辨率重建的判别器 (VGG-Style)
    特点：
    1. 去掉了 Batch Norm (BN)，避免伪影，适合色彩丰富的重建任务。
    2. 使用 LeakyReLU 增加稳定性。
    3. 结尾使用 AdaptiveAvgPool，支持任意分辨率输入 (64, 128, 256 均可)。
    """
    def __init__(self, input_channels=3, base_channels=64):
        super(ReconstructionDiscriminator, self).__init__()

        self.features = nn.Sequential(
            # input: [batch, 3, H, W]
            
            # 第一层不需要 BN
            nn.Conv2d(input_channels, base_channels, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),

            # 下采样 1
            nn.Conv2d(base_channels, base_channels, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            # 特征提取
            nn.Conv2d(base_channels, base_channels * 2, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),

            # 下采样 2
            nn.Conv2d(base_channels * 2, base_channels * 2, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            # 特征提取
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),

            # 下采样 3
            nn.Conv2d(base_channels * 4, base_channels * 4, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            # 特征提取
            nn.Conv2d(base_channels * 4, base_channels * 8, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 下采样 4 (感受野显著增大)
            nn.Conv2d(base_channels * 8, base_channels * 8, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            # 特征提取
            nn.Conv2d(base_channels * 8, base_channels * 8, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # 分类器
        self.classifier = nn.Sequential(
            # 关键：自适应池化，强行将任意尺寸压缩为 1x1
            # 这样无论输入是 128x128 还是 256x256，这里出来的都是 [B, 512, 1, 1]
            nn.AdaptiveAvgPool2d((1, 1)), 
            nn.Flatten(),
            nn.Linear(base_channels * 8, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1) # 输出 Logits
        )

    def forward(self, x):
        features = self.features(x)
        output = self.classifier(features)
        return output
