import lpips
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision.utils import save_image
import pandas as pd
import numpy as np
from tqdm import tqdm
import random
from ..config_runtime import config
import csv
from ..models.color_spaces import (RGB2Lab, Lab2RGB)
from ..models.networks import DegradationModule, DiscriminatorModule, ReconstructionModule, ReconstructionDiscriminator
from ..losses import VGGPerceptualLoss, LowFrequencyLoss, MultiBranchWaterLoss, EdgeLoss, ct_loss, mmd_rbf, \
    calculate_psnr, calculate_ssim, DINOv2PerceptualLoss, GradientLoss, FocalFrequencyLoss, SRVGGPerceptualLoss
from ..utils.common import GeometricAugmentation, ParameterMonitor
from ..data.datasets import UnpairedUnalignedDataset, PairedReferenceHRDataset

# class DiscriminatorFeatureCTLoss(nn.Module):
#     """使用判别器特征的对比损失，专门针对退化任务优化"""
#     def __init__(self, discriminator_module, use_multiscale=True):
#         super().__init__()
#         self.discriminator = discriminator_module
#         self.use_multiscale = use_multiscale
#     def forward(self, anchor, positive, negative):
#         """
#         anchor: 生成的LR (G(HR))
#         positive: 真实LR (真实退化)
#         negative: 双三次LR (简单退化)
#         """
#         # 确保输入在正确范围内
#         # anchor = torch.clamp(anchor, 0, 1)
#         positive = torch.clamp(positive, 0, 1)
#         negative = torch.clamp(negative, 0, 1)
        
#         # 提取多尺度特征
#         _, anchor_features = self.discriminator(anchor, return_features=True)
#         _, positive_features = self.discriminator(positive, return_features=True)
#         _, negative_features = self.discriminator(negative, return_features=True)
        
#         total_ct_loss = 0
#         num_layers = 0
        
#         for feat_anchor, feat_pos, feat_neg in zip(anchor_features, positive_features, negative_features):
#             # 保持空间信息，展平为特征向量
#             B, C, H, W = feat_anchor.shape
#             feat_anchor_flat = feat_anchor.view(B, C, -1).mean(dim=2)  # [B, C] - 空间平均
#             feat_pos_flat = feat_pos.view(B, C, -1).mean(dim=2)
#             feat_neg_flat = feat_neg.view(B, C, -1).mean(dim=2)
            
#             # 归一化特征向量
#             feat_anchor_norm = F.normalize(feat_anchor_flat + 1e-5, p=2, dim=1)
#             feat_pos_norm = F.normalize(feat_pos_flat + 1e-5, p=2, dim=1)
#             feat_neg_norm = F.normalize(feat_neg_flat + 1e-5, p=2, dim=1)
#             # feat_anchor_norm = F.normalize(feat_anchor_flat, p=2, dim=1)
#             # feat_pos_norm = F.normalize(feat_pos_flat, p=2, dim=1)
#             # feat_neg_norm = F.normalize(feat_neg_flat, p=2, dim=1)
            
#             # 余弦相似度计算
#             sim_pos = (feat_anchor_norm * feat_pos_norm).sum(dim=1)  # [B]
#             sim_neg = (feat_anchor_norm * feat_neg_norm).sum(dim=1)  # [B]
            
#             # 对比损失：拉近与真实退化，推远与双三次
#             # 使用margin-based损失，希望sim_pos > 0.8, sim_neg < 0.2
#             layer_loss = (F.relu(0.8 - sim_pos) +  # 希望相似度 > 0.8
#                          F.relu(sim_neg - 0.2)).mean()  # 希望相似度 < 0.2
            
#             total_ct_loss += layer_loss
#             num_layers += 1
        
#         return total_ct_loss / num_layers if num_layers > 0 else torch.tensor(0.0, device=anchor.device)
class DiscriminatorFeatureCTLoss(nn.Module):
    def __init__(self, discriminator_module, use_multiscale=True):
        super().__init__()
        self.discriminator = discriminator_module
        self.use_multiscale = use_multiscale

    def forward(self, anchor, positive, negative):
        # 0. 极其关键：防止输入包含极小值或 NAN
        anchor = torch.nan_to_num(anchor, nan=0.5, posinf=1.0, neginf=0.0)
        positive = torch.nan_to_num(positive, nan=0.5, posinf=1.0, neginf=0.0)
        negative = torch.nan_to_num(negative, nan=0.5, posinf=1.0, neginf=0.0)

        _, anchor_features = self.discriminator(anchor, return_features=True)
        _, positive_features = self.discriminator(positive, return_features=True)
        _, negative_features = self.discriminator(negative, return_features=True)
        
        total_ct_loss = 0
        num_layers = 0
        
        for feat_anchor, feat_pos, feat_neg in zip(anchor_features, positive_features, negative_features):
            B, C, H, W = feat_anchor.shape
            # 增加平滑处理，防止某一层特征图全为 0 导致 norm 为 0
            feat_anchor_flat = feat_anchor.view(B, C, -1).mean(dim=2) + 1e-6
            feat_pos_flat = feat_pos.view(B, C, -1).mean(dim=2) + 1e-6
            feat_neg_flat = feat_neg.view(B, C, -1).mean(dim=2) + 1e-6
            
            # 1. 增加 eps 并使用更稳定的归一化方式
            feat_anchor_norm = F.normalize(feat_anchor_flat, p=2, dim=1, eps=1e-8)
            feat_pos_norm = F.normalize(feat_pos_flat, p=2, dim=1, eps=1e-8)
            feat_neg_norm = F.normalize(feat_neg_flat, p=2, dim=1, eps=1e-8)
            
            # 2. 余弦相似度计算后进行 clamp，防止浮点误差导致其超出 [-1, 1]
            sim_pos = torch.clamp((feat_anchor_norm * feat_pos_norm).sum(dim=1), -1.0, 1.0)
            sim_neg = torch.clamp((feat_anchor_norm * feat_neg_norm).sum(dim=1), -1.0, 1.0)
            
            # 3. 如果 sim_pos 是 NAN (万一归一化还是失败)，替换为 0
            sim_pos = torch.nan_to_num(sim_pos, nan=0.0)
            sim_neg = torch.nan_to_num(sim_neg, nan=0.0)

            layer_loss = (F.relu(0.8 - sim_pos) + F.relu(sim_neg - 0.2)).mean()
            total_ct_loss += layer_loss
            num_layers += 1
        
        return total_ct_loss / num_layers if num_layers > 0 else torch.tensor(0.0, device=anchor.device)


    
# ================= [关键修改] 将 Hook 定义在类外面，作为全局函数 =================
def nan_checker_hook(module, inp, out):
    # 检查输出是否包含 NaN
    if isinstance(out, (list, tuple)):
        check_targets = out
    else:
        check_targets = [out]
    
    for i, t in enumerate(check_targets):
        if isinstance(t, torch.Tensor) and (torch.isnan(t).any() or torch.isinf(t).any()):
            print(f"\n[!!!] 捕获到 NaN/Inf! 出现在层: {module.__class__.__name__}")
            print(f"层对象信息: {module}")
            # 检查输入，看看是输入就坏了还是这一层算坏的
            if isinstance(inp, (list, tuple)) and len(inp) > 0:
                if isinstance(inp[0], torch.Tensor):
                    print(f"该层输入是否存在 NaN: {torch.isnan(inp[0]).any()}")
                    print(f"该层输入是否存在 Inf: {torch.isinf(inp[0]).any()}")
            raise RuntimeError("终止训练以排查 NaN")
# ===========================================================================
    
    
class JointTrainingSystem:
    def __init__(self, degradation_config, reconstruction_config, device='cuda'):
        self.device = device

        # 创建独立的RGB2Lab和Lab2RGB实例
        self.rgb2lab = RGB2Lab().to(device)
        self.lab2rgb = Lab2RGB().to(device)

        # 初始化模块
        self.degradation_module = DegradationModule(degradation_config, device)
        self.discriminator_module = DiscriminatorModule(degradation_config, device)
        self.reconstruction_module = ReconstructionModule(**reconstruction_config, device=device)
        
#         print("正在初始化重建专用判别器 (ReconstructionDiscriminator)...")

#         # 不需要传 config，直接传通道数 (RGB=3) 和基础通道数 (通常 64)
#         self.discriminator_hr = ReconstructionDiscriminator(input_channels=3, base_channels=64).to(device)

        # 多GPU包装
        from ..config_runtime import config
        if config.NUM_GPUS > 1:
            print(f"使用 {config.NUM_GPUS} 个GPU进行训练")
            self.degradation_module = nn.DataParallel(self.degradation_module)
            self.discriminator_module = nn.DataParallel(self.discriminator_module)
            self.reconstruction_module = nn.DataParallel(self.reconstruction_module)
            # self.discriminator_hr = nn.DataParallel(self.discriminator_hr)

        # 确保所有子模块都在正确的设备上
        self.degradation_module = self.degradation_module.to(device)
        self.discriminator_module = self.discriminator_module.to(device)
        self.reconstruction_module = self.reconstruction_module.to(device)
        
        # AMP梯度缩放器初始化
        from ..config_runtime import config
        if config.USE_AMP:
            self.scaler_g_degrade = torch.cuda.amp.GradScaler()
            self.scaler_g_reconstruct = torch.cuda.amp.GradScaler()
            self.scaler_d = torch.cuda.amp.GradScaler()
            print("AMP混合精度训练已启用")
        else:
            self.scaler_g_degrade = None
            self.scaler_g_reconstruct = None
            self.scaler_d = None
            print("AMP混合精度训练未启用")

        # 损失函数
        self.vgg_perceptual_loss = VGGPerceptualLoss().to(device)
        
        # === [新增] 给重建专用的 VGG Loss ===
        print("正在初始化重建专用 VGG Loss (Layer 16 - relu3_4)...")
        # feature_layer=16 (relu3_4) 是去模糊的关键！
        self.recon_vgg_loss = SRVGGPerceptualLoss(feature_layer=16, use_l1=True, device=device)
        
        self.dinov2_perceptual_loss = DINOv2PerceptualLoss(num_patches=4, patch_size=224, l1_weight=0.5).to(device)
        
        # ================= [新增] 初始化 LPIPS =================
        # net='alex' 是最推荐的配置，计算速度快且与人类感知最接近
        print("正在初始化 LPIPS Loss (AlexNet)...")
        self.lpips_loss = lpips.LPIPS(net='alex').to(device)
        self.lpips_loss.eval() # 设置为评估模式
        # ======================================================
        
        # 新增：使用判别器特征的CT Loss
        self.discriminator_ct_loss = DiscriminatorFeatureCTLoss(self.discriminator_module)

        self.low_freq_loss_fn = LowFrequencyLoss(scale_factor=config.UPSCALE_FACTOR, filter_size=5)
        self.reconstruction_criterion = MultiBranchWaterLoss()
        # self.edge_loss_fn = EdgeLoss(device=device).to(device)
        self.l1_loss = nn.L1Loss().to(device)
        
        self.fft_loss = FocalFrequencyLoss(loss_weight=1.0, alpha=1.0).to(device)
        
        self.gradient_loss_fn = GradientLoss(device=device).to(device)
        
        # # [新增] 初始化数据增强变换 (Flip/Rotate)
        # # 必须先导入: import torchvision.transforms as T (如果文件头没有请加上)
        # import torchvision.transforms as T 
        # self.aug_transform = T.Compose([
        #     T.RandomHorizontalFlip(p=0.5),
        #     T.RandomVerticalFlip(p=0.5)
        # ])
        # 数据增强
        self.geometric_augmentation = GeometricAugmentation(prob=config.AUGMENTATION_PROBABILITY)

        # 优化器
        from ..config_runtime import config
        self.optimizer_g_degrade = optim.Adam(self.degradation_module.parameters(), lr=config.GENERATOR_LR,
                                              betas=(0.5, 0.999))
        # self.optimizer_g_reconstruct = optim.Adam(self.reconstruction_module.parameters(), lr=config.RECONSTRUCTION_LR,
        #                                           betas=(0.5, 0.999))
        
        # ========== [修改开始 1：优化器分组，保护 Tau] ==========
        # 1. 将 Tau 参数单独提取出来
        recon_params_tau = []
        recon_params_weight = []
        
        for name, param in self.reconstruction_module.named_parameters():
            if 'tau' in name:
                recon_params_tau.append(param)
            else:
                recon_params_weight.append(param)
        
        # 2. 为 Tau 设置更小的学习率 (基础学习率的 1%)
        # 这样即使梯度很大，步子也不会迈得太大
        self.optimizer_g_reconstruct = optim.Adam([
            {'params': recon_params_weight, 'lr': config.RECONSTRUCTION_LR},
            {'params': recon_params_tau,    'lr': config.RECONSTRUCTION_LR * 0.01} 
        ], betas=(0.5, 0.999))
        # ========== [修改结束 1] ==========
        
        
        self.optimizer_d = optim.Adam(self.discriminator_module.parameters(), lr=config.DISCRIMINATOR_LR,
                                      betas=(0.5, 0.999))
        
    #     self.optimizer_d_hr = optim.Adam(
    #     self.discriminator_hr.parameters(), 
    #     lr=config.DISCRIMINATOR_LR,  # 或者手动写 1e-4
    #     betas=(0.5, 0.999)
    # )

        # 学习率调度器
        self.scheduler_g_degrade = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_g_degrade, T_max=config.DECAY_EPOCHS, eta_min=config.GENERATOR_LR / 10.0)
        self.scheduler_g_reconstruct = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_g_reconstruct, T_max=config.DECAY_EPOCHS, eta_min=config.RECONSTRUCTION_LR / 10.0)
        self.scheduler_d = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_d, T_max=config.DECAY_EPOCHS, eta_min=config.DISCRIMINATOR_LR / 10.0)

        self.lr_decay_started = False

        # 数据增强
        self.geometric_augmentation = GeometricAugmentation(prob=config.AUGMENTATION_PROBABILITY)

        # ================= 修改点1：独立的梯度累积计数器 =================
        self.accumulation_steps = config.ACCUMULATION_STEPS
        # 每个优化器独立的梯度累积计数器
        self.degrade_accum_count = 0
        self.reconstruct_accum_count = 0
        self.discriminator_accum_count = 0
        # ==============================================================

        # CT-Loss权重
        self.lambda_ct = config.CT_WEIGHT

        print("初始化学习率调度器完成")
        print(f"梯度累积步数: {self.accumulation_steps}, 有效批次大小: {config.BATCH_SIZE * self.accumulation_steps}")
        print(f"CT-Loss权重: {self.lambda_ct}")
        print("使用DiscriminatorModule特征进行CT-Loss计算")
        print("独立的梯度累积计数器已初始化：退化模块、重建模块、判别器")
    
    def freeze_degradation_model(self):
        """冻结退化模块和判别器，用于阶段二训练"""
        print(">>> 正在冻结退化模块和判别器参数... <<<")
        
        # 冻结退化模块
        self.degradation_module.eval()
        for param in self.degradation_module.parameters():
            param.requires_grad = False
        
        # 冻结判别器
        self.discriminator_module.eval()
        for param in self.discriminator_module.parameters():
            param.requires_grad = False
            
        
    def compute_degradation_ct_loss(self, generated_lab_lr, real_lr_rgb, hr_rgb):
        """使用判别器特征的CT Loss计算"""
        # 转换为RGB
        generated_rgb = self.lab2rgb(generated_lab_lr)
        
        # [核心修复 1]：必须在这里 Clamp！
        # x3 插值很容易产生负值，如果不 Clamp，CT Loss 计算相似度时会产生 NaN
        generated_rgb = torch.clamp(generated_rgb, 0, 1)
        
        # 确保真实图像也在范围内
        real_lr_rgb = torch.clamp(real_lr_rgb, 0, 1)
        
        # 对 Bicubic 结果也要对齐和 Clamp
        # 这里的 interpolate 建议也加上 size=generated_lab_lr.shape[-2:] 的逻辑，
        # 但由于它是从 HR 下采样，只要 generated_lab_lr 也是从 HR 来的，通常没问题。
        # 为了保险，可以使用 size 参数：
        bicubic_rgb = F.interpolate(hr_rgb, 
                                   size=generated_lab_lr.shape[-2:], 
                                   mode='bicubic', 
                                   align_corners=False)
        bicubic_rgb = torch.clamp(bicubic_rgb, 0, 1)
        
        # 使用判别器特征进行对比学习
        return self.discriminator_ct_loss(generated_rgb, real_lr_rgb, bicubic_rgb)
            
            
    
    
#     def compute_degradation_ct_loss(self, generated_lab_lr, real_lr_rgb, hr_rgb):
#         """使用判别器特征的CT Loss计算"""
#         # 转换为RGB（判别器需要RGB输入）
#         generated_rgb = self.lab2rgb(generated_lab_lr)
#         bicubic_rgb = F.interpolate(hr_rgb, size=generated_lab_lr.shape[2:], 
#                                    mode='bicubic', align_corners=False)
        
#         # 确保输入在正确范围内
#         # generated_rgb = torch.clamp(generated_rgb, 0, 1)
#         real_lr_rgb = torch.clamp(real_lr_rgb, 0, 1)
#         bicubic_rgb = torch.clamp(bicubic_rgb, 0, 1)
        
#         # 使用判别器特征进行对比学习
#         return self.discriminator_ct_loss(generated_rgb, real_lr_rgb, bicubic_rgb)

    def validate_l_channel_inputs(self):
        """验证所有判别器输入都是L通道"""
        print("验证L通道输入一致性...")

        # 测试数据
        test_hr = torch.rand(2, 3, 64, 64).to(self.device)
        test_lr = torch.rand(2, 3, 32, 32).to(self.device)

        # 生成LR
        generated_lab_lr = self.degradation_module(test_hr)

        # 检查生成LR的通道数
        assert generated_lab_lr.shape[1] == 3, "生成LR应该是Lab三通道"

        # 提取L通道
        generated_l = generated_lab_lr[:, 0:1, :, :]
        assert generated_l.shape[1] == 1, "L通道应该是单通道"

        # 检查CT-Loss输入
        with torch.no_grad():
            ct_loss = self.compute_degradation_ct_loss(generated_lab_lr, test_lr, test_hr)
            print(f"CT-Loss计算成功: {ct_loss.item()}")

        print("✓ 所有L通道输入验证通过")
    
    def step_schedulers(self, epoch):
        from ..config_runtime import config
        if epoch >= config.WARMUP_EPOCHS:
            if not self.lr_decay_started:
                print(f"\n=== 第{epoch + 1}轮开始学习率衰减 ===")
                self.lr_decay_started = True

            self.scheduler_g_degrade.step()
            self.scheduler_g_reconstruct.step()
            self.scheduler_d.step()

    def get_current_lr(self):
        return {
            'degradation_lr': self.optimizer_g_degrade.param_groups[0]['lr'],
            'reconstruction_lr': self.optimizer_g_reconstruct.param_groups[0]['lr'],
            'discriminator_lr': self.optimizer_d.param_groups[0]['lr']
        }

    def save_checkpoint(self, filepath, epoch, best_psnr, best_epoch, best_degradation_mmd, best_degradation_niqe):
        from ..config_runtime import config
        if config.NUM_GPUS > 1:
            degradation_state = self.degradation_module.module.state_dict()
            discriminator_state = self.discriminator_module.module.state_dict()
            reconstruction_state = self.reconstruction_module.module.state_dict()
        else:
            degradation_state = self.degradation_module.state_dict()
            discriminator_state = self.discriminator_module.state_dict()
            reconstruction_state = self.reconstruction_module.state_dict()

        # ================= 修改点2：保存累积计数器状态 =================
        checkpoint = {
            'epoch': epoch,
            'best_psnr': best_psnr,
            'best_epoch': best_epoch,
            'best_degradation_mmd': best_degradation_mmd,
            'best_degradation_niqe': best_degradation_niqe,
            'degradation_module_state_dict': degradation_state,
            'discriminator_module_state_dict': discriminator_state,
            'reconstruction_module_state_dict': reconstruction_state,
            'optimizer_g_degrade_state_dict': self.optimizer_g_degrade.state_dict(),
            'optimizer_g_reconstruct_state_dict': self.optimizer_g_reconstruct.state_dict(),
            'optimizer_d_state_dict': self.optimizer_d.state_dict(),
            'scheduler_g_degrade_state_dict': self.scheduler_g_degrade.state_dict(),
            'scheduler_g_reconstruct_state_dict': self.scheduler_g_reconstruct.state_dict(),
            'scheduler_d_state_dict': self.scheduler_d.state_dict(),
            'lr_decay_started': self.lr_decay_started,
            'accumulation_steps': self.accumulation_steps,
            'degrade_accum_count': self.degrade_accum_count,
            'reconstruct_accum_count': self.reconstruct_accum_count,
            'discriminator_accum_count': self.discriminator_accum_count,
            'scaler_g_degrade_state_dict': self.scaler_g_degrade.state_dict() if config.USE_AMP else None,
            'scaler_g_reconstruct_state_dict': self.scaler_g_reconstruct.state_dict() if config.USE_AMP else None,
            'scaler_d_state_dict': self.scaler_d.state_dict() if config.USE_AMP else None,
            # 'discriminator_hr_state_dict': self.discriminator_hr.state_dict(), # [新增]
            # 'optimizer_d_hr_state_dict': self.optimizer_d_hr.state_dict(),     # [新增]
        }
        # =============================================================
        
        torch.save(checkpoint, filepath)
        print(f"训练状态已保存到: {filepath}")

    def load_checkpoint(self, filepath, device):
        from ..config_runtime import config
        if not os.path.exists(filepath):
            print(f"警告: checkpoint文件不存在: {filepath}")
            return None

        # 1. 必须先加载文件，才能读取内容
        checkpoint = torch.load(filepath, map_location=device)

        # 2. 加载基础模块 (退化、重建、LR判别器)
        if config.NUM_GPUS > 1:
            self.degradation_module.module.load_state_dict(checkpoint['degradation_module_state_dict'])
            self.discriminator_module.module.load_state_dict(checkpoint['discriminator_module_state_dict'])
            self.reconstruction_module.module.load_state_dict(checkpoint['reconstruction_module_state_dict'])
        else:
            self.degradation_module.load_state_dict(checkpoint['degradation_module_state_dict'])
            self.discriminator_module.load_state_dict(checkpoint['discriminator_module_state_dict'])
            self.reconstruction_module.load_state_dict(checkpoint['reconstruction_module_state_dict'])

        self.optimizer_g_degrade.load_state_dict(checkpoint['optimizer_g_degrade_state_dict'])
        self.optimizer_g_reconstruct.load_state_dict(checkpoint['optimizer_g_reconstruct_state_dict'])
        self.optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])

        self.scheduler_g_degrade.load_state_dict(checkpoint['scheduler_g_degrade_state_dict'])
        self.scheduler_g_reconstruct.load_state_dict(checkpoint['scheduler_g_reconstruct_state_dict'])
        self.scheduler_d.load_state_dict(checkpoint['scheduler_d_state_dict'])

        # 3. [关键修改] 加载 HR 判别器 (必须放在 torch.load 之后)
#         if 'discriminator_hr_state_dict' in checkpoint:
#             if config.NUM_GPUS > 1:
#                 self.discriminator_hr.module.load_state_dict(checkpoint['discriminator_hr_state_dict'])
#             else:
#                 self.discriminator_hr.load_state_dict(checkpoint['discriminator_hr_state_dict'])
            
#             # 如果你有保存优化器状态，也应该加载
#             if 'optimizer_d_hr_state_dict' in checkpoint:
#                 self.optimizer_d_hr.load_state_dict(checkpoint['optimizer_d_hr_state_dict'])
            
        #     print(">>> 成功加载 HR 判别器 (ReconstructionDiscriminator) 状态 <<<")
        # else:
        #     print("!!! 警告: Checkpoint 中未包含 HR 判别器，将使用随机初始化 (如果是第一次加GAN这是正常的) !!!")

        self.lr_decay_started = checkpoint['lr_decay_started']
        self.accumulation_steps = checkpoint.get('accumulation_steps', config.ACCUMULATION_STEPS)
        
        # 加载计数器
        self.degrade_accum_count = checkpoint.get('degrade_accum_count', 0)
        self.reconstruct_accum_count = checkpoint.get('reconstruct_accum_count', 0)
        self.discriminator_accum_count = checkpoint.get('discriminator_accum_count', 0)

        if config.USE_AMP and 'scaler_g_degrade_state_dict' in checkpoint:
            if checkpoint['scaler_g_degrade_state_dict'] is not None:
                self.scaler_g_degrade.load_state_dict(checkpoint['scaler_g_degrade_state_dict'])
                self.scaler_g_reconstruct.load_state_dict(checkpoint['scaler_g_reconstruct_state_dict'])
                self.scaler_d.load_state_dict(checkpoint['scaler_d_state_dict'])
                print("AMP梯度缩放器状态已恢复")
        self._reset_all_neurons()

        print(f"训练状态已从 {filepath} 加载")
        return checkpoint

    def _reset_all_neurons(self):
        for m in self.degradation_module.modules():
            if hasattr(m, "reset"):
                m.reset()
        for m in self.discriminator_module.modules():
            if hasattr(m, "reset"):
                m.reset()
        for m in self.reconstruction_module.modules():
            if hasattr(m, "reset"):
                m.reset()

    @staticmethod
    def compute_fm_loss(real_features, fake_features):
        fm = 0.0
        count = 0
        for rf, ff in zip(real_features, fake_features):
            if rf.shape != ff.shape:
                ff = F.interpolate(ff, size=rf.shape[2:], mode='bicubic', align_corners=False)
            fm += F.l1_loss(ff, rf.detach())
            count += 1
        if count == 0:
            return torch.tensor(0.0, device='cpu')
        return fm / count

    def calculate_validation_mmd(self, val_loader, max_batches=40):
        """计算验证集的MMD损失"""
        total_mmd = 0.0
        batch_count = 0
        
        self.degradation_module.eval()
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                if batch_idx >= max_batches:
                    break
                    
                hr = batch["hr"].to(self.device)
                lr_gt = batch["lr"].to(self.device)
                
                # 生成LR图像
                generated_lab_lr = self.degradation_module(hr)
                
                # 计算MMD
                real_lab_lr = self.rgb2lab(lr_gt)
                real_l = real_lab_lr[:, 0:1, :, :]
                fake_l = generated_lab_lr[:, 0:1, :, :]
                
                batch_mmd = mmd_rbf(real_l, fake_l)
                total_mmd += batch_mmd.item()
                batch_count += 1
        
        return total_mmd / batch_count if batch_count > 0 else 0.0
    
    def relativistic_gan_loss(self, real_pred, fake_pred):
        """RaGAN Loss 计算函数 (ESRGAN 核心)"""
        # Real 应该比 Fake 真 -> log(sigmoid(Real - mean(Fake)))
        real_loss = F.binary_cross_entropy_with_logits(
            real_pred - fake_pred.mean(), torch.ones_like(real_pred))
        # Fake 应该比 Real 假 -> log(1 - sigmoid(Fake - mean(Real)))
        fake_loss = F.binary_cross_entropy_with_logits(
            fake_pred - real_pred.mean(), torch.zeros_like(fake_pred))
        return (real_loss + fake_loss) / 2
    
    def train_degradation_step(self, hr_rgb, real_lr_rgb):
        """修复后的退化模块预训练阶段，包含正确的梯度累积和AMP上下文"""
        from ..config_runtime import config

        # ====================== Step 1: 判别器更新阶段 ======================
        self.discriminator_accum_count += 1
        
        if self.discriminator_accum_count == 1:
            self.optimizer_d.zero_grad()
        
        # ✅ [修复] 添加 autocast 上下文
        with torch.cuda.amp.autocast(enabled=config.USE_AMP):
            # generated_lab_lr = self.degradation_module(hr_rgb).detach() 
            # fake_rgb = self.lab2rgb(generated_lab_lr).detach().clone()
            generated_lab_lr = self.degradation_module(hr_rgb)
            fake_rgb = self.lab2rgb(generated_lab_lr)
            
            
            # ######## [修改 1] 必须立即 Clamp，防止 NaN ########
            # Bicubic 插值会导致数值略微越界 (如 -0.01 或 1.01)，必须钳位到 0-1
            fake_rgb = torch.clamp(fake_rgb, 0, 1) 
            # fake_rgb = torch.clamp(fake_rgb, 0.001, 0.999)
            real_lr_rgb = torch.clamp(real_lr_rgb, 0, 1)
            # ################################################

            # ######## [修改 2] 判别器更新需 Detach ########
            # 显式切断梯度给判别器用，防止梯度流向退化模块
            fake_rgb_for_d = fake_rgb.detach()
            

            # real_pred = self.discriminator_module(real_lr_rgb)
            # fake_pred = self.discriminator_module(fake_rgb)
            real_pred = self.discriminator_module(real_lr_rgb)
            fake_pred = self.discriminator_module(fake_rgb_for_d) # 使用 detach 的变量
            # #############################################
            # real_pred = self.discriminator_module(torch.clamp(real_lr_rgb, 0, 1))
            # fake_pred = self.discriminator_module(torch.clamp(fake_rgb, 0, 1))

            # 标签平滑
            real_labels = torch.ones_like(real_pred) * 0.9
            fake_labels = torch.zeros_like(fake_pred) * 0.1

            # 判别器损失
            d_loss = (F.binary_cross_entropy_with_logits(real_pred, real_labels) +
                      F.binary_cross_entropy_with_logits(fake_pred, fake_labels)) / 2
            
            # 梯度累积
            d_loss = d_loss / self.accumulation_steps

        # AMP训练 - 判别器
        if config.USE_AMP:
            self.scaler_d.scale(d_loss).backward()
        else:
            d_loss.backward()
        
        if self.discriminator_accum_count == self.accumulation_steps:
            if config.USE_AMP:
                self.scaler_d.unscale_(self.optimizer_d)
                torch.nn.utils.clip_grad_norm_(self.discriminator_module.parameters(), max_norm=0.05)
                self.scaler_d.step(self.optimizer_d)
                self.scaler_d.update()
            else:
                torch.nn.utils.clip_grad_norm_(self.discriminator_module.parameters(), max_norm=0.05)
                self.optimizer_d.step()
            self.discriminator_accum_count = 0

        # ====================== Step 2: 退化模块更新阶段 ======================
        self.degrade_accum_count += 1
        
        if self.degrade_accum_count == 1:
            self.optimizer_g_degrade.zero_grad()

        # 重置脉冲神经元
        self._reset_all_neurons() # 建议使用封装好的 reset 方法

        # 冻结判别器梯度
        d_params = list(self.discriminator_module.parameters())
        d_requires = [p.requires_grad for p in d_params]
        for p in d_params:
            p.requires_grad = False

        # ✅ [修复] 添加 autocast 上下文
        with torch.cuda.amp.autocast(enabled=config.USE_AMP):
            # 重新 forward 退化模块
            # generated_lab_lr = self.degradation_module(hr_rgb)
            
            
            # generated_lab_lr[:, 0, :, :] = torch.clamp(generated_lab_lr[:, 0, :, :], min=1e-5)
            
            
            # fake_rgb = self.lab2rgb(generated_lab_lr)

            # 判别器特征
            real_logits, real_features = self.discriminator_module(real_lr_rgb, return_features=True)
            fake_logits, fake_features = self.discriminator_module(fake_rgb, return_features=True)

            # 各项损失计算
            adv_loss = F.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits) * 0.9)
            fm_loss = self.compute_fm_loss(real_features, fake_features)

            hr_lab = self.rgb2lab(hr_rgb)
            L_hr = hr_lab[:, 0:1, :, :]
            fake_l = generated_lab_lr[:, 0:1, :, :]
            L_hr_down = F.interpolate(
                L_hr, 
                size=fake_l.shape[-2:], # 关键修改：强制对齐
                mode='bicubic', 
                align_corners=False
            )
            # L_hr_down = F.interpolate(L_hr, scale_factor=1 / config.UPSCALE_FACTOR, mode='bicubic', align_corners=False)

            low_freq_loss = self.low_freq_loss_fn(L_hr, fake_l)
            
            # # ================= [插入在这里] =================
            # # 放在这里最合适，刚算完低频 Loss，正准备算感知 Loss
            # print(f"DEBUG Check: L_channel max={fake_l.max().item():.4f}, min={fake_l.min().item():.4f}")
            # # ===============================================
            
            # L_fake_norm = torch.clamp(fake_l, 0, 1)
            # L_hr_norm = torch.clamp(L_hr_down, 0, 1)

            L_hr_3ch = L_hr_down.repeat(1, 3, 1, 1)
            # fake_l_clamped = torch.clamp(fake_l, 0, 1)      # 1. 先把数据限制在 0-1 之间
            # L_fake_3ch = fake_l_clamped.repeat(1, 3, 1, 1)  # 2. 再复制成 3 通道
            L_fake_3ch = fake_l.repeat(1, 3, 1, 1)
            perceptual_loss = self.vgg_perceptual_loss(L_fake_3ch, L_hr_3ch)

            mmd_loss = torch.tensor(0.0, device=self.device)
            loss_ct = self.compute_degradation_ct_loss(generated_lab_lr, real_lr_rgb, hr_rgb)
            current_ct_weight = self.lambda_ct * (0.5 if not self.lr_decay_started else 1.0)

            degradation_loss = (config.ADV_WEIGHT * adv_loss +
                                config.FM_WEIGHT * fm_loss +
                                config.LOWFREQ_WEIGHT * low_freq_loss +
                                config.PERCEPT_WEIGHT * perceptual_loss +
                                0.0 * mmd_loss +
                                current_ct_weight * loss_ct)

            degradation_loss = degradation_loss / self.accumulation_steps
            
            # ================= [插入排查代码：精确排查哪个 Loss 是罪魁祸首] =================
            # 逐项检查子损失张量
            for name, loss_val in [('adv', adv_loss), ('fm', fm_loss), ('low_freq', low_freq_loss), 
                                   ('perceptual', perceptual_loss), ('ct', loss_ct)]:
                if torch.isnan(loss_val).any() or torch.isinf(loss_val).any():
                    print(f"\n[!!!] 警告：发现异常子损失 -> {name} = {loss_val.item()}")
                    # 如果这个子损失炸了，我们可以看看输入是不是有问题
                    # 比如：print(f"检查该层输入是否存在 NaN...")

            # 检查除以累积步数后的总损失
            if torch.isnan(degradation_loss).any() or torch.isinf(degradation_loss).any():
                print(f"[!!!] 警告：总损失 degradation_loss 在 backward 之前已变为 NaN/Inf!")
                # 强制打印当前的权重权重配置，确认是否有权重设置错误
                print(f"当前权重：ADV={config.ADV_WEIGHT}, FM={config.FM_WEIGHT}, LF={config.LOWFREQ_WEIGHT}")
            # =============================================================================

        # AMP训练 - 退化模块
        if config.USE_AMP:
            self.scaler_g_degrade.scale(degradation_loss).backward()
        else:
            degradation_loss.backward()
        
        # 恢复判别器梯度状态
        for p, req in zip(d_params, d_requires):
            p.requires_grad = req
        
        if self.degrade_accum_count == self.accumulation_steps:
            if config.USE_AMP:
                self.scaler_g_degrade.unscale_(self.optimizer_g_degrade)
                torch.nn.utils.clip_grad_norm_(self.degradation_module.parameters(), max_norm=1.0)
                self.scaler_g_degrade.step(self.optimizer_g_degrade)
                self.scaler_g_degrade.update()
            else:
                torch.nn.utils.clip_grad_norm_(self.degradation_module.parameters(), max_norm=1.0)
                self.optimizer_g_degrade.step()
            self.degrade_accum_count = 0

        return {
            'd_loss': d_loss.item() * self.accumulation_steps,
            'degradation_loss': degradation_loss.item() * self.accumulation_steps,
            'adv_loss': adv_loss.item(),
            # ... 其他返回保持不变
            'fm_loss': fm_loss.item() if isinstance(fm_loss, torch.Tensor) else float(fm_loss),
            'mmd_loss': mmd_loss.item() if isinstance(mmd_loss, torch.Tensor) else float(mmd_loss),
            'low_freq_loss': low_freq_loss.item(),
            'perceptual_loss': perceptual_loss.item(),
            'loss_ct': loss_ct.item(),
            'ct_weight': current_ct_weight
        }
    
    
    
    def train_step(self, hr_rgb, real_lr_rgb):
        from ..config_runtime import config
        import torchvision.transforms.functional as TF
        
        use_frozen_degradation = getattr(config, 'USE_PRETRAINED_RLGM', False)
        
        # 1. 判别器训练 ------------------------------------------------------
        d_loss = torch.tensor(0.0, device=self.device)
        
        if not use_frozen_degradation:
            self.discriminator_accum_count += 1
            if self.discriminator_accum_count == 1:
                self.optimizer_d.zero_grad()
            
            with torch.cuda.amp.autocast(enabled=config.USE_AMP):
                self._reset_all_neurons()
                with torch.no_grad():
                     generated_lab_lr = self.degradation_module(hr_rgb).detach()
                
                fake_rgb = self.lab2rgb(generated_lab_lr).detach().clone()
                
                self._reset_all_neurons()
                # real_pred = self.discriminator_module(real_lr_rgb)
                real_pred = self.discriminator_module(torch.clamp(real_lr_rgb, 0, 1))
                self._reset_all_neurons()
                # fake_pred = self.discriminator_module(fake_rgb)
                fake_pred = self.discriminator_module(torch.clamp(fake_rgb, 0, 1))

                real_labels = torch.ones_like(real_pred) * 0.9
                fake_labels = torch.zeros_like(fake_pred) * 0.1
                loss_d_calc = (F.binary_cross_entropy_with_logits(real_pred, real_labels) +
                               F.binary_cross_entropy_with_logits(fake_pred, fake_labels)) / 2
                d_loss = loss_d_calc / self.accumulation_steps

            if config.USE_AMP:
                self.scaler_d.scale(d_loss).backward()
            else:
                d_loss.backward()
            
            if self.discriminator_accum_count == self.accumulation_steps:
                if config.USE_AMP:
                    self.scaler_d.unscale_(self.optimizer_d)
                    torch.nn.utils.clip_grad_norm_(self.discriminator_module.parameters(), max_norm=1.0)
                    self.scaler_d.step(self.optimizer_d)
                    self.scaler_d.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.discriminator_module.parameters(), max_norm=1.0)
                    self.optimizer_d.step()
                self.discriminator_accum_count = 0

        # 2. 生成器联合训练 --------------------------------------------------
        self.degrade_accum_count += 1
        self.reconstruct_accum_count += 1
        
        if self.degrade_accum_count == 1 and not use_frozen_degradation:
            self.optimizer_g_degrade.zero_grad()
        if self.reconstruct_accum_count == 1:
            self.optimizer_g_reconstruct.zero_grad()

        # [修复重点] 临时冻结判别器，防止梯度污染
        d_params = list(self.discriminator_module.parameters())
        d_requires_grad_states = [p.requires_grad for p in d_params]
        for p in d_params:
            p.requires_grad = False

        try:
            self._reset_all_neurons()

            with torch.cuda.amp.autocast(enabled=config.USE_AMP):
                # A. 退化 (Forward)
                if use_frozen_degradation:
                    with torch.no_grad():
                        generated_lab_lr = self.degradation_module(hr_rgb)
                else:
                    generated_lab_lr = self.degradation_module(hr_rgb)
                    
                    
                # ======= [在这里插入排查代码 A] =======
                if torch.isnan(generated_lab_lr).any():
                    print("\n[!!!] NaN 诞生于退化模块 (Degradation Module) 输出!")
                    print(f"输入 HR 范围: min={hr_rgb.min().item():.4f}, max={hr_rgb.max().item():.4f}")
                    raise ValueError("Degradation Module Output NaN")
                # ======================================
                    
                    
                # B. 增强 (Augmentation)
                # 直接调用修改后的 geometric_augmentation，传入 (LR, HR)
                # 它会自动保证两者发生完全一样的翻转/旋转
                
                # 注意：generated_lab_lr 是生成的LR，hr_rgb 是原始HR
                
                
                lr_original = generated_lab_lr
                hr_original = hr_rgb
                
                # 1. 执行同步增强：生成新的变量 (aug_lr, aug_hr)，它们是旋转后的
                # 您的 utils.py 确保了 aug_lr 和 aug_hr 拥有完全相同的旋转/翻转
                lr_augmented, hr_augmented = self.geometric_augmentation(
                    lr_original, 
                    target=hr_original
                )
                

#                 # B. 增强 (Augmentation)
                # ================= [修改：自动梯度阀门] =================
                if use_frozen_degradation:
                    # 模式 A (冻结): 切断梯度，防止重建模块“作弊”修改退化模块
                    reconstruction_input = lr_augmented.detach()
                else:
                    # 模式 B (联合): 保留梯度，允许端到端微调
                    reconstruction_input = lr_augmented
                # ======================================================


                # C. 重建 (Forward)
                
                # ======= [在这里插入排查代码 B] =======
                if torch.isnan(reconstruction_input).any():
                    print("\n[!!!] NaN 进入了重建模块! 检查增强 (Augmentation) 或退化输出")
                    raise ValueError("Reconstruction Input NaN")
                # ======================================
                
                # reconstruction_input = lr_augmented.detach()
                hr_reconstructed = self.reconstruction_module(reconstruction_input)
                # hr_reconstructed = torch.clamp(hr_reconstructed, 0, 1)
                
               # ==================== 【插入点：HR 判别器训练】 ====================
        
                # 1. 定义热身步数 (前 5000 步只练 L1，不练 GAN)
                # 简单粗暴的方式：用 optimizer 的 step 计数，或者传入 global_step
                # 这里假设你有个 self.global_step 计数器，如果没有，暂且设 warmup 为 0 或手动控制
                # warmup_steps = 0 
                # # 获取当前步数 (大概估算)
                # current_step = self.optimizer_g_reconstruct.state_dict()['state'].get(
                #     next(iter(self.reconstruction_module.parameters())), {}).get('step', 0)
                # d_hr_loss_val = 0.0

#                 # 2. 训练 HR 判别器 (D_HR Update)
#                 if current_step > warmup_steps:
#                     # 开启梯度
#                     for p in self.discriminator_hr.parameters(): p.requires_grad = True
#                     self.optimizer_d_hr.zero_grad()
            
#                     # 判别真假 (注意：fake 需要 detach)
#                     # 使用增强后的 HR (hr_augmented) 作为真值
#                     real_hr_pred = self.discriminator_hr(torch.clamp(hr_augmented, 0, 1))
#                     fake_hr_pred = self.discriminator_hr(torch.clamp(hr_reconstructed.detach(), 0, 1))

#                     # 计算 RaGAN Loss 并更新
#                     d_hr_loss = self.relativistic_gan_loss(real_hr_pred, fake_hr_pred)
#                     d_hr_loss.backward()
#                     self.optimizer_d_hr.step()
#                     d_hr_loss_val = d_hr_loss.item()

#                     # 关闭梯度 (为生成器训练做准备)
#                     for p in self.discriminator_hr.parameters(): p.requires_grad = False 
                

                # D. Loss 计算
                # --- Degrade Loss ---
                if use_frozen_degradation:
                    total_degradation_loss = torch.tensor(0.0, device=self.device)
                    adv_loss = torch.tensor(0.0, device=self.device)
                    fm_loss = torch.tensor(0.0, device=self.device)
                    mmd_loss_degrade = torch.tensor(0.0, device=self.device)
                    low_freq_loss_degrade = torch.tensor(0.0, device=self.device)
                    perceptual_loss_degrade = torch.tensor(0.0, device=self.device)
                    loss_ct = torch.tensor(0.0, device=self.device)
                    current_ct_weight = 0.0
                else:
                    fake_rgb = self.lab2rgb(generated_lab_lr)

                    self._reset_all_neurons()
                    # _, real_features = self.discriminator_module(real_lr_rgb, return_features=True)
                    _, real_features = self.discriminator_module(torch.clamp(real_lr_rgb, 0, 1), return_features=True)
                    self._reset_all_neurons()
                    fake_logits, fake_features = self.discriminator_module(fake_rgb, return_features=True)
                    # fake_logits, fake_features = self.discriminator_module(torch.clamp(fake_rgb, 0, 1), return_features=True)

                    adv_loss = F.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits) * 0.9)
                    fm_loss = self.compute_fm_loss(real_features, fake_features)

                    hr_lab = self.rgb2lab(hr_rgb)
                    L_hr = hr_lab[:, 0:1, :, :]
                    fake_l = generated_lab_lr[:, 0:1, :, :]
                    L_hr_down = F.interpolate(L_hr, scale_factor=1 / config.UPSCALE_FACTOR, mode='bicubic', align_corners=False)
                    
                    low_freq_loss_degrade = self.low_freq_loss_fn(L_hr, fake_l)

                    L_hr_3ch = L_hr_down.repeat(1, 3, 1, 1)
                    L_fake_3ch = fake_l.repeat(1, 3, 1, 1)
                    perceptual_loss_degrade = self.vgg_perceptual_loss(L_fake_3ch, L_hr_3ch)
                    
                    real_lab_lr = self.rgb2lab(real_lr_rgb)
                    # ======= [在这里插入排查代码 C] =======
                    if torch.isnan(real_lab_lr).any():
                        print("\n[!!!] NaN 诞生于真实 LR 的 RGB->Lab 转换!")
                        print(f"real_lr_rgb 范围: min={real_lr_rgb.min().item():.4f}, max={real_lr_rgb.max().item():.4f}")
                        raise ValueError("Real LR Lab conversion NaN")
                    # ======================================
                    real_l = real_lab_lr[:, 0:1, :, :]
                    mmd_loss_degrade = torch.tensor(0.0, device=self.device)

                    loss_ct = self.compute_degradation_ct_loss(generated_lab_lr, real_lr_rgb, hr_rgb)
                    current_ct_weight = self.lambda_ct * (0.5 if not self.lr_decay_started else 1.0)

                    total_degradation_loss = (config.ADV_WEIGHT * adv_loss +
                                              config.FM_WEIGHT * fm_loss +
                                              config.LOWFREQ_WEIGHT * low_freq_loss_degrade +
                                              config.PERCEPT_WEIGHT * perceptual_loss_degrade +
                                              0.0 * mmd_loss_degrade +
                                              current_ct_weight * loss_ct)
                
                
                # --- Reconstruct Loss ---
                reconstruction_loss = self.l1_loss(hr_reconstructed, hr_augmented)
                
                loss_fft = self.fft_loss(hr_reconstructed, hr_augmented)
                
                grad_loss_val = self.gradient_loss_fn(hr_reconstructed, hr_augmented)
                
                perceptual_loss_reconstruct = self.recon_vgg_loss(
                    hr_reconstructed, 
                    torch.clamp(hr_augmented, 0, 1) # 确保 GT 也是干净的
                )
                
                # perceptual_loss_reconstruct = self.dinov2_perceptual_loss(
                #     # torch.clamp(hr_reconstructed, 0, 1), 
                #     hr_reconstructed,
                #     torch.clamp(hr_augmented, 0, 1)
                # )
                
                
                # perceptual_loss_reconstruct = self.dinov2_perceptual_loss(hr_reconstructed, hr_augmented)
                # edge_loss = self.edge_loss_fn(hr_reconstructed, hr_augmented)
                
                # recon_norm = torch.clamp(hr_reconstructed, 0, 1) * 2 - 1
                recon_norm = hr_reconstructed * 2 - 1
                gt_norm = torch.clamp(hr_augmented, 0, 1) * 2 - 1
                lpips_val = self.lpips_loss(recon_norm, gt_norm).mean()
                
                # 计算 HR 对抗损失 (G_GAN_Loss)
                adv_loss_hr = torch.tensor(0.0, device=self.device)

#                 if current_step > warmup_steps:
#                     # 重新判别 (带梯度)
#                     fake_hr_pred_g = self.discriminator_hr(hr_reconstructed) 
#                     # fake_hr_pred_g = self.discriminator_hr(torch.clamp(hr_reconstructed, 0, 1))
#                     real_hr_pred_detached = self.discriminator_hr(hr_augmented).detach()

#                     # RaGAN Generator Loss: 骗判别器说"我也很真"
#                     adv_loss_hr = self.relativistic_gan_loss(fake_hr_pred_g, real_hr_pred_detached)

                
                total_reconstruction_loss = (
                    0.8 * reconstruction_loss +         # 【改】L1 恢复为 1.0 (保底色准)
                    1.0 * perceptual_loss_reconstruct + # DINO (保结构)
                    1.0* lpips_val +                   # LPIPS (保观感)
                    0.0 * loss_fft +                    # 【新增】FFT (核心去糊手段)
                    0.5 * grad_loss_val              # Gradient (保边缘锐度)
                    # 0.0 * adv_loss_hr                  # 【改】GAN 提至 0.02 (核心加纹理手段)
                )
            #     total_reconstruction_loss = (
            #     0.01 * reconstruction_loss +        # L1 降权
            #     1.0 * perceptual_loss_reconstruct + # DINO 保持
            #     1.0 * lpips_val +                   # LPIPS 保持
            #     0.005 * adv_loss_hr                 # <--- 新增对抗损失
            # )
                
                
                
                # 修改后 (强制像素对齐)
#                 total_reconstruction_loss = (0.1 * reconstruction_loss +

#                                              1.0 * perceptual_loss_reconstruct +

#                                              0.2 * edge_loss + 

#                                              1.0 * lpips_val)

                
                degradation_loss_scaled = total_degradation_loss / self.accumulation_steps
                reconstruction_loss_scaled = total_reconstruction_loss / self.accumulation_steps

            # Backward (在 try 块内执行，确保 finally 会被调用)
            if config.USE_AMP:
                if not use_frozen_degradation:
                    self.scaler_g_degrade.scale(degradation_loss_scaled).backward()
                self.scaler_g_reconstruct.scale(reconstruction_loss_scaled).backward()
            else:
                if not use_frozen_degradation:
                    degradation_loss_scaled.backward()
                reconstruction_loss_scaled.backward()
        
        finally:
            # [修复重点] 必须恢复判别器的梯度状态，否则下一次 train_step 的第一阶段判别器无法更新
            for p, state in zip(d_params, d_requires_grad_states):
                p.requires_grad = state
        
        # 3. 优化器更新 ------------------------------------------------------
        degrade_update = False
        reconstruct_update = False
        
        if self.degrade_accum_count == self.accumulation_steps and not use_frozen_degradation:
            degrade_update = True
            self.degrade_accum_count = 0
        if self.reconstruct_accum_count == self.accumulation_steps:
            reconstruct_update = True
            self.reconstruct_accum_count = 0
            
            
        
        if degrade_update and not use_frozen_degradation:
            if config.USE_AMP:
                self.scaler_g_degrade.unscale_(self.optimizer_g_degrade)
                torch.nn.utils.clip_grad_norm_(self.degradation_module.parameters(), max_norm=1.0)
                self.scaler_g_degrade.step(self.optimizer_g_degrade)
                self.scaler_g_degrade.update()
            else:
                torch.nn.utils.clip_grad_norm_(self.degradation_module.parameters(), max_norm=1.0)
                self.optimizer_g_degrade.step()

        # --- B. 重建模块更新 (核心修复！) ---
        if reconstruct_update:
            # 1. 准备梯度 (Unscale)
            if config.USE_AMP:
                self.scaler_g_reconstruct.unscale_(self.optimizer_g_reconstruct)
            
            # 2. 先按数值截断 Tau (防止 500+ 的梯度炸飞全场)
            # 这一步会把 Tau 的梯度强行压到 [-0.1, 0.1]
            tau_params = [p for n, p in self.reconstruction_module.named_parameters() 
                          if 'tau' in n and p.grad is not None]
            if tau_params:
                torch.nn.utils.clip_grad_value_(tau_params, clip_value=0.1)
            
            # 3. 再按范数截断全局 (保护卷积层)
            # 现在 max_norm=1.0 是安全的，因为 Tau 已经被压住了
            torch.nn.utils.clip_grad_norm_(self.reconstruction_module.parameters(), max_norm=1.0)
            
#             # ================= [插入开始：打印梯度范数] =================
#             # 每隔 100 个 update 打印一次，防止刷屏
#             # 我们用一个临时计数器或者简单判断随机数，这里为了简单直接打印
#             # 建议仅打印前几层和后几层，观察梯度传递情况
#             print_grad = False
#             try:
#                 # 尝试获取当前的 step (如果优化器里有的话)
#                 current_step = self.optimizer_g_reconstruct.state_dict()['state'].get(
#                     next(iter(self.reconstruction_module.parameters())), {}).get('step', 0)
#                 if current_step % 50 == 0: # 每 50 步打印一次
#                     print_grad = True
#             except:
#                 print_grad = True # 获取失败就默认打印，或者你可以手动控制

#             if print_grad:
#                 print(f"\n--- [Debug] 卷积层权重梯度检查 (Step {current_step}) ---")
#                 conv_count = 0
#                 for name, param in self.reconstruction_module.named_parameters():
#                     # 只看卷积层的权重，且必须有梯度
#                     if 'weight' in name and 'conv' in name and param.grad is not None:
#                         # 只打印网络头部(input)、中部、尾部(output)的几层
#                         if conv_count < 2 or 'fusion' in name or conv_count > 50: 
#                             grad_norm = param.grad.norm().item()
#                             print(f"Layer: {name} | Grad Norm: {grad_norm:.4e}")
#                         conv_count += 1
#                 print("------------------------------------------------------\n")
#             # ================= [插入结束] =================
            
            # 4. 执行更新
            if config.USE_AMP:
                self.scaler_g_reconstruct.step(self.optimizer_g_reconstruct)
                self.scaler_g_reconstruct.update()
            else:
                self.optimizer_g_reconstruct.step()

            # 5. [最后保险] 强制将 Tau 限制在合理物理范围内
            # 防止多次迭代后 Tau 变成负数或极大值
            with torch.no_grad():
                for name, param in self.reconstruction_module.named_parameters():
                    if 'tau' in name:
                        # 0.01 ~ 5.0 是 SNN 时间常数的常见物理范围
                        param.clamp_(0.01, 5.0) 
        # ========== [修改结束 2] ==========
            
        
        
        # if config.USE_AMP:
        #     if degrade_update:
        #         self.scaler_g_degrade.unscale_(self.optimizer_g_degrade)
        #         torch.nn.utils.clip_grad_norm_(self.degradation_module.parameters(), max_norm=1.0)
        #         self.scaler_g_degrade.step(self.optimizer_g_degrade)
        #         self.scaler_g_degrade.update()
        #     if reconstruct_update:
        #         self.scaler_g_reconstruct.unscale_(self.optimizer_g_reconstruct)
        #         torch.nn.utils.clip_grad_norm_(self.reconstruction_module.parameters(), max_norm=20.0)
        #         self.scaler_g_reconstruct.step(self.optimizer_g_reconstruct)
        #         self.scaler_g_reconstruct.update()
        # else:
        #     if degrade_update:
        #         torch.nn.utils.clip_grad_norm_(self.degradation_module.parameters(), max_norm=1.0)
        #         self.optimizer_g_degrade.step()
        #     if reconstruct_update:
        #         torch.nn.utils.clip_grad_norm_(self.reconstruction_module.parameters(), max_norm=20.0)
        #         self.optimizer_g_reconstruct.step()


        return {
            'd_loss': d_loss.item() * self.accumulation_steps if isinstance(d_loss, torch.Tensor) else d_loss,
            'g_loss': (total_degradation_loss + total_reconstruction_loss).item(),
            'adv_loss': adv_loss.item() if not use_frozen_degradation else 0.0,
            'adv_loss_hr': adv_loss_hr.item(),
            'fm_loss': fm_loss.item() if isinstance(fm_loss, torch.Tensor) else float(fm_loss),
            'mmd_loss': mmd_loss_degrade.item() if isinstance(mmd_loss_degrade, torch.Tensor) else float(mmd_loss_degrade),
            'reconstruction_loss': reconstruction_loss.item(),
            'perceptual_loss_reconstruct': perceptual_loss_reconstruct.item(),
            # 'edge_loss': edge_loss.item(),
            'gradient_loss': grad_loss_val.item(),
            'low_freq_loss_degrade': low_freq_loss_degrade.item() if not use_frozen_degradation else 0.0,
            'perceptual_loss_degrade': perceptual_loss_degrade.item() if not use_frozen_degradation else 0.0,
            'degradation_loss': total_degradation_loss.item(),
            'reconstruction_total_loss': total_reconstruction_loss.item(),
            'hr_reconstructed': hr_reconstructed.detach(),
            'generated_lab_lr': generated_lab_lr.detach(),
            'augmented_lab_lr': lr_augmented.detach(),
            'augmented_hr_rgb': hr_augmented.detach(),
            'loss_ct': loss_ct.item() if not use_frozen_degradation else 0.0,
            'ct_weight': current_ct_weight,
            'lpips_loss': lpips_val.item()
        }
    
    
