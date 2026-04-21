import os
import torch
import torch.nn as nn
import torch.optim as optim
import lpips
from ..models.networks import ReconstructionModule
from ..losses import SRVGGPerceptualLoss, GradientLoss
from ..models.color_spaces import RGB2Lab, Lab2RGB
from ..utils.common import GeometricAugmentation

class JointTrainingSystem:
    def __init__(self, device='cuda'):
        self.device = device
        
        # 导入配置 (延迟导入避免循环依赖)
        from ..config_runtime import config

        # ==================== 1. 初始化模型 ====================
        print("正在初始化重建模块 (Reconstruction Module)...")
        self.reconstruction_module = ReconstructionModule(**config.RECONSTRUCTION_CONFIG, device=device)
        
        # 多GPU处理
        if config.NUM_GPUS > 1:
            print(f"使用 {config.NUM_GPUS} 个GPU进行训练")
            self.reconstruction_module = nn.DataParallel(self.reconstruction_module)
        
        self.reconstruction_module = self.reconstruction_module.to(device)

        # ==================== 2. 工具类初始化 ====================
        # 颜色空间转换
        self.rgb2lab = RGB2Lab().to(device)
        self.lab2rgb = Lab2RGB().to(device) # 虽然重建网络内部有转换，但这在可视化时可能用到

        # 数据增强 (几何变换：翻转/旋转)
        self.geometric_augmentation = GeometricAugmentation(prob=config.AUGMENTATION_PROBABILITY)

        # ==================== 3. 损失函数初始化 ====================
        # (1) 像素损失
        self.l1_loss = nn.L1Loss().to(device)
        
        # (2) 感知损失 (VGG19 - Layer relu3_4)
        print("初始化 SRVGGPerceptualLoss...")
        self.recon_vgg_loss = SRVGGPerceptualLoss(feature_layer=16, use_l1=True, device=device)
        
        # (3) LPIPS 损失 (AlexNet)
        print("初始化 LPIPS Loss...")
        self.lpips_loss = lpips.LPIPS(net='alex').to(device)
        self.lpips_loss.eval() 
        
        # (4) 梯度损失 (边缘锐度)
        self.gradient_loss_fn = GradientLoss(device=device).to(device)

        # ==================== 4. 优化器配置 (SNN 特有) ====================
        # 将 Tau 参数单独提取，给予更小的学习率，防止 SNN 时间常数剧烈波动
        recon_params_tau = []
        recon_params_weight = []
        
        for name, param in self.reconstruction_module.named_parameters():
            if 'tau' in name:
                recon_params_tau.append(param)
            else:
                recon_params_weight.append(param)
        
        self.optimizer_g_reconstruct = optim.Adam([
            {'params': recon_params_weight, 'lr': config.RECONSTRUCTION_LR},
            {'params': recon_params_tau,    'lr': config.RECONSTRUCTION_LR * 0.01} 
        ], betas=(0.5, 0.999))

        # ==================== 5. 学习率调度器 ====================
        self.scheduler_g_reconstruct = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_g_reconstruct, 
            T_max=config.DECAY_EPOCHS, 
            eta_min=config.RECONSTRUCTION_LR / 10.0
        )

        # ==================== 6. AMP 混合精度 ====================
        if config.USE_AMP:
            self.scaler_g_reconstruct = torch.cuda.amp.GradScaler()
            print("AMP 混合精度训练已启用")
        else:
            self.scaler_g_reconstruct = None

        # 梯度累积计数器
        self.accumulation_steps = config.ACCUMULATION_STEPS
        self.reconstruct_accum_count = 0
        self.lr_decay_started = False

    def train_step(self, hr_rgb, real_lr_rgb):
        """
        配对数据有监督训练步
        :param hr_rgb: [B, 3, H, W] 真实高清图
        :param real_lr_rgb: [B, 3, H/s, W/s] 真实低清图
        """
        from ..config_runtime import config

        # 1. 优化器梯度清零 (处理梯度累积)
        self.reconstruct_accum_count += 1
        if self.reconstruct_accum_count == 1:
            self.optimizer_g_reconstruct.zero_grad()

        # 2. 前向传播与 Loss 计算
        with torch.cuda.amp.autocast(enabled=config.USE_AMP):
            # A. 重置 SNN 神经元状态
            self._reset_all_neurons()

            # B. 格式转换: RGB -> LAB (重建网络需要 LAB 输入)
            input_lr_lab = self.rgb2lab(real_lr_rgb)

            # C. 同步几何增强 (Flip/Rotate)
            # 确保 LR 和 HR 做完全相同的变换
            lr_augmented, hr_augmented = self.geometric_augmentation(input_lr_lab, target=hr_rgb)
            
            # D. 网络推理
            # 输入: Augmented LR (LAB) -> 输出: Reconstructed HR (RGB)
            hr_reconstructed = self.reconstruction_module(lr_augmented)
            
            # E. 损失计算
            # 1. L1 Loss (像素级)
            loss_l1 = self.l1_loss(hr_reconstructed, hr_augmented)
            
            # 准备数据用于感知计算 (确保在 0-1 之间)
            recon_clamped = torch.clamp(hr_reconstructed, 0, 1)
            gt_clamped = torch.clamp(hr_augmented, 0, 1)
            
            # 2. VGG Loss (结构纹理)
            loss_percep = self.recon_vgg_loss(recon_clamped, gt_clamped)
            
            # 3. LPIPS Loss (视觉感知)
            # LPIPS 推荐输入范围 [-1, 1]
            loss_lpips = self.lpips_loss(recon_clamped * 2 - 1, gt_clamped * 2 - 1).mean()
            
            # 4. Gradient Loss (边缘)
            loss_grad = self.gradient_loss_fn(hr_reconstructed, hr_augmented)

            # F. 总损失加权
            # 注意：如果 config 中没有定义这些 WEIGHT 常量，这里使用默认值
            w_l1 = getattr(config, 'RECON_L1_WEIGHT', 1.0)
            w_percep = getattr(config, 'RECON_PERCEP_WEIGHT', 1.0)
            w_lpips = getattr(config, 'RECON_LPIPS_WEIGHT', 1.0)
            w_grad = getattr(config, 'RECON_GRAD_WEIGHT', 0.5)

            total_loss = (w_l1 * loss_l1 + 
                          w_percep * loss_percep + 
                          w_lpips * loss_lpips + 
                          w_grad * loss_grad)
            
            # 梯度累积缩放
            loss_scaled = total_loss / self.accumulation_steps

        # 3. 反向传播
        if config.USE_AMP:
            self.scaler_g_reconstruct.scale(loss_scaled).backward()
        else:
            loss_scaled.backward()

        # 4. 优化器更新
        if self.reconstruct_accum_count == self.accumulation_steps:
            if config.USE_AMP:
                self.scaler_g_reconstruct.unscale_(self.optimizer_g_reconstruct)
                
                # SNN 梯度保护：先截断 Tau 的梯度值
                tau_params = [p for n, p in self.reconstruction_module.named_parameters() 
                              if 'tau' in n and p.grad is not None]
                if tau_params:
                    torch.nn.utils.clip_grad_value_(tau_params, clip_value=0.1)
                
                # 全局梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.reconstruction_module.parameters(), max_norm=1.0)
                
                self.scaler_g_reconstruct.step(self.optimizer_g_reconstruct)
                self.scaler_g_reconstruct.update()
            else:
                torch.nn.utils.clip_grad_norm_(self.reconstruction_module.parameters(), max_norm=1.0)
                self.optimizer_g_reconstruct.step()
            
            # 计数器重置
            self.reconstruct_accum_count = 0
            
            # [物理约束] 强制限制 Tau 在合理范围内 (0.01 ~ 5.0)
            with torch.no_grad():
                for name, param in self.reconstruction_module.named_parameters():
                    if 'tau' in name:
                        param.clamp_(0.01, 5.0)

        return {
            'reconstruction_loss': loss_l1.item(),
            'perceptual_loss_reconstruct': loss_percep.item(),
            'lpips_loss': loss_lpips.item(),
            'gradient_loss': loss_grad.item(),
            'reconstruction_total_loss': total_loss.item(),
            'hr_reconstructed': hr_reconstructed.detach(), # 用于可视化
            'augmented_hr_rgb': hr_augmented.detach()      # 用于可视化对比 GT
        }

    def _reset_all_neurons(self):
        """重置所有脉冲神经元的膜电位状态"""
        for m in self.reconstruction_module.modules():
            if hasattr(m, "reset"):
                m.reset()

    def step_schedulers(self, epoch):
        from ..config_runtime import config
        if epoch >= config.WARMUP_EPOCHS:
            self.scheduler_g_reconstruct.step()

    def get_current_lr(self):
        return {'reconstruction_lr': self.optimizer_g_reconstruct.param_groups[0]['lr']}

    def save_checkpoint(self, filepath, epoch, best_psnr, best_epoch, val_ssim, val_lpips):
        from ..config_runtime import config
        
        # 处理多 GPU 保存逻辑
        if config.NUM_GPUS > 1:
            reconstruction_state = self.reconstruction_module.module.state_dict()
        else:
            reconstruction_state = self.reconstruction_module.state_dict()

        checkpoint = {
            'epoch': epoch,
            'best_psnr': best_psnr,
            'best_epoch': best_epoch,
            'val_ssim': val_ssim,
            'val_lpips': val_lpips,
            'reconstruction_module_state_dict': reconstruction_state,
            'optimizer_g_reconstruct_state_dict': self.optimizer_g_reconstruct.state_dict(),
            'scheduler_g_reconstruct_state_dict': self.scheduler_g_reconstruct.state_dict(),
            'scaler_g_reconstruct_state_dict': self.scaler_g_reconstruct.state_dict() if config.USE_AMP else None,
        }
        
        torch.save(checkpoint, filepath)
        print(f"Checkpoint 已保存: {filepath}")

    def load_checkpoint(self, filepath, device):
        if not os.path.exists(filepath):
            print(f"警告: 无法找到 checkpoint: {filepath}")
            return None

        print(f"正在加载 checkpoint: {filepath}")
        checkpoint = torch.load(filepath, map_location=device)
        
        from ..config_runtime import config
        if config.NUM_GPUS > 1:
            self.reconstruction_module.module.load_state_dict(checkpoint['reconstruction_module_state_dict'])
        else:
            self.reconstruction_module.load_state_dict(checkpoint['reconstruction_module_state_dict'])

        self.optimizer_g_reconstruct.load_state_dict(checkpoint['optimizer_g_reconstruct_state_dict'])
        self.scheduler_g_reconstruct.load_state_dict(checkpoint['scheduler_g_reconstruct_state_dict'])
        
        if config.USE_AMP and 'scaler_g_reconstruct_state_dict' in checkpoint:
             if checkpoint['scaler_g_reconstruct_state_dict'] is not None:
                self.scaler_g_reconstruct.load_state_dict(checkpoint['scaler_g_reconstruct_state_dict'])

        self._reset_all_neurons()
        return checkpoint
