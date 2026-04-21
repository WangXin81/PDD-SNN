import os

# 设置显存碎片化管理
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image
import sys

# 导入自定义模块
from ..config_runtime import config
from ..losses import calculate_psnr, calculate_ssim
from ..data.datasets import PairedReferenceHRDataset
from ..utils.common import set_seed, ParameterMonitor, save_metrics_to_csv, \
    batch_calculate_metrics_reconstruction, EnergyMeter
# 注意：这里导入的是我们将要修改的 trainer 文件
from .recon_trainer import JointTrainingSystem


# ================= [工具类：用于特征图可视化] =================
class LayerActivations:
    """一个用来'钩住'中间层输出的小钩子"""
    def __init__(self):
        self.features = None
    
    def hook_fn(self, module, input, output):
        self.features = output.detach()

def save_feature_map(model, input_tensor, layer_name_fragment, save_path):
    """可视化指定层的特征图"""
    model.eval()
    activations = LayerActivations()
    
    target_layer = None
    for name, module in model.named_modules():
        if layer_name_fragment in name and isinstance(module, torch.nn.Conv2d):
            target_layer = module
            # print(f"-> 成功定位可视化目标层: {name}")
            break
            
    if target_layer is None: return

    handle = target_layer.register_forward_hook(activations.hook_fn)
    
    try:
        with torch.no_grad():
            if input_tensor.dim() == 3: img = input_tensor.unsqueeze(0)
            else: img = input_tensor
            model(img)
    except Exception as e:
        pass

    handle.remove()
    
    features = activations.features
    if features is None: return

    if features.dim() == 5: features = features.mean(dim=0) 
    feature_map = features[0]
    heatmap = feature_map.mean(dim=0)
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        save_image(heatmap, save_path)
    except Exception as e:
        print(f"保存特征图失败: {e}")

# =============================================================

def prepare_environment():
    set_seed(getattr(config, "SEED", 42))
    os.makedirs(config.SAVE_DIR, exist_ok=True)

def main():
    prepare_environment()
    # ==================== 1. 数据加载器配置 (纯配对模式) ====================
    print("="*50)
    print("正在初始化配对数据加载器 (Paired Mode)...")
    print(f"  - 训练集 HR: {config.HR_DIR}")
    print(f"  - 训练集 LR: {config.LR_DIR}")
    
    # 训练集加载器
    train_dataset = PairedReferenceHRDataset(
        hr_dir=config.HR_DIR,
        lr_dir=config.LR_DIR,
        upscale_factor=getattr(config, 'UPSCALE_FACTOR', 4),
        patch_size=config.LR_PATCH_SIZE
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True # 训练集通常丢弃最后不足的batch
    )

    # 验证集加载器
    if hasattr(config, 'VAL_LR_DIR') and config.VAL_LR_DIR is not None:
        print(f"  - 验证集 HR: {config.VAL_HR_DIR}")
        print(f"  - 验证集 LR: {config.VAL_LR_DIR}")
        val_dataset = PairedReferenceHRDataset(
            hr_dir=config.VAL_HR_DIR,
            lr_dir=config.VAL_LR_DIR,
            upscale_factor=getattr(config, 'UPSCALE_FACTOR', 4),
            patch_size=config.LR_PATCH_SIZE
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.VAL_BATCH_SIZE, # 通常为 1
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=True,
            drop_last=False
        )
    else:
        raise ValueError("配对训练模式下，必须在 Config 中指定 VAL_LR_DIR！")

    print(f"数据加载完成。训练集 batches: {len(train_loader)}, 验证集 batches: {len(val_loader)}")
    print("="*50)

    # ==================== 2. 初始化模型 ====================
    print("初始化联合训练系统...")
    # 注意：trainer_loss_gan_copy1.py 中的 JointTrainingSystem 需要适配这种模式
    joint_system = JointTrainingSystem(
        # degradation_config=config.DEGRADATION_CONFIG,
        # reconstruction_config=config.RECONSTRUCTION_CONFIG,
        device=config.DEVICE
    )
    
    # 初始化参数监控器
    param_monitor = ParameterMonitor()

    # 初始化评估器 (可选)
    reconstruction_evaluator = None
    try:
        from ..evaluation.metrics import ImageQualityEvaluator
        reconstruction_evaluator = ImageQualityEvaluator(device=config.DEVICE)
        print("重建模块图像质量评估器 (NIQE/BRISQUE) 已初始化")
    except ImportError:
        print("提示: 未找到 evaluate 模块，将跳过 NIQE/BRISQUE 计算")

    # ==================== 3. 初始化能耗计算器 ====================
    # 获取 LR 尺寸用于能耗估算
    # 假设输入是正方形，取 config 中的某个默认值或者写死一个典型值
    # 这里我们简单取 64x64 作为 LR 基准
    lr_h, lr_w = 64, 64 
    t_steps = config.RECONSTRUCTION_CONFIG.get('time_steps', 5)
    
    energy_meter = EnergyMeter(
        model=joint_system.reconstruction_module, 
        input_size=(1, 3, lr_h, lr_w), 
        device=config.DEVICE, 
        time_steps=t_steps
    )

    # ==================== 4. 准备记录文件 ====================
    metrics_csv_path = os.path.join(config.SAVE_DIR, "training_metrics.csv")
    csv_headers = [
        'epoch', 
        # Train
        'train_l1', 'train_percep', 'train_grad', 'train_lpips', 'train_total',
        'train_niqe', # 如果有
        # Val
        'val_psnr', 'val_ssim', 'val_lpips', 'val_niqe', 'val_brisque', 'val_uiqm',
        # Energy
        'val_energy_J', 'val_ops_G',
        # LR
        'lr_recon'
    ]
    
    best_psnr = 0.0
    best_epoch = 0
    start_epoch = 0

    # ==================== 5. 恢复训练逻辑 ====================
    if config.RESUME_TRAINING and os.path.exists(config.RESUME_CHECKPOINT_PATH):
        print(f"正在从checkpoint恢复训练: {config.RESUME_CHECKPOINT_PATH}")
        checkpoint_data = joint_system.load_checkpoint(config.RESUME_CHECKPOINT_PATH, config.DEVICE)
        if checkpoint_data:
            start_epoch = checkpoint_data['epoch']
            best_psnr = checkpoint_data.get('best_psnr', 0.0)
            best_epoch = checkpoint_data.get('best_epoch', 0)
            print(f"已恢复至 Epoch {start_epoch + 1}")
    else:
        print("从头开始训练 (Scratch)")

    # 创建必要的保存目录
    os.makedirs(os.path.join(config.SAVE_DIR, "train_vis"), exist_ok=True)
    os.makedirs(os.path.join(config.SAVE_DIR, "val_vis"), exist_ok=True)

    # ==================== 6. 主训练循环 (纯重建) ====================
    print("\n" + "="*50)
    print(f"开始重建模块有监督训练 (Supervised Training) - 共 {config.EPOCHS} Epochs")
    print("="*50)

    for epoch in range(start_epoch, config.EPOCHS):
        # --- 训练阶段 ---
        joint_system.reconstruction_module.train()
        
        epoch_stats = {
            'l1_loss': 0.0, 'percep_loss': 0.0, 'grad_loss': 0.0, 
            'lpips_loss': 0.0, 'total_loss': 0.0
        }
        
        train_pbar = tqdm(train_loader, desc=f"Train Epoch {epoch + 1}/{config.EPOCHS}")
        last_train_recon_img = None # 用于计算NIQE

        for i, batch in enumerate(train_pbar):
            # 获取数据: HR 和 真实的 LR
            hr = batch["hr"].to(config.DEVICE)
            real_lr = batch["lr"].to(config.DEVICE) # 直接从数据集读取

            # 调用 Train Step
            # 注意：这里的 train_step 必须是修改后只接受 (hr, real_lr) 的版本
            losses = joint_system.train_step(hr, real_lr)

            # 记录 Loss
            epoch_stats['l1_loss'] += losses.get('reconstruction_loss', 0.0)
            epoch_stats['percep_loss'] += losses.get('perceptual_loss_reconstruct', 0.0)
            epoch_stats['grad_loss'] += losses.get('gradient_loss', 0.0)
            epoch_stats['lpips_loss'] += losses.get('lpips_loss', 0.0)
            epoch_stats['total_loss'] += losses.get('reconstruction_total_loss', 0.0)
            
            # 用于后续可视化或指标计算
            hr_reconstructed = losses.get('hr_reconstructed')
            if i == len(train_loader) - 1:
                last_train_recon_img = hr_reconstructed.detach().clamp(0, 1)

            # 进度条显示
            train_pbar.set_postfix({
                'L1': f"{losses.get('reconstruction_loss', 0.0):.4f}",
                'LPIPS': f"{losses.get('lpips_loss', 0.0):.4f}"
            })

            # 训练集可视化 (每100个batch保存一次)
            if i % 100 == 0:
                vis_dir = os.path.join(config.SAVE_DIR, "train_vis", f"epoch_{epoch + 1}")
                os.makedirs(vis_dir, exist_ok=True)
                
                # 取 batch 中第一张图
                save_lr = torch.nn.functional.interpolate(real_lr[0:1], size=hr.shape[2:], mode='nearest')
                save_recon = torch.clamp(hr_reconstructed[0:1], 0, 1)
                save_hr = torch.clamp(hr[0:1], 0, 1)
                
                combined = torch.cat([save_lr, save_recon, save_hr], dim=3) # 左右拼接
                save_image(combined, os.path.join(vis_dir, f"iter_{i}.png"))

        # 计算 Epoch 平均 Loss
        num_batches = len(train_loader)
        avg_stats = {k: v / num_batches for k, v in epoch_stats.items()}
        
        # 调整学习率
        joint_system.step_schedulers(epoch)
        current_lr = joint_system.get_current_lr()['reconstruction_lr']

        # 计算训练集 NIQE (可选)
        train_niqe = 0.0
        if reconstruction_evaluator and last_train_recon_img is not None:
            metrics = batch_calculate_metrics_reconstruction(last_train_recon_img, reconstruction_evaluator)
            train_niqe = metrics.get('niqe', 0.0)

        print(f"\nEpoch {epoch + 1} Stats: Total Loss={avg_stats['total_loss']:.4f}, L1={avg_stats['l1_loss']:.4f}, LR={current_lr:.2e}")

        # --- 验证阶段 ---
        joint_system.reconstruction_module.eval()
        val_psnr_total = 0.0
        val_ssim_total = 0.0
        val_lpips_total = 0.0
        # val_niqe_total = 0.0
        # val_count = 0
        val_niqe_total = 0.0
        val_brisque_total = 0.0
        val_uiqm_total = 0.0
        val_count = 0
        
        # 准备能耗计算
        energy_meter.reset()
        energy_meter.register_hooks()
        
        val_save_dir = os.path.join(config.SAVE_DIR, "val_vis", f"epoch_{epoch+1}")
        os.makedirs(val_save_dir, exist_ok=True)

        print("开始验证...")
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                if batch_idx >= config.MAX_VAL_BATCHES: break
                
                val_hr = batch["hr"].to(config.DEVICE)
                val_lr = batch["lr"].to(config.DEVICE)
                
                # 1. 格式转换 (RGB -> LAB)
                val_lr_lab = joint_system.rgb2lab(val_lr)
                joint_system._reset_all_neurons()
                
                # 2. 推理
                val_recon = joint_system.reconstruction_module(val_lr_lab)
                
                # 3. 后处理
                val_recon_clamped = torch.clamp(val_recon, 0, 1)
                val_hr_clamped = torch.clamp(val_hr, 0, 1)
                
                # 4. 计算基础指标
                # PSNR/SSIM 需要 RGB [0, 255]
                # 注意：calculate_psnr 内部会自己处理 0-1 到 0-255 的转换
                current_psnr = 0.0
                current_ssim = 0.0
                batch_size = val_hr.shape[0]
                
                for b in range(batch_size):
                    current_psnr += calculate_psnr(val_recon_clamped[b], val_hr_clamped[b], crop_border=config.UPSCALE_FACTOR, test_y_channel=True)
                    current_ssim += calculate_ssim(val_recon_clamped[b], val_hr_clamped[b], crop_border=config.UPSCALE_FACTOR, test_y_channel=True)
                
                val_psnr_total += current_psnr
                val_ssim_total += current_ssim
                
                # 5. 计算 LPIPS
                recon_norm = val_recon_clamped * 2 - 1
                hr_norm = val_hr_clamped * 2 - 1
                val_lpips_total += joint_system.lpips_loss(recon_norm, hr_norm).mean().item() * batch_size
                
                # ================= [修改 2] 计算 NIQE, BRISQUE, UIQM =================
                # 只有当评估器初始化成功时才计算
                if reconstruction_evaluator is not None:
                    # 1. 计算 NIQE
                    # calculate_batch_niqe 接受 Tensor [B, 3, H, W] (0-1)
                    batch_niqe = reconstruction_evaluator.calculate_batch_niqe(val_recon_clamped)
                    val_niqe_total += batch_niqe * batch_size
                    
                    # 2. 计算 BRISQUE
                    batch_brisque = reconstruction_evaluator.calculate_batch_brisque(val_recon_clamped)
                    val_brisque_total += batch_brisque * batch_size
                    
                    # 3. 计算 UIQM
                    batch_uiqm = reconstruction_evaluator.calculate_batch_uiqm(val_recon_clamped)
                    val_uiqm_total += batch_uiqm * batch_size
                # ====================================================================
                
                val_count += batch_size

                # 6. 保存部分验证图片
                if batch_idx < 3:
                    for k in range(min(2, batch_size)):
                        save_image(val_recon_clamped[k], os.path.join(val_save_dir, f"b{batch_idx}_img{k}_recon.png"))
                        # if epoch == 0: # GT 只存一次
                        save_image(val_hr_clamped[k], os.path.join(val_save_dir, f"b{batch_idx}_img{k}_hr.png"))
                
                # 7. 可视化特征图 (仅第一张)
                if batch_idx == 0:
                    vis_feat_dir = os.path.join(config.SAVE_DIR, "feature_maps", f"epoch_{epoch+1}")
                    save_feature_map(joint_system.reconstruction_module, val_lr_lab[0:1], "fusion", 
                                     os.path.join(vis_feat_dir, "layer_fusion.png"))

        # 计算验证集平均指标
        avg_val_psnr = val_psnr_total / val_count if val_count > 0 else 0
        avg_val_ssim = val_ssim_total / val_count if val_count > 0 else 0
        avg_val_lpips = val_lpips_total / val_count if val_count > 0 else 0
        
        avg_val_niqe = val_niqe_total / val_count if val_count > 0 else 0
        avg_val_brisque = val_brisque_total / val_count if val_count > 0 else 0
        avg_val_uiqm = val_uiqm_total / val_count if val_count > 0 else 0
        
        energy_metrics = energy_meter.calculate_metrics()
        energy_meter.remove_hooks()

        print("-" * 30)
        print(f"验证集结果 (Epoch {epoch+1}):")
        print(f"  PSNR: {avg_val_psnr:.4f} | SSIM: {avg_val_ssim:.4f} | LPIPS: {avg_val_lpips:.4f}")
        print(f"  NIQE: {avg_val_niqe:.4f} | BRISQUE: {avg_val_brisque:.4f} | UIQM: {avg_val_uiqm:.4f}")
        print(f"  能耗: {energy_metrics.get('Energy_SNN (J)', 0):.4f} J")
        print("-" * 30)

        # 保存 CSV
        metrics_row = [
            epoch + 1,
            avg_stats['l1_loss'], avg_stats['percep_loss'], avg_stats['grad_loss'], avg_stats['lpips_loss'], avg_stats['total_loss'],
            train_niqe,
            avg_val_psnr, avg_val_ssim, avg_val_lpips, avg_val_niqe, avg_val_brisque, avg_val_uiqm,
            energy_metrics.get('Energy_SNN (J)', 0), energy_metrics.get('GSOPs', 0),
            current_lr
        ]
        save_metrics_to_csv(metrics_csv_path, metrics_row, csv_headers)

        # 保存最佳模型
        if avg_val_psnr > best_psnr:
            best_psnr = avg_val_psnr
            best_epoch = epoch + 1
            save_path = os.path.join(config.SAVE_DIR, "best_reconstruction_net.pth")
            if config.NUM_GPUS > 1:
                torch.save(joint_system.reconstruction_module.module.state_dict(), save_path)
            else:
                torch.save(joint_system.reconstruction_module.state_dict(), save_path)
            print(f"发现新最佳 PSNR: {best_psnr:.4f}，模型已保存。")

        # 定期保存 Checkpoint
        if (epoch + 1) % 5 == 0:
            latest_path = os.path.join(config.SAVE_DIR, "latest_checkpoint.pth")
            joint_system.save_checkpoint(latest_path, epoch+1, best_psnr, best_epoch, 0.0, 0.0)

    print("所有训练完成！")

if __name__ == "__main__":
    main()
