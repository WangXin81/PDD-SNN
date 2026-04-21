import os

# 设置显存碎片化管理
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.utils import save_image
import pandas as pd
import numpy as np
import csv
from tqdm import tqdm
from PIL import Image
import sys

# 导入自定义模块
from config import config
from models import DegradationModule, DiscriminatorModule, ReconstructionModule
from losses1 import calculate_psnr, calculate_ssim
from data_loader import create_data_loaders, PairedReferenceHRDataset, UnpairedUnalignedDataset
from utils import set_seed, ParameterMonitor, create_niqe_prior_if_needed, calculate_distribution_metrics, \
    save_metrics_to_csv, batch_calculate_metrics_degradation, \
    batch_calculate_metrics_reconstruction, EnergyMeter
# from trainer_loss_gan import JointTrainingSystem
from trainer_loss_gan_copy1 import JointTrainingSystem





# ================= [新增工具类：用于特征图可视化] =================
class LayerActivations:
    """一个用来'钩住'中间层输出的小钩子"""
    def __init__(self):
        self.features = None
    
    def hook_fn(self, module, input, output):
        # 这里的 output 就是该层的特征图
        self.features = output.detach()

def save_feature_map(model, input_tensor, layer_name_fragment, save_path):
    """
    可视化指定层的特征图
    input_tensor: 输入图像 [B, C, H, W]
    layer_name_fragment: 你想看的层名字的一部分，比如 'spatial_extractor'
    """
    model.eval()
    activations = LayerActivations()
    
    # 1. 自动寻找你想看的层
    target_layer = None
    for name, module in model.named_modules():
        # 只要名字里包含你写的片段，就钩住这一层
        if layer_name_fragment in name and isinstance(module, torch.nn.Conv2d):
            target_layer = module
            print(f"-> 成功定位可视化目标层: {name}")
            break
            
    if target_layer is None:
        return # 没找到就不画了

    # 2. 注册钩子
    handle = target_layer.register_forward_hook(activations.hook_fn)
    
    # 3. 跑一次前向传播 (Forward)
    try:
        with torch.no_grad():
            # 确保输入维度正确，如果是 [C, H, W] 加上 Batch 维
            if input_tensor.dim() == 3:
                img = input_tensor.unsqueeze(0)
            else:
                img = input_tensor
            
            # 跑模型 (只需跑通即可，结果不重要，重要的是钩子拿到了中间层)
            model(img)
    except Exception as e:
        print(f"可视化前向传播时忽略错误: {e}")
        pass

    # 4. 移除钩子 (清理现场)
    handle.remove()
    
    # 5. 处理特征图
    features = activations.features
    if features is None:
        return

    # SNN 输出可能是 5D [T, B, C, H, W] 或 4D [B, C, H, W]
    # 我们需要在 Time 维度和 Channel 维度做平均，把它变成一张热力图
    
    # 如果是 5D (含时间步)，先对时间 T 取平均
    if features.dim() == 5:
        features = features.mean(dim=0) # -> [B, C, H, W]
    
    # 取第一张图
    feature_map = features[0] # -> [C, H, W]
    
    # 对所有通道取平均 (Mean Projection)，变成单通道热力图
    # 也可以取最大值 (Max Projection)，看你喜好
    heatmap = feature_map.mean(dim=0) # -> [H, W]
    
    # 归一化到 0-1 以便保存图片
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    
    # 保存
    try:
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        save_image(heatmap, save_path)
        print(f"特征图已保存: {save_path}")
    except Exception as e:
        print(f"保存特征图失败: {e}")

# =============================================================




def prepare_synthetic_validation_data(joint_system, config):
    """
    使用当前加载的退化模型，将验证集HR转换为对应的合成LR，并保存到磁盘。
    返回生成的LR文件夹路径。
    """
    print("\n" + "="*40)
    print(">>> [系统] 正在构建合成验证集 (Synthetic Validation Set)...")
    
    val_hr_dir = config.VAL_HR_DIR
    # 在保存目录下创建一个专门存放生成的LR验证集文件夹
    generated_lr_dir = os.path.join(config.SAVE_DIR, "val_generated_lrs_stage2")
    
    # 确保文件夹存在
    os.makedirs(generated_lr_dir, exist_ok=True)
    
    # 获取所有 HR 图片文件，并排序
    exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    hr_files = sorted([f for f in os.listdir(val_hr_dir) if f.lower().endswith(exts)])
    
    # 确保模型处于评估模式
    joint_system.degradation_module.eval()
    
    # 定义简单的 Transform (只转 Tensor)
    to_tensor = transforms.Compose([transforms.ToTensor()])
    
    count = 0
    print(f"  - 源 HR 路径: {val_hr_dir}")
    print(f"  - 目标 LR 路径: {generated_lr_dir}")
    
    with torch.no_grad():
        for filename in tqdm(hr_files, desc="Generating Val LRs"):
            # 1. 读取 HR
            hr_path = os.path.join(val_hr_dir, filename)
            try:
                hr_img = Image.open(hr_path).convert('RGB')
            except Exception as e:
                print(f"跳过损坏文件 {filename}: {e}")
                continue
            
            # 2. 关键：调整尺寸为 upscale_factor 的倍数
            # (这是为了防止退化后 LR 与 HR 尺寸比例不是整数，导致 PSNR 计算报错)
            w, h = hr_img.size
            factor = getattr(config, 'UPSCALE_FACTOR', 4) 
            new_w = w - (w % factor)
            new_h = h - (h % factor)
            
            # 只有当尺寸不匹配时才裁剪
            if new_w != w or new_h != h:
                hr_img = hr_img.crop((0, 0, new_w, new_h))
            
            hr_tensor = to_tensor(hr_img).unsqueeze(0).to(config.DEVICE)
            
            # 3. 模型推理 (生成 LR)
            joint_system._reset_all_neurons()
            generated_lab_lr = joint_system.degradation_module(hr_tensor)
            
            # 4. 转回 RGB、Clamp 并保存
            generated_rgb_lr = joint_system.lab2rgb(generated_lab_lr)
            generated_rgb_lr = torch.clamp(generated_rgb_lr, 0, 1)
            
            # 5. 保存 (文件名保持完全一致！)
            save_path = os.path.join(generated_lr_dir, filename)
            save_image(generated_rgb_lr, save_path)
            count += 1
            
    print(f">>> 合成验证集构建完成！共生成 {count} 张 LR 图片。")
    print("="*40 + "\n")
    
    return generated_lr_dir




def get_dataset_image_size(directory):
    """
    尝试从指定目录读取第一张图片的尺寸 (H, W)
    """
    if not os.path.exists(directory):
        print(f"警告: 目录不存在 {directory}")
        return None, None
        
    exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    
    for filename in os.listdir(directory):
        if filename.lower().endswith(exts):
            file_path = os.path.join(directory, filename)
            try:
                with Image.open(file_path) as img:
                    w, h = img.size # PIL 返回的是 (W, H)
                    print(f"从数据集检测到图像尺寸: {h}x{w} (文件名: {filename})")
                    return h, w # 返回 (H, W) 给 PyTorch 使用
            except Exception as e:
                continue
    
    print(f"警告: 在 {directory} 中未找到有效图像")
    return None, None

# 设置随机种子
set_seed(42)

# 创建保存目录
os.makedirs(config.SAVE_DIR, exist_ok=True)


# 检查NIQE先验文件
print("=" * 50)
print("检查NIQE先验文件...")
# 请确保这里的路径是你实际的路径，如果找不到文件会自动创建
NIQE_PRIOR_PATH = create_niqe_prior_if_needed(config.LR_DIR, "/root/autodl-tmp/SNN/整合完整/整合分类/zhenghe2_3/niqe_water_params.mat")
print("=" * 50)

def main():
    # 创建数据加载器
    # print("创建数据加载器...")
    # data_loaders = create_data_loaders(config)
    # train_loader = data_loaders['train_loader']
    # val_loader = data_loaders['val_loader']
    
    # ==================== 数据加载器配置 (修改版) ====================
    print("创建数据加载器...")
    
    # 1. 训练集加载器 (保持不变)
    # train_loader = create_data_loaders(config) 
    train_loader = create_data_loaders(config)['train_loader']

    # 2. 验证集加载器 (关键修改点！)
    # 检查 config 中是否配置了 LR 路径
    if hasattr(config, 'VAL_LR_DIR') and config.VAL_LR_DIR is not None and os.path.exists(config.VAL_LR_DIR):
        print(f"验证模式: 配对数据 (Paired) \n  - HR: {config.VAL_HR_DIR}\n  - LR: {config.VAL_LR_DIR}")
        from data_loader import PairedReferenceHRDataset # 确保导入了这个类
        
        val_dataset = PairedReferenceHRDataset(
            hr_dir=config.VAL_HR_DIR,
            lr_dir=config.VAL_LR_DIR, # 传入 LR 路径
            upscale_factor=getattr(config, 'UPSCALE_FACTOR', 4)
        )
    else:
        print(f"验证模式: 非配对数据 (Unpaired) - 将使用 Bicubic 模拟退化")
        from data_loader import UnpairedUnalignedDataset
        
        val_dataset = UnpairedUnalignedDataset(
            hr_dir=config.VAL_HR_DIR,
            lr_dir=None, 
            upscale_factor=getattr(config, 'UPSCALE_FACTOR', 4)
        )

    # 创建 DataLoader 实例
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.VAL_BATCH_SIZE, # 注意这里通常是 1
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False  # 验证集不要丢弃最后的数据
    )

    # 初始化模型
    print("初始化模型...")
    joint_system = JointTrainingSystem(
        degradation_config=config.DEGRADATION_CONFIG,
        reconstruction_config=config.RECONSTRUCTION_CONFIG,
        device=config.DEVICE
    )
    
#     print("\n" + "="*20 + " [深度诊断] 神经元阈值检查 " + "="*20)
    
#     def check_thresholds(module, module_name):
#         print(f"\n--- 检查模块: {module_name} ---")
#         found_any = False
#         # 遍历所有子层
#         for name, layer in module.named_modules():
#             # 检查是否有 v_th 参数 (SpikingJelly 的神经元通常都有这个 buffer)
#             if hasattr(layer, 'v_th'):
#                 # 获取数值
#                 th_val = layer.v_th
#                 if isinstance(th_val, torch.Tensor):
#                     th_val = th_val.item()
                
#                 print(f"层: {name} | 阈值 v_th = {th_val:.4f}")
#                 found_any = True
                
#                 # 为了不刷屏，每个大模块只打印前 3 个和后 3 个神经元
#                 # (你可以根据需要去掉这个限制，看全部)
#                 # break 
        
#         if not found_any:
#             print(f"⚠️ 警告: 在 {module_name} 中未找到任何包含 'v_th' 的层！")

#     # 1. 检查重建网络 (这是你最关心的)
#     check_thresholds(joint_system.reconstruction_module, "Reconstruction Module")

#     # 2. 检查判别器 (这是可能有硬编码 0.3 的)
#     check_thresholds(joint_system.discriminator_module, "Discriminator Module")

#     # 3. 检查退化网络 (这是可能有默认值 0.5 的)
#     check_thresholds(joint_system.degradation_module, "Degradation Module")

#     print("="*60 + "\n")
#     # ================================================================
    
    
    print("验证CT-Loss修复...")
    joint_system.validate_l_channel_inputs()

    # 初始化参数监控器
    param_monitor = ParameterMonitor()

    # 初始化评估器
    reconstruction_evaluator = None
    degradation_evaluator = None
    distribution_comparator = None

    try:
        from evaluate import ImageQualityEvaluator
        reconstruction_evaluator = ImageQualityEvaluator(device=config.DEVICE)
        print("重建模块图像质量评估器已初始化")
    except ImportError:
        print("警告: 无法导入 evaluate 模块，重建模块将跳过质量评估")

    try:
        from 获取先验计算 import ImageQualityEvaluator as DegradationImageQualityEvaluator
        degradation_evaluator = DegradationImageQualityEvaluator(device=config.DEVICE, prior_path=NIQE_PRIOR_PATH)
        print("退化模块图像质量评估器已初始化")
    except ImportError:
        print("警告: 无法导入 获取先验计算 模块，退化模块将跳过质量评估")

    try:
        from 计算生成模块1 import LRImageDistributionComparator
        distribution_comparator = LRImageDistributionComparator()
        print("分布比较器已初始化")
    except ImportError:
        print("警告: 无法导入 计算生成模块1，将跳过分布距离计算")
        
        
    # ================= [EnergyMeter 初始化] =================
    # 获取原始模型（处理多GPU情况）
    model_to_profile = joint_system.reconstruction_module
    if isinstance(model_to_profile, nn.DataParallel) or isinstance(model_to_profile, nn.parallel.DistributedDataParallel):
        model_to_profile = model_to_profile.module

    # 1. 尝试从数据集目录动态读取 HR 尺寸
    detected_h, detected_w = get_dataset_image_size(config.HR_DIR)
    
    if detected_h is not None and detected_w is not None:
        hr_h, hr_w = detected_h, detected_w
    else:
        print("无法从数据集读取尺寸，使用默认值 256")
        hr_h = getattr(config, 'HR_IMG_SIZE', 256)
        hr_w = getattr(config, 'HR_IMG_SIZE', 256)

    # 2. 获取上采样因子
    upscale_factor = getattr(config, 'UPSCALE_FACTOR', 2)
    
    # 3. 计算 LR 尺寸
    lr_h = hr_h // upscale_factor
    lr_w = hr_w // upscale_factor
    
    # 4. 获取时间步
    t_steps = config.RECONSTRUCTION_CONFIG.get('time_steps', 5)
    print(f"初始化能耗计算器:")
    print(f"  - 目标 HR 尺寸: {hr_h}x{hr_w} (来自数据集)")
    print(f"  - 上采样倍数: {upscale_factor}")
    print(f"  - 计算基准输入尺寸 (LR): {lr_h}x{lr_w}")
    
    energy_meter = EnergyMeter(
        model=model_to_profile, 
        input_size=(1, 3, lr_h, lr_w), 
        device=config.DEVICE, 
        time_steps=t_steps
    )

    # 创建CSV文件头
    metrics_csv_path = os.path.join(config.SAVE_DIR, "training_metrics.csv")
    
    csv_headers = [
        'epoch', 
        # --- Train Losses ---
        'train_recon_l1', 'train_recon_percep', 'train_recon_grad', 
        'train_recon_lpips', 'train_recon_total',
        'train_degrade_loss', # 还可以保留退化总Loss稍微看一眼收敛情况
        
        # --- Train Metrics ---
        'train_recon_niqe', 'train_recon_brisque',
        
        # --- Val Metrics ---
        'val_psnr', 'val_ssim', 'val_lpips',
        'val_niqe', 'val_brisque', 'val_uiqm',
        
        # --- Energy ---
        'val_gsops', 'val_energy_snn_J', 'val_energy_cnn_J', 'val_delta_E_percent',
        
        # --- LR ---
        'lr_recon', 'lr_degrade', 'lr_disc'
    ]
    

    # 初始化追踪变量
    best_psnr = 0.0
    best_epoch = 0
    best_degradation_mmd = float('inf')
    best_degradation_niqe = float('inf')
    start_epoch = 0
    
    # [关键变量] 用于全自动模式下在内存中传递最佳模型路径
    best_rlgm_path_memory = "" 

    # 恢复训练逻辑
    if config.RESUME_TRAINING and os.path.exists(config.RESUME_CHECKPOINT_PATH):
        print(f"正在从checkpoint恢复训练: {config.RESUME_CHECKPOINT_PATH}")
        checkpoint_data = joint_system.load_checkpoint(config.RESUME_CHECKPOINT_PATH, config.DEVICE)
        if checkpoint_data:
            start_epoch = checkpoint_data['epoch']-1
            best_psnr = checkpoint_data['best_psnr']
            best_epoch = checkpoint_data['best_epoch']
            best_degradation_mmd = checkpoint_data.get('best_degradation_mmd', float('inf'))
            best_degradation_niqe = checkpoint_data.get('best_degradation_niqe', float('inf'))

            param_history_path = os.path.join(config.SAVE_DIR, "learnable_parameters_history.csv")
            param_monitor.load_from_csv(param_history_path)

            print(f"恢复训练: 从epoch {start_epoch + 1} 开始")
    else:
        print("从头开始训练")

    # =========================================================================
    # [阶段一]：退化模块预训练 (Degradation Pretraining)
    # =========================================================================
    if config.PRETRAIN_DEGRADATION_ONLY and start_epoch == 0:
        print("\n" + "="*50)
        print(f" [阶段一] 开始退化模块预训练 ({config.PRETRAIN_EPOCHS} Epochs)")
        print("="*50)
        
        
        # [修改 1.1] 定义阶段一专属 CSV 路径和表头
        pretrain_csv_path = os.path.join(config.SAVE_DIR, "pretrain_metrics.csv")
        pretrain_headers = [
            'epoch', 
            'avg_d_loss', 'avg_g_loss', 'avg_fm_loss', 'avg_ct_loss', # 关注 Loss
            'train_deg_niqe', 'train_deg_brisque', 'train_deg_uiqm',  # 关注生成质量
            'dist_bhattacharyya', 'dist_chi_square', 'dist_cosine', 'dist_niqe' # 关注分布距离
        ]
        
        
        
        # 初始化阶段一的最佳指标
        best_fm_stage1 = float('inf')
        
        for pre_epoch in range(config.PRETRAIN_EPOCHS):
            joint_system.degradation_module.train()
            joint_system.discriminator_module.train()
            
            running_d_loss = 0.0
            running_deg_loss = 0.0
            running_fm_loss = 0.0 
            running_ct_loss = 0.0
            
            # 用于计算分布指标的临时变量
            last_real_lr = None
            last_fake_lr = None
            
            pretrain_pbar = tqdm(train_loader, desc=f"Pretrain {pre_epoch + 1}/{config.PRETRAIN_EPOCHS}")

            for i, batch in enumerate(pretrain_pbar):
                hr = batch["hr"].to(config.DEVICE)
                lr_gt = batch["lr"].to(config.DEVICE)

                stats = joint_system.train_degradation_step(hr, lr_gt)

                running_d_loss += stats['d_loss']
                running_deg_loss += stats['degradation_loss']
                current_fm = stats.get('fm_loss', 0.0)
                running_ct_loss += stats.get('loss_ct', 0.0) # [修改 1.3] 累加 CT Loss
                running_fm_loss += current_fm
                
                # ================= 阶段一图像保存逻辑 =================
                if i % 100 == 0:
                    # 创建保存目录
                    iter_dir = os.path.join(config.SAVE_DIR, "train_degradations", f"epoch_{pre_epoch + 1}_iter_{i}")
                    os.makedirs(iter_dir, exist_ok=True)
                    
                    with torch.no_grad():
                        # 生成用于可视化的 RGB 图像
                        generated_lab = joint_system.degradation_module(hr)
                        fake_rgb = joint_system.lab2rgb(generated_lab)
                        #新增clamp
                        fake_rgb = torch.clamp(fake_rgb, 0, 1)
                    
                    # 保存前 2 张样本
                    for j in range(min(2, hr.shape[0])):
                        save_image(hr[j].cpu(), os.path.join(iter_dir, f"sample_{j}_hr.png"))
                        save_image(lr_gt[j].cpu(), os.path.join(iter_dir, f"sample_{j}_real_lr.png"))
                        save_image(fake_rgb[j].cpu(), os.path.join(iter_dir, f"sample_{j}_fake_lr.png"))
                # ==========================================================

                # 保存最后一个Batch的数据用于epoch结束时的指标计算
                if i == len(train_loader) - 1:
                    with torch.no_grad():
                        last_real_lr = lr_gt.detach()
                        generated_lab = joint_system.degradation_module(hr)
                        last_fake_lr = joint_system.lab2rgb(generated_lab).detach()

                pretrain_pbar.set_postfix({
                    'D': f"{stats.get('d_loss', 0.0):.4f}",
                    'G': f"{stats.get('degradation_loss', 0.0):.4f}",
                    'FM': f"{current_fm:.4f}",
                    'CT': f"{stats.get('loss_ct', 0.0):.4f}"
                })
                
            # --- Epoch 结束：计算平均指标 ---
            num_batches = len(train_loader)
            avg_d = running_d_loss / num_batches
            avg_g = running_deg_loss / num_batches
            avg_fm = running_fm_loss / num_batches
            avg_ct = running_ct_loss / num_batches
            avg_fm = running_fm_loss / len(train_loader)
            print(f"[Pretrain {pre_epoch + 1}] Avg FM Loss: {avg_fm:.6f}")
            
            # === 计算并打印退化模块质量指标与分布指标 ===
            train_degradation_niqe = 0.0
            train_degradation_brisque = 0.0
            train_degradation_uiqm = 0.0
            
            bhattacharyya_distance = 0.0
            chi_square_distance = 0.0
            cosine_similarity = 0.0
            degradation_niqe = 0.0

            if last_fake_lr is not None and last_real_lr is not None:
                # 1. 计算质量指标 (NIQE, BRISQUE, UIQM)
                if degradation_evaluator:
                    metrics = batch_calculate_metrics_degradation(last_fake_lr, degradation_evaluator, NIQE_PRIOR_PATH)
                    train_degradation_niqe = metrics.get('niqe', 0.0)
                    train_degradation_brisque = metrics.get('brisque', 0.0)
                    train_degradation_uiqm = metrics.get('uiqm', 0.0)
                    
                    print(f"训练集退化模块质量指标 - NIQE: {train_degradation_niqe:.4f}, "
                          f"BRISQUE: {train_degradation_brisque:.4f}, UIQM: {train_degradation_uiqm:.4f}")

                # 2. 计算分布指标
                if distribution_comparator:
                    dist_metrics = calculate_distribution_metrics(last_real_lr, last_fake_lr, distribution_comparator, NIQE_PRIOR_PATH)
                    bhattacharyya_distance = dist_metrics.get('bhattacharyya', 0.0)
                    chi_square_distance = dist_metrics.get('chi_square', 0.0)
                    cosine_similarity = dist_metrics.get('cosine', 0.0)
                    degradation_niqe = dist_metrics.get('niqe', 0.0)
                    
                    print("分布指标:")
                    print(f"  - Bhattacharyya距离: {bhattacharyya_distance:.4f}")
                    print(f"  - Chi-square距离: {chi_square_distance:.4f}")
                    print(f"  - Cosine相似度: {cosine_similarity:.4f}")
                    print(f"  - 退化NIQE: {degradation_niqe:.4f}")
                
                pretrain_row = [
                pre_epoch + 1, avg_d, avg_g, avg_fm, avg_ct,
                train_degradation_niqe, train_degradation_brisque, train_degradation_uiqm,
                bhattacharyya_distance, chi_square_distance, cosine_similarity, degradation_niqe
            ]
            save_metrics_to_csv(pretrain_csv_path, pretrain_row, pretrain_headers)
                
                    
                    

            # 自动保存最佳模型
            if train_degradation_niqe > 0 and train_degradation_niqe < best_degradation_niqe:
                print(f"发现更优退化模型！NIQE 从 {best_degradation_niqe:.4f} 降至 {train_degradation_niqe:.4f}")
                best_degradation_niqe = train_degradation_niqe
                
                save_path = os.path.join(config.SAVE_DIR, "best_degradation_module.pth")
                
                if config.NUM_GPUS > 1:
                    torch.save(joint_system.degradation_module.module.state_dict(), save_path)
                else:
                    torch.save(joint_system.degradation_module.state_dict(), save_path)
                    
                best_rlgm_path_memory = save_path
                print(f"已保存最佳预训练权重: {save_path}")

        print("\n 阶段一预训练完成。")
        print(f"准备无缝切换进入阶段二 (重建训练)...")
        
        # ================= [自动切换逻辑] =================
        print(">>> 正在自动配置阶段二环境...")
        config.USE_PRETRAINED_RLGM = True
        if best_rlgm_path_memory:
            config.PRETRAINED_RLGM_PATH = best_rlgm_path_memory
            
    # =========================================================================
    # [中间操作]：加载最佳权重 + 冻结参数 (Freeze Logic)
    # 
    
    # =========================================================================
    # [中间操作]：加载最佳权重 + 生成数据 + 刷新 DataLoader
    # =========================================================================
    if hasattr(config, 'USE_PRETRAINED_RLGM') and config.USE_PRETRAINED_RLGM:
        target_path = getattr(config, 'PRETRAINED_RLGM_PATH', "")
        # 如果内存中有刚才阶段一训练好的最佳路径，优先使用
        if best_rlgm_path_memory: 
            target_path = best_rlgm_path_memory

        if os.path.exists(target_path):
            print(f"\n[系统切换] 正在加载最佳退化模型权重: {target_path}")
            
            # 1. 加载权重
            checkpoint = torch.load(target_path, map_location=config.DEVICE)
            if isinstance(joint_system.degradation_module, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
                joint_system.degradation_module.module.load_state_dict(checkpoint)
            else:
                joint_system.degradation_module.load_state_dict(checkpoint)
            
            # 2. 冻结退化网络 & 判别器
            joint_system.degradation_module.eval()
            for param in joint_system.degradation_module.parameters():
                param.requires_grad = False
            
            joint_system.discriminator_module.eval()
            for param in joint_system.discriminator_module.parameters():
                param.requires_grad = False
                
            print(">>> 退化模块已加载最佳权重并冻结 (Frozen)。")

            # ================= [关键新增：生成配对验证集] =================
            # 调用我们在上面定义的函数
            synthetic_lr_dir = prepare_synthetic_validation_data(joint_system, config)
            
            print(f">>> 正在刷新验证集 DataLoader...")
            
            # 3. 重新实例化 DataLoader
            # 必须使用 PairedReferenceHRDataset 确保文件名对齐
            from data_loader import PairedReferenceHRDataset 
            
            # 注意：这里的 hr_dir 必须是你原始的验证集 HR 路径
            new_val_dataset = PairedReferenceHRDataset(
                hr_dir=config.VAL_HR_DIR,
                lr_dir=synthetic_lr_dir,  # 指向新生成的 LR 文件夹
                upscale_factor=getattr(config, 'UPSCALE_FACTOR', 4)
            )
            
            # 覆盖旧的 val_loader
            val_loader = DataLoader(
                new_val_dataset,
                batch_size=config.VAL_BATCH_SIZE, 
                shuffle=False,
                num_workers=config.NUM_WORKERS,
                pin_memory=True,
                drop_last=False
            )
            print(">>> 验证环境刷新完成：现在使用 [真实HR - 模型合成LR] 进行评估。\n")
            # ============================================================

        else:
            print(f"警告：设置了 USE_PRETRAINED_RLGM=True，但找不到权重文件: {target_path}")
            print("将继续使用当前的 val_loader (可能是 Bicubic 数据)。")
            
    
#     if hasattr(config, 'USE_PRETRAINED_RLGM') and config.USE_PRETRAINED_RLGM:
#         target_path = getattr(config, 'PRETRAINED_RLGM_PATH', "")
#         if best_rlgm_path_memory: 
#             target_path = best_rlgm_path_memory

#         if os.path.exists(target_path):
#             print(f"\n[系统切换] 正在加载/回读 最佳退化模型权重: {target_path}")
            
#             checkpoint = torch.load(target_path, map_location=config.DEVICE)
#             if isinstance(joint_system.degradation_module, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
#                 joint_system.degradation_module.module.load_state_dict(checkpoint)
#             else:
#                 joint_system.degradation_module.load_state_dict(checkpoint)
            
#             joint_system.degradation_module.eval()
#             # joint_system.degradation_module.train()
#             for param in joint_system.degradation_module.parameters():
#                 param.requires_grad = False
            
#             joint_system.discriminator_module.eval()
#             for param in joint_system.discriminator_module.parameters():
#                 param.requires_grad = False
                
#             print("退化模块已加载最佳权重并冻结 (Frozen)。")
#         else:
#             print(f"警告：设置了 USE_PRETRAINED_RLGM=True，但找不到权重文件: {target_path}")

    # 创建保存目录
    os.makedirs(os.path.join(config.SAVE_DIR, "train_reconstructions"), exist_ok=True)
    os.makedirs(os.path.join(config.SAVE_DIR, "val_reconstructions"), exist_ok=True)
    os.makedirs(os.path.join(config.SAVE_DIR, "final_results"), exist_ok=True)

    # =========================================================================
    # [阶段二]：正式联合训练 / 重建模块训练 (Reconstruction Training)
    # =========================================================================
    print("\n" + "="*50)
    print(f"[阶段二] 开始重建模块正式训练 ({config.EPOCHS} Epochs)")
    print("="*50)

    for epoch in range(start_epoch, config.EPOCHS):
        latest_checkpoint_path = os.path.join(config.SAVE_DIR, "latest_checkpoint.pth")
        joint_system.save_checkpoint(
            latest_checkpoint_path, epoch + 1, best_psnr, best_epoch, 0.0, 0.0
        )

        # ==================== 训练阶段 ====================
        joint_system.reconstruction_module.train()

        epoch_stats = {
            'd_loss': 0.0, 'g_loss': 0.0, 'recon_loss': 0.0, 
            'lpips_loss': 0.0, 'degradation_loss': 0.0,
            'gradient_loss': 0.0,           
            'perceptual_loss': 0.0,
            'reconstruction_total_loss': 0.0
        }

        train_reconstructed_images = []
        train_generated_lr_images = []
        
        # 记录最后一个batch的重建图，用于计算训练集指标
        last_train_recon_img = None
        
        train_pbar = tqdm(train_loader, desc=f"Train Epoch {epoch + 1}/{config.EPOCHS}")

        for i, batch in enumerate(train_pbar):
            hr = batch["hr"].to(config.DEVICE)
            lr_gt = batch["lr"].to(config.DEVICE)

            losses = joint_system.train_step(hr, lr_gt)
            
            
            # # ================= [插入在这里] =================
            # # 检查 Tau 的梯度 (仅在第一个 Batch 打印，避免刷屏)
            # if i == 0: 
            #     print("\n--- [Debug] Checking Tau Gradients ---")
            #     found_tau = False
            #     for name, param in joint_system.reconstruction_module.named_parameters():
            #         if 'tau' in name:
            #             found_tau = True
            #             if param.grad is not None:
            #                 grad_mean = param.grad.abs().mean().item()
            #                 print(f"Layer: {name} | Tau Value: {param.data.item():.4f} | Grad: {grad_mean:.8f}")
            #             else:
            #                 print(f"Layer: {name} | Tau Value: {param.data.item():.4f} | Grad: None (No Gradient!)")
            #             # break # 建议先不要 break，看一眼所有层的 tau 是否都有梯度
            #     if not found_tau:
            #         print("Warning: No parameter named 'tau' found in reconstruction_module!")
            #     print("----------------------------------------\n")
            # ===============================================
            

            hr_reconstructed = losses['hr_reconstructed']
            
            # 从 losses 中提取增强后的数据
            aug_lab_lr = losses.get('augmented_lab_lr', losses['generated_lab_lr'])
            aug_hr_rgb = losses.get('augmented_hr_rgb', hr) 
            
            # 保存最后一个batch用于指标计算
            if i == len(train_loader) - 1:
                last_train_recon_img = hr_reconstructed.detach().clamp(0, 1)

            if i < config.MAX_VAL_BATCHES:
                hr_reconstructed_processed = hr_reconstructed.detach().cpu().clamp(0, 1)
                train_reconstructed_images.extend([img for img in hr_reconstructed_processed])
                
                generated_lr_rgb = joint_system.lab2rgb(aug_lab_lr)
                generated_lr_rgb_processed = generated_lr_rgb.detach().cpu().clamp(0, 1)
                train_generated_lr_images.extend([img for img in generated_lr_rgb_processed])

            epoch_stats['d_loss'] += losses.get('d_loss', 0.0)
            epoch_stats['g_loss'] += losses.get('g_loss', 0.0)
            epoch_stats['recon_loss'] += losses.get('reconstruction_loss', 0.0)
            epoch_stats['lpips_loss'] += losses.get('lpips_loss', 0.0)
            epoch_stats['degradation_loss'] += losses.get('degradation_loss', 0.0)
            epoch_stats['gradient_loss'] += losses.get('gradient_loss', 0.0) 
            epoch_stats['perceptual_loss'] += losses.get('perceptual_loss_reconstruct', 0.0) 
            epoch_stats['reconstruction_total_loss'] += losses.get('reconstruction_total_loss', 0.0)

            current_accum = (i % config.ACCUMULATION_STEPS) + 1
            train_pbar.set_postfix({
                'Recon': f"{losses.get('reconstruction_loss', 0.0):.4f}",
                'Deg': f"{losses.get('degradation_loss', 0.0):.4f}",
                'Accum': f"{current_accum}/{config.ACCUMULATION_STEPS}"
            })

            if i % 100 == 0:
                iter_dir = os.path.join(config.SAVE_DIR, "train_reconstructions", f"epoch_{epoch + 1}_iter_{i}")
                os.makedirs(iter_dir, exist_ok=True)
                
                # 检查增强键是否存在
                use_aug_keys = 'augmented_hr_rgb' in losses

                if use_aug_keys:
                    real_target_rgb = losses['augmented_hr_rgb']
                    real_input_lab = losses['augmented_lab_lr']
                else:
                    real_target_rgb = hr
                    real_input_lab = losses.get('generated_lab_lr')

                augmented_lr_rgb = joint_system.lab2rgb(real_input_lab)
                augmented_lr_rgb = torch.clamp(augmented_lr_rgb, 0, 1)  # 加上这句
                recon_img = losses['hr_reconstructed']
                recon_img = torch.clamp(recon_img, 0, 1)                # 加上这句
                
                real_target_rgb = torch.clamp(real_target_rgb, 0, 1)    # 建议顺手把 Target 也加上
                
                for j in range(min(2, hr.shape[0])):
                    save_image(real_target_rgb[j].cpu(), os.path.join(iter_dir, f"sample_{j}_hr.png"))
                    save_image(recon_img[j].cpu(), os.path.join(iter_dir, f"sample_{j}_recon.png"))
                    save_image(augmented_lr_rgb[j].cpu(), os.path.join(iter_dir, f"sample_{j}_input.png"))
        
        # 训练 Epoch 结束，清理残留梯度 (处理 accumulate steps 边界情况)
        joint_system.optimizer_g_degrade.zero_grad()
        joint_system.optimizer_g_reconstruct.zero_grad()
        joint_system.optimizer_d.zero_grad()

        num_batches = len(train_loader)
        avg_stats = {k: v / num_batches for k, v in epoch_stats.items()}

        joint_system.step_schedulers(epoch)
        current_lr = joint_system.get_current_lr()

        print(f"\nEpoch {epoch + 1} Stats: Recon Loss={avg_stats['recon_loss']:.4f}, LPIPS={avg_stats['lpips_loss']:.4f}")
        print(f"LR: Recon={current_lr['reconstruction_lr']:.2e}, Degrade={current_lr['degradation_lr']:.2e}")
        
        # === 计算并打印重建模块训练集质量指标 ===
        train_reconstruction_niqe = 0.0
        train_reconstruction_brisque = 0.0
        train_reconstruction_uiqm = 0.0
        train_reconstruction_uism = 0.0
        
        if reconstruction_evaluator and last_train_recon_img is not None:
            metrics = batch_calculate_metrics_reconstruction(last_train_recon_img, reconstruction_evaluator)
            train_reconstruction_niqe = metrics.get('niqe', 0.0)
            train_reconstruction_brisque = metrics.get('brisque', 0.0)
            train_reconstruction_uiqm = metrics.get('uiqm', 0.0)
            train_reconstruction_uism = metrics.get('uism', 0.0)
            
            print(f"训练集重建模块质量指标 - NIQE: {train_reconstruction_niqe:.4f}, "
                  f"BRISQUE: {train_reconstruction_brisque:.4f}, UIQM: {train_reconstruction_uiqm:.4f}")

        # ==================== 验证阶段 (合并优化版) ====================
        joint_system.reconstruction_module.eval()

        # 1. 初始化统计变量
        # val_psnr_cumulative = 0.0
        # val_ssim_cumulative = 0.0
        # val_lpips_cumulative = 0.0
        
        val_psnr_total_score = 0.0
        val_ssim_total_score = 0.0
        val_lpips_total_score = 0.0
        
        # [新增] 初始化 Bicubic 基准统计变量
        val_bicubic_psnr_total = 0.0
        val_bicubic_ssim_total = 0.0
        
        total_val_images = 0  # 记录总图片数
        
        # 感知指标统计
        val_niqe_cumulative = 0.0
        val_brisque_cumulative = 0.0
        val_uiqm_cumulative = 0.0
        val_uism_cumulative = 0.0
        
        val_count = 0        # 记录处理了多少个 Batch
        val_metric_count = 0 # 记录成功计算感知指标的次数

        # 2. 准备能耗计算与保存目录
        energy_meter.reset()
        energy_meter.register_hooks()
        
        val_save_dir = os.path.join(config.SAVE_DIR, "val_reconstructions", f"epoch_{epoch+1}")
        os.makedirs(val_save_dir, exist_ok=True)

        print(f"开始验证 (前 {config.MAX_VAL_BATCHES} Batches)...")
 
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                if batch_idx >= config.MAX_VAL_BATCHES: break
                
                hr = batch["hr"].to(config.DEVICE)
                
                
                # B. 获取 LR (考题) - 直接从文件夹读！
                if "lr" in batch:
                    val_lr_rgb = batch["lr"].to(config.DEVICE)
                else:
                    # 保底措施：万一路径没对，自动退化防止报错
                    h, w = hr.shape[2], hr.shape[3]
                    factor = getattr(config, 'UPSCALE_FACTOR', 4) # 动态获取倍率
                    val_lr_rgb = torch.nn.functional.interpolate(
                        hr, size=(h//factor, w//factor), mode='bicubic', align_corners=False)
                    # val_lr_rgb = torch.nn.functional.interpolate(
                    #     hr, size=(h//4, w//4), mode='bicubic', align_corners=False)

                # C. 格式转换 (RGB -> LAB)
                # 你的 SNN 重建网络必须吃 LAB 格式
                val_lr_lab = joint_system.rgb2lab(val_lr_rgb)
                
                joint_system._reset_all_neurons()
                
                
                # --- A. 模型推理 ---
#                 # 1. 退化：生成对齐的 LR
#                 # generated_lab_lr = joint_system.degradation_module(hr)
                
#                 # 2. 重建：恢复 HR
#                 hr_reconstructed = joint_system.reconstruction_module(val_lr_lab)
                
#                 # --- B. 数据准备 ---
#                 # 钳位到 [0, 1] 用于 PSNR/SSIM/NIQE
#                 recon_clamped = torch.clamp(hr_reconstructed, 0, 1)
#                 hr_clamped = torch.clamp(hr, 0, 1)

                # 2. 重建：恢复 HR (输入是 RGB)
                hr_reconstructed = joint_system.reconstruction_module(val_lr_lab)
                
                # 3. 后处理：输出必然是 RGB，只需 Clamp
                # (注意：losses1.py 的 calculate_psnr 需要 RGB 输入)
                recon_clamped = torch.clamp(hr_reconstructed, 0, 1)
                hr_clamped = torch.clamp(hr, 0, 1)
                # ================= [修改结束] =================

                
                # 归一化到 [-1, 1] 用于 LPIPS
                recon_norm = recon_clamped * 2 - 1
                hr_norm = hr_clamped * 2 - 1
                
                # --- [修正 1] 数据范围转换 (0-1 -> 0-255) ---
                recon_255 = recon_clamped * 255.0
                hr_255 = hr_clamped * 255.0
                
                # ================= [新增] 生成 Bicubic 基准图 =================
                # 1. 使用双三次插值强行放大 LR 到 HR 尺寸
                bicubic_img = torch.nn.functional.interpolate(
                    val_lr_rgb, 
                    size=(hr.shape[2], hr.shape[3]), # 目标尺寸 (H, W)
                    mode='bicubic', 
                    align_corners=False
                )
                # 2. 同样的 Clamp + Scale 处理 (确保对比公平)
                bicubic_255 = torch.clamp(bicubic_img, 0, 1) * 255.0
                # ============================================================
                
                
                # --- C. 计算指标 (修改版：启用学术标准 Y通道 + 切边) ---
                # 1. 常规指标 (PSNR, SSIM)
                # 必须遍历 Batch 逐张计算，因为标准的 calculate_psnr 是针对单张图设计的
                current_batch_psnr = 0.0
                current_batch_ssim = 0.0
                
                # [新增] 当前 Batch 的 Bicubic 分数
                current_bicubic_psnr_batch = 0.0
                current_bicubic_ssim_batch = 0.0
                
                batch_val_size = recon_clamped.shape[0]
                total_val_images += batch_val_size # 累加图片数

                # 获取之前定义的放大倍数 (如果在 main 函数开头定义了 upscale_factor，直接用即可)
                # 如果找不到变量，可以写死：crop_size = 4 (对应x4任务)
                crop_size = upscale_factor 

                for b_idx in range(batch_val_size):
                    current_batch_psnr += calculate_psnr(
                        recon_clamped[b_idx], 
                        hr_clamped[b_idx], 
                        crop_border=crop_size,  # 关键：切除边缘，去掉 padding 伪影
                        test_y_channel=True,     # 关键：开启 Y 通道，过滤水下色度噪声
                        data_range=255.0
                    )
                    current_batch_ssim += calculate_ssim(
                        recon_clamped[b_idx], 
                        hr_clamped[b_idx], 
                        crop_border=crop_size, 
                        test_y_channel=True,
                        data_range=255.0
                    )
                    
                    
                    # B. [新增] 计算 Bicubic (笨办法) 的得分
                    current_bicubic_psnr_batch += calculate_psnr(
                        bicubic_255[b_idx], 
                        hr_255[b_idx], 
                        crop_border=crop_size, 
                        test_y_channel=True,
                        data_range=255.0
                    )
                    current_bicubic_ssim_batch += calculate_ssim(
                        bicubic_255[b_idx], 
                        hr_255[b_idx], 
                        crop_border=crop_size, 
                        test_y_channel=True,
                        data_range=255.0
                    )
                    
                
                # 计算当前 Batch 平均分
                batch_psnr = current_batch_psnr / batch_val_size
                batch_ssim = current_batch_ssim / batch_val_size
                
                # 累加到全局统计
                # val_psnr_cumulative += batch_psnr
                # val_ssim_cumulative += batch_ssim
                
                val_psnr_total_score += current_batch_psnr
                val_ssim_total_score += current_batch_ssim
                # [新增] 累加 Bicubic 得分
                val_bicubic_psnr_total += current_bicubic_psnr_batch
                val_bicubic_ssim_total += current_bicubic_ssim_batch
                

                # 2. 感知距离 (LPIPS)
                # LPIPS 返回的是 batch 平均值，乘以 batch_size 还原为总分2
                batch_lpips_avg = joint_system.lpips_loss(recon_norm, hr_norm).mean().item()
                val_lpips_total_score += batch_lpips_avg * batch_val_size
                # batch_lpips = joint_system.lpips_loss(recon_norm, hr_norm).mean().item()
                # val_lpips_cumulative += batch_lpips

                # 3. 无参考质量指标 (NIQE, BRISQUE 等)
                if reconstruction_evaluator:
                    # 注意：evaluate模块内部通常需要[0,1]输入
                    metrics = batch_calculate_metrics_reconstruction(recon_clamped, reconstruction_evaluator)
                    if metrics:
                        val_niqe_cumulative += metrics.get('niqe', 0.0)
                        val_brisque_cumulative += metrics.get('brisque', 0.0)
                        val_uiqm_cumulative += metrics.get('uiqm', 0.0)
                        val_uism_cumulative += metrics.get('uism', 0.0)
                        val_metric_count += 1
                
                val_count += 1 

                # --- D. 保存图片 (仅前 2 个 Batch) ---

                if batch_idx < 2: 
                    current_batch_size = hr.shape[0]
                    
                    
                    # 准备 LR (为了可视化，转回 RGB 并钳位)
                    # val_lr_rgb = joint_system.lab2rgb(generated_lab_lr)
                    val_lr_clamped = torch.clamp(val_lr_rgb, 0, 1)
                    
                    # 简单插值 LR 以便拼接 (防止尺寸不匹配报错)
                    # if val_lr_clamped.shape[2:] != hr_clamped.shape[2:]:
                    #     val_lr_vis = torch.nn.functional.interpolate(
                    #         val_lr_clamped, size=hr_clamped.shape[2:], mode='bicubic', align_corners=False
                    #     )
                    # else:
                    #     val_lr_vis = val_lr_clamped
                    
                    
                    for img_idx in range(current_batch_size):
                        # 基础文件名
                        base_name = f"val_b{batch_idx}_img{img_idx}"
                        
                        # --- 计算误差图 (Difference Map) ---
                        # 计算绝对差值
                        diff = torch.abs(recon_clamped[img_idx] - hr_clamped[img_idx])
                        # 放大 10 倍亮度，让人眼能看清微小的错误 (否则是一片黑)
                        diff_amplified = torch.clamp(diff * 10, 0, 1)
                        
                        
                        # --- 分别保存三张图 (和 train_reconstructions 一样) ---
                        
                        # A. 保存原始 LR (小尺寸，原汁原味)
                        save_image(
                            val_lr_clamped[img_idx], 
                            os.path.join(val_save_dir, f"{base_name}_1_LR.png") 
                        )
                        
                        # B. 保存模型重建 SR (大尺寸)
                        save_image(
                            recon_clamped[img_idx], 
                            os.path.join(val_save_dir, f"{base_name}_2_SR.png")
                        )

                        # C. 保存高清 GT (大尺寸)
                        save_image(
                            hr_clamped[img_idx], 
                            os.path.join(val_save_dir, f"{base_name}_3_HR.png")
                        )
                        
                        # D. 保存误差图 Diff (大尺寸，越亮代表误差越大)
                        save_image(
                            diff_amplified, 
                            os.path.join(val_save_dir, f"{base_name}_4_Diff.png")
                        )
                        
                        # ================= [插入开始：调用可视化] =================
                        # 只有第一张图才做这个，太费时间了
                        if img_idx == 0:
                            vis_save_dir = os.path.join(config.SAVE_DIR, "feature_maps", f"epoch_{epoch+1}")

                            # 1. 看看浅层特征 (提取到了边缘吗？)
                            # 这里的名字 'spatial_extractor' 需要根据你的 models.py 里的实际命名调整
                            # 如果你不确定名字，可以先填 'conv1' 或者 'initial'
                            save_feature_map(
                                joint_system.reconstruction_module, 
                                val_lr_lab[0:1], # 只传第一张图的 LAB 数据
                                layer_name_fragment="initial",  # 尝试匹配第一层
                                save_path=os.path.join(vis_save_dir, f"b{batch_idx}_layer_initial.png")
                            )

                            # 2. 看看深层特征 (是抽象的形状吗？还是死寂的黑色？)
                            save_feature_map(
                                joint_system.reconstruction_module, 
                                val_lr_lab[0:1], 
                                layer_name_fragment="fusion", # 尝试匹配融合层
                                save_path=os.path.join(vis_save_dir, f"b{batch_idx}_layer_fusion.png")
                            )
                        # ================= [插入结束] =================
                        
                        
                        
#                         # 计算绝对误差，并放大 10 倍，让人眼能看见微小的修正
#                         diff = torch.abs(recon_clamped[img_idx] - hr_clamped[img_idx])
#                         diff_amplified = torch.clamp(diff * 10, 0, 1)
                        
                        
#                         # 拼接图片：左边是原图 HR，右边是重建图
#                         # comparison = torch.cat([hr[img_idx], hr_reconstructed[img_idx]], dim=2)
                        
#                         comparison = torch.cat([
#                             val_lr_vis[img_idx],       # Input
#                             hr_clamped[img_idx],       # GT
#                             recon_clamped[img_idx],    # Output
#                             diff_amplified             # Diff (由黑变亮表示误差大)
#                         ], dim=2)
                        
#                         save_path = os.path.join(val_save_dir, f"val_b{batch_idx}_img{img_idx}.png")
#                         save_image(comparison, save_path)
                    
                    if batch_idx == 0:
                        print(f"已保存验证对比图至: {val_save_dir}")

        # 3. 结束验证，计算平均值
        energy_metrics = energy_meter.calculate_metrics()
        energy_meter.remove_hooks()
        
        # 计算基础指标平均值
        if total_val_images > 0:
            val_psnr = val_psnr_total_score / total_val_images
            val_ssim = val_ssim_total_score / total_val_images
            avg_val_lpips = val_lpips_total_score / total_val_images
            # [新增] 计算 Bicubic 平均分
            avg_bicubic_psnr = val_bicubic_psnr_total / total_val_images
            avg_bicubic_ssim = val_bicubic_ssim_total / total_val_images
        else:
            val_psnr, val_ssim, avg_val_lpips = 0.0, 0.0, 0.0
            avg_bicubic_psnr, avg_bicubic_ssim = 0.0, 0.0
        # if val_count > 0:
        #     val_psnr = val_psnr_cumulative / val_count
        #     val_ssim = val_ssim_cumulative / val_count
        #     avg_val_lpips = val_lpips_cumulative / val_count
        # else:
        #     val_psnr, val_ssim, avg_val_lpips = 0.0, 0.0, 0.0
        
        # 计算感知指标平均值
        if val_metric_count > 0:
            val_niqe = val_niqe_cumulative / val_metric_count
            val_brisque = val_brisque_cumulative / val_metric_count
            val_uiqm = val_uiqm_cumulative / val_metric_count
            val_uism = val_uism_cumulative / val_metric_count
        else:
            val_niqe, val_brisque, val_uiqm, val_uism = 0.0, 0.0, 0.0, 0.0
        
        # --- E. 打印日志 ---
        print("-" * 30)
        print(f"基准测试 (Baseline Check):")
        print(f"  - Bicubic (纯插值): PSNR={avg_bicubic_psnr:.4f}, SSIM={avg_bicubic_ssim:.4f}")
        print(f"  - Model   (你的SNN): PSNR={val_psnr:.4f},         SSIM={val_ssim:.4f}")
        
        if val_ssim < avg_bicubic_ssim + 0.01:
            print(">>> 警告: 模型表现不如/接近纯插值，模型可能正在'装死' (Model Collapse) <<<")
        else:
            print(">>> 恭喜: 模型表现优于纯插值，正在有效学习细节 <<<")
        print(f"能耗评估 (Energy Efficiency):")
        print(f"  - GSOPs: {energy_metrics.get('GSOPs', 0):.4f} G")
        print(f"  - Energy (SNN): {energy_metrics.get('Energy_SNN (J)', 0):.4f} J")
        print(f"  - Energy (CNN Baseline): {energy_metrics.get('Energy_CNN (J)', 0):.4f} J")
        print(f"  - Delta_E: {energy_metrics.get('Delta_E (%)', 0):.2f} %")
        print(f"质量指标 (Validation):")
        print(f"  - PSNR: {val_psnr:.4f} / Best: {max(best_psnr, val_psnr):.4f}")
        print(f"  - SSIM: {val_ssim:.4f}")
        print(f"  - LPIPS: {avg_val_lpips:.4f}")
        if reconstruction_evaluator:
            print(f"  - NIQE: {val_niqe:.4f} | UIQM: {val_uiqm:.4f}")
        print("-" * 30)

        # 保存最佳模型
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            best_epoch = epoch + 1
            save_path = os.path.join(config.SAVE_DIR, "best_reconstruction_net.pth")
            if config.NUM_GPUS > 1:
                torch.save(joint_system.reconstruction_module.module.state_dict(), save_path)
            else:
                torch.save(joint_system.reconstruction_module.state_dict(), save_path)
            print(f"新最佳 PSNR! 模型已保存: {save_path}")
            
        # 构建 CSV 行
        
        metrics_row = [
            epoch + 1,
            
            # --- Train Losses ---
            avg_stats['recon_loss'],         # L1
            avg_stats['perceptual_loss'],    # Perceptual
            avg_stats['gradient_loss'],      # Gradient
            avg_stats['lpips_loss'],         # LPIPS
            avg_stats['reconstruction_total_loss'], # Total
            avg_stats['degradation_loss'],   # Degrade Total
            
            # --- Train Metrics ---
            train_reconstruction_niqe, 
            train_reconstruction_brisque,
            
            # --- Val Metrics ---
            val_psnr,
            val_ssim,
            avg_val_lpips,
            val_niqe, 
            val_brisque, 
            val_uiqm,
            
            # --- Energy ---
            energy_metrics.get('GSOPs', 0),
            energy_metrics.get('Energy_SNN (J)', 0),
            energy_metrics.get('Energy_CNN (J)', 0),
            energy_metrics.get('Delta_E (%)', 0),
            
            # --- LR ---
            current_lr['reconstruction_lr'],
            current_lr['degradation_lr'],
            current_lr['discriminator_lr']
        ]
        
#         metrics_row = [
#             epoch + 1,
#             avg_stats['d_loss'],
#             avg_stats['g_loss'],
#             0.0, 0.0, 0.0, # train_adv, fm, mmd (未统计细节)
#             avg_stats['recon_loss'],
#             avg_stats['perceptual_loss'],  
#             avg_stats['gradient_loss'],    
#             0.0, 0.0, # low, percep_deg
#             avg_stats['degradation_loss'],
#             0.0, 0.0, # recon_total, ct
#             avg_stats['lpips_loss'],
            
#             0.0, 0.0, 0.0, 0.0, 0.0, # Val Losses (这些暂时没有在验证集计算)
#             0.0, 0.0, 0.0, 0.0, 0.0,
#             0.0, 0.0, 0.0,
            
#             avg_val_lpips,
            
#             energy_metrics.get('GSOPs', 0),
#             energy_metrics.get('Energy_SNN (J)', 0),
#             energy_metrics.get('Energy_CNN (J)', 0),
#             energy_metrics.get('Delta_E (%)', 0),
            
#             val_psnr,
#             val_ssim,
            
#             current_lr['degradation_lr'],
#             current_lr['reconstruction_lr'],
#             current_lr['discriminator_lr']
#         ]

#         if degradation_evaluator:
#             # 阶段二不更新退化指标，填0或保持上一轮
#             metrics_row.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

#         if reconstruction_evaluator:
#             metrics_row.extend([
#                 train_reconstruction_niqe, train_reconstruction_brisque, train_reconstruction_uiqm, train_reconstruction_uism, 
#                 val_niqe, val_brisque, val_uiqm, val_uism
#             ])

#         if distribution_comparator:
#             # 阶段二不更新分布指标
#             metrics_row.extend([0.0, 0.0, 0.0, 0.0])

        save_metrics_to_csv(metrics_csv_path, metrics_row, csv_headers)

        if epoch % 2 == 0:
            param_monitor.log_parameters(joint_system, epoch + 1)

        if (epoch + 1) % 5 == 0:
            degradation_save_path = os.path.join(config.SAVE_DIR, f"degradation_module_epoch{epoch + 1}.pth")
            discriminator_save_path = os.path.join(config.SAVE_DIR, f"discriminator_epoch{epoch + 1}.pth")
            reconstruction_save_path = os.path.join(config.SAVE_DIR, f"reconstruction_net_epoch{epoch + 1}.pth")

            if config.NUM_GPUS > 1:
                torch.save(joint_system.degradation_module.module.state_dict(), degradation_save_path)
                torch.save(joint_system.discriminator_module.module.state_dict(), discriminator_save_path)
                torch.save(joint_system.reconstruction_module.module.state_dict(), reconstruction_save_path)
            else:
                torch.save(joint_system.degradation_module.state_dict(), degradation_save_path)
                torch.save(joint_system.discriminator_module.state_dict(), discriminator_save_path)
                torch.save(joint_system.reconstruction_module.state_dict(), reconstruction_save_path)

            print(f"模型已保存: {degradation_save_path}, {discriminator_save_path}, {reconstruction_save_path}")

    final_path = os.path.join(config.SAVE_DIR, "final_reconstruction_net.pth")
    if config.NUM_GPUS > 1:
        torch.save(joint_system.reconstruction_module.module.state_dict(), final_path)
    else:
        torch.save(joint_system.reconstruction_module.state_dict(), final_path)
    
    print("所有训练完成！")

if __name__ == "__main__":
    main()