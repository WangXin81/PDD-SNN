import io
import copy
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import save_image
import pandas as pd
import csv
from tqdm import tqdm
from losses1 import calculate_psnr, calculate_ssim
import cv2
import math
try:
    from thop import profile
except ImportError:
    print("未安装 thop 库，无法自动计算 FLOPs。请运行: pip install thop")
    profile = None

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"随机种子已设置为: {seed}")

    
# def calculate_psnr_ssim_with_paired_data(joint_system, paired_loader, device, max_batches=2):
#     """使用配对数据计算PSNR和SSIM"""
#     psnr_values = []
#     ssim_values = []
#     count = 0

#     with torch.no_grad():
#         for batch_idx, batch in enumerate(paired_loader):
#             if batch_idx >= max_batches:
#                 break

#             hr = batch["hr"].to(device)
#             ref_hr = batch["ref_hr"].to(device)

#             generated_lab_lr = joint_system.degradation_module(hr)
#             hr_reconstructed = joint_system.reconstruction_module(generated_lab_lr)

#             batch_psnr = 0.0
#             batch_ssim = 0.0
#             for j in range(hr_reconstructed.shape[0]):
#                 batch_psnr += calculate_psnr(hr_reconstructed[j], ref_hr[j])
#                 batch_ssim += calculate_ssim(hr_reconstructed[j], ref_hr[j])

#             psnr_values.append(batch_psnr / hr_reconstructed.shape[0])
#             ssim_values.append(batch_ssim / hr_reconstructed.shape[0])
#             count += 1

#     if count > 0:
#         return sum(psnr_values) / count, sum(ssim_values) / count
#     else:
#         return 0.0, 0.0


# def calculate_psnr_ssim_with_paired_data(joint_system, paired_loader, device, max_batches=2):
#     """使用配对数据计算PSNR和SSIM"""
#     total_psnr = 0.0
#     total_ssim = 0.0
#     total_images = 0 # 改为统计图片总数

#     with torch.no_grad():
#         for batch_idx, batch in enumerate(paired_loader):
#             if batch_idx >= max_batches:
#                 break

#             hr = batch["hr"].to(device)
#             ref_hr = batch["ref_hr"].to(device)

#             generated_lab_lr = joint_system.degradation_module(hr)
#             hr_reconstructed = joint_system.reconstruction_module(generated_lab_lr)
            
#             batch_size = hr_reconstructed.shape[0]

#             for j in range(batch_size):
#                 total_psnr += calculate_psnr(hr_reconstructed[j], ref_hr[j])
#                 total_ssim += calculate_ssim(hr_reconstructed[j], ref_hr[j])
#             total_images += batch_size

#             # psnr_values.append(batch_psnr / hr_reconstructed.shape[0])
#             # ssim_values.append(batch_ssim / hr_reconstructed.shape[0])
#             # count += 1

#     if total_images > 0:
#         return total_psnr / total_images, total_ssim / total_images
#     else:
#         return 0.0, 0.0



# ============================================================
# 请将此函数复制并替换 utils.py 中原本的同名函数
# ============================================================

def calculate_psnr_ssim_with_paired_data(joint_system, paired_loader, device, scale=4, max_batches=2):
    """
    使用配对数据计算PSNR和SSIM (已启用学术标准: Y通道 + 切边)
    注意: scale 参数默认设为 4，如果您是跑 x2 任务，请在 main.py 调用时传入 scale=2
    """
    total_psnr = 0.0
    total_ssim = 0.0
    total_images = 0 

    with torch.no_grad():
        for batch_idx, batch in enumerate(paired_loader):
            if batch_idx >= max_batches:
                break

            hr = batch["hr"].to(device)
            ref_hr = batch["ref_hr"].to(device)

            generated_lab_lr = joint_system.degradation_module(hr)
            hr_reconstructed = joint_system.reconstruction_module(generated_lab_lr)
            hr_reconstructed = torch.clamp(hr_reconstructed, 0, 1)
            
            batch_size = hr_reconstructed.shape[0]

            for j in range(batch_size):
                # ================= 核心修改 =================
                # 显式传递 crop_border 和 test_y_channel
                # 1. crop_border=scale: x4任务切4个像素，x2切2个
                # 2. test_y_channel=True: 只计算亮度通道，排除颜色干扰
                # ============================================
                total_psnr += calculate_psnr(
                    hr_reconstructed[j], 
                    ref_hr[j], 
                    crop_border=scale, 
                    test_y_channel=True
                )
                total_ssim += calculate_ssim(
                    hr_reconstructed[j], 
                    ref_hr[j], 
                    crop_border=scale, 
                    test_y_channel=True
                )
            
            total_images += batch_size

    if total_images > 0:
        return total_psnr / total_images, total_ssim / total_images
    else:
        return 0.0, 0.0


# def batch_calculate_metrics_degradation(images, evaluator, max_images=4):
#     """批量计算退化模块图像质量指标"""
#     if not images or len(images) == 0:
#         print("警告: 图像列表为空，无法计算退化指标")
#         return 0.0, 0.0, 0.0
#     try:
#         # 防守式图像预处理
#         images_to_eval = []
#         for img in images[:max_images]:
#             if isinstance(img, torch.Tensor):
#                 img_processed = img.detach().cpu().clamp(0, 1)
#                 if img_processed.dim() == 3 and img_processed.shape[0] in [1, 3]:
#                     images_to_eval.append(img_processed)
#                 else:
#                     print(f"警告: 跳过形状异常的图像: {img_processed.shape}")
#             else:
#                 print(f"警告: 跳过非张量图像: {type(img)}")

#         if len(images_to_eval) == 0:
#             print("警告: 无有效图像用于退化指标计算")
#             return 0.0, 0.0, 0.0

#         print(f"计算退化指标: 使用 {len(images_to_eval)} 张图像")

#         # 使用获取先验计算1.py中的方法
#         from 获取先验计算 import batch_niqe_complete_scores, batch_brisque_scores, batch_uiqm_scores
#         niqe_scores = batch_niqe_complete_scores(images_to_eval, NIQE_PRIOR_PATH)
#         brisque_scores = batch_brisque_scores(images_to_eval)
#         uiqm_scores = batch_uiqm_scores(images_to_eval)

#         niqe = np.mean(niqe_scores) if niqe_scores and len(niqe_scores) > 0 else 0.0
#         brisque = np.mean(brisque_scores) if brisque_scores and len(brisque_scores) > 0 else 0.0
#         uiqm = np.mean(uiqm_scores) if uiqm_scores and len(uiqm_scores) > 0 else 0.0

#         return niqe, brisque, uiqm

#     except Exception as e:
#         print(f"批量计算退化模块指标时出错: {e}")
#         return 0.0, 0.0, 0.0


# def batch_calculate_metrics_degradation(images, evaluator, prior_path, max_images=4):
#     """
#     批量计算退化模块图像质量指标
#     新增参数: prior_path (NIQE计算需要的先验文件路径)
#     """
#     if not images or len(images) == 0:
#         print("警告: 图像列表为空，无法计算退化指标")
#         return 0.0, 0.0, 0.0
#     try:
#         # 防守式图像预处理
#         images_to_eval = []
#         for img in images[:max_images]:
#             if isinstance(img, torch.Tensor):
#                 img_processed = img.detach().cpu().clamp(0, 1)
#                 if img_processed.dim() == 3 and img_processed.shape[0] in [1, 3]:
#                     images_to_eval.append(img_processed)
#                 else:
#                     print(f"警告: 跳过形状异常的图像: {img_processed.shape}")
#             else:
#                 print(f"警告: 跳过非张量图像: {type(img)}")

#         if len(images_to_eval) == 0:
#             print("警告: 无有效图像用于退化指标计算")
#             return 0.0, 0.0, 0.0

#         print(f"计算退化指标: 使用 {len(images_to_eval)} 张图像")

#         # 使用获取先验计算.py中的方法
#         from 获取先验计算 import batch_niqe_complete_scores, batch_brisque_scores, batch_uiqm_scores
        
#         # --- 修改点在这里：使用传入的 prior_path，而不是未定义的 NIQE_PRIOR_PATH ---
#         niqe_scores = batch_niqe_complete_scores(images_to_eval, prior_path) 
#         # ---------------------------------------------------------------------
        
#         brisque_scores = batch_brisque_scores(images_to_eval)
#         uiqm_scores = batch_uiqm_scores(images_to_eval)

#         niqe = np.mean(niqe_scores) if niqe_scores and len(niqe_scores) > 0 else 0.0
#         brisque = np.mean(brisque_scores) if brisque_scores and len(brisque_scores) > 0 else 0.0
#         uiqm = np.mean(uiqm_scores) if uiqm_scores and len(uiqm_scores) > 0 else 0.0

#         return niqe, brisque, uiqm

#     except Exception as e:
#         print(f"批量计算退化模块指标时出错: {e}")
#         return 0.0, 0.0, 0.0


import torch
import numpy as np

def batch_calculate_metrics_degradation(images, evaluator, prior_path, max_images=4):
    """
    批量计算退化模块图像质量指标
    修复了 Tensor 布尔判断错误，并统一返回字典格式。
    """
    
    # ================= [修复核心] 安全的空值检查 =================
    # 1. 检查是否为 None
    if images is None:
        return {}
        
    # 2. 如果是 Tensor，使用 numel() 检查元素个数
    if isinstance(images, torch.Tensor):
        if images.numel() == 0:
            print("警告: 图像张量为空")
            return {}
            
    # 3. 如果是 List，检查长度
    elif isinstance(images, list):
        if len(images) == 0:
            print("警告: 图像列表为空")
            return {}
    # ==========================================================

    try:
        # 防守式图像预处理
        images_to_eval = []
        
        # 处理 Tensor 切片时注意保持维度
        # 如果 images 是 Tensor [B, C, H, W]，遍历它会得到 [C, H, W]
        current_images = images[:max_images]

        for img in current_images:
            if isinstance(img, torch.Tensor):
                img_processed = img.detach().cpu().clamp(0, 1)
                
                # 检查维度，确保是 [C, H, W]
                if img_processed.dim() == 3 and img_processed.shape[0] in [1, 3]:
                    images_to_eval.append(img_processed)
                elif img_processed.dim() == 4 and img_processed.shape[0] == 1:
                    # 处理 [1, C, H, W] 的情况
                    images_to_eval.append(img_processed.squeeze(0))
                else:
                    # print(f"警告: 跳过形状异常的图像: {img_processed.shape}")
                    pass
            else:
                # print(f"警告: 跳过非张量图像: {type(img)}")
                pass

        if len(images_to_eval) == 0:
            # print("警告: 无有效图像用于退化指标计算")
            return {}

        # print(f"计算退化指标: 使用 {len(images_to_eval)} 张图像")

        # 动态导入 (如果您的项目结构允许，建议放在文件开头)
        from 获取先验计算 import batch_niqe_complete_scores, batch_brisque_scores, batch_uiqm_scores
        
        # 计算各指标
        # 传入 prior_path 给 NIQE
        niqe_scores = batch_niqe_complete_scores(images_to_eval, prior_path) 
        brisque_scores = batch_brisque_scores(images_to_eval)
        uiqm_scores = batch_uiqm_scores(images_to_eval)

        # 计算平均值
        niqe = np.mean(niqe_scores) if niqe_scores and len(niqe_scores) > 0 else 0.0
        brisque = np.mean(brisque_scores) if brisque_scores and len(brisque_scores) > 0 else 0.0
        uiqm = np.mean(uiqm_scores) if uiqm_scores and len(uiqm_scores) > 0 else 0.0

        # ================= [修复] 返回字典而非元组 =================
        return {
            'niqe': float(niqe),
            'brisque': float(brisque),
            'uiqm': float(uiqm)
        }
        # ==========================================================

    except Exception as e:
        print(f"批量计算退化模块指标时出错: {e}")
        return {}



def batch_calculate_metrics_reconstruction(images, evaluator, max_images=4):
    """批量计算重建模块图像质量指标"""
    # if not images or len(images) == 0:
    #     print("警告: 图像列表为空，无法计算重建指标")
    #     return 0.0, 0.0, 0.0
    
    # 先判断是否为 None
    if images is None:
        return {}
    
    # 如果是 Tensor，检查元素个数
    if isinstance(images, torch.Tensor):
        if images.numel() == 0: # 使用 .numel() 检查是否为空
            return {}
    # 如果是列表，检查长度
    elif isinstance(images, list):
        if len(images) == 0:
            return {}
    
    try:
        # 防守式图像预处理
        images_to_eval = []
        for img in images[:max_images]:
            if isinstance(img, torch.Tensor):
                img_processed = img.detach().cpu().clamp(0, 1)
                
                if img_processed.dim() == 3 and img_processed.shape[0] in [1, 3]:
                    images_to_eval.append(img_processed)
                else:
                    print(f"警告: 跳过形状异常的图像: {img_processed.shape}")
            else:
                print(f"警告: 跳过非张量图像: {type(img)}")

        if len(images_to_eval) == 0:
            print("警告: 无有效图像用于重建指标计算")
            # return 0.0, 0.0, 0.0
            return {}

        # print(f"计算重建指标: 使用 {len(images_to_eval)} 张图像")

        # 使用evaluate.py中的方法
        niqe = evaluator.calculate_batch_niqe(images_to_eval)
        brisque = evaluator.calculate_batch_brisque(images_to_eval)
        uiqm = evaluator.calculate_batch_uiqm(images_to_eval)
        uiqm, uism, uicm, uiconm = evaluator.calculate_batch_uiqm(images_to_eval, return_components=True)
        
        return {
            'niqe': niqe,
            'brisque': brisque,
            'uiqm': uiqm,
            'uism': uism
        }
        
    except Exception as e:
        print(f"批量计算重建模块指标时出错: {e}")
        return {} # 出错时返回空字典
        
    #     return niqe, brisque, uiqm
    # except Exception as e:
    #     print(f"批量计算重建模块指标时出错: {e}")
    #     return 0.0, 0.0, 0.0
    
    
# class GeometricAugmentation:
#     def __init__(self, prob=0.5):
#         self.prob = prob
#         self.last_transform_types = None

#     def __call__(self, img, transform_types=None):
#         batch_size = img.shape[0]

#         if transform_types is None:
#             transform_types = []
#             for i in range(batch_size):
#                 if random.random() < self.prob:
#                     transform_types.append(random.choice([
#                         'flip_h', 'flip_v', 'rotate_90', 'rotate_180', 'rotate_270'
#                     ]))
#                 else:
#                     transform_types.append('identity')

#         self.last_transform_types = transform_types
#         augmented_imgs = img.clone()

#         for i, transform_type in enumerate(transform_types):
#             if transform_type == 'flip_h':
#                 augmented_imgs[i] = torch.flip(img[i], [2])
#             elif transform_type == 'flip_v':
#                 augmented_imgs[i] = torch.flip(img[i], [1])
#             elif transform_type == 'rotate_90':
#                 augmented_imgs[i] = torch.rot90(img[i], 1, [1, 2])
#             elif transform_type == 'rotate_180':
#                 augmented_imgs[i] = torch.rot90(img[i], 2, [1, 2])
#             elif transform_type == 'rotate_270':
#                 augmented_imgs[i] = torch.rot90(img[i], 3, [1, 2])

#         return augmented_imgs


# utils.py

# class GeometricAugmentation:
#     def __init__(self, prob=0.5):
#         self.prob = prob

#     def __call__(self, img, target=None):
#         """
#         同时对 img (LR) 和 target (HR) 进行完全相同的几何变换
#         Args:
#             img: 输入图像 Tensor [B, C, H, W]
#             target: 目标图像 Tensor [B, C, H*s, W*s] (可选)
#         Returns:
#             如果 target 为 None: 返回 augmented_img
#             如果 target 不为 None: 返回 (augmented_img, augmented_target)
#         """
#         # 如果概率未命中，直接返回原图
#         if random.random() >= self.prob:
#             if target is not None:
#                 return img, target
#             return img

#         batch_size = img.shape[0]
        
#         # 1. 预先生成变换列表 (确保 LR 和 HR 使用相同的随机决策)
#         transforms_list = []
#         for i in range(batch_size):
#             # 随机选择一种变换: 水平翻转, 垂直翻转, 旋转90/180/270
#             t_type = random.choice(['flip_h', 'flip_v', 'rotate_90', 'rotate_180', 'rotate_270'])
#             transforms_list.append(t_type)

#         # 2. 对 img 应用变换
#         img_aug = self._apply_transforms(img, transforms_list)
        
#         # 3. 如果有 target，应用完全相同的变换
#         if target is not None:
#             target_aug = self._apply_transforms(target, transforms_list)
#             return img_aug, target_aug
            
#         return img_aug

#     def _apply_transforms(self, tensor, transforms_list):
#         """内部函数：执行具体的 Tensor 变换"""
#         augmented = tensor.clone()
#         for i, t_type in enumerate(transforms_list):
#             if t_type == 'flip_h':
#                 augmented[i] = torch.flip(tensor[i], [2]) # 翻转宽 (W)
#             elif t_type == 'flip_v':
#                 augmented[i] = torch.flip(tensor[i], [1]) # 翻转高 (H)
#             elif t_type == 'rotate_90':
#                 augmented[i] = torch.rot90(tensor[i], 1, [1, 2])
#             elif t_type == 'rotate_180':
#                 augmented[i] = torch.rot90(tensor[i], 2, [1, 2])
#             elif t_type == 'rotate_270':
#                 augmented[i] = torch.rot90(tensor[i], 3, [1, 2])
#         return augmented



#     def get_last_transform_types(self):
#         return self.last_transform_types



import torch
import random

class GeometricAugmentation:
    """
    几何增强：随机翻转和旋转。
    [修复版]：针对长方形图像优化，防止旋转导致的维度崩溃。
    """
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img, target=None):
        """
        同时对 img (LR) 和 target (HR) 进行完全相同的几何变换
        img: 输入图像 Tensor [B, C, H, W]
        target: 目标图像 Tensor [B, C, H*s, W*s] (可选)
        """
        # 1. 随机概率：如果没中，就直接返回原图
        if random.random() > self.prob:
            return (img, target) if target is not None else img

        # 2. 检查形状：判断是否为正方形
        # 图像维度通常是 [Batch, Channel, Height, Width]
        h, w = img.shape[-2], img.shape[-1]
        is_square = (h == w)

        # 3. 确定允许的操作索引
        # 0: 原图, 1: 水平翻转, 2: 垂直翻转, 3: 旋转180度
        # 4: 旋转90度, 5: 旋转270度
        if is_square:
            # 正方形随便转，形状都不会变
            aug_idx = random.randint(0, 5)
        else:
            # 长方形禁止 90/270 度旋转 (4和5)，否则宽高会对调，导致张量维度报错
            valid_choices = [0, 1, 2, 3] 
            aug_idx = random.choice(valid_choices)

        # 4. 执行变换
        # 直接对整个 Batch 进行操作，速度更快，且能保证 Batch 内所有图片形状一致
        img_aug = self._apply_op(img, aug_idx)
        
        if target is not None:
            # 对 HR 目标应用完全一致的变换，确保像素对齐
            target_aug = self._apply_op(target, aug_idx)
            return img_aug, target_aug
        
        return img_aug

    def _apply_op(self, tensor, idx):
        """内部函数：执行具体的 Tensor 变换"""
        # 图像维度通常是最后两维 (H, W)
        dims = [-2, -1]
        
        if idx == 0:   # 无操作
            return tensor
        elif idx == 1: # 水平翻转 (翻转最后一位 Width)
            return torch.flip(tensor, [-1])
        elif idx == 2: # 垂直翻转 (翻转倒数第二位 Height)
            return torch.flip(tensor, [-2])
        elif idx == 3: # 旋转180度
            return torch.rot90(tensor, 2, dims)
        elif idx == 4: # 旋转90度 (仅正方形会进这里)
            return torch.rot90(tensor, 1, dims)
        elif idx == 5: # 旋转270度 (仅正方形会进这里)
            return torch.rot90(tensor, 3, dims)
            
        return tensor




class ParameterMonitor:
    def __init__(self):
        self.history = {}

    def log_parameters(self, joint_system, epoch):
        epoch_data = {}
        modules_to_check = [
            joint_system.degradation_module,
            joint_system.discriminator_module,
            joint_system.reconstruction_module
        ]

        for module in modules_to_check:
            self._collect_module_parameters(module, epoch_data)

        self.history[epoch] = epoch_data

    def _collect_module_parameters(self, module, epoch_data, prefix=""):
        if hasattr(module, 'named_modules'):
            for name, submodule in module.named_modules():
                if prefix:
                    full_name = f"{prefix}.{name}"
                else:
                    full_name = name

                params = {}
                if hasattr(submodule, 'tau') and isinstance(submodule.tau, nn.Parameter):
                    params['tau'] = submodule.tau.data.item()

                if params:
                    epoch_data[full_name] = params
        else:
            params = {}
            if hasattr(module, 'tau') and isinstance(module.tau, nn.Parameter):
                params['tau'] = module.tau.data.item()

            if params:
                module_name = prefix if prefix else module.__class__.__name__
                epoch_data[module_name] = params

    def save_to_csv(self, filename):
        rows = []
        for epoch, data in self.history.items():
            for module_name, params in data.items():
                for param_name, value in params.items():
                    rows.append({
                        'epoch': epoch,
                        'module': module_name,
                        'parameter': param_name,
                        'value': value
                    })

        df = pd.DataFrame(rows)
        df.to_csv(filename, index=False)
        print(f"参数历史已保存到: {filename}")

    def load_from_csv(self, filename):
        if not os.path.exists(filename):
            print(f"参数历史文件不存在: {filename}")
            return

        try:
            df = pd.read_csv(filename)
            self.history = {}
            for _, row in df.iterrows():
                epoch = int(row['epoch'])
                module_name = row['module']
                param_name = row['parameter']
                value = row['value']

                if epoch not in self.history:
                    self.history[epoch] = {}
                if module_name not in self.history[epoch]:
                    self.history[epoch][module_name] = {}

                self.history[epoch][module_name][param_name] = value
            print(f"参数历史已从 {filename} 加载")
        except Exception as e:
            print(f"加载参数历史时出错: {e}")

def create_niqe_prior_if_needed(lr_dir, save_path):
    if not os.path.exists(save_path):
        print("NIQE先验文件不存在，正在创建...")

        if not os.path.exists(lr_dir):
            print(f"警告: LR图像目录 {lr_dir} 不存在，无法创建NIQE先验")
            try:
                import scipy.io
                scipy.io.savemat(save_path, {
                    'pop_mu': np.zeros(18),
                    'pop_cov': np.eye(18)
                })
                print(f"创建了空的NIQE先验文件: {save_path}")
            except Exception as e:
                print(f"创建空先验文件失败: {e}")
            return save_path

        try:
            from 获取先验计算 import build_niqe_prior_from_images
            lr_files = [f for f in os.listdir(lr_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            print(f"在 {lr_dir} 中找到 {len(lr_files)} 个LR图像文件")

            if len(lr_files) == 0:
                print("警告: LR目录中没有图像文件，创建空的NIQE先验")
                import scipy.io
                scipy.io.savemat(save_path, {
                    'pop_mu': np.zeros(18),
                    'pop_cov': np.eye(18)
                })
            else:
                print("开始从真实LR图像构建NIQE先验...")
                build_niqe_prior_from_images(lr_dir, save_path=save_path, patch_size=32)

                if os.path.exists(save_path):
                    print(f"NIQE先验文件创建成功: {save_path}")
                else:
                    print("NIQE先验文件创建失败")

        except Exception as e:
            print(f"创建NIQE先验文件时出错: {e}")
            try:
                import scipy.io
                scipy.io.savemat(save_path, {
                    'pop_mu': np.zeros(18),
                    'pop_cov': np.eye(18)
                })
                print("创建了空的NIQE先验文件作为后备")
            except Exception as e2:
                print(f"创建空先验文件也失败: {e2}")
    else:
        print(f"NIQE先验文件已存在: {save_path}")

    return save_path

# def calculate_distribution_metrics(real_lr_images, generated_lr_images, distribution_comparator):
#     if (distribution_comparator is None or real_lr_images is None or
#         generated_lr_images is None or not isinstance(real_lr_images, torch.Tensor) or
#         not isinstance(generated_lr_images, torch.Tensor)):
#         return 0.0, 0.0, 0.0

#     if real_lr_images.numel() == 0 or generated_lr_images.numel() == 0:
#         return 0.0, 0.0, 0.0

#     try:
#         real_lr_np = real_lr_images.cpu().numpy()
#         generated_lr_np = generated_lr_images.cpu().numpy()
#         real_l = real_lr_np[:, 0, :, :]
#         generated_l = generated_lr_np[:, 0, :, :]
#         if len(real_l) > 0 and len(generated_l) > 0:
#             real_sample = real_l[0]
#             generated_sample = generated_l[0]
#             real_sample_uint8 = (real_sample * 255).astype(np.uint8)
#             generated_sample_uint8 = (generated_sample * 255).astype(np.uint8)
#             real_l_dist = distribution_comparator._compute_L_channel_distribution(real_sample_uint8)
#             gen_l_dist = distribution_comparator._compute_L_channel_distribution(generated_sample_uint8)
#             cosine_sim = distribution_comparator._cosine_similarity(real_l_dist, gen_l_dist)
#             cosine_distance = 1.0 - cosine_sim  # 将相似度转换为距离
#             bhattacharyya_distance = distribution_comparator._bhattacharyya_distance(real_l_dist, gen_l_dist)
#             chi_square_distance = distribution_comparator._chi_square_distance(real_l_dist, gen_l_dist)
#             return bhattacharyya_distance, chi_square_distance, cosine_distance
#             # cosine_sim = distribution_comparator._cosine_similarity(real_l_dist, gen_l_dist)
#             # bhattacharyya_distance = distribution_comparator._bhattacharyya_distance(real_l_dist, gen_l_dist)
#             # chi_square_distance = distribution_comparator._chi_square_distance(real_l_dist, gen_l_dist)
#             # return bhattacharyya_distance, chi_square_distance, cosine_sim
#     except Exception as e:
#         print(f"计算分布距离时出错: {e}")
#     return 0.0, 0.0, 0.0


import numpy as np
import torch

# def calculate_distribution_metrics(real_lr_images, generated_lr_images, distribution_comparator, prior_path=None):
#     """
#     计算分布差异指标 (Bhattacharyya, Chi-square, Cosine) 和 NIQE
#     """
#     # 1. 安全检查：处理 None 或 类型错误
#     if (distribution_comparator is None or real_lr_images is None or
#         generated_lr_images is None or not isinstance(real_lr_images, torch.Tensor) or
#         not isinstance(generated_lr_images, torch.Tensor)):
#         return {} # 返回空字典，避免 main.py 报错

#     # 检查是否为空 Tensor
#     if real_lr_images.numel() == 0 or generated_lr_images.numel() == 0:
#         return {}

#     try:
#         # ==========================================
#         # Part 1: 计算分布距离 (原有逻辑)
#         # ==========================================
#         real_lr_np = real_lr_images.cpu().numpy()
#         generated_lr_np = generated_lr_images.cpu().numpy()
        
#         # 提取第 0 个样本进行近似计算
#         # 注意：这里仅取 batch 中的第一张图计算分布距离
#         bhattacharyya_distance = 0.0
#         chi_square_distance = 0.0
#         cosine_sim = 0.0
        
#         if real_lr_np.shape[0] > 0 and generated_lr_np.shape[0] > 0:
#             # 假设格式为 [B, C, H, W]，取第 0 个样本的第 0 通道 (L通道/灰度)
#             real_sample = real_lr_np[0, 0, :, :]
#             generated_sample = generated_lr_np[0, 0, :, :]
            
#             # 转换为 uint8 [0, 255]
#             real_sample_uint8 = (np.clip(real_sample, 0, 1) * 255).astype(np.uint8)
#             generated_sample_uint8 = (np.clip(generated_sample, 0, 1) * 255).astype(np.uint8)
            
#             # 计算分布特征
#             real_l_dist = distribution_comparator._compute_L_channel_distribution(real_sample_uint8)
#             gen_l_dist = distribution_comparator._compute_L_channel_distribution(generated_sample_uint8)
            
#             # 计算距离指标
#             cosine_sim = distribution_comparator._cosine_similarity(real_l_dist, gen_l_dist)
#             bhattacharyya_distance = distribution_comparator._bhattacharyya_distance(real_l_dist, gen_l_dist)
#             chi_square_distance = distribution_comparator._chi_square_distance(real_l_dist, gen_l_dist)

#         # ==========================================
#         # Part 2: 计算 NIQE (使用 '获取先验计算' 模块)
#         # ==========================================
#         degradation_niqe = 0.0
#         try:
#             # 动态导入，确保使用您指定的模块方法
#             from 获取先验计算 import batch_niqe_complete
            
#             # batch_niqe_complete 通常接受 Tensor 列表或 Batch Tensor
#             # 确保数据在 CPU 上
#             imgs_for_niqe = generated_lr_images.detach().cpu()
            
#             # 计算 Batch 的平均 NIQE
#             degradation_niqe = batch_niqe_complete(imgs_for_niqe)
            
#         except ImportError:
#             print("警告: 无法导入 '获取先验计算.batch_niqe_complete'，NIQE 设为 0")
#         except Exception as niqe_err:
#             print(f"NIQE 计算出错: {niqe_err}")

#         # ==========================================
#         # Part 3: 返回字典 (关键修复)
#         # ==========================================
#         return {
#             'bhattacharyya': float(bhattacharyya_distance), # 对应 main.py: .get('bhattacharyya')
#             'chi_square': float(chi_square_distance),       # 对应 main.py: .get('chi_square')
#             'cosine_similarity': float(cosine_sim),         # 对应 main.py: .get('cosine_similarity')
#             'cosine': float(1.0 - cosine_sim),              # 备用：余弦距离
#             'degradation_niqe': float(degradation_niqe),    # 对应 main.py: .get('degradation_niqe')
#             'niqe': float(degradation_niqe)                 # 备用
#         }

#     except Exception as e:
#         print(f"计算分布指标综合出错: {e}")
#         return {} # 出错返回空字典


def calculate_distribution_metrics(real_lr_images, generated_lr_images, distribution_comparator, prior_path=None):
    """
    计算分布差异指标 (Bhattacharyya, Chi-square, Cosine) 和 NIQE
    改进点：计算 Batch 中所有图片的平均指标，而非仅取第一张
    """
    # 1. 安全检查
    if (distribution_comparator is None or real_lr_images is None or
        generated_lr_images is None or not isinstance(real_lr_images, torch.Tensor) or
        not isinstance(generated_lr_images, torch.Tensor)):
        return {}

    if real_lr_images.numel() == 0 or generated_lr_images.numel() == 0:
        return {}

    try:
        # Part 1: 计算分布距离 (Batch 平均模式)
        real_lr_np = real_lr_images.cpu().numpy()
        generated_lr_np = generated_lr_images.cpu().numpy()
        
        batch_size = real_lr_np.shape[0]
        
        # 用于累加各项指标
        total_bhattacharyya = 0.0
        total_chi_square = 0.0
        total_cosine_sim = 0.0
        valid_samples = 0
        
        for i in range(batch_size):
            try:
                # 提取单张图像 (C, H, W)
                real_img = real_lr_np[i]
                gen_img = generated_lr_np[i]
                
                # --- [改进] 更稳健的 L 通道/灰度图提取 ---
                # 假设输入是 [C, H, W] 且值域 [0, 1]
                # 方案 A: 如果是 RGB (3通道)，转为灰度
                if real_img.shape[0] == 3:
                    # 简单转换: 0.299*R + 0.587*G + 0.114*B
                    real_gray = 0.299*real_img[0] + 0.587*real_img[1] + 0.114*real_img[2]
                    gen_gray = 0.299*gen_img[0] + 0.587*gen_img[1] + 0.114*gen_img[2]
                # 方案 B: 如果本身就是灰度 (1通道) 或其他
                else:
                    real_gray = real_img[0]
                    gen_gray = gen_img[0]
                
                # 转换为 uint8 [0, 255] 以供直方图计算
                real_sample_uint8 = (np.clip(real_gray, 0, 1) * 255).astype(np.uint8)
                generated_sample_uint8 = (np.clip(gen_gray, 0, 1) * 255).astype(np.uint8)
                
                # 计算分布特征
                real_l_dist = distribution_comparator._compute_L_channel_distribution(real_sample_uint8)
                gen_l_dist = distribution_comparator._compute_L_channel_distribution(generated_sample_uint8)
                
                # 累加指标
                total_cosine_sim += distribution_comparator._cosine_similarity(real_l_dist, gen_l_dist)
                total_bhattacharyya += distribution_comparator._bhattacharyya_distance(real_l_dist, gen_l_dist)
                total_chi_square += distribution_comparator._chi_square_distance(real_l_dist, gen_l_dist)
                valid_samples += 1
                
            except Exception as loop_e:
                # 忽略单张计算错误，继续下一张
                continue
        
        # 计算平均值
        if valid_samples > 0:
            avg_bhattacharyya = total_bhattacharyya / valid_samples
            avg_chi_square = total_chi_square / valid_samples
            avg_cosine_sim = total_cosine_sim / valid_samples
        else:
            avg_bhattacharyya = 0.0
            avg_chi_square = 0.0
            avg_cosine_sim = 0.0

        # Part 2: 计算 NIQE (保持不变)
        degradation_niqe = 5.0
        try:
            from 获取先验计算 import batch_niqe_complete, batch_niqe_complete_scores
            imgs_for_niqe = generated_lr_images.detach().cpu()
            
            # 优先使用传入 prior_path 的方式
            # 注意：请确保您已按之前建议修改了 utils.py 的参数列表，加入了 prior_path
            scores = batch_niqe_complete_scores(imgs_for_niqe, prior_path)
            if scores:
                degradation_niqe = np.mean(scores)
        except Exception:
            pass

        # Part 3: 返回字典
        return {
            'bhattacharyya': float(avg_bhattacharyya),
            'chi_square': float(avg_chi_square),
            'cosine_similarity': float(avg_cosine_sim),
            'cosine': float(1.0 - avg_cosine_sim), # 存为余弦距离
            'degradation_niqe': float(degradation_niqe),
            'niqe': float(degradation_niqe)
        }

    except Exception as e:
        print(f"计算分布指标综合出错: {e}")
        return {}



def save_metrics_to_csv(metrics_csv_path, metrics_row, headers):
    file_exists = os.path.isfile(metrics_csv_path)
    with open(metrics_csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(headers)
        writer.writerow(metrics_row)
        
        
        
# =========================================================================
# [新增] EnergyMeter 类：用于计算论文中的 GSOPs, Energy, ΔE(%)
# =========================================================================
class EnergyMeter:
    def __init__(self, model, input_size=(1, 3, 512, 512), device='cuda', time_steps=5):
        """
        初始化能量计算器
        :param model: 你的 SNN 重建模型
        :param input_size: 单帧图像的维度 (B=1, C, H, W)，用于计算基准 CNN FLOPs
        :param time_steps: SNN 的时间步数 (T)
        """
        self.model = model
        self.device = device
        self.input_size = input_size
        self.time_steps = time_steps
        
        # 引用论文 Table 1 中的能耗常数 (45nm CMOS) [cite: 526, 527]
        self.E_MAC = 4.6  # pJ per MAC (for CNN)
        self.E_ACC = 0.9  # pJ per ACC (for SNN)
        
        # 统计变量
        self.total_spikes = 0
        self.total_neurons = 0  # 神经元总数 (C * H * W * T)
        self.hooks = []
        
        # 1. 计算基准 CNN FLOPs (GFLOPs)
        # 论文定义: FLOPs_CNN 是非脉冲对应模型(Non-spiking counterpart)的计算量 [cite: 530]。
        # thop 计算的是 MACs，论文中的计算公式 (17) 实际上也是 MACs 的数量。
        self.cnn_flops = self._calculate_cnn_flops(device='cpu')
        if self.cnn_flops > 0:
            print(f"[Baseline CNN] Estimated FLOPs per inference: {self.cnn_flops / 1e9:.4f} G")

    def _calculate_cnn_flops(self, device='cpu'):
        """使用 thop 计算 FLOPs，并归一化为单帧 CNN 的计算量"""
        if profile is None:
            print("警告: 未安装 thop，无法计算 FLOPs。GSOPs 和 Energy 将为 0。")
            return 0
            
        try:
            with torch.no_grad():
                # 1. 将模型保存到内存缓冲区 (Buffer)
                buffer = io.BytesIO()
                torch.save(self.model, buffer)
                buffer.seek(0)
                # 2. 从缓冲区加载模型，并直接映射到 CPU
                # 这一步会创建全新的 Tensor，绝对不会影响原始的 GPU 模型
                model_copy = torch.load(buffer, map_location=device)
                model_copy.eval()
            
                # 创建假输入 (Batch=1)
                dummy_input = torch.randn(self.input_size).to(device)

                # 使用 thop.profile 计算 MACs
                # 注意：如果你的模型 forward 包含 T 次时间步循环，thop 会统计 T 次的总量。
                # 为了得到“非脉冲对应版(CNN)”的计算量，我们需要除以 T [cite: 560]。
                macs, _ = profile(model_copy, inputs=(dummy_input, ), verbose=False)

                del model_copy
                del dummy_input
                del buffer
                # torch.cuda.empty_cache()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
                # 论文公式 (19) E_CNN = FLOPs * E_MAC，这里的 FLOPs 实际上指 MAC 操作数。
                # 所以我们直接用 thop 返回的 macs 作为 FLOPs。
                flops = macs 

                # 归一化：除以时间步 T，得到静态 CNN 的基准计算量
                baseline_cnn_flops = flops / self.time_steps
                print(f"[EnergyMeter] CNN FLOPs calculated on CPU: {baseline_cnn_flops / 1e9:.4f} G")
                return baseline_cnn_flops
            
        except Exception as e:
            print(f"FLOPs 计算失败 (可能是输入维度问题): {e}")
            # import traceback
            # traceback.print_exc()
            return 0

    def register_hooks(self):
        """注册 Hook 监听所有脉冲神经元"""
        # self.hooks = []
        # self.reset()
        self.remove_hooks() 
        self.reset()
        
        # 遍历模型寻找脉冲神经元层
        # 这里列出了 neurons1.py 中用到的所有脉冲神经元类名
        neuron_types = ['MultiStepPmLIFNode', 'MultiStepTernaryPmLIFNode', 'LIFNode']
        
        count = 0
        for name, module in self.model.named_modules():
            if module.__class__.__name__ in neuron_types:
                h = module.register_forward_hook(self._spike_hook)
                self.hooks.append(h)
                count += 1
        
        print(f"EnergyMeter: 已注册 {count} 个神经元层的监听器")

    def remove_hooks(self):
        """移除 Hook"""
        for h in self.hooks:
            h.remove()
        self.hooks = []
        
        
    def _spike_hook(self, module, input, output):
        """
        Hook 函数：统计脉冲发放情况
        Output shape: [T, B, C, H, W]
        """
        # [核心修正]
        # 1. output.detach() : 切断梯度，节省显存
        # 2. .float()        : 【防溢出】转为浮点数，防止超过 Int32 (21亿) 变成负数
        # 3. .abs()          : 【防抵消】三元脉冲必须取绝对值！-1 和 1 都算 1 次能耗
        # 4. .sum().item()   : 求和并转为 Python 数字
        
        spike_count = output.detach().float().abs().sum().item()
        
        # 累加
        self.total_spikes += spike_count
        self.total_neurons += output.numel()

#     def _spike_hook(self, module, input, output):
#         """
#         Hook 函数：统计脉冲发放情况
#         Output shape: [T, B, C, H, W]
#         """
#         # output 是脉冲张量 (0 或 1)
#         # 1. 累加发放的脉冲总数 (Sum of 1s) [cite: 535]
#         self.total_spikes += output.detach().sum().item()
        
#         # 2. 累加神经元总容量 (T * B * C * H * W)
#         self.total_neurons += output.numel()

    def reset(self):
        self.total_spikes = 0
        self.total_neurons = 0

    def calculate_metrics(self):
        """
        根据论文 4.2.2 节计算最终指标
        """
        if self.total_neurons == 0:
            return {}

        # 1. 计算平均脉冲率 Spike Rate (S_r)
        # S_r = 总脉冲数 / 总神经元时空容量 [cite: 534]
        avg_spike_rate = self.total_spikes / self.total_neurons
        
        # 2. 计算 GSOPs (Giga Synaptic Operations)
        # SNN SOPs = CNN FLOPs * Spike Rate [cite: 531]
        # 注意：我们要计算的是 SNN 在 T 个时间步内的总操作数
        # 恢复包含 T 的总计算量
        total_cnn_ops = self.cnn_flops * self.time_steps 
        sops = total_cnn_ops * avg_spike_rate
        gsops = sops / 1e9  # 转换为 Giga

        # 3. 计算 Energy (J)
        # E_CNN = FLOPs_CNN * 4.6 pJ (CNN 只推理一次，不乘 T) [cite: 540]
        energy_cnn_pj = self.cnn_flops * self.E_MAC
        energy_cnn_joule = energy_cnn_pj * 1e-12

        # E_SNN = SOPs * 0.9 pJ (SNN 是 T 次累加) [cite: 541]
        energy_snn_pj = sops * self.E_ACC
        energy_snn_joule = energy_snn_pj * 1e-12

        # 4. 计算 ΔE (%)
        # Delta E = (E_CNN - E_SNN) / E_CNN * 100 [cite: 546]
        if energy_cnn_joule > 0:
            delta_e = (energy_cnn_joule - energy_snn_joule) / energy_cnn_joule * 100
        else:
            delta_e = 0.0

        return {
            "GSOPs": gsops,
            "Energy_SNN (J)": energy_snn_joule,
            "Energy_CNN (J)": energy_cnn_joule,
            "Delta_E (%)": delta_e,
            "Avg_Spike_Rate": avg_spike_rate
        }
    
# def calculate_uciqe(img_tensor):
#     """
#     计算 UCIQE (Underwater Color Image Quality Evaluation)
#     img_tensor: Tensor [B, 3, H, W] 或 [3, H, W], 范围 [0, 1]
#     """
#     # 1. 转换为 Numpy [H, W, C] 并缩放到 [0, 255]
#     if isinstance(img_tensor, torch.Tensor):
#         img = img_tensor.detach().cpu()
#         if img.dim() == 4:
#             img = img[0]
#         img = img.permute(1, 2, 0).numpy()
#     else:
#         img = img_tensor

#     img = (np.clip(img, 0, 1) * 255.0).astype(np.uint8)
    
#     # 2. 转换颜色空间 RGB -> BGR -> LAB
#     img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#     lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab).astype(np.float64)
#     l = lab[:, :, 0] * (100.0 / 255.0)  # [0, 255] -> [0, 100]
#     a = lab[:, :, 1] - 128.0           # [0, 255] -> [-128, 127]
#     b = lab[:, :, 2] - 128.0           # [0, 255] -> [-128, 127]
    
#     # l, a, b = cv2.split(lab)
#     # l, a, b = l.astype(np.float64), a.astype(np.float64), b.astype(np.float64)

#     # 3. 计算 UCIQE
#     chroma = np.sqrt(a**2 + b**2)
#     sigma_c = np.std(chroma)
#     con_l = l.max() - l.min()
#     saturation = chroma / (l + 1e-6)
#     mu_s = np.mean(saturation)

#     # 论文系数
#     uciqe = 0.4680 * sigma_c + 0.2745 * con_l + 0.2576 * mu_s
#     return uciqe    
    

    
    
def calculate_uciqe(img_tensor, crop_border=4):
    """
    改进版 UCIQE 计算：增加数值稳定性处理
    """
    if isinstance(img_tensor, torch.Tensor):
        img = img_tensor.detach().cpu().numpy()
        # 维度转换 [C, H, W] -> [H, W, C]
        if img.ndim == 3:
            img = img.transpose(1, 2, 0)
    else:
        img = img_tensor

    # 1. 必须切除边缘 (就像 PSNR 一样)，防止补零导致的数值爆炸
    if crop_border > 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, :]

    # 2. 转换范围并转为 LAB
    img = (img * 255.0).clip(0, 255).astype(np.uint8)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float64)
    l, a, b = cv2.split(lab)

    # 3. 计算色度和亮度对比度
    chroma = np.sqrt(a**2 + b**2)
    sigma_c = np.std(chroma)
    
    # 亮度对比度使用 1% 和 99% 分位数差值 (更稳健)
    l_flat = l.flatten()
    con_l = np.percentile(l_flat, 99) - np.percentile(l_flat, 1)

    # 4. 【核心修复】计算饱和度时增加阈值过滤和范围限制
    # 只有亮度 > 1 的像素才参与饱和度计算，防止除以近零值
    mask = l > 1.0 
    if np.sum(mask) > 0:
        # 饱和度 = 色度 / 亮度
        saturation = chroma[mask] / (l[mask] + 1e-5)
        # 饱和度通常不会超过 1.0 (Lab空间特殊，这里限制在合理范围)
        saturation = np.clip(saturation, 0, 10) 
        mu_s = np.mean(saturation)
    else:
        mu_s = 0.0

    # 5. 组合公式 (系数使用典型论文值)
    uciqe = 0.4680 * sigma_c + 0.2745 * con_l + 0.2576 * mu_s
    
    # 归一化到 0-1 范围 (如果结果还是很大，可以检查系数或尝试除以 100)
    # 大多数论文展示的是归一化后的值
    return uciqe / 100.0

