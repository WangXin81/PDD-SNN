import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16
from torchvision.models import VGG16_Weights
from torchvision import transforms
import numpy as np
import os
from typing import Optional
import re
import torch.fft

# 边缘损失
class EdgeLoss(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3).repeat(3, 1, 1, 1))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3).repeat(3, 1, 1, 1))
        self.l1_loss = nn.L1Loss()
        # self.sobel_x = sobel_x.view(1, 1, 3, 3).repeat(3, 1, 1, 1).to(device)
        # self.sobel_y = sobel_y.view(1, 1, 3, 3).repeat(3, 1, 1, 1).to(device)
        # self.l1_loss = nn.L1Loss()

    def forward(self, pred, target):
        pred = pred.float()
        target = target.float()
        grad_x_pred = F.conv2d(pred, self.sobel_x, padding=1, groups=3)
        grad_y_pred = F.conv2d(pred, self.sobel_y, padding=1, groups=3)
        grad_pred = torch.sqrt(grad_x_pred ** 2 + grad_y_pred ** 2 + 1e-8)
        grad_x_target = F.conv2d(target, self.sobel_x, padding=1, groups=3)
        grad_y_target = F.conv2d(target, self.sobel_y, padding=1, groups=3)
        grad_target = torch.sqrt(grad_x_target ** 2 + grad_y_target ** 2 + 1e-8)
        edge_loss = self.l1_loss(grad_pred, grad_target)
        return edge_loss


# VGG 感知损失（固定权重）

class VGGPerceptualLoss(nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        # vgg = vgg16(pretrained=True).features[:16]  # 使用前 16 层
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:16]
        self.vgg = vgg.eval()
        for param in self.vgg.parameters():
            param.requires_grad = False
        self.resize = resize
        
        # 新增：ImageNet 归一化参数
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        # 使用计算得到的水下图像统计值
        # self.register_buffer('mean', torch.tensor([0.1920, 0.5102, 0.5267]).view(1, 3, 1, 1))
        # self.register_buffer('std', torch.tensor([0.1904, 0.1966, 0.2047]).view(1, 3, 1, 1))

    def forward(self, x, y):
        x = torch.clamp(x, 0, 1) # 强制限制在 0-1 之间
        y = torch.clamp(y, 0, 1) # 强制限制在 0-1 之间
        # 输入必须是 [0,1] 的 3 通道
        if x.shape[1] != 3:
            x = x.repeat(1, 3, 1, 1)
            y = y.repeat(1, 3, 1, 1)
        
        # 新增：ImageNet 归一化
        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std
        
        return F.l1_loss(self.vgg(x), self.vgg(y))

# class VGGPerceptualLoss(nn.Module):
#     def __init__(self, resize=True):
#         super(VGGPerceptualLoss, self).__init__()
#         # vgg = vgg16(pretrained=True).features[:16]  # 使用前 16 层
#         vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:16]
#         self.vgg = vgg.eval()
#         for param in self.vgg.parameters():
#             param.requires_grad = False
#         self.resize = resize

#     def forward(self, x, y):
#         # 输入必须是 [0,1] 的 3 通道
#         if x.shape[1] != 3:
#             x = x.repeat(1, 3, 1, 1)
#             y = y.repeat(1, 3, 1, 1)
#         return F.l1_loss(self.vgg(x), self.vgg(y))


class DINOv2PerceptualLoss(nn.Module):
    """
    DINOv2 感知损失函数，用于处理高分辨率图像。

    (已更新：支持 L1 和 L2 的混合权重，并兼容 Python 3.8)

    参数:
        num_patches (int): 每个训练步骤中，从每张图像中随机采样的图块数量。
        patch_size (int): DINOv2模型所需的图块大小 (例如 224)。
        l1_weight (float): L1 损失的权重 (0.0 ~ 1.0)。L2 权重将是 (1.0 - l1_weight)。
                           1.0 = 纯 L1 (MAE), 0.0 = 纯 L2 (MSE)。
        model_name (str): 要加载的DINOv2模型。
    """

    def __init__(self, num_patches=4, patch_size=224, l1_weight=0.5, model_name='dinov2_vits14'):
        super().__init__()
        self.num_patches = num_patches
        self.patch_size = patch_size

        if not (0.0 <= l1_weight <= 1.0):
            raise ValueError("l1_weight 必须在 0.0 和 1.0 之间")

        self.l1_weight = l1_weight
        self.l2_weight = 1.0 - l1_weight

        if self.l1_weight == 1.0:
            self.p_str = "L1"
        elif self.l1_weight == 0.0:
            self.p_str = "L2 (MSE)"
        else:
            self.p_str = f"L1({self.l1_weight}) + L2({self.l2_weight})"

        print(f"Loading {model_name} for DINOv2PerceptualLoss...")

        # ============================================================
        # 🧩 Python 3.8 兼容性修复：遍历所有 dinov2/layers 下文件
        # ============================================================
        cache_dir = os.path.expanduser("~/.cache/torch/hub/facebookresearch_dinov2_main")
        layers_dir = os.path.join(cache_dir, "dinov2/layers")

        if os.path.isdir(layers_dir):
            print("正在检查 DINOv2/layers 模块的兼容性问题...")
            for fname in os.listdir(layers_dir):
                if not fname.endswith(".py"):
                    continue
                fpath = os.path.join(layers_dir, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        content = f.read()

                    modified = False

                    # 替换 "float | None" 和 "None | float"
                    if "float | None" in content or "None | float" in content:
                        content = re.sub(r"float\s*\|\s*None", "Optional[float]", content)
                        content = re.sub(r"None\s*\|\s*float", "Optional[float]", content)
                        modified = True

                    # 若使用 Optional 但无导入语句，则在顶部补上
                    if "Optional[" in content and "from typing import Optional" not in content:
                        content = "from typing import Optional\n" + content
                        modified = True

                    if modified:
                        with open(fpath, "w", encoding="utf-8") as f:
                            f.write(content)
                        print(f"已修复: {fname}")
                except Exception as e:
                    print(f"修复 {fname} 失败: {e}")
            print("所有 DINOv2/layers 文件已检查并修复。")
        else:
            print("未找到 DINOv2/layers 文件夹，可能模型尚未下载。")

        # ============================================================
        # ✅ 加载 DINOv2 模型
        # ============================================================
        self.dinov2 = torch.hub.load('facebookresearch/dinov2', model_name)
        self.dinov2.eval()

        for param in self.dinov2.parameters():
            param.requires_grad = False

        # ============================================================
        # ✅ 归一化层（与 DINOv2 训练时一致）
        # ============================================================
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        print(f"DINOv2PerceptualLoss ({self.p_str}) initialized. Ready.")

    def _extract_synced_patches(self, pred_img, target_img):
        """
        从预测图像和目标图像中提取同步的随机图块。
        """
        batch_size, _, H, W = pred_img.shape

        if H < self.patch_size or W < self.patch_size:
            raise ValueError(
                f"Image size ({H}, {W}) is smaller than patch size ({self.patch_size}). "
                "Consider using smaller patches or resizing."
            )

        all_tops = torch.randint(0, H - self.patch_size + 1, (batch_size, self.num_patches), device=pred_img.device)
        all_lefts = torch.randint(0, W - self.patch_size + 1, (batch_size, self.num_patches), device=pred_img.device)

        pred_patches_list = []
        target_patches_list = []

        for i in range(batch_size):
            for j in range(self.num_patches):
                top = all_tops[i, j].item()
                left = all_lefts[i, j].item()

                pred_patch = pred_img[i:i + 1, :, top:top + self.patch_size, left:left + self.patch_size]
                target_patch = target_img[i:i + 1, :, top:top + self.patch_size, left:left + self.patch_size]

                pred_patches_list.append(pred_patch)
                target_patches_list.append(target_patch)

        pred_batch = torch.cat(pred_patches_list, dim=0)
        target_batch = torch.cat(target_patches_list, dim=0)

        return pred_batch, target_batch

    def forward(self, pred_img, target_img):
        """
        计算损失。
        """
        if pred_img.shape[1] == 1:
            pred_img = pred_img.repeat(1, 3, 1, 1)
            target_img = target_img.repeat(1, 3, 1, 1)

        pred_batch, target_batch = self._extract_synced_patches(pred_img, target_img)

        pred_batch_norm = self.normalize(pred_batch)
        target_batch_norm = self.normalize(target_batch)

        pred_features = self.dinov2.forward_features(pred_batch_norm)['x_norm_patchtokens']
        target_features = self.dinov2.forward_features(target_batch_norm)['x_norm_patchtokens']

        loss_l1 = torch.tensor(0.0, device=pred_features.device)
        loss_l2 = torch.tensor(0.0, device=pred_features.device)

        if self.l1_weight > 0:
            loss_l1 = F.l1_loss(pred_features, target_features)

        if self.l2_weight > 0:
            loss_l2 = F.mse_loss(pred_features, target_features)

        loss = (self.l1_weight * loss_l1) + (self.l2_weight * loss_l2)
        return loss


# --- 测试模块 ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        print("\n--- 测试 混合L1/L2 (l1_weight=0.5) ---")
        dino_loss_mixed = DINOv2PerceptualLoss(num_patches=2, l1_weight=0.5).to(device)

        B, C, H, W = 2, 1, 512, 512
        clean_docs = torch.rand(B, C, H, W, device=device)
        repaired_docs = F.avg_pool2d(clean_docs, kernel_size=3, stride=1, padding=1)

        loss_mixed = dino_loss_mixed(repaired_docs, clean_docs)
        print(f"DINOv2 Perceptual Loss (Mixed 0.5/0.5): {loss_mixed.item()}")

    except Exception as e:
        print(f"\n发生错误: {e}")
        print("提示：请确认已安装 torch、torchvision、dinov2 等依赖。")


# ========== 低频损失（论文实现） ==========
class LowFrequencyLoss(nn.Module):
    def __init__(self, scale_factor=2, filter_size=5):
        super().__init__()
        self.scale_factor = scale_factor
        self.filter_size = filter_size

        # 创建高斯低通滤波器
        gaussian_filter = self._create_gaussian_filter(filter_size)
        self.register_buffer('gaussian_kernel', gaussian_filter)

    def _create_gaussian_filter(self, size, sigma=1.0):
        """创建高斯低通滤波器"""
        coords = torch.arange(size).float() - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g.unsqueeze(0) * g.unsqueeze(1)
        g = g / g.sum()
        return g.unsqueeze(0).unsqueeze(0)  # (1, 1, size, size)

    def low_pass_filter(self, x):
        """应用低通滤波"""
        # 对每个通道应用相同的滤波器
        kernel = self.gaussian_kernel.to(x.device)
        kernel = kernel.repeat(x.size(1), 1, 1, 1)
        padding = self.filter_size // 2
        return F.conv2d(x, kernel, padding=padding, groups=x.size(1))
    
    def forward(self, hr_l_channel, generated_lr_l_channel):
        """
        hr_l_channel: HR图像的L通道
        generated_lr_l_channel: 生成LR图像的L通道
        """
        # 对HR图像应用低通滤波
        hr_low_pass = self.low_pass_filter(hr_l_channel)
        
        # [核心修复]：不使用 scale_factor，直接使用 generated_lr 的实际尺寸
        # 避免 256/3 = 85.333 导致的对齐误差
        target_size = generated_lr_l_channel.shape[-2:]
        
        hr_low_pass_down = F.interpolate(hr_low_pass,
                                         size=target_size, # 强制对齐
                                         mode='bicubic',
                                         align_corners=False)

        # 对生成LR图像应用低通滤波
        lr_low_pass = self.low_pass_filter(generated_lr_l_channel)

        # 计算L1损失
        return F.l1_loss(lr_low_pass, hr_low_pass_down)

#     def forward(self, hr_l_channel, generated_lr_l_channel):
#         """
#         hr_l_channel: HR图像的L通道
#         generated_lr_l_channel: 生成LR图像的L通道
#         """
#         # 对HR图像应用低通滤波和下采样
#         hr_low_pass = self.low_pass_filter(hr_l_channel)
#         hr_low_pass_down = F.interpolate(hr_low_pass,
#                                          scale_factor=1 / self.scale_factor,
#                                          mode='bicubic')

#         # 对生成LR图像应用低通滤波
#         lr_low_pass = self.low_pass_filter(generated_lr_l_channel)

#         # 计算L1损失（论文使用L1）
#         return F.l1_loss(lr_low_pass, hr_low_pass_down)


class MultiBranchWaterLoss(nn.Module):
    """
    Loss where spatial_out/freq_out are upsampled only for loss calculation.
    - spatial_out, freq_out may be LR (e.g. 128x128)
    - fusion_out is expected to be HR (e.g. 256x256)
    - hr is HR ground-truth
    """

    def __init__(self, l1_weight=1.0, spatial_weight=0.25, freq_weight=0.25, use_perceptual=True, device='cuda'):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.l1_weight = l1_weight
        self.spatial_weight = spatial_weight
        self.freq_weight = freq_weight
        self.use_perceptual = use_perceptual
        self.device = device
        if self.use_perceptual:
            self.perc = VGGPerceptualLoss().to(device)

    def forward(self, spatial_out, freq_out, fusion_out, hr):
        """
        spatial_out: [B, C, H_lr, W_lr]  (可能 LR)
        freq_out:    [B, C, H_lr, W_lr]  (可能 LR)
        fusion_out:  [B, C, H_hr, W_hr]  (应为 HR)
        hr:          [B, C, H_hr, W_hr]  (GT)
        """

        losses = {}

        # --- 1. check fusion_out vs hr ---
        if fusion_out.shape[-2:] != hr.shape[-2:]:
            # 强烈建议fusion_out与hr一致；若不一致，可上采样 fusion_out
            fusion_out = F.interpolate(fusion_out, size=hr.shape[-2:], mode='bicubic', align_corners=False)

        # --- 2. 上采样 spatial/freq 到 HR（仅用于 loss） ---
        if spatial_out.shape[-2:] != hr.shape[-2:]:
            spatial_up = F.interpolate(spatial_out, size=hr.shape[-2:], mode='bicubic', align_corners=False)
        else:
            spatial_up = spatial_out

        if freq_out.shape[-2:] != hr.shape[-2:]:
            freq_up = F.interpolate(freq_out, size=hr.shape[-2:], mode='bicubic', align_corners=False)
        else:
            freq_up = freq_out

        # --- 3. 计算 L1 loss ---
        loss_fusion_l1 = self.l1(fusion_out, hr)
        loss_spatial_l1 = self.l1(spatial_up, hr)
        loss_freq_l1 = self.l1(freq_up, hr)

        losses['loss_fusion_l1'] = loss_fusion_l1.item()
        losses['loss_spatial_l1'] = loss_spatial_l1.item()
        losses['loss_freq_l1'] = loss_freq_l1.item()

        # --- 4. 可选感知损失（在 HR 空间） ---
        loss_fusion_perc = torch.tensor(0.0, device=self.device)
        loss_spatial_perc = torch.tensor(0.0, device=self.device)
        if self.use_perceptual:
            loss_fusion_perc = self.perc(fusion_out, hr)
            loss_spatial_perc = self.perc(spatial_up, hr)
            losses['loss_fusion_perc'] = float(loss_fusion_perc)
            losses['loss_spatial_perc'] = float(loss_spatial_perc)

        # --- 5. 权重合并（示例） ---
        total_loss = (self.l1_weight * loss_fusion_l1 +
                      self.spatial_weight * loss_spatial_l1 +
                      self.freq_weight * loss_freq_l1)

        # 把感知损失按需加上（给 fusion 主分支更高权重）
        if self.use_perceptual:
            total_loss = total_loss + 0.2 * loss_fusion_perc + 0.05 * loss_spatial_perc

        losses['total_loss'] = float(total_loss)

        return total_loss, losses


# def gradient_loss(pred, target):
#     sobel_x = torch.tensor([[1, 0, -1],
#                             [2, 0, -2],
#                             [1, 0, -1]], dtype=torch.float32, device=pred.device).view(1, 1, 3, 3)
#     sobel_y = torch.tensor([[1, 2, 1],
#                             [0, 0, 0],
#                             [-1, -2, -1]], dtype=torch.float32, device=pred.device).view(1, 1, 3, 3)
#     # 多通道逐通道计算
#     pred_x = F.conv2d(pred, sobel_x.repeat(pred.size(1), 1, 1, 1), padding=1, groups=pred.size(1))
#     pred_y = F.conv2d(pred, sobel_y.repeat(pred.size(1), 1, 1, 1), padding=1, groups=pred.size(1))
#     targ_x = F.conv2d(target, sobel_x.repeat(target.size(1), 1, 1, 1), padding=1, groups=target.size(1))
#     targ_y = F.conv2d(target, sobel_y.repeat(target.size(1), 1, 1, 1), padding=1, groups=target.size(1))
#     return F.l1_loss(pred_x, targ_x) + F.l1_loss(pred_y, targ_y)



class GradientLoss(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        # 定义 Sobel 算子
        # X方向: 检测垂直边缘
        kernel_x = torch.tensor([[1, 0, -1], 
                                 [2, 0, -2], 
                                 [1, 0, -1]], dtype=torch.float32)
        # Y方向: 检测水平边缘
        kernel_y = torch.tensor([[1, 2, 1], 
                                 [0, 0, 0], 
                                 [-1, -2, -1]], dtype=torch.float32)
        
        # 将核 reshape 为 (1, 1, 3, 3) 并注册为 buffer
        # register_buffer 可以让 tensor 随模型自动移动到 GPU，且不作为可训练参数保存
        self.register_buffer('kernel_x', kernel_x.view(1, 1, 3, 3))
        self.register_buffer('kernel_y', kernel_y.view(1, 1, 3, 3))
        
        self.l1_loss = nn.L1Loss()

    def forward(self, pred, target):
        b, c, h, w = pred.shape
        
        # 动态适配通道数
        # expand 不会分配新内存，比 repeat 更高效
        kx = self.kernel_x.expand(c, 1, 3, 3)
        ky = self.kernel_y.expand(c, 1, 3, 3)

        # 计算梯度 (Depthwise Conv)
        pred_x = F.conv2d(pred, kx, padding=1, groups=c)
        pred_y = F.conv2d(pred, ky, padding=1, groups=c)
        
        # 目标梯度无需计算梯度，节省显存
        with torch.no_grad():
            targ_x = F.conv2d(target, kx, padding=1, groups=c)
            targ_y = F.conv2d(target, ky, padding=1, groups=c)
        
        # 计算 Loss
        loss = self.l1_loss(pred_x, targ_x) + self.l1_loss(pred_y, targ_y)
        return loss



# 辅助损失与工具函数：MMD实现（可微）
def _pairwise_sq_dists(x, y):
    # x: [m, D], y: [n, D]
    xx = (x * x).sum(dim=1, keepdim=True)  # [m,1]
    yy = (y * y).sum(dim=1, keepdim=True)  # [n,1]
    return xx - 2.0 * (x @ y.t()) + yy.t()


def gaussian_kernel_matrix(x, y, sigmas=(0.5, 1.0, 2.0, 4.0)):
    dist_sq = _pairwise_sq_dists(x, y)  # [m, n]
    K = 0.0
    for sigma in sigmas:
        gamma = 1.0 / (2.0 * (sigma ** 2))
        K = K + torch.exp(-gamma * dist_sq)
    return K


def mmd_rbf(F, G, sigmas=(0.5, 1.0, 2.0, 4.0), sample_pixels=1024):
    """
    计算 MMD（RBF核）在 F 与 G 上，支持对像素采样以降低内存。
    F,G: [B, C, H, W]
    """
    if F is None or G is None:
        return torch.tensor(0.0, device=F.device if F is not None else 'cpu')
    # 使用 L通道或传入的张量直接展平
    f = F.view(F.shape[0], -1)  # [B, D]
    g = G.view(G.shape[0], -1)
    # 若特征维度太大，随机采样若干位置以降低计算量（按列采样）
    D = f.shape[1]
    if D > sample_pixels:
        idx = torch.randperm(D, device=f.device)[:sample_pixels]
        f = f[:, idx]
        g = g[:, idx]
    K_ff = gaussian_kernel_matrix(f, f, sigmas)
    K_gg = gaussian_kernel_matrix(g, g, sigmas)
    K_fg = gaussian_kernel_matrix(f, g, sigmas)
    return K_ff.mean() + K_gg.mean() - 2.0 * K_fg.mean()


# def calculate_psnr(img1, img2, data_range=1.0):
#     """计算PSNR"""
#     img1 = torch.clamp(img1, 0, data_range)
#     img2 = torch.clamp(img2, 0, data_range)
#     mse = torch.mean((img1 - img2) ** 2)
#     if mse == 0:
#         return float('inf')
#     return 20 * torch.log10(data_range / torch.sqrt(mse)).item()


import torch
import numpy as np
import math
from skimage.metrics import structural_similarity as ssim

# def calculate_psnr(img1, img2, crop_border=0, test_y_channel=False):
#     # 1. 统一转为 Numpy
#     if isinstance(img1, torch.Tensor): img1 = img1.cpu().detach().numpy()
#     if isinstance(img2, torch.Tensor): img2 = img2.cpu().detach().numpy()
    
#     # 2. 自动修正数值范围到 [0, 255]
#     # 如果数据是 [0, 1] (max < 1.1), 乘以 255
#     if img1.max() <= 1.1: img1 = (img1 * 255.0).round()
#     if img2.max() <= 1.1: img2 = (img2 * 255.0).round()
        
#     img1, img2 = img1.astype(np.float64), img2.astype(np.float64)

#     # 3. 计算 MSE
#     mse = np.mean((img1 - img2) ** 2)
#     if mse == 0: return float('inf')
    
#     # 4. 计算 PSNR (现在用 255 做分子是正确的了)
#     return 20 * math.log10(255.0 / math.sqrt(mse))


# def calculate_psnr(img1, img2, crop_border=0, test_y_channel=False):
#     # 1. 统一转为 Numpy
#     if isinstance(img1, torch.Tensor): img1 = img1.cpu().detach().numpy()
#     if isinstance(img2, torch.Tensor): img2 = img2.cpu().detach().numpy()
    
#     # 2. 自动修正数值范围到 [0, 255]
#     # 如果数据是 [0, 1] (max < 1.1), 乘以 255
# #     if img1.max() <= 1.1: img1 = (img1 * 255.0).round()
# #     if img2.max() <= 1.1: img2 = (img2 * 255.0).round()
        
# #     img1, img2 = img1.astype(np.float64), img2.astype(np.float64)
#     img1 = img1.astype(np.float64) * 255.0
#     img2 = img2.astype(np.float64) * 255.0

#     # 3. 计算 MSE
#     mse = np.mean((img1 - img2) ** 2)
#     if mse == 0: return float('inf')
    
#     # 4. 计算 PSNR (现在用 255 做分子是正确的了)
#     return 20 * math.log10(255.0 / math.sqrt(mse))


# def calculate_ssim(img1, img2, data_range=1.0):
#     """计算SSIM"""
#     try:
#         import piq
#         if img1.dim() == 3:
#             img1 = img1.unsqueeze(0)
#         if img2.dim() == 3:
#             img2 = img2.unsqueeze(0)
#         return piq.ssim(img1, img2, data_range=data_range).item()
#     except ImportError:
#         if img1.dim() == 3:
#             img1 = img1.unsqueeze(0)
#         if img2.dim() == 3:
#             img2 = img2.unsqueeze(0)
#         window_size = 11
#         C1 = (0.01 * data_range) ** 2
#         C2 = (0.03 * data_range) ** 2
#         mu1 = F.avg_pool2d(img1, window_size, 1, window_size // 2)
#         mu2 = F.avg_pool2d(img2, window_size, 1, window_size // 2)
#         mu1_sq = mu1.pow(2)
#         mu2_sq = mu2.pow(2)
#         mu1_mu2 = mu1 * mu2
#         sigma1_sq = F.avg_pool2d(img1 * img1, window_size, 1, window_size // 2) - mu1_sq
#         sigma2_sq = F.avg_pool2d(img2 * img2, window_size, 1, window_size // 2) - mu2_sq
#         sigma12 = F.avg_pool2d(img1 * img2, window_size, 1, window_size // 2) - mu1_mu2
#         ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
#         return ssim_map.mean().item()


def to_y_channel(img):
    """
    将图像转换到 Y 通道 (亮度通道)
    img: Numpy array (H, W, C) 或 (H, W), 范围 [0, 255], RGB顺序
    返回: Numpy array (H, W), 范围 [0, 255]
    """
    img = img.astype(np.float64)
    if img.ndim == 3 and img.shape[2] == 3:
        # 使用 Matlab 标准系数转换 RGB -> Y
        # 公式来源: BasicSR / SwinIR 标准
        # img = np.dot(img, [24.966, 128.553, 65.481]) / 255.0 + 16.0
        img = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    return img

# def calculate_psnr(img1, img2, crop_border=0, test_y_channel=False):
#     """
#     计算 PSNR (学术标准版)
#     参数:
#         img1, img2: Tensor 或 Numpy
#         crop_border: 需要切除的边缘像素数 (scale)
#         test_y_channel: 是否只计算 Y 通道
#     """
#     # 1. 统一转为 Numpy
#     if isinstance(img1, torch.Tensor): img1 = img1.cpu().detach().numpy()
#     if isinstance(img2, torch.Tensor): img2 = img2.cpu().detach().numpy()
    
#     # 2. 维度处理 (CHW -> HWC)
#     if img1.ndim == 3 and img1.shape[0] in [1, 3]: img1 = img1.transpose(1, 2, 0)
#     if img2.ndim == 3 and img2.shape[0] in [1, 3]: img2 = img2.transpose(1, 2, 0)

#     # 3. 智能范围修正 (确保是 [0, 255] float)
#     img1 = img1.astype(np.float64)
#     img2 = img2.astype(np.float64)
    
#     if img1.max() <= 1.0:
#         img1 = img1 * 255.0
#     # 否则认为是 0-255，直接截断
#     img1 = np.clip(img1, 0, 255.0)

#     if img2.max() <= 1.0:
#         img2 = img2 * 255.0
#     img2 = np.clip(img2, 0, 255.0)
    
# #     if img1.mean() < 2.0:
# #         img1 = np.clip(img1, 0, 1) * 255.0
# #     else:
# #         img1 = np.clip(img1, 0, 255.0)

# #     if img2.mean() < 2.0:
# #         img2 = np.clip(img2, 0, 1) * 255.0
# #     else:
# #         img2 = np.clip(img2, 0, 255.0)
    
#     # if img1.max() <= 1.1: img1 = img1 * 255.0
#     # if img2.max() <= 1.1: img2 = img2 * 255.0
    
#     # 4. 转 Y 通道 (关键步骤)
#     if test_y_channel:
#         img1 = to_y_channel(img1)
#         img2 = to_y_channel(img2)
    
#     # 5. 切除边缘 (关键步骤)
#     if crop_border != 0:
#         img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
#         img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]
        
#     # 6. 计算 MSE 和 PSNR
#     mse = np.mean((img1 - img2) ** 2)
#     if mse == 0: return float('inf')
    
#     return 20. * math.log10(255.0 / math.sqrt(mse))

# def calculate_ssim(img1, img2, crop_border=0, test_y_channel=False):
#     """
#     计算 SSIM (学术标准版)
#     """
#     # 1. 统一转为 Numpy
#     if isinstance(img1, torch.Tensor): img1 = img1.cpu().detach().numpy()
#     if isinstance(img2, torch.Tensor): img2 = img2.cpu().detach().numpy()
    
#     # 2. 维度处理 (CHW -> HWC)
#     if img1.ndim == 3 and img1.shape[0] in [1, 3]: img1 = img1.transpose(1, 2, 0)
#     if img2.ndim == 3 and img2.shape[0] in [1, 3]: img2 = img2.transpose(1, 2, 0)
        
#     # 3. 智能范围修正
#     img1 = img1.astype(np.float64)
#     img2 = img2.astype(np.float64)
    
#     if img1.max() <= 1.0:
#         img1 = img1 * 255.0
#     # 否则认为是 0-255，直接截断
#     img1 = np.clip(img1, 0, 255.0)

#     if img2.max() <= 1.0:
#         img2 = img2 * 255.0
#     img2 = np.clip(img2, 0, 255.0)
    
# #     if img1.mean() < 2.0:
# #         img1 = np.clip(img1, 0, 1) * 255.0
# #     else:
# #         img1 = np.clip(img1, 0, 255.0)

# #     if img2.mean() < 2.0:
# #         img2 = np.clip(img2, 0, 1) * 255.0
# #     else:
# #         img2 = np.clip(img2, 0, 255.0)
    
    
#     # if img1.max() <= 1.1: img1 = img1 * 255.0
#     # if img2.max() <= 1.1: img2 = img2 * 255.0

#     # 4. 转 Y 通道
#     if test_y_channel:
#         img1 = to_y_channel(img1)
#         img2 = to_y_channel(img2)
        
#     # 5. 切除边缘
#     if crop_border != 0:
#         img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
#         img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]
    
#     input_channel_axis = 2 if img1.ndim == 3 else None
    
#     return ssim(img1, img2, data_range=255, channel_axis=input_channel_axis)
    
#     # 6. 计算 SSIM
#     # 兼容不同版本的 skimage 参数
#     # is_channel_last = (img1.ndim == 3)
#     # try:
#     #     # 新版 skimage 写法
#     #     return ssim(img1, img2, data_range=255, gaussian_weights=True, 
#     #                 channel_axis=-1 if is_channel_last else None)
#     # except TypeError:
#     #     # 旧版 skimage 写法 (multichannel 参数)
#     #     return ssim(img1, img2, data_range=255, gaussian_weights=True, 
#     #                 multichannel=is_channel_last)


def calculate_psnr(img1, img2, crop_border=0, test_y_channel=False, data_range=255.0):
    """
    学术标准 PSNR 计算
    """
    # 1. Tensor -> Numpy
    if isinstance(img1, torch.Tensor): img1 = img1.cpu().detach().numpy()
    if isinstance(img2, torch.Tensor): img2 = img2.cpu().detach().numpy()
    
    # 2. 维度调整 (CHW -> HWC)
    if img1.ndim == 3 and img1.shape[0] in [1, 3]: img1 = img1.transpose(1, 2, 0)
    if img2.ndim == 3 and img2.shape[0] in [1, 3]: img2 = img2.transpose(1, 2, 0)

    # 3. 类型转换
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    # 4. 范围修正 (如果不指定 data_range=255，这里不会触发，所以必须指定)
    if data_range == 255.0 and img1.max() <= 1.1:
        img1 = img1 * 255.0
    if data_range == 255.0 and img2.max() <= 1.1:
        img2 = img2 * 255.0
        
    # 5. 转 Y 通道
    if test_y_channel:
        img1 = to_y_channel(img1)
        img2 = to_y_channel(img2)
    
    # 6. 切边
    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]
        
    # 7. 计算
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0: return float('inf')
    
    return 20. * math.log10(data_range / math.sqrt(mse))


def calculate_ssim(img1, img2, crop_border=0, test_y_channel=False, data_range=255.0):
    """
    学术标准 SSIM 计算
    """
    if isinstance(img1, torch.Tensor): img1 = img1.cpu().detach().numpy()
    if isinstance(img2, torch.Tensor): img2 = img2.cpu().detach().numpy()
    
    if img1.ndim == 3 and img1.shape[0] in [1, 3]: img1 = img1.transpose(1, 2, 0)
    if img2.ndim == 3 and img2.shape[0] in [1, 3]: img2 = img2.transpose(1, 2, 0)
        
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    if data_range == 255.0 and img1.max() <= 1.1:
        img1 = img1 * 255.0
    if data_range == 255.0 and img2.max() <= 1.1:
        img2 = img2 * 255.0

    if test_y_channel:
        img1 = to_y_channel(img1)
        img2 = to_y_channel(img2)
        
    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]
    
    input_channel_axis = 2 if img1.ndim == 3 else None
    
    # 显式传入 data_range=255
    return ssim(img1, img2, data_range=data_range, channel_axis=input_channel_axis)




# CT-Loss 对比损失函数
def ct_loss(anchor_feat, pos_feat, neg_feat):
    """
    对比损失函数 (Contrastive Loss)
    参数:
        anchor_feat: 锚点特征 [B, C, H, W] 或 [B, D]
        pos_feat: 正样本特征 [B, C, H, W] 或 [B, D]
        neg_feat: 负样本特征 [B, C, H, W] 或 [B, D]
    返回:
        ct_loss: 对比损失值
    """
    # 如果输入是4D特征图，进行全局平均池化
    if anchor_feat.ndim == 4:
        anchor = anchor_feat.mean(dim=[2, 3])  # [B, C]
        pos = pos_feat.mean(dim=[2, 3])  # [B, C]
        neg = neg_feat.mean(dim=[2, 3])  # [B, C]
    else:
        anchor, pos, neg = anchor_feat, pos_feat, neg_feat

    # 计算正样本和负样本的距离
    d_pos = torch.norm(pos - anchor, dim=1)  # [B]
    d_neg = torch.norm(neg - anchor, dim=1)  # [B]

    # 对比损失公式
    loss = (torch.log1p(torch.exp(d_pos)) + torch.log1p(torch.exp(-d_neg))).mean()

    return loss


# ============ 新增：使用判别器特征的CT Loss ============
class DiscriminatorFeatureCTLoss(nn.Module):
    """
    使用判别器特征的对比损失
    在特征空间进行对比学习，专门学习退化特征
    """

    def __init__(self, discriminator_module, use_multiscale=True, margin=0.8):
        super().__init__()
        self.discriminator = discriminator_module
        self.use_multiscale = use_multiscale
        self.margin = margin

    def forward(self, anchor, positive, negative):
        """
        anchor: 生成的LR (G(HR))
        positive: 真实LR (真实退化)
        negative: 双三次LR (简单退化)
        """
        # 提取多尺度特征
        _, anchor_features = self.discriminator(anchor, return_features=True)
        _, positive_features = self.discriminator(positive, return_features=True)
        _, negative_features = self.discriminator(negative, return_features=True)

        total_ct_loss = 0
        num_layers = 0

        for feat_anchor, feat_pos, feat_neg in zip(anchor_features, positive_features, negative_features):
            # 保持空间信息，展平为特征向量
            B, C, H, W = feat_anchor.shape
            feat_anchor_flat = feat_anchor.view(B, C, -1).mean(dim=2)  # [B, C] - 空间平均
            feat_pos_flat = feat_pos.view(B, C, -1).mean(dim=2)
            feat_neg_flat = feat_neg.view(B, C, -1).mean(dim=2)

            # 归一化特征向量
            feat_anchor_norm = F.normalize(feat_anchor_flat, p=2, dim=1)
            feat_pos_norm = F.normalize(feat_pos_flat, p=2, dim=1)
            feat_neg_norm = F.normalize(feat_neg_flat, p=2, dim=1)

            # 余弦相似度计算
            sim_pos = (feat_anchor_norm * feat_pos_norm).sum(dim=1)  # [B]
            sim_neg = (feat_anchor_norm * feat_neg_norm).sum(dim=1)  # [B]

            # 对比损失：拉近与真实退化，推远与双三次
            # 希望 sim_pos 接近 1，sim_neg 接近 -1
            pos_loss = F.relu(self.margin - sim_pos)  # 如果sim_pos < margin，则惩罚
            neg_loss = F.relu(sim_neg - 0.2)
            # neg_loss = F.relu(sim_neg + (1 - self.margin))  # 如果sim_neg > -(1-margin)，则惩罚

            layer_loss = (pos_loss + neg_loss).mean()

            total_ct_loss += layer_loss
            num_layers += 1

        return total_ct_loss / num_layers if num_layers > 0 else total_ct_loss


class DiscriminatorHighLevelCTLoss(nn.Module):
    """只使用判别器的高层特征，更关注整体退化风格"""

    def __init__(self, discriminator_module, layer_indices=[-2, -1], margin=0.8):
        super().__init__()
        self.discriminator = discriminator_module
        self.layer_indices = layer_indices
        self.margin = margin

    def forward(self, anchor, positive, negative):
        # 提取特征
        _, anchor_features = self.discriminator(anchor, return_features=True)
        _, positive_features = self.discriminator(positive, return_features=True)
        _, negative_features = self.discriminator(negative, return_features=True)

        total_ct_loss = 0
        valid_layers = 0

        # 只使用指定的高层特征层
        for idx in self.layer_indices:
            if idx < len(anchor_features) and idx >= 0:
                feat_anchor = anchor_features[idx]
                feat_pos = positive_features[idx]
                feat_neg = negative_features[idx]

                # 高层特征通常已经包含丰富的语义信息
                B, C = feat_anchor.shape[0], feat_anchor.shape[1]
                feat_anchor_flat = feat_anchor.reshape(B, -1)
                feat_pos_flat = feat_pos.reshape(B, -1)
                feat_neg_flat = feat_neg.reshape(B, -1)

                # 归一化特征向量
                feat_anchor_norm = F.normalize(feat_anchor_flat, p=2, dim=1)
                feat_pos_norm = F.normalize(feat_pos_flat, p=2, dim=1)
                feat_neg_norm = F.normalize(feat_neg_flat, p=2, dim=1)

                # 计算相似度
                sim_pos = (feat_anchor_norm * feat_pos_norm).sum(dim=1)
                sim_neg = (feat_anchor_norm * feat_neg_norm).sum(dim=1)

                # 对比损失
                pos_loss = F.relu(self.margin - sim_pos)
                neg_loss = F.relu(sim_neg + (1 - self.margin))

                layer_loss = (pos_loss + neg_loss).mean()
                total_ct_loss += layer_loss
                valid_layers += 1

        return total_ct_loss / valid_layers if valid_layers > 0 else total_ct_loss
    

class FocalFrequencyLoss(nn.Module):
    """
    频域损失 (Focal Frequency Loss)
    无需第三方库，直接使用 PyTorch 原生 FFT 实现。
    """
    def __init__(self, loss_weight=1.0, alpha=1.0):
        super(FocalFrequencyLoss, self).__init__()
        self.loss_weight = loss_weight
        self.alpha = alpha

    def forward(self, pred, target):
        # 1. 快速傅里叶变换 (FFT)
        # dim=(-2, -1) 表示只在空间维度 (H, W) 上进行变换
        pred_freq = torch.fft.fftn(pred, dim=(-2, -1))
        target_freq = torch.fft.fftn(target, dim=(-2, -1))
        
        # 2. 计算频谱距离 (平方欧氏距离)
        # abs() 计算复数的模长
        freq_distance = (pred_freq - target_freq).abs().pow(2)
        
        # 3. Focal 动态权重机制
        # 差异越大的频率点，权重越高
        weight_matrix = freq_distance / (freq_distance.max().detach() + 1e-8)
        weight_matrix = weight_matrix.pow(self.alpha)
        
        # 4. 计算加权后的损失
        loss = (freq_distance * weight_matrix).mean()
        
        return self.loss_weight * loss
    
import torchvision.models as models    
class SRVGGPerceptualLoss(nn.Module):
    """
    专门用于超分辨率重建的 VGG Loss (ESRGAN 标准)
    特点：
    1. 默认使用 relu3_4 (feature_layer=16) 之前的层，专注于纹理恢复。
    2. 使用 L1 Loss 计算特征距离 (比 MSE 更能产生锐利边缘)。
    3. 内置 ImageNet 归一化，无需外部处理。
    """
    def __init__(self, feature_layer=16, use_l1=True, device='cuda'):
        """
        feature_layer: 
            34 -> relu5_4 (更深，关注整体结构，适合退化模块)
            16 -> relu3_4 (【超分黄金标准】，更浅，极度关注纹理细节)
        """
        super(SRVGGPerceptualLoss, self).__init__()
        
        # 加载预训练 VGG19
        vgg = models.vgg19(pretrained=True)
        
        # 截取到指定的层 (feature_layer)
        # relu3_4 大约在第 16 层，relu5_4 大约在第 34 层
        model_list = list(vgg.features.children())
        self.features = nn.Sequential(*model_list[:feature_layer+1]).to(device).eval()
        
        # 冻结参数，不参与训练
        for param in self.features.parameters():
            param.requires_grad = False
            
        # VGG 标准化参数 (ImageNet Mean/Std)
        self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device))
        self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device))
        
        self.criterion = nn.L1Loss() if use_l1 else nn.MSELoss()

    def normalize(self, x):
        # 假设输入 x 在 [0, 1] 之间
        return (x - self.mean) / self.std

    def forward(self, pred, target):
        # 1. 归一化 (Input: [0, 1] RGB)
        pred_norm = self.normalize(pred)
        target_norm = self.normalize(target)
        
        # 2. 提取特征
        pred_feat = self.features(pred_norm)
        with torch.no_grad():
            target_feat = self.features(target_norm)
            
        # 3. 计算损失 (L1)
        return self.criterion(pred_feat, target_feat)