# kernel_noise_extractor.py - 模糊核和噪声提取功能
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import logging
from tqdm import tqdm
from glob import glob
import random

# 尝试导入KLSNN1模块，如果失败则使用备用方案
try:
    from KLSNN1 import KernelGAN_UnpairedLR, LRPatchDataset, set_seed

    KLSNN_AVAILABLE = True
except ImportError as e:
    print(f"警告: 无法导入KLSNN1模块: {e}")
    print("将使用简化版本的核提取功能")
    KLSNN_AVAILABLE = False


    # 提供简化的set_seed函数
    def set_seed(seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

# 配置日志
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


# ========== 噪声提取和加载工具类 ==========
class NoiseExtractor:
    @staticmethod
    def rgb_to_lab(rgb_tensor):
        """RGB转LAB颜色空间"""
        device = rgb_tensor.device
        imgs = rgb_tensor.permute(0, 2, 3, 1).cpu().numpy() * 255.0
        labs = []
        for img in imgs:
            img_clamped = np.clip(img, 0, 255).astype(np.uint8)
            lab = cv2.cvtColor(img_clamped, cv2.COLOR_RGB2LAB).astype(np.float32)
            L = lab[..., 0] / 100.0
            A = (lab[..., 1] - 128.0) / 127.0
            B = (lab[..., 2] - 128.0) / 127.0
            L = np.clip(L, 0.0, 1.0)
            A = np.clip(A, -1.0, 1.0)
            B = np.clip(B, -1.0, 1.0)
            lab_normalized = np.stack([L, A, B], axis=-1)
            labs.append(torch.from_numpy(lab_normalized).permute(2, 0, 1).to(device))
        return torch.stack(labs).to(device)

    @staticmethod
    def extract_noise_patches(lr_image_dir, save_dir, patch_size=32, max_var=0.001, min_mean=0.01, num_patches=1000):
        """从LR图像的平滑区域提取噪声补丁"""
        os.makedirs(save_dir, exist_ok=True)
        lr_paths = [os.path.join(lr_image_dir, f) for f in os.listdir(lr_image_dir)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', 'bmp'))]

        patch_count = 0
        transform = transforms.ToTensor()

        for path in tqdm(lr_paths, desc="提取噪声补丁"):
            if patch_count >= num_patches:
                break
            img = Image.open(path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0)  # (1,3,H,W)

            # 转换为LAB并提取L通道
            lab_tensor = NoiseExtractor.rgb_to_lab(img_tensor)
            l_channel = lab_tensor[:, 0:1, :, :]  # (1,1,H,W)

            # 滑动窗口寻找平滑区域
            h, w = l_channel.shape[2], l_channel.shape[3]
            for i in range(0, h - patch_size, patch_size // 2):
                for j in range(0, w - patch_size, patch_size // 2):
                    if patch_count >= num_patches:
                        break
                    patch = l_channel[:, :, i:i + patch_size, j:j + patch_size]
                    patch_var = torch.var(patch)
                    patch_mean = torch.mean(patch)

                    # 筛选低方差（平滑区域）且均值符合条件的补丁作为噪声样本
                    if patch_var < max_var and patch_mean > min_mean:
                        # 假设平滑区域的像素波动为噪声
                        noise_patch = patch - torch.mean(patch)
                        save_path = os.path.join(save_dir, f"noise_{patch_count}.pt")
                        torch.save(noise_patch, save_path)
                        patch_count += 1
                if patch_count >= num_patches:
                    break
        logger.info(f"已提取 {patch_count} 个噪声补丁到 {save_dir}")


class NoiseDataset(Dataset):
    """噪声补丁数据集"""

    def __init__(self, noise_dir):
        self.noise_paths = glob(os.path.join(noise_dir, "*.pt"))
        if not self.noise_paths:
            raise ValueError(f"未在 {noise_dir} 找到噪声补丁文件，请先运行噪声提取")

    def __len__(self):
        return len(self.noise_paths)

    def __getitem__(self, idx):
        noise = torch.load(self.noise_paths[idx])
        # 确保返回的是4维张量 (1, channel, height, width)
        if noise.dim() == 2:
            noise = noise.unsqueeze(0)  # 添加通道维度
        return noise


# ========== 单图像数据集类（用于多核训练） ==========
class SingleImagePatchDataset(Dataset):
    """单图像patch数据集，用于为每张图像单独训练KernelGAN"""

    def __init__(self, image_path, patch_size=64, num_patches=50):
        self.image_path = image_path
        self.patch_size = patch_size
        self.num_patches = num_patches

        # 加载图像
        self.image = Image.open(image_path).convert('RGB')
        self.img_tensor = transforms.ToTensor()(self.image)  # (3, H, W)

        # 预计算可能的patch位置
        self.h, self.w = self.img_tensor.shape[1], self.img_tensor.shape[2]
        self.positions = []

        for i in range(0, self.h - patch_size, patch_size // 4):
            for j in range(0, self.w - patch_size, patch_size // 4):
                self.positions.append((i, j))

        # 如果位置不够，重复一些位置
        while len(self.positions) < num_patches:
            self.positions.extend(self.positions[:num_patches - len(self.positions)])

    def __len__(self):
        return self.num_patches

    def __getitem__(self, idx):
        i, j = self.positions[idx % len(self.positions)]
        patch = self.img_tensor[:, i:i + self.patch_size, j:j + self.patch_size]
        return patch


# ========== KernelGAN 模糊核估计器 ==========
class KernelGANEstimator:
    def __init__(self, kernel_size=15, device='cuda'):
        self.device = device
        self.kernel_size = kernel_size
        if not KLSNN_AVAILABLE:
            logger.warning("KLSNN1不可用，KernelGANEstimator将无法正常工作")
            return

        self.trainer = KernelGAN_UnpairedLR(
            kernel_size=kernel_size,
            scale=2,
            patch_size=64,
            batch_size=32,
            device=device
        )

    def train(self, image_dir, epochs=50):
        """训练KernelGAN"""
        if not KLSNN_AVAILABLE:
            logger.error("KLSNN1不可用，无法训练KernelGAN")
            return None

        dataset = LRPatchDataset(
            image_dir=image_dir,
            patch_size=64,
            num_patches_per_image=10
        )
        print(f"Starting KernelGAN training with {len(dataset)} samples...")
        self.trainer.train(dataset, epochs=epochs)

    def estimate_kernel(self, image_path):
        """估计单个图像的模糊核"""
        if not KLSNN_AVAILABLE:
            logger.error("KLSNN1不可用，无法估计模糊核")
            return np.random.randn(self.kernel_size, self.kernel_size) if hasattr(self,
                                                                                  'kernel_size') else np.random.randn(
                15, 15)

        return self.trainer.estimate_kernel(image_path)

    def estimate_kernels_from_dir(self, image_dir, num_kernels=100):
        """从目录中估计多个模糊核"""
        image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg', 'bmp'))]
        image_paths = image_paths[:min(num_kernels, len(image_paths))]

        kernels = []
        for path in tqdm(image_paths, desc="Estimating kernels"):
            kernel = self.estimate_kernel(path)
            kernels.append(kernel)

        return kernels

    def analyze_kernels(self, kernels):
        """分析模糊核的统计特性"""
        kernels = np.array(kernels)
        mean_kernel = np.mean(kernels, axis=0)
        std_kernel = np.std(kernels, axis=0)

        return {
            'mean': mean_kernel,
            'std': std_kernel,
            'kernels': kernels
        }


# ========== 多核KernelGAN训练和估计函数 ==========
def train_multiple_kernelgans(lr_image_dir, scale=4, kernel_size=15, epochs_per_image=5, num_images=None, device='cuda',
                              seed=42):
    """为多张图像分别训练KernelGAN，获得多个不同的核"""
    if not KLSNN_AVAILABLE:
        logger.error("KLSNN1不可用，无法进行多核训练")
        # 返回模拟结果用于测试
        dummy_kernels = np.random.rand(10, kernel_size, kernel_size)
        dummy_stats = {
            "kernel_mean": float(np.mean(dummy_kernels)),
            "kernel_std": float(np.std(dummy_kernels)),
            "noise_mean": 0.0,
            "noise_std": 0.01,
        }
        return dummy_kernels, dummy_stats

    # 设置种子确保图像选择的可重复性
    set_seed(seed)

    # 添加目录创建代码
    os.makedirs("autodl-tmp/SNN/整合完整/整合分类/KL/scale4", exist_ok=True)  # 添加这一行

    image_paths = [os.path.join(lr_image_dir, f) for f in os.listdir(lr_image_dir)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', 'bmp'))]

    # 修改：使用所有图像而不是随机选择
    if num_images is None:
        # 如果num_images为None，使用所有图像
        selected_paths = image_paths
    else:
        # 如果指定了num_images，限制使用的图像数量
        if len(image_paths) > num_images:
            selected_paths = np.random.choice(image_paths, size=num_images, replace=False)
        else:
            selected_paths = image_paths

    kernels = []
    noise_mus = []
    noise_rhos = []

    logger.info(f"开始为 {len(selected_paths)} 张图像训练KernelGAN...")

    for i, img_path in enumerate(tqdm(selected_paths, desc="训练多个KernelGAN")):
        try:
            # 为每个KernelGAN设置不同的种子（基于基础种子）
            img_seed = seed + i * 1000
            set_seed(img_seed)

            # 为每张图像创建独立的KernelGAN实例
            kg = KernelGAN_UnpairedLR(
                kernel_size=kernel_size,
                # scale=2,
                scale=scale,
                patch_size=64,
                batch_size=16,
                device=device
            )

            # 创建单图像数据集
            single_img_dataset = SingleImagePatchDataset(
                image_path=img_path,
                patch_size=64,
                num_patches=20  # 从单张图像提取多个patch
            )

            # 训练这个图像的KernelGAN
            kg.train(single_img_dataset, epochs=epochs_per_image)

            # 获取该图像特有的核
            kernel = kg.kn.normalized_kernel().squeeze().detach().cpu().numpy()
            kernels.append(kernel)

            # 收集噪声参数
            noise_mus.append(kg.kn.noise_mu.detach().cpu().numpy())
            noise_rhos.append(kg.kn.noise_rho.detach().cpu().numpy())

            logger.info(f"图像 {i + 1}/{len(selected_paths)} 核学习完成")

            # 释放内存
            del kg, single_img_dataset
            if device == 'cuda':
                torch.cuda.empty_cache()

        except Exception as e:
            logger.warning(f"处理图像 {img_path} 时出错: {str(e)}")
            continue

    # 转换为numpy数组
    kernels = np.array(kernels)
    noise_mus = np.array(noise_mus)
    noise_rhos = np.array(noise_rhos)

    # 计算噪声统计量
    noise_sigmas = F.softplus(torch.tensor(noise_rhos)).numpy()
    noise_mean = float(np.mean(noise_mus))
    noise_std = float(np.mean(noise_sigmas))

    # 保存核和统计量
    kernels_path = "autodl-tmp/SNN/整合完整/整合分类/KL/scale4/kernelgan_kernels.npy"
    np.save(kernels_path, kernels)

    stats = {
        "kernel_mean": float(np.mean(kernels)),
        "kernel_std": float(np.std(kernels)),  # 这个std现在反映真正的多样性
        "noise_mean": noise_mean,
        "noise_std": noise_std,
    }
    stats_path = "autodl-tmp/SNN/整合完整/整合分类/KL/scale4/degradation_stats.pth"
    torch.save(stats, stats_path)

    logger.info(f"KernelGAN估计的{len(kernels)}个不同模糊核已保存到 {kernels_path}")
    logger.info(f"退化统计量已保存到 {stats_path}")

    # 分析核的多样性
    analyze_kernel_diversity(kernels)

    return kernels, stats


def analyze_kernel_diversity(kernels):
    """分析核的多样性"""
    logger.info("=== 核多样性分析 ===")
    logger.info(f"核数量: {len(kernels)}")
    logger.info(f"核形状: {kernels[0].shape}")
    logger.info(f"核均值范围: {np.min(kernels):.4f} - {np.max(kernels):.4f}")
    logger.info(f"核标准差: {np.std(kernels):.4f}")

    # 计算核之间的差异
    if len(kernels) > 1:
        differences = []
        for i in range(len(kernels)):
            for j in range(i + 1, len(kernels)):
                diff = np.mean(np.abs(kernels[i] - kernels[j]))
                differences.append(diff)
        logger.info(f"核间平均差异: {np.mean(differences):.4f} (±{np.std(differences):.4f})")

    # 可视化前几个核
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        for i, ax in enumerate(axes.flat):
            if i < len(kernels):
                ax.imshow(kernels[i], cmap='hot')
                ax.set_title(f'Kernel {i + 1}')
                ax.axis('off')
        plt.tight_layout()
        plt.savefig('autodl-tmp/SNN/整合完整/整合分类/KL/scale4/kernel_diversity.png')
        plt.close()
        logger.info("核多样性可视化已保存到 autodl-tmp/SNN/整合完整/整合分类/KL/scale4/kernel_diversity.png")
    except Exception as e:
        logger.warning(f"保存核可视化时出错: {str(e)}")


# ========== 简化的核提取函数（当KLSNN不可用时使用） ==========
def extract_kernels_simple(lr_image_dir, kernel_size=15, num_kernels=100):
    """简化的核提取方法，当KLSNN不可用时使用"""
    logger.info("使用简化方法提取模糊核...")

    # 创建高斯核作为示例
    kernels = []
    for i in range(num_kernels):
        # 创建随机的高斯核
        kernel = np.random.randn(kernel_size, kernel_size)
        kernel = np.exp(-(kernel ** 2) / 2)  # 高斯形状
        kernel = kernel / np.sum(kernel)  # 归一化
        kernels.append(kernel)

    kernels = np.array(kernels)

    # 保存核
    os.makedirs("autodl-tmp/SNN/整合完整/整合分类/KL/scale4", exist_ok=True)
    kernels_path = "autodl-tmp/SNN/整合完整/整合分类/KL/scale4/simple_kernels.npy"
    np.save(kernels_path, kernels)

    stats = {
        "kernel_mean": float(np.mean(kernels)),
        "kernel_std": float(np.std(kernels)),
        "noise_mean": 0.0,
        "noise_std": 0.01,
    }
    stats_path = "autodl-tmp/SNN/整合完整/整合分类/KL/scale4/degradation_stats.pth"
    torch.save(stats, stats_path)

    logger.info(f"简化方法提取的{len(kernels)}个模糊核已保存到 {kernels_path}")
    return kernels, stats


# ========== 主函数示例 ==========
if __name__ == "__main__":
    # 设置全局随机种子
    set_seed(42)

    # 配置参数
    LR_IMAGE_DIR = "autodl-tmp/SNN/整合完整/Train-1360/LR"
    NOISE_SAVE_DIR = "autodl-tmp/SNN/整合完整/整合分类/KL/scale4"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # 创建结果目录
    os.makedirs("autodl-tmp/SNN/整合完整/整合分类/KL/scale4", exist_ok=True)

    # 步骤1: 使用多核KernelGAN训练并估计模糊核
    logger.info("开始使用多核KernelGAN估计模糊核...")

    if KLSNN_AVAILABLE:
        target_scale = 4
        k_size = 21 if target_scale == 4 else 15
        kernels, kernel_stats = train_multiple_kernelgans(
            lr_image_dir=LR_IMAGE_DIR,
            scale=target_scale,    # 传入倍率
            kernel_size=k_size,    # 传入适配的核大小
            # kernel_size=15,
            epochs_per_image=500,  # 每张图像训练轮次
            num_images=50,  # 使用所有图像，不限制数量
            device=DEVICE,
            seed=42
        )
    else:
        logger.warning("KLSNN1不可用，使用简化方法提取模糊核")
        kernels, kernel_stats = extract_kernels_simple(
            lr_image_dir=LR_IMAGE_DIR,
            kernel_size=15,
            num_kernels=100
        )

    # 步骤2: 提取噪声补丁
    logger.info("开始从LR图像提取噪声补丁...")
    NoiseExtractor.extract_noise_patches(
        lr_image_dir=LR_IMAGE_DIR,
        save_dir=NOISE_SAVE_DIR,
        patch_size=32,
        num_patches=10000
    )

    logger.info("模糊核和噪声提取完成！")
    logger.info(f"- 生成的模糊核数量: {len(kernels)}")
    logger.info(f"- 噪声补丁保存位置: {NOISE_SAVE_DIR}")
    logger.info(f"- 模糊核统计信息: {kernel_stats}")