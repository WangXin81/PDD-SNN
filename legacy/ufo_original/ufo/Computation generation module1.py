import numpy as np
import math
import cv2
from typing import Union, Tuple, List, Dict
import os
from pathlib import Path


class LRImageDistributionComparator:
    """专门用于真实LR vs 生成LR图像分布比较（L通道专用）"""

    def __init__(self, bins: int = 256, hist_range: Tuple = (0, 255)):
        self.bins = bins
        self.hist_range = hist_range

    def compare_image_distributions(self, real_lr_path: str, generated_lr_path: str) -> Dict:
        """
        比较真实LR和生成LR图像的L通道分布
        """
        # 读取图像 - 假设输入都是RGB格式
        real_img = self._read_image(real_lr_path)
        gen_img = self._read_image(generated_lr_path)

        # 计算L通道分布
        real_l_dist = self._compute_L_channel_distribution(real_img)
        gen_l_dist = self._compute_L_channel_distribution(gen_img)

        # 计算分布距离
        cosine_sim = self._cosine_similarity(real_l_dist, gen_l_dist)
        results = {
            'L_channel': {
                'bhattacharyya': self._bhattacharyya_distance(real_l_dist, gen_l_dist),
                'chi_square': self._chi_square_distance(real_l_dist, gen_l_dist),
                'cosine_similarity': cosine_sim,
                'cosine_distance': 1 - cosine_sim
            }
        }

        return results

    def _read_image(self, image_path: str) -> np.ndarray:
        """读取RGB图像"""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图像: {image_path}")

        # 如果确认输入是RGB格式，直接返回（不移除这行转换）
        # 因为OpenCV默认读取为BGR，需要转换为RGB
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    
    def _compute_L_channel_distribution(self, image: np.ndarray) -> np.ndarray:
        """计算L通道分布，支持单通道和三通道图像"""
    
        # 检查图像通道数
        if len(image.shape) == 2:
            # 单通道图像，直接作为L通道
            l_channel = image
        elif len(image.shape) == 3 and image.shape[2] == 3:
            # 三通道RGB图像，转换为Lab颜色空间并提取L通道
            lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l_channel_metrics = lab_image[:, :, 0]
        else:
            raise ValueError(f"不支持的图像形状: {image.shape}")
    
        # 计算L通道直方图
        hist = cv2.calcHist([l_channel], [0], None, [self.bins], self.hist_range)
        return self._normalize_distribution(hist)

#     def _compute_L_channel_distribution(self, image: np.ndarray) -> np.ndarray:
#         """计算L通道分布"""
#         # 从RGB转换为Lab颜色空间
#         lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

#         # 提取L通道（亮度通道）
#         l_channel = lab_image[:, :, 0]

#         # 计算L通道直方图
#         hist = cv2.calcHist([l_channel], [0], None, [self.bins], self.hist_range)
#         return self._normalize_distribution(hist)

    def _normalize_distribution(self, distribution: np.ndarray) -> np.ndarray:
        """归一化分布"""
        distribution = distribution.flatten().astype(np.float64)
        distribution = distribution / (np.sum(distribution) + 1e-10)
        return distribution

    def _bhattacharyya_distance(self, p: np.ndarray, q: np.ndarray) -> float:
        """计算Bhattacharyya距离"""
        bc = np.sum(np.sqrt(p * q))
        bc = max(min(bc, 1.0), 0.0)
        return -math.log(bc + 1e-10)

    def _chi_square_distance(self, p: np.ndarray, q: np.ndarray) -> float:
        """计算卡方距离"""
        epsilon = 1e-10
        return 0.5 * np.sum(((p - q) ** 2) / (p + q + epsilon))

    def _cosine_similarity(self, p: np.ndarray, q: np.ndarray) -> float:
        """计算余弦相似度"""
        dot_product = np.dot(p, q)
        norm_p = np.linalg.norm(p)
        norm_q = np.linalg.norm(q)

        if norm_p == 0 or norm_q == 0:
            return 0.0

        return dot_product / (norm_p * norm_q)


class BatchLRComparator:
    """批量比较真实LR和生成LR图像（L通道专用）"""

    def __init__(self, real_lr_dir: str, generated_lr_dir: str):
        self.real_lr_dir = Path(real_lr_dir)
        self.generated_lr_dir = Path(generated_lr_dir)
        self.comparator = LRImageDistributionComparator()

    def compare_all_images(self) -> Dict:
        """比较所有图像对"""
        real_images = list(self.real_lr_dir.glob("*.png")) + list(self.real_lr_dir.glob("*.jpg")) + list(self.real_lr_dir.glob("*.bmp"))
        results = {}

        for real_img_path in real_images:
            gen_img_path = self.generated_lr_dir / real_img_path.name

            if gen_img_path.exists():
                try:
                    comparison_result = self.comparator.compare_image_distributions(
                        str(real_img_path), str(gen_img_path)
                    )
                    results[real_img_path.name] = comparison_result
                except Exception as e:
                    print(f"处理 {real_img_path.name} 时出错: {e}")

        return results

    def generate_summary_report(self, results: Dict) -> Dict:
        """生成汇总报告"""
        summary = {
            'L_channel': {},
            'overall': {}
        }

        metrics = ['bhattacharyya', 'chi_square', 'cosine_similarity', 'cosine_distance']

        for metric in metrics:
            summary['L_channel'][metric] = []

        for image_result in results.values():
            l_channel_metrics = image_result['L_channel']
            for metric, value in l_channel_metrics.items():
                summary['L_channel'][metric].append(value)

        for metric in metrics:
            values = summary['L_channel'][metric]
            if values:
                summary['L_channel'][f'{metric}_mean'] = np.mean(values)
                summary['L_channel'][f'{metric}_std'] = np.std(values)
                summary['L_channel'][f'{metric}_min'] = np.min(values)
                summary['L_channel'][f'{metric}_max'] = np.max(values)

        summary['overall'] = self._compute_overall_score(summary['L_channel'])
        return summary

    def _compute_overall_score(self, l_channel_metrics: Dict) -> Dict:
        """计算总体相似度分数"""
        cosine_mean = l_channel_metrics.get('cosine_similarity_mean', 0)
        cosine_distance_mean = l_channel_metrics.get('cosine_distance_mean', 0)

        return {
            'similarity_score': cosine_mean,
            'distance_score': cosine_distance_mean,
            'bhattacharyya_mean': l_channel_metrics.get('bhattacharyya_mean', 0),
            'chi_square_mean': l_channel_metrics.get('chi_square_mean', 0)
        }