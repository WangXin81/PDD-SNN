import math
import os
from pathlib import Path

import cv2
import numpy as np


class LRImageDistributionComparator:
    def __init__(self, bins=256, hist_range=(0, 255)):
        self.bins = bins
        self.hist_range = hist_range

    def compare_image_distributions(self, real_lr_path, generated_lr_path):
        real_img = self._read_image(real_lr_path)
        gen_img = self._read_image(generated_lr_path)
        real_dist = self._compute_l_channel_distribution(real_img)
        gen_dist = self._compute_l_channel_distribution(gen_img)
        cosine_sim = self._cosine_similarity(real_dist, gen_dist)
        return {
            "L_channel": {
                "bhattacharyya": self._bhattacharyya_distance(real_dist, gen_dist),
                "chi_square": self._chi_square_distance(real_dist, gen_dist),
                "cosine_similarity": cosine_sim,
                "cosine_distance": 1 - cosine_sim,
            }
        }

    def _read_image(self, image_path):
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Unable to read image: {image_path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _compute_l_channel_distribution(self, image):
        if len(image.shape) == 2:
            l_channel = image
        elif len(image.shape) == 3 and image.shape[2] == 3:
            l_channel = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)[:, :, 0]
        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")
        hist = cv2.calcHist([l_channel], [0], None, [self.bins], self.hist_range)
        hist = hist.flatten().astype(np.float64)
        return hist / (np.sum(hist) + 1e-10)

    def _bhattacharyya_distance(self, p, q):
        bc = np.sum(np.sqrt(p * q))
        bc = max(min(bc, 1.0), 0.0)
        return -math.log(bc + 1e-10)

    def _chi_square_distance(self, p, q):
        epsilon = 1e-10
        return 0.5 * np.sum(((p - q) ** 2) / (p + q + epsilon))

    def _cosine_similarity(self, p, q):
        dot_product = np.dot(p, q)
        norm_p = np.linalg.norm(p)
        norm_q = np.linalg.norm(q)
        if norm_p == 0 or norm_q == 0:
            return 0.0
        return dot_product / (norm_p * norm_q)


class BatchLRComparator:
    def __init__(self, real_lr_dir, generated_lr_dir):
        self.real_lr_dir = Path(real_lr_dir)
        self.generated_lr_dir = Path(generated_lr_dir)
        self.comparator = LRImageDistributionComparator()

    def compare_all_images(self):
        results = {}
        for real_img_path in sorted(self.real_lr_dir.iterdir()):
            if not real_img_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}:
                continue
            gen_img_path = self.generated_lr_dir / real_img_path.name
            if not gen_img_path.exists():
                continue
            try:
                results[real_img_path.name] = self.comparator.compare_image_distributions(
                    str(real_img_path), str(gen_img_path)
                )
            except Exception:
                continue
        return results
