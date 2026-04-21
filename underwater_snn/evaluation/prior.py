import os

import numpy as np
import scipy.io
from PIL import Image
from skimage import color
from skimage.transform import resize
from tqdm import tqdm

from .metrics import compute_image_mscn_transform, extract_on_patches


def build_niqe_prior_from_images(dataset_dir, save_path, patch_size=32):
    all_feats = []
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")

    valid_files = [
        name for name in os.listdir(dataset_dir)
        if name.lower().endswith((".jpg", ".png", ".jpeg", ".bmp"))
    ]
    if not valid_files:
        raise ValueError(f"No valid image files found in {dataset_dir}")

    for filename in tqdm(valid_files, desc="Building NIQE prior"):
        img_path = os.path.join(dataset_dir, filename)
        try:
            img = np.array(Image.open(img_path))
            if img.ndim == 3:
                img = color.rgb2gray(img)
            img = resize(img, (128, 128), anti_aliasing=True).astype(np.float32)
            mscn, _, _ = compute_image_mscn_transform(img)
            feats = extract_on_patches(mscn, patch_size=patch_size)
            if len(feats) > 0:
                all_feats.append(feats)
        except Exception:
            continue

    if not all_feats:
        raise RuntimeError("No NIQE features were extracted from the dataset.")

    all_feats = np.vstack(all_feats)
    mu_pris = np.mean(all_feats, axis=0)
    cov_pris = np.cov(all_feats.T)
    scipy.io.savemat(save_path, {"pop_mu": mu_pris, "pop_cov": cov_pris})
    return mu_pris, cov_pris

