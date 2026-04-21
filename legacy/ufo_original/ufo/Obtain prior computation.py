import numpy as np
import scipy.misc
import scipy.io
import scipy
from PIL import Image
import scipy.ndimage
import scipy.special
import math
from skimage.transform import resize
import torch
from skimage import color, filters, feature
from scipy import stats
from torchvision import transforms
import piq
import warnings
import os
import logging
from skimage.color import rgb2gray
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 确保中文显示正常
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]

# 忽略特定警告
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

gamma_range = np.arange(0.2, 10, 0.001)
a = scipy.special.gamma(2.0 / gamma_range)
a *= a
b = scipy.special.gamma(1.0 / gamma_range)
c = scipy.special.gamma(3.0 / gamma_range)
prec_gammas = a / (b * c)


def aggd_features(imdata):
    # flatten imdata
    imdata.shape = (len(imdata.flat),)
    imdata2 = imdata * imdata
    left_data = imdata2[imdata < 0]
    right_data = imdata2[imdata >= 0]
    left_mean_sqrt = 0
    right_mean_sqrt = 0
    if len(left_data) > 0:
        left_mean_sqrt = np.sqrt(np.average(left_data))
    if len(right_data) > 0:
        right_mean_sqrt = np.sqrt(np.average(right_data))

    if right_mean_sqrt != 0:
        gamma_hat = left_mean_sqrt / right_mean_sqrt
    else:
        gamma_hat = np.inf
    # solve r-hat norm

    imdata2_mean = np.mean(imdata2)
    if imdata2_mean != 0:
        r_hat = (np.average(np.abs(imdata)) ** 2) / (np.average(imdata2))
    else:
        r_hat = np.inf
    rhat_norm = r_hat * (((math.pow(gamma_hat, 3) + 1) * (gamma_hat + 1)) / math.pow(math.pow(gamma_hat, 2) + 1, 2))

    # solve alpha by guessing values that minimize ro
    pos = np.argmin((prec_gammas - rhat_norm) ** 2);
    alpha = gamma_range[pos]

    gam1 = scipy.special.gamma(1.0 / alpha)
    gam2 = scipy.special.gamma(2.0 / alpha)
    gam3 = scipy.special.gamma(3.0 / alpha)

    aggdratio = np.sqrt(gam1) / np.sqrt(gam3)
    bl = aggdratio * left_mean_sqrt
    br = aggdratio * right_mean_sqrt

    # mean parameter
    N = (br - bl) * (gam2 / gam1)  # *aggdratio
    return (alpha, N, bl, br, left_mean_sqrt, right_mean_sqrt)


def ggd_features(imdata):
    nr_gam = 1 / prec_gammas
    sigma_sq = np.var(imdata)
    E = np.mean(np.abs(imdata))
    rho = sigma_sq / E ** 2
    pos = np.argmin(np.abs(nr_gam - rho));
    return gamma_range[pos], sigma_sq


def paired_product(new_im):
    shift1 = np.roll(new_im.copy(), 1, axis=1)
    shift2 = np.roll(new_im.copy(), 1, axis=0)
    shift3 = np.roll(np.roll(new_im.copy(), 1, axis=0), 1, axis=1)
    shift4 = np.roll(np.roll(new_im.copy(), 1, axis=0), -1, axis=1)

    H_img = shift1 * new_im
    V_img = shift2 * new_im
    D1_img = shift3 * new_im
    D2_img = shift4 * new_im

    return (H_img, V_img, D1_img, D2_img)


def gen_gauss_window(lw, sigma):
    sd = np.float32(sigma)
    lw = int(lw)
    weights = [0.0] * (2 * lw + 1)
    weights[lw] = 1.0
    sum = 1.0
    sd *= sd
    for ii in range(1, lw + 1):
        tmp = np.exp(-0.5 * np.float32(ii * ii) / sd)
        weights[lw + ii] = tmp
        weights[lw - ii] = tmp
        sum += 2.0 * tmp
    for ii in range(2 * lw + 1):
        weights[ii] /= sum
    return weights


def compute_image_mscn_transform(image, C=1, avg_window=None, extend_mode='constant'):
    if avg_window is None:
        avg_window = gen_gauss_window(3, 7.0 / 6.0)
    assert len(np.shape(image)) == 2
    h, w = np.shape(image)
    mu_image = np.zeros((h, w), dtype=np.float32)
    var_image = np.zeros((h, w), dtype=np.float32)
    image = np.array(image).astype('float32')
    scipy.ndimage.correlate1d(image, avg_window, 0, mu_image, mode=extend_mode)
    scipy.ndimage.correlate1d(mu_image, avg_window, 1, mu_image, mode=extend_mode)
    scipy.ndimage.correlate1d(image ** 2, avg_window, 0, var_image, mode=extend_mode)
    scipy.ndimage.correlate1d(var_image, avg_window, 1, var_image, mode=extend_mode)
    var_image = np.sqrt(np.abs(var_image - mu_image ** 2))
    return (image - mu_image) / (var_image + C), var_image, mu_image


def _niqe_extract_subband_feats(mscncoefs):
    # alpha_m,  = extract_ggd_features(mscncoefs)
    alpha_m, N, bl, br, lsq, rsq = aggd_features(mscncoefs.copy())
    pps1, pps2, pps3, pps4 = paired_product(mscncoefs)
    alpha1, N1, bl1, br1, lsq1, rsq1 = aggd_features(pps1)
    alpha2, N2, bl2, br2, lsq2, rsq2 = aggd_features(pps2)
    alpha3, N3, bl3, br3, lsq3, rsq3 = aggd_features(pps3)
    alpha4, N4, bl4, br4, lsq4, rsq4 = aggd_features(pps4)
    return np.array([alpha_m, (bl + br) / 2.0,
                     alpha1, N1, bl1, br1,  # (V)
                     alpha2, N2, bl2, br2,  # (H)
                     alpha3, N3, bl3, br3,  # (D1)
                     alpha4, N4, bl4, br4,  # (D2)
                     ])


def get_patches_train_features(img, patch_size, stride=8):
    return _get_patches_generic(img, patch_size, 1, stride)


def get_patches_test_features(img, patch_size, stride=8):
    return _get_patches_generic(img, patch_size, 0, stride)


# def extract_on_patches(img, patch_size):
#     h, w = img.shape
#     patch_size = int(patch_size)  # 修复：将 np.int 改为 int
#     patches = []
#     for j in range(0, h - patch_size + 1, patch_size):
#         for i in range(0, w - patch_size + 1, patch_size):
#             patch = img[j:j + patch_size, i:i + patch_size]
#             patches.append(patch)

def extract_on_patches(img, patch_size, stride=8):
    h, w = img.shape
    patch_size = int(patch_size)  # 修复：将 np.int 改为 int
    patches = []
    for j in range(0, h - patch_size + 1, stride):
        for i in range(0, w - patch_size + 1, stride):
            patch = img[j:j + patch_size, i:i + patch_size]
            patches.append(patch)
    if not patches: # 防止空列表报错
        return np.array([])

    patches = np.array(patches)

    patch_features = []
    for p in patches:
        patch_features.append(_niqe_extract_subband_feats(p))
    patch_features = np.array(patch_features)

    return patch_features


def _get_patches_generic(img, patch_size, is_train, stride):
    h, w = np.shape(img)
    if h < patch_size or w < patch_size:
        logger.warning(f"Input image too small: {h}x{w}, required at least {patch_size}x{patch_size}")
        return np.array([])  # 返回空数组而不是退出程序

    # ensure that the patch divides evenly into img
    hoffset = (h % patch_size)
    woffset = (w % patch_size)

    if hoffset > 0:
        img = img[:-hoffset, :]
    if woffset > 0:
        img = img[:, :-woffset]

    img = img.astype(np.float32)

    # 对于128×128图像，不进行下采样
    if h > 128 and w > 128:
        # 如果图像大于128×128，调整到128×128
        img = resize(img, (128, 128), order=3, mode='reflect', anti_aliasing=True)
        img = img.astype(np.float32)
        # 同时调整patch_size以适应新尺寸
        patch_size = min(patch_size, 32)  # 使用更小的patch_size

    mscn1, var, mu = compute_image_mscn_transform(img)
    mscn1 = mscn1.astype(np.float32)

    # 对于128×128图像，跳过下采样步骤
    if h <= 128 and w <= 128:
        feats_lvl1 = extract_on_patches(mscn1, patch_size, stride)
        feats = feats_lvl1
    else:
        # 对于大图像，仍然使用多尺度特征
        img2 = resize(img, (img.shape[0] // 2, img.shape[1] // 2), order=3, mode='reflect', anti_aliasing=True)
        img2 = img2.astype(np.float32)
        mscn2, _, _ = compute_image_mscn_transform(img2)
        mscn2 = mscn2.astype(np.float32)
        feats_lvl1 = extract_on_patches(mscn1, patch_size, stride)
        feats_lvl2 = extract_on_patches(mscn2, patch_size // 2, stride)
        feats = np.hstack((feats_lvl1, feats_lvl2))

    return feats


# ================== 核心功能：构建NIQE先验函数 ==================
def build_niqe_prior_from_images(dataset_dir, save_path="/root/autodl-tmp/SNN/整合完整/整合分类/zhenghe2/niqe_water_params.mat", patch_size=32):
    """从真实LR图像构建NIQE先验（mu_pris, cov_pris）"""
    all_feats = []
    print(f"Building NIQE prior from dataset: {dataset_dir}")

    if not os.path.exists(dataset_dir):
        logger.error(f"Dataset directory {dataset_dir} does not exist!")
        return None, None

    valid_files = [f for f in os.listdir(dataset_dir) if f.endswith((".jpg", ".png", ".jpeg", ".bmp"))]
    if not valid_files:
        logger.error(f"No valid image files found in {dataset_dir}")
        return None, None

    for filename in tqdm(valid_files):
        try:
            img_path = os.path.join(dataset_dir, filename)
            img = Image.open(img_path)
            img = np.array(img)

            if img.ndim == 3:
                img = color.rgb2gray(img)

            # 调整图像大小为128×128
            img = resize(img, (128, 128), anti_aliasing=True).astype(np.float32)

            # 计算MSCN系数
            mscn, _, _ = compute_image_mscn_transform(img)

            # 提取特征
            feats = extract_on_patches(mscn, patch_size=patch_size)
            if len(feats) > 0:
                all_feats.append(feats)

        except Exception as e:
            logger.warning(f"Error processing {filename}: {e}")
            continue

    if not all_feats:
        logger.error("No features extracted from the dataset!")
        return None, None

    all_feats = np.vstack(all_feats)
    mu_pris = np.mean(all_feats, axis=0)
    cov_pris = np.cov(all_feats.T)

    # 保存先验参数
    scipy.io.savemat(save_path, {'pop_mu': mu_pris, 'pop_cov': cov_pris})
    print(f"NIQE prior saved to: {save_path}")
    print(f"Feature dimension: {mu_pris.shape[0]}")

    return mu_pris, cov_pris


def niqe(inputImgData, prior_path="niqe_water_params.mat"):
    """计算图像的 NIQE 分数，若先验不存在则自动构建"""
    patch_size = 32  # 为128×128图像使用更小的patch_size

    # 检查先验文件是否存在，如果不存在则构建
    if not os.path.exists(prior_path):
        logger.warning(f"NIQE prior file {prior_path} not found, building new one...")
        dataset_dir = "/root/autodl-tmp/SNN/整合完整/Train-1360/LR"  # 修改为你的真实LR图像路径
        build_niqe_prior_from_images(dataset_dir, save_path=prior_path, patch_size=patch_size)

        if not os.path.exists(prior_path):
            logger.error("Failed to build NIQE prior, using default values")
            return 5.0

    try:
        params = scipy.io.loadmat(prior_path)
        pop_mu = np.ravel(params["pop_mu"])
        pop_cov = params["pop_cov"]
    except Exception as e:
        logger.error(f"Error loading NIQE prior: {e}")
        return 5.0

    M, N = inputImgData.shape

    # 对于128×128图像，调整最小尺寸要求
    min_required_size = 32
    if M < min_required_size or N < min_required_size:
        logger.warning(f"Image too small for NIQE: {M}x{N}")
        return 5.0

    feats = get_patches_test_features(inputImgData, patch_size)
    if len(feats) == 0:
        logger.warning("No features extracted for NIQE calculation")
        return 5.0

    sample_mu = np.mean(feats, axis=0)
    sample_cov = np.cov(feats.T)

    X = sample_mu - pop_mu
    covmat = ((pop_cov + sample_cov) / 2.0)
    pinvmat = scipy.linalg.pinv(covmat)
    niqe_score = np.sqrt(np.dot(np.dot(X, pinvmat), X))

    return niqe_score


def calculate_niqe_complete(img_tensor, prior_path="/root/autodl-tmp/SNN/整合完整/整合分类/zhenghe2/niqe_water_params.mat"):
    """完整的NIQE计算函数 - 适配128×128图像"""
    try:
        # 将张量转换为numpy数组
        if torch.is_tensor(img_tensor):
            img_np = img_tensor.squeeze().cpu().numpy()
        else:
            img_np = img_tensor

        # 如果图像是RGB，转换为灰度
        if len(img_np.shape) == 3 and img_np.shape[0] == 3:
            img_gray = rgb2gray(img_np.transpose(1, 2, 0))
        elif len(img_np.shape) == 3 and img_np.shape[2] == 3:
            img_gray = rgb2gray(img_np)
        else:
            img_gray = img_np

        # 确保图像数据类型正确（NIQE要求）
        img_gray = img_gray.astype(np.float32)

        # 调整图像大小为128×128（如果必要）
        H, W = img_gray.shape
        if H != 128 or W != 128:
            img_gray = resize(img_gray, (128, 128), order=3, mode='reflect', anti_aliasing=True)
            img_gray = img_gray.astype(np.float32)

        # 确保图像值范围合理
        if img_gray.max() > 255 or img_gray.min() < 0:
            img_gray = np.clip(img_gray, 0, 255)
            img_gray = img_gray.astype(np.float32)

        logger.debug(f"NIQE: 图像尺寸 {img_gray.shape}, 准备计算NIQE")

        # 调用niqe函数，传递先验文件路径
        return niqe(img_gray, prior_path)

    except Exception as e:
        logger.error(f"Error in calculate_niqe_complete: {e}")
        return 5.0


# ========= 改进的批量NIQE计算 =========
def batch_niqe_complete_parallel(imgs, prior_path="/root/autodl-tmp/SNN/整合完整/整合分类/zhenghe2/niqe_water_params.mat", max_workers=None):
    """并行批量NIQE计算 - 使用多线程"""
    try:
        if max_workers is None:
            max_workers = min(32, (os.cpu_count() or 1) + 4)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_img = {
                executor.submit(calculate_niqe_complete, img, prior_path): i
                for i, img in enumerate(imgs)
            }

            # 收集结果
            scores = [0.0] * len(imgs)
            for future in as_completed(future_to_img):
                img_idx = future_to_img[future]
                try:
                    scores[img_idx] = future.result()
                except Exception as e:
                    logger.error(f"Error processing image {img_idx}: {e}")
                    scores[img_idx] = 5.0  # 默认值

        return float(np.mean(scores)), scores
    except Exception as e:
        logger.error(f"Error in batch_niqe_complete_parallel: {e}")
        return 0.0, [5.0] * len(imgs)


def batch_niqe_complete(imgs, prior_path="/root/autodl-tmp/SNN/整合完整/整合分类/zhenghe2/niqe_water_params.mat"):
    """批量NIQE计算 - 保持向后兼容"""
    try:
        _, scores = batch_niqe_complete_parallel(imgs, prior_path)
        return float(np.mean(scores))
    except Exception as e:
        logger.error(f"Error in batch_niqe_complete: {e}")
        return 0.0


def batch_niqe_complete_scores(imgs, prior_path="/root/autodl-tmp/SNN/整合完整/整合分类/zhenghe2/niqe_water_params.mat"):
    """批量NIQE计算返回所有分数"""
    try:
        _, scores = batch_niqe_complete_parallel(imgs, prior_path)
        return scores
    except Exception as e:
        logger.error(f"Error in batch_niqe_complete_scores: {e}")
        return []


# ========= 改进的UISM实现 =========
def _calculate_gradient_entropy(gradient_map, bins=256):
    """计算梯度图的熵"""
    try:
        # 计算梯度直方图
        hist, _ = np.histogram(gradient_map, bins=bins, range=(0, np.max(gradient_map)))
        # 归一化
        hist = hist.astype(np.float32)
        hist_sum = np.sum(hist)
        if hist_sum > 0:
            hist = hist / hist_sum
        else:
            return 0.0

        # 计算熵（添加小值避免log(0)）
        hist = hist + 1e-10
        entropy_val = -np.sum(hist * np.log2(hist))
        return entropy_val
    except Exception as e:
        logger.error(f"Error in _calculate_gradient_entropy: {e}")
        return 0.0


def _calculate_edge_contrast(edge_map, original_intensity):
    """基于边缘强度的对比度计算"""
    try:
        # 二值化边缘图
        edge_threshold = 0.1 * np.max(edge_map)
        edge_mask = edge_map > edge_threshold

        if np.sum(edge_mask) == 0:
            return 0.0

        # 计算边缘区域和非边缘区域的强度对比
        edge_intensity = np.mean(original_intensity[edge_mask])
        non_edge_intensity = np.mean(original_intensity[~edge_mask])

        if non_edge_intensity > 0:
            contrast = edge_intensity / non_edge_intensity
        else:
            contrast = edge_intensity

        return contrast
    except Exception as e:
        logger.error(f"Error in _calculate_edge_contrast: {e}")
        return 0.0


def _uism_improved(img):
    """改进的UISM实现 - 使用梯度强度直方图的熵和边缘对比度"""
    try:
        # 转换为灰度图
        gray = color.rgb2gray(img)

        # 计算梯度幅值（使用Sobel算子）
        grad_x = filters.sobel_v(gray)
        grad_y = filters.sobel_h(gray)
        gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)

        # 方法1: 梯度强度直方图的熵
        gradient_entropy = _calculate_gradient_entropy(gradient_magnitude)

        # 方法2: 基于边缘强度的对比度
        edge_contrast = _calculate_edge_contrast(gradient_magnitude, gray)

        # 组合两种方法（权重可以根据实际情况调整）
        uism_score = 0.6 * gradient_entropy + 0.4 * edge_contrast

        return uism_score
    except Exception as e:
        logger.error(f"Error in _uism_improved: {e}")
        return 0.0


# ========= 改进的UICM实现 =========
def _calculate_color_distribution_features(rg, yb):
    """计算颜色分布的特征"""
    try:
        # 计算基本统计量
        rg_mean = np.mean(rg)
        rg_std = np.std(rg)
        yb_mean = np.mean(yb)
        yb_std = np.std(yb)

        # 计算偏度和峰度（反映分布形状）
        rg_skew = stats.skew(rg.flatten()) if len(rg.flatten()) > 2 else 0
        yb_skew = stats.skew(yb.flatten()) if len(yb.flatten()) > 2 else 0
        rg_kurtosis = stats.kurtosis(rg.flatten()) if len(rg.flatten()) > 2 else 0
        yb_kurtosis = stats.kurtosis(yb.flatten()) if len(yb.flatten()) > 2 else 0

        # 计算分位数
        rg_q1, rg_q2, rg_q3 = np.percentile(rg, [25, 50, 75])
        yb_q1, yb_q2, yb_q3 = np.percentile(yb, [25, 50, 75])

        return {
            'rg_mean': rg_mean, 'rg_std': rg_std, 'rg_skew': rg_skew, 'rg_kurtosis': rg_kurtosis,
            'rg_q1': rg_q1, 'rg_q2': rg_q2, 'rg_q3': rg_q3,
            'yb_mean': yb_mean, 'yb_std': yb_std, 'yb_skew': yb_skew, 'yb_kurtosis': yb_kurtosis,
            'yb_q1': yb_q1, 'yb_q2': yb_q2, 'yb_q3': yb_q3
        }
    except Exception as e:
        logger.error(f"Error in _calculate_color_distribution_features: {e}")
        return {
            'rg_mean': 0, 'rg_std': 0, 'rg_skew': 0, 'rg_kurtosis': 0,
            'rg_q1': 0, 'rg_q2': 0, 'rg_q3': 0,
            'yb_mean': 0, 'yb_std': 0, 'yb_skew': 0, 'yb_kurtosis': 0,
            'yb_q1': 0, 'yb_q2': 0, 'yb_q3': 0
        }


def _uicm_improved(img):
    """改进的UICM实现 - 考虑色彩失真方向性分布"""
    try:
        # 计算RG和YB分量
        rg = img[:, :, 0] - img[:, :, 1]
        yb = (img[:, :, 0] + img[:, :, 1]) / 2 - img[:, :, 2]

        # 计算详细的颜色分布特征
        color_features = _calculate_color_distribution_features(rg, yb)

        # 计算颜色分量的相关性（反映色彩失真方向性）
        try:
            color_correlation = np.corrcoef(rg.flatten(), yb.flatten())[0, 1]
            if np.isnan(color_correlation):
                color_correlation = 0
        except:
            color_correlation = 0

        # 计算色彩分布的离散程度
        rg_discreteness = color_features['rg_std'] / (abs(color_features['rg_mean']) + 1e-10)
        yb_discreteness = color_features['yb_std'] / (abs(color_features['yb_mean']) + 1e-10)

        # 改进的UICM公式，考虑更多统计特征
        uicm = (-0.0268 * np.sqrt(color_features['rg_mean'] ** 2 + color_features['yb_mean'] ** 2) +
                0.1586 * np.sqrt(color_features['rg_std'] + color_features['yb_std']) +
                0.05 * (1 - abs(color_correlation)) +  # 考虑颜色相关性
                0.03 * (rg_discreteness + yb_discreteness))  # 考虑离散程度

        return uicm
    except Exception as e:
        logger.error(f"Error in _uicm_improved: {e}")
        return 0.0


# ========= 改进的UIConM实现 =========
def _uiconm_improved(img):
    """改进的UIConM实现 - 使用更复杂的对比度度量"""
    try:
        gray = color.rgb2gray(img)

        # 方法1: 标准差（原有方法）
        std_contrast = np.std(gray)

        # 方法2: 使用局部对比度（更符合人类视觉）
        from skimage.filters import laplace
        local_contrast = np.std(laplace(gray))

        # 方法3: 使用Michelson对比度
        min_intensity = np.min(gray)
        max_intensity = np.max(gray)
        if (max_intensity + min_intensity) > 0:
            michelson_contrast = (max_intensity - min_intensity) / (max_intensity + min_intensity)
        else:
            michelson_contrast = 0

        # 组合多种对比度度量
        uiconm = 0.5 * std_contrast + 0.3 * local_contrast + 0.2 * michelson_contrast

        return uiconm
    except Exception as e:
        logger.error(f"Error in _uiconm_improved: {e}")
        return 0.0


# ========= 改进的批量UIQM计算 =========
def _calculate_uiqm_single(img):
    """计算单张图像的UIQM分数"""
    try:
        if img is None:
            raise ValueError("Input image cannot be None")

        if isinstance(img, torch.Tensor):
            img = transforms.ToPILImage()(img.cpu())
        if isinstance(img, Image.Image):
            img = np.array(img)
        if img.dtype != np.float32:
            img = img.astype(np.float32) / 255.0

        # 使用改进的实现
        uicm = _uicm_improved(img)
        uism = _uism_improved(img)
        uiconm = _uiconm_improved(img)

        # UIQM公式保持不变，但各个分量的计算更准确
        uiqm = 0.0282 * uicm + 0.2953 * uism + 3.5753 * uiconm
        return float(uiqm)
    except Exception as e:
        logger.error(f"Error in _calculate_uiqm_single: {e}")
        return 0.0


def batch_uiqm_parallel(images, max_workers=None):
    """并行批量UIQM计算 - 使用多线程"""
    try:
        if max_workers is None:
            max_workers = min(32, (os.cpu_count() or 1) + 4)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_img = {
                executor.submit(_calculate_uiqm_single, img): i
                for i, img in enumerate(images)
            }

            # 收集结果
            scores = [0.0] * len(images)
            for future in as_completed(future_to_img):
                img_idx = future_to_img[future]
                try:
                    scores[img_idx] = future.result()
                except Exception as e:
                    logger.error(f"Error processing image {img_idx}: {e}")
                    scores[img_idx] = 0.0  # 默认值

        return float(np.mean(scores)), scores
    except Exception as e:
        logger.error(f"Error in batch_uiqm_parallel: {e}")
        return 0.0, [0.0] * len(images)


def batch_uiqm(images):
    """批量计算UIQM分数 - 保持向后兼容"""
    try:
        mean_score, _ = batch_uiqm_parallel(images)
        return mean_score
    except Exception as e:
        logger.error(f"Error in batch_uiqm: {e}")
        return 0.0


def batch_uiqm_scores(images):
    """批量计算UIQM分数，返回所有分数列表"""
    try:
        _, scores = batch_uiqm_parallel(images)
        return scores
    except Exception as e:
        logger.error(f"Error in batch_uiqm_scores: {e}")
        return []


# ========= BRISQUE 计算相关函数 =========
def calculate_brisque(image, device='cuda'):
    try:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        elif isinstance(image, torch.Tensor):
            image = transforms.ToPILImage()(image.cpu())
        if image.mode != 'RGB':
            image = image.convert('RGB')
        transform = transforms.Compose([transforms.ToTensor()])
        img_tensor = transform(image).unsqueeze(0).to(device)
        brisque_score = piq.brisque(img_tensor, data_range=1.0, reduction='none')
        return brisque_score.item()
    except Exception as e:
        logger.error(f"Error in calculate_brisque: {e}")
        return 0.0


def batch_brisque(images, device='cuda', batch_size=32):
    try:
        scores = []
        transform = transforms.Compose([transforms.ToTensor()])
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            batch_tensors = []
            for img in batch:
                if isinstance(img, torch.Tensor):
                    img = transforms.ToPILImage()(img.cpu())
                elif isinstance(img, np.ndarray):
                    img = Image.fromarray(img)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img_tensor = transform(img).to(device)
                batch_tensors.append(img_tensor)
            batch_tensor = torch.stack(batch_tensors).to(device)
            batch_scores = piq.brisque(batch_tensor, data_range=1.0, reduction='none')
            scores.extend(batch_scores.cpu().numpy())

        # 清理GPU内存
        if device == 'cuda':
            torch.cuda.empty_cache()

        return float(np.mean(scores))
    except Exception as e:
        logger.error(f"Error in batch_brisque: {e}")
        if device == 'cuda':
            torch.cuda.empty_cache()
        return 0.0


def batch_brisque_scores(images, device='cuda', batch_size=32):
    """批量计算BRISQUE分数，返回所有分数列表"""
    try:
        scores = []
        transform = transforms.Compose([transforms.ToTensor()])
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            batch_tensors = []
            for img in batch:
                if isinstance(img, torch.Tensor):
                    img = transforms.ToPILImage()(img.cpu())
                elif isinstance(img, np.ndarray):
                    img = Image.fromarray(img)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img_tensor = transform(img).to(device)
                batch_tensors.append(img_tensor)
            batch_tensor = torch.stack(batch_tensors).to(device)
            batch_scores = piq.brisque(batch_tensor, data_range=1.0, reduction='none')
            scores.extend(batch_scores.cpu().numpy().tolist())

        # 清理GPU内存
        if device == 'cuda':
            torch.cuda.empty_cache()

        return scores
    except Exception as e:
        logger.error(f"Error in batch_brisque_scores: {e}")
        if device == 'cuda':
            torch.cuda.empty_cache()
        return []


# ========= UIQM 计算相关函数（使用改进的实现） =========
def calculate_uiqm(image):
    try:
        if image is None:
            raise ValueError("Input image cannot be None")

        if isinstance(image, torch.Tensor):
            image = transforms.ToPILImage()(image.cpu())
        if isinstance(image, Image.Image):
            image = np.array(image)
        if image.dtype != np.float32:
            image = image.astype(np.float32) / 255.0

        # 使用改进的实现
        uicm = _uicm_improved(image)
        uism = _uism_improved(image)
        uiconm = _uiconm_improved(image)

        # UIQM公式保持不变，但各个分量的计算更准确
        uiqm = 0.0282 * uicm + 0.2953 * uism + 3.5753 * uiconm
        return float(uiqm)
    except Exception as e:
        logger.error(f"Error in calculate_uiqm: {e}")
        return 0.0


# ========= 简化的指标计算工具类 =========
class ImageQualityEvaluator:
    def __init__(self, device='cuda', prior_path="niqe_water_params.mat", max_workers=None):
        self.device = device
        self.prior_path = prior_path
        self.max_workers = max_workers

    def calculate_niqe(self, img_tensor):
        """计算完整的NIQE分数（使用skimage实现）"""
        return calculate_niqe_complete(img_tensor, self.prior_path)

    def calculate_batch_niqe(self, imgs):
        """批量计算完整的NIQE分数"""
        return batch_niqe_complete(imgs, self.prior_path)

    def calculate_batch_niqe_scores(self, imgs):
        """批量计算完整的NIQE分数，返回所有分数列表"""
        return batch_niqe_complete_scores(imgs, self.prior_path)

    def calculate_brisque(self, image):
        return calculate_brisque(image, self.device)

    def calculate_batch_brisque(self, images, batch_size=32):
        return batch_brisque(images, self.device, batch_size)

    def calculate_batch_brisque_scores(self, images, batch_size=32):
        """批量计算BRISQUE分数，返回所有分数列表"""
        return batch_brisque_scores(images, self.device, batch_size)

    def calculate_uiqm(self, image):
        return calculate_uiqm(image)

    def calculate_batch_uiqm(self, images):
        return batch_uiqm(images)

    def calculate_batch_uiqm_scores(self, images):
        """批量计算UIQM分数，返回所有分数列表"""
        return batch_uiqm_scores(images)

    def evaluate_all_metrics(self, images):
        """计算所有三个指标的平均值"""
        try:
            # 并行计算所有指标
            with ThreadPoolExecutor(max_workers=3) as executor:
                niqe_future = executor.submit(batch_niqe_complete_scores, images, self.prior_path)
                brisque_future = executor.submit(batch_brisque_scores, images, self.device)
                uiqm_future = executor.submit(batch_uiqm_scores, images)

                niqe_scores = niqe_future.result()
                brisque_scores = brisque_future.result()
                uiqm_scores = uiqm_future.result()

            return {
                'niqe_mean': np.mean(niqe_scores) if niqe_scores else 0,
                'brisque_mean': np.mean(brisque_scores) if brisque_scores else 0,
                'uiqm_mean': np.mean(uiqm_scores) if uiqm_scores else 0,
                'niqe_scores': niqe_scores,
                'brisque_scores': brisque_scores,
                'uiqm_scores': uiqm_scores
            }
        except Exception as e:
            logger.error(f"Error in evaluate_all_metrics: {e}")
            return {
                'niqe_mean': 0, 'brisque_mean': 0, 'uiqm_mean': 0,
                'niqe_scores': [], 'brisque_scores': [], 'uiqm_scores': []
            }


# ========= 使用示例 =========
# if __name__ == "__main__":
#     # 示例：生成NIQE先验
#     build_niqe_prior_from_images("./real_LR_images", save_path="niqe_water_params.mat", patch_size=32)

#     # 示例：计算单张图像NIQE
#     test_img = Image.open("./real_LR_images/sample.png")
#     test_img = np.array(test_img)
#     score = calculate_niqe_complete(test_img, prior_path="niqe_water_params.mat")
#     print("NIQE score:", score)

#     # 示例：使用评估器
#     evaluator = ImageQualityEvaluator()

#     # 计算单张图像的所有指标
#     test_images = [test_img]  # 假设有一组测试图像
#     metrics = evaluator.evaluate_all_metrics(test_images)
#     print("All metrics:", metrics)