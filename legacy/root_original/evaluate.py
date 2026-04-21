# import numpy as np
# import scipy.misc
# import scipy.io
# from os.path import dirname
# from os.path import join
# import scipy
# from PIL import Image
# import scipy.ndimage
# import scipy.special
# import math
# from skimage.transform import resize
# import torch
# from skimage import color, filters, feature
# from scipy.ndimage import gaussian_filter
# from scipy import stats
# from torchvision import transforms
# import piq
# import warnings
# import os
# import logging

# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)

# import matplotlib.pyplot as plt
# plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]

# warnings.filterwarnings("ignore", category=RuntimeWarning)
# warnings.filterwarnings("ignore", category=UserWarning)

# # ================= NIQE 辅助函数 (保持不变) =================
# gamma_range = np.arange(0.2, 10, 0.001)
# a = scipy.special.gamma(2.0 / gamma_range)
# a *= a
# b = scipy.special.gamma(1.0 / gamma_range)
# c = scipy.special.gamma(3.0 / gamma_range)
# prec_gammas = a / (b * c)

# def aggd_features(imdata):
#     imdata.shape = (len(imdata.flat),)
#     imdata2 = imdata * imdata
#     left_data = imdata2[imdata < 0]
#     right_data = imdata2[imdata >= 0]
#     left_mean_sqrt = 0
#     right_mean_sqrt = 0
#     if len(left_data) > 0:
#         left_mean_sqrt = np.sqrt(np.average(left_data))
#     if len(right_data) > 0:
#         right_mean_sqrt = np.sqrt(np.average(right_data))

#     if right_mean_sqrt != 0:
#         gamma_hat = left_mean_sqrt / right_mean_sqrt
#     else:
#         gamma_hat = np.inf

#     imdata2_mean = np.mean(imdata2)
#     if imdata2_mean != 0:
#         r_hat = (np.average(np.abs(imdata)) ** 2) / (np.average(imdata2))
#     else:
#         r_hat = np.inf
#     rhat_norm = r_hat * (((math.pow(gamma_hat, 3) + 1) * (gamma_hat + 1)) / math.pow(math.pow(gamma_hat, 2) + 1, 2))

#     pos = np.argmin((prec_gammas - rhat_norm) ** 2);
#     alpha = gamma_range[pos]

#     gam1 = scipy.special.gamma(1.0 / alpha)
#     gam2 = scipy.special.gamma(2.0 / alpha)
#     gam3 = scipy.special.gamma(3.0 / alpha)

#     aggdratio = np.sqrt(gam1) / np.sqrt(gam3)
#     bl = aggdratio * left_mean_sqrt
#     br = aggdratio * right_mean_sqrt
#     N = (br - bl) * (gam2 / gam1)
#     return (alpha, N, bl, br, left_mean_sqrt, right_mean_sqrt)

# def paired_product(new_im):
#     shift1 = np.roll(new_im.copy(), 1, axis=1)
#     shift2 = np.roll(new_im.copy(), 1, axis=0)
#     shift3 = np.roll(np.roll(new_im.copy(), 1, axis=0), 1, axis=1)
#     shift4 = np.roll(np.roll(new_im.copy(), 1, axis=0), -1, axis=1)
#     H_img = shift1 * new_im
#     V_img = shift2 * new_im
#     D1_img = shift3 * new_im
#     D2_img = shift4 * new_im
#     return (H_img, V_img, D1_img, D2_img)

# def gen_gauss_window(lw, sigma):
#     sd = np.float32(sigma)
#     lw = int(lw)
#     weights = [0.0] * (2 * lw + 1)
#     weights[lw] = 1.0
#     sum = 1.0
#     sd *= sd
#     for ii in range(1, lw + 1):
#         tmp = np.exp(-0.5 * np.float32(ii * ii) / sd)
#         weights[lw + ii] = tmp
#         weights[lw - ii] = tmp
#         sum += 2.0 * tmp
#     for ii in range(2 * lw + 1):
#         weights[ii] /= sum
#     return weights

# def compute_image_mscn_transform(image, C=1, avg_window=None, extend_mode='constant'):
#     if avg_window is None:
#         avg_window = gen_gauss_window(3, 7.0 / 6.0)
#     assert len(np.shape(image)) == 2
#     h, w = np.shape(image)
#     mu_image = np.zeros((h, w), dtype=np.float32)
#     var_image = np.zeros((h, w), dtype=np.float32)
#     image = np.array(image).astype('float32')
#     scipy.ndimage.correlate1d(image, avg_window, 0, mu_image, mode=extend_mode)
#     scipy.ndimage.correlate1d(mu_image, avg_window, 1, mu_image, mode=extend_mode)
#     scipy.ndimage.correlate1d(image ** 2, avg_window, 0, var_image, mode=extend_mode)
#     scipy.ndimage.correlate1d(var_image, avg_window, 1, var_image, mode=extend_mode)
#     var_image = np.sqrt(np.abs(var_image - mu_image ** 2))
#     return (image - mu_image) / (var_image + C), var_image, mu_image

# def _niqe_extract_subband_feats(mscncoefs):
#     alpha_m, N, bl, br, lsq, rsq = aggd_features(mscncoefs.copy())
#     pps1, pps2, pps3, pps4 = paired_product(mscncoefs)
#     alpha1, N1, bl1, br1, lsq1, rsq1 = aggd_features(pps1)
#     alpha2, N2, bl2, br2, lsq2, rsq2 = aggd_features(pps2)
#     alpha3, N3, bl3, br3, lsq3, rsq3 = aggd_features(pps3)
#     alpha4, N4, bl4, br4, lsq4, rsq4 = aggd_features(pps4)
#     return np.array([alpha_m, (bl + br) / 2.0,
#                      alpha1, N1, bl1, br1,
#                      alpha2, N2, bl2, br2,
#                      alpha3, N3, bl3, br3,
#                      alpha4, N4, bl4, br4])

# def get_patches_test_features(img, patch_size, stride=8):
#     return _get_patches_generic(img, patch_size, 0, stride)

# def extract_on_patches(img, patch_size, stride=8):
#     h, w = img.shape
#     patch_size = int(patch_size)
#     patches = []
#     for j in range(0, h - patch_size + 1, stride):
#         for i in range(0, w - patch_size + 1, stride):
#             patch = img[j:j + patch_size, i:i + patch_size]
#             patches.append(patch)
#     patches = np.array(patches)
#     patch_features = []
#     for p in patches:
#         patch_features.append(_niqe_extract_subband_feats(p))
#     patch_features = np.array(patch_features)
#     return patch_features

# def _get_patches_generic(img, patch_size, is_train, stride):
#     h, w = np.shape(img)
#     if h < patch_size or w < patch_size:
#         return np.array([]) 

#     hoffset = (h % patch_size)
#     woffset = (w % patch_size)
#     if hoffset > 0: img = img[:-hoffset, :]
#     if woffset > 0: img = img[:, :-woffset]

#     img = img.astype(np.float32)
#     # [修正]: preserve_range=True 确保 scale 2 的范围保持 [0, 255]
#     img2 = resize(img, (img.shape[0] // 2, img.shape[1] // 2), 
#                   order=3, mode='reflect', anti_aliasing=True, preserve_range=True)
#     img2 = img2.astype(np.float32)

#     mscn1, var, mu = compute_image_mscn_transform(img)
#     mscn1 = mscn1.astype(np.float32)

#     mscn2, _, _ = compute_image_mscn_transform(img2)
#     mscn2 = mscn2.astype(np.float32)

#     feats_lvl1 = extract_on_patches(mscn1, patch_size, stride)
#     feats_lvl2 = extract_on_patches(mscn2, patch_size // 2, stride // 2)

#     min_len = min(len(feats_lvl1), len(feats_lvl2))
#     if min_len == 0: return np.array([])
        
#     feats_lvl1 = feats_lvl1[:min_len]
#     feats_lvl2 = feats_lvl2[:min_len]
#     feats = np.hstack((feats_lvl1, feats_lvl2)) 
#     return feats

# def niqe(inputImgData):
#     patch_size = 96 # 建议维持 96 以获得更稳定的统计
#     module_path = dirname(__file__)
#     params = scipy.io.loadmat(join(module_path,'niqe_image_params.mat'))
#     pop_mu = np.ravel(params["pop_mu"])
#     pop_cov = params["pop_cov"]

#     M, N = inputImgData.shape
#     if M < patch_size or N < patch_size: return 5.0 # 尺寸过小后的回退值

#     feats = get_patches_test_features(inputImgData, patch_size)
#     if len(feats) == 0: return 5.0

#     sample_mu = np.mean(feats, axis=0)
#     sample_cov = np.cov(feats.T)
#     X = sample_mu - pop_mu
#     covmat = ((pop_cov + sample_cov) / 2.0)
#     pinvmat = scipy.linalg.pinv(covmat)
#     niqe_score = np.sqrt(np.dot(np.dot(X, pinvmat), X))
#     return niqe_score

# # ==================== 修复后的 NIQE 计算入口 ====================

# def safe_rgb2gray(img):
#     """手动实现 RGB 转灰度，确保输入输出范围一致 (即 [0, 255] 输入 -> [0, 255] 输出)"""
#     # img shape: [H, W, C]
#     return 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]

# def calculate_niqe_complete(img_tensor):
#     """
#     完整的NIQE计算函数 - 修正版
#     核心修复：手动计算灰度，防止 skimage.rgb2gray 隐式归一化到 [0, 1]
#     """
#     try:
#         # 1. 转 Numpy
#         if torch.is_tensor(img_tensor):
#             img_np = img_tensor.squeeze().detach().cpu().numpy()
#         else:
#             img_np = img_tensor
            
#             print(f"[DEBUG VERIFY NIQE] 原始输入范围: Min={img_np.min():.4f}, Max={img_np.max():.4f}")

#         # 2. [关键] 范围检查与拉伸
#         # 如果是 float 且最大值 <= 1.1，说明是 [0, 1] 范围，拉伸到 255
#         if img_np.dtype != np.uint8 and img_np.max() <= 1.1:
#             img_np = img_np * 255.0
            
#         # 3. 转灰度 (手动实现，确保保持 [0, 255] 范围)
#         if len(img_np.shape) == 3:
#             # 确保是 (H, W, C)
#             if img_np.shape[0] == 3:
#                 img_np = img_np.transpose(1, 2, 0)
            
#             # 手动计算灰度: Y = 0.299R + 0.587G + 0.114B
#             # 这样输出的 img_gray 依然是 [0, 255] 范围
#             if img_np.shape[2] >= 3:
#                 img_gray = (0.299 * img_np[:, :, 0] + 
#                             0.587 * img_np[:, :, 1] + 
#                             0.114 * img_np[:, :, 2])
#             else:
#                 img_gray = img_np[:, :, 0]
#         else:
#             img_gray = img_np

#         # 4. 类型转换与截断
#         img_gray = img_gray.astype(np.float32)
#         # 防止双三次插值等操作导致的轻微越界
#         img_gray = np.clip(img_gray, 0, 255) 
        
#         print(f"[DEBUG VERIFY NIQE] 最终计算范围: Min={img_gray.min():.4f}, Max={img_gray.max():.4f}")

#         # 5. 尺寸检查
#         min_required_size = 96 
#         H, W = img_gray.shape
#         if H < min_required_size or W < min_required_size:
#             return 5.0 # 返回平均值

#         # 6. 计算
#         return niqe(img_gray)

#     except Exception as e:
#         logger.error(f"Error in calculate_niqe_complete: {e}")
#         return 5.0

# def batch_niqe_complete(imgs):
#     try:
#         scores = [calculate_niqe_complete(img) for img in imgs]
#         return float(np.mean(scores))
#     except Exception:
#         return 0.0

# def batch_niqe_complete_scores(imgs):
#     try:
#         return [calculate_niqe_complete(img) for img in imgs]
#     except Exception:
#         return []


# # ==================== 修复后的 UISM/UICM (UIQM 组件) ====================
# # 关键修复：确保所有计算基于 [0, 255] 范围，以匹配论文公式权重

# def _calculate_gradient_entropy(gradient_map, bins=256):
#     try:
#         # 增加 range 参数确保直方图稳定
#         hist, _ = np.histogram(gradient_map, bins=bins, range=(0, max(np.max(gradient_map), 1e-6)))
#         hist = hist.astype(np.float32)
#         hist_sum = np.sum(hist) + 1e-10 
#         hist = hist / hist_sum
#         hist = hist + 1e-10 
#         entropy_val = -np.sum(hist * np.log2(hist))
#         return entropy_val
#     except Exception:
#         return 0.0

# def _calculate_edge_contrast(edge_map, original_intensity):
#     try:
#         edge_threshold = 0.1 * np.max(edge_map)
#         edge_mask = edge_map > edge_threshold
#         if np.sum(edge_mask) == 0: return 0.0
        
#         edge_intensity = np.mean(original_intensity[edge_mask])
#         non_edge_intensity = np.mean(original_intensity[~edge_mask])
        
#         if non_edge_intensity > 1e-6:
#             contrast = edge_intensity / non_edge_intensity
#         else:
#             contrast = edge_intensity
#         return contrast
#     except Exception:
#         return 0.0

# def _uism_improved(img):
#     """img: [H, W, 3] float32 in range [0, 255]"""
#     try:
#         # 我们这里也需要安全的灰度转换
#         if len(img.shape) == 3 and img.shape[2] == 3:
#              gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
#         else:
#              gray = img

#         # 梯度幅值
#         grad_x = filters.sobel_v(gray)
#         grad_y = filters.sobel_h(gray)
#         gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)

#         gradient_entropy = _calculate_gradient_entropy(gradient_magnitude)
#         edge_contrast = _calculate_edge_contrast(gradient_magnitude, gray)

#         return 0.6 * gradient_entropy + 0.4 * edge_contrast
#     except Exception:
#         return 0.0

# def _calculate_color_distribution_features(rg, yb):
#     try:
#         return {
#             'rg_mean': np.mean(rg), 'rg_std': np.std(rg), 
#             'yb_mean': np.mean(yb), 'yb_std': np.std(yb)
#         }
#     except Exception:
#         return {'rg_mean': 0, 'rg_std': 0, 'yb_mean': 0, 'yb_std': 0}

# def _uicm_improved(img):
#     """img: [H, W, 3] float32 in range [0, 255]"""
#     try:
#         R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]
#         rg = R - G
#         yb = (R + G) / 2 - B
        
#         stats_data = _calculate_color_distribution_features(rg, yb)
        
#         try:
#             color_correlation = np.corrcoef(rg.flatten(), yb.flatten())[0, 1]
#             if np.isnan(color_correlation): color_correlation = 0
#         except: color_correlation = 0
            
#         uicm = (-0.0268 * np.sqrt(stats_data['rg_mean'] ** 2 + stats_data['yb_mean'] ** 2) +
#                 0.1586 * np.sqrt(stats_data['rg_std'] + stats_data['yb_std']) +
#                 0.05 * (1 - abs(color_correlation)))
#         return uicm
#     except Exception:
#         return 0.0

# def _uiconm_improved(img):
#     """img: [H, W, 3] float32 in range [0, 255]"""
#     try:
#         if len(img.shape) == 3 and img.shape[2] == 3:
#              gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
#         else:
#              gray = img

#         std_contrast = np.std(gray)
        
#         grad_x = filters.sobel_v(gray)
#         grad_y = filters.sobel_h(gray)
#         grad_mag = np.sqrt(grad_x**2 + grad_y**2)
#         local_contrast = np.mean(grad_mag)

#         return 0.5 * (std_contrast / 128.0) + 0.5 * (local_contrast / 10.0)
#     except Exception:
#         return 0.0

# # ==================== 修复后的 UIQM 入口 ====================

# def calculate_uiqm(image):
#     """
#     UIQM 计算入口。
#     [关键修复]: 始终确保输入是 numpy float 且在 [0, 255] 范围。
#     """
#     try:
#         if image is None: return 0.0

#         # 1. 统一转为 Numpy HWC
#         if isinstance(image, torch.Tensor):
#             img_np = image.squeeze().detach().cpu().numpy()
#             if img_np.ndim == 3 and img_np.shape[0] == 3: # CHW -> HWC
#                 img_np = img_np.transpose(1, 2, 0)
#         elif isinstance(image, Image.Image):
#             img_np = np.array(image)
#         else:
#             img_np = image
            
#             print(f"[DEBUG VERIFY UIQM] 原始输入范围: Min={img_np.min():.4f}, Max={img_np.max():.4f}")

#         # 2. 统一转为 Float32 并拉伸到 [0, 255]
#         img_np = img_np.astype(np.float32)
#         if img_np.max() <= 1.1: 
#             print(f"[DEBUG VERIFY UIQM] -> 检测到 [0, 1] 范围，正在拉伸至 [0, 255]...")
#             img_np = img_np * 255.0
            
#         # 3. 截断保护
#         img_np = np.clip(img_np, 0, 255)
#         print(f"[DEBUG VERIFY UIQM] 最终计算范围: Min={img_np.min():.4f}, Max={img_np.max():.4f}")

#         # 4. 计算
#         uicm = _uicm_improved(img_np)
#         uism = _uism_improved(img_np)
#         uiconm = _uiconm_improved(img_np)

#         uiqm = 0.0282 * uicm + 0.2953 * uism + 3.5753 * uiconm
#         return float(uiqm)
#     except Exception as e:
#         logger.error(f"Error in calculate_uiqm: {e}")
#         return 0.0

# def batch_uiqm(images):
#     try:
#         scores = [calculate_uiqm(img) for img in images]
#         return float(np.mean(scores))
#     except Exception:
#         return 0.0

# def batch_uiqm_scores(images):
#     try:
#         return [calculate_uiqm(img) for img in images]
#     except Exception:
#         return []

# # ==================== 修复后的 BRISQUE 计算 ====================

# def batch_brisque(images, device='cuda', batch_size=32):
#     """
#     [关键修复]: 直接使用 Tensor (0-1) 进行计算，移除 PIL 转换，提高精度和速度。
#     """
#     try:
#         scores = []
#         processed_imgs = []
#         # 预处理：统一转为 [C, H, W] 的 Tensor，且范围 [0, 1]
#         for img in images:
#             if isinstance(img, torch.Tensor):
#                 t = img.detach()
#                 if t.dim() == 4: t = t.squeeze(0) 
                
#                 if not debug_printed:
#                     print(f"[DEBUG VERIFY BRISQUE] 原始Tensor范围: {t.min():.4f} - {t.max():.4f}")
                
                
#                 if t.max() > 1.1: 
                    
#                     if not debug_printed:
#                         print(f"[DEBUG VERIFY BRISQUE] -> 检测到 [0, 255]，归一化至 [0, 1]...")
                    
#                     t = t / 255.0 # 如果意外传入了 0-255，归一化
#                 processed_imgs.append(t)
#                 debug_printed = True
#             elif isinstance(img, np.ndarray):
#                 t = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
#                 processed_imgs.append(t)
                
#         # 分批计算
#         for i in range(0, len(processed_imgs), batch_size):
#             batch_list = processed_imgs[i:i + batch_size]
#             if not batch_list: continue
            
#             batch_tensor = torch.stack(batch_list).to(device)
            
#             # PIQ 需要 RGB
#             if batch_tensor.shape[1] == 1:
#                 batch_tensor = batch_tensor.repeat(1, 3, 1, 1)
            
#             batch_tensor = torch.clamp(batch_tensor, 0, 1)

#             with torch.no_grad():
#                 # data_range=1.0 对应 [0, 1] 输入
#                 batch_scores = piq.brisque(batch_tensor, data_range=1.0, reduction='none')
                
#             scores.extend(batch_scores.cpu().numpy().tolist())

#         if device == 'cuda': torch.cuda.empty_cache()
#         return float(np.mean(scores)) if scores else 0.0
#     except Exception as e:
#         logger.error(f"Error in batch_brisque: {e}")
#         return 0.0

# def batch_brisque_scores(images, device='cuda', batch_size=32):
#     try:
#         scores = []
#         processed_imgs = []
#         for img in images:
#             if isinstance(img, torch.Tensor):
#                 t = img.detach()
#                 if t.dim() == 4: t = t.squeeze(0)
#                 if t.max() > 1.1: t = t / 255.0
#                 processed_imgs.append(t)
#             elif isinstance(img, np.ndarray):
#                 t = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
#                 processed_imgs.append(t)
        
#         for i in range(0, len(processed_imgs), batch_size):
#             batch_list = processed_imgs[i:i + batch_size]
#             if not batch_list: continue
#             batch_tensor = torch.stack(batch_list).to(device)
#             if batch_tensor.shape[1] == 1: batch_tensor = batch_tensor.repeat(1, 3, 1, 1)
#             batch_tensor = torch.clamp(batch_tensor, 0, 1)
#             with torch.no_grad():
#                 batch_scores = piq.brisque(batch_tensor, data_range=1.0, reduction='none')
#             scores.extend(batch_scores.cpu().numpy().tolist())
            
#         if device == 'cuda': torch.cuda.empty_cache()
#         return scores
#     except Exception:
#         return []

# # ========= 评估器类接口 =========
# class ImageQualityEvaluator:
#     def __init__(self, device='cuda'):
#         self.device = device

#     def calculate_batch_niqe(self, imgs):
#         return batch_niqe_complete(imgs)

#     def calculate_batch_brisque(self, images, batch_size=32):
#         return batch_brisque(images, self.device, batch_size)

#     def calculate_batch_uiqm(self, images):
#         return batch_uiqm(images)
    
#     def calculate_niqe(self, img): return calculate_niqe_complete(img)
#     def calculate_brisque(self, img): return batch_brisque([img], self.device)
#     def calculate_uiqm(self, img): return calculate_uiqm(img)
    
#     def calculate_batch_niqe_scores(self, imgs): return batch_niqe_complete_scores(imgs)
#     def calculate_batch_brisque_scores(self, imgs): return batch_brisque_scores(imgs, self.device)
#     def calculate_batch_uiqm_scores(self, imgs): return batch_uiqm_scores(imgs)






import numpy as np
import scipy.misc
import scipy.io
from os.path import dirname
from os.path import join
import scipy
from PIL import Image
import scipy.ndimage
import scipy.special
import math
from skimage.transform import resize
import torch
from skimage import color, filters, feature
from scipy.ndimage import gaussian_filter
from scipy import stats
from torchvision import transforms
import piq
import warnings
import os
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ================= NIQE 辅助函数 (保持不变) =================
gamma_range = np.arange(0.2, 10, 0.001)
a = scipy.special.gamma(2.0 / gamma_range)
a *= a
b = scipy.special.gamma(1.0 / gamma_range)
c = scipy.special.gamma(3.0 / gamma_range)
prec_gammas = a / (b * c)

def aggd_features(imdata):
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

    imdata2_mean = np.mean(imdata2)
    if imdata2_mean != 0:
        r_hat = (np.average(np.abs(imdata)) ** 2) / (np.average(imdata2))
    else:
        r_hat = np.inf
    rhat_norm = r_hat * (((math.pow(gamma_hat, 3) + 1) * (gamma_hat + 1)) / math.pow(math.pow(gamma_hat, 2) + 1, 2))

    pos = np.argmin((prec_gammas - rhat_norm) ** 2);
    alpha = gamma_range[pos]

    gam1 = scipy.special.gamma(1.0 / alpha)
    gam2 = scipy.special.gamma(2.0 / alpha)
    gam3 = scipy.special.gamma(3.0 / alpha)

    aggdratio = np.sqrt(gam1) / np.sqrt(gam3)
    bl = aggdratio * left_mean_sqrt
    br = aggdratio * right_mean_sqrt
    N = (br - bl) * (gam2 / gam1)
    return (alpha, N, bl, br, left_mean_sqrt, right_mean_sqrt)

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
    alpha_m, N, bl, br, lsq, rsq = aggd_features(mscncoefs.copy())
    pps1, pps2, pps3, pps4 = paired_product(mscncoefs)
    alpha1, N1, bl1, br1, lsq1, rsq1 = aggd_features(pps1)
    alpha2, N2, bl2, br2, lsq2, rsq2 = aggd_features(pps2)
    alpha3, N3, bl3, br3, lsq3, rsq3 = aggd_features(pps3)
    alpha4, N4, bl4, br4, lsq4, rsq4 = aggd_features(pps4)
    return np.array([alpha_m, (bl + br) / 2.0,
                     alpha1, N1, bl1, br1,
                     alpha2, N2, bl2, br2,
                     alpha3, N3, bl3, br3,
                     alpha4, N4, bl4, br4])

def get_patches_test_features(img, patch_size, stride=8):
    return _get_patches_generic(img, patch_size, 0, stride)

def extract_on_patches(img, patch_size, stride=8):
    h, w = img.shape
    patch_size = int(patch_size)
    patches = []
    for j in range(0, h - patch_size + 1, stride):
        for i in range(0, w - patch_size + 1, stride):
            patch = img[j:j + patch_size, i:i + patch_size]
            patches.append(patch)
    patches = np.array(patches)
    patch_features = []
    for p in patches:
        patch_features.append(_niqe_extract_subband_feats(p))
    patch_features = np.array(patch_features)
    return patch_features

def _get_patches_generic(img, patch_size, is_train, stride):
    h, w = np.shape(img)
    if h < patch_size or w < patch_size:
        return np.array([]) 

    hoffset = (h % patch_size)
    woffset = (w % patch_size)
    if hoffset > 0: img = img[:-hoffset, :]
    if woffset > 0: img = img[:, :-woffset]

    img = img.astype(np.float32)
    # [修正]: preserve_range=True 确保 scale 2 的范围保持 [0, 255]
    img2 = resize(img, (img.shape[0] // 2, img.shape[1] // 2), 
                  order=3, mode='reflect', anti_aliasing=True, preserve_range=True)
    img2 = img2.astype(np.float32)

    mscn1, var, mu = compute_image_mscn_transform(img)
    mscn1 = mscn1.astype(np.float32)

    mscn2, _, _ = compute_image_mscn_transform(img2)
    mscn2 = mscn2.astype(np.float32)

    feats_lvl1 = extract_on_patches(mscn1, patch_size, stride)
    feats_lvl2 = extract_on_patches(mscn2, patch_size // 2, stride // 2)

    min_len = min(len(feats_lvl1), len(feats_lvl2))
    if min_len == 0: return np.array([])
        
    feats_lvl1 = feats_lvl1[:min_len]
    feats_lvl2 = feats_lvl2[:min_len]
    feats = np.hstack((feats_lvl1, feats_lvl2)) 
    return feats

def niqe(inputImgData):
    patch_size = 96 # 建议维持 96 以获得更稳定的统计
    module_path = dirname(__file__)
    params = scipy.io.loadmat(join(module_path,'niqe_image_params.mat'))
    pop_mu = np.ravel(params["pop_mu"])
    pop_cov = params["pop_cov"]

    M, N = inputImgData.shape
    if M < patch_size or N < patch_size: return 5.0 # 尺寸过小后的回退值

    feats = get_patches_test_features(inputImgData, patch_size)
    if len(feats) == 0: return 5.0

    sample_mu = np.mean(feats, axis=0)
    sample_cov = np.cov(feats.T)
    X = sample_mu - pop_mu
    covmat = ((pop_cov + sample_cov) / 2.0)
    pinvmat = scipy.linalg.pinv(covmat)
    niqe_score = np.sqrt(np.dot(np.dot(X, pinvmat), X))
    return niqe_score

# ==================== 修复后的 NIQE 计算入口 ====================

def calculate_niqe_complete(img_tensor):
    """
    完整的NIQE计算函数 - 修正版
    核心修复：手动计算灰度，防止 skimage.rgb2gray 隐式归一化到 [0, 1]
    """
    try:
        # 1. 转 Numpy
        if torch.is_tensor(img_tensor):
            img_np = img_tensor.squeeze().detach().cpu().numpy()
        else:
            img_np = img_tensor

        # 2. [关键] 范围检查与拉伸
        if img_np.dtype != np.uint8 and img_np.max() <= 1.1:
            img_np = img_np * 255.0
            
        # 3. 转灰度 (手动实现，确保保持 [0, 255] 范围)
        if len(img_np.shape) == 3:
            if img_np.shape[0] == 3:
                img_np = img_np.transpose(1, 2, 0)
            
            # 手动计算灰度: Y = 0.299R + 0.587G + 0.114B
            if img_np.shape[2] >= 3:
                img_gray = (0.299 * img_np[:, :, 0] + 
                            0.587 * img_np[:, :, 1] + 
                            0.114 * img_np[:, :, 2])
            else:
                img_gray = img_np[:, :, 0]
        else:
            img_gray = img_np

        # 4. 类型转换与截断
        img_gray = img_gray.astype(np.float32)
        img_gray = np.clip(img_gray, 0, 255) 

        # 5. 尺寸检查
        min_required_size = 96 
        H, W = img_gray.shape
        if H < min_required_size or W < min_required_size:
            return 5.0 # 返回平均值

        # 6. 计算
        return niqe(img_gray)

    except Exception as e:
        logger.error(f"Error in calculate_niqe_complete: {e}")
        return 5.0

def batch_niqe_complete(imgs):
    try:
        scores = [calculate_niqe_complete(img) for img in imgs]
        return float(np.mean(scores))
    except Exception:
        return 0.0

def batch_niqe_complete_scores(imgs):
    try:
        return [calculate_niqe_complete(img) for img in imgs]
    except Exception:
        return []


# ==================== 修复后的 UISM/UICM (UIQM 组件) ====================
# 关键修复：确保所有计算基于 [0, 255] 范围

def _calculate_gradient_entropy(gradient_map, bins=256):
    try:
        hist, _ = np.histogram(gradient_map, bins=bins, range=(0, max(np.max(gradient_map), 1e-6)))
        hist = hist.astype(np.float32)
        hist_sum = np.sum(hist) + 1e-10 
        hist = hist / hist_sum
        hist = hist + 1e-10 
        entropy_val = -np.sum(hist * np.log2(hist))
        return entropy_val
    except Exception:
        return 0.0

def _calculate_edge_contrast(edge_map, original_intensity):
    try:
        edge_threshold = 0.1 * np.max(edge_map)
        edge_mask = edge_map > edge_threshold
        if np.sum(edge_mask) == 0: return 0.0
        
        edge_intensity = np.mean(original_intensity[edge_mask])
        non_edge_intensity = np.mean(original_intensity[~edge_mask])
        
        if non_edge_intensity > 1e-6:
            contrast = edge_intensity / non_edge_intensity
        else:
            contrast = edge_intensity
        return contrast
    except Exception:
        return 0.0

def _uism_improved(img):
    """img: [H, W, 3] float32 in range [0, 255]"""
    try:
        # 安全灰度转换
        if len(img.shape) == 3 and img.shape[2] == 3:
             gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
        else:
             gray = img

        # 梯度幅值
        grad_x = filters.sobel_v(gray)
        grad_y = filters.sobel_h(gray)
        gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)

        gradient_entropy = _calculate_gradient_entropy(gradient_magnitude)
        edge_contrast = _calculate_edge_contrast(gradient_magnitude, gray)

        return 0.6 * gradient_entropy + 0.4 * edge_contrast
    except Exception:
        return 0.0

def _calculate_color_distribution_features(rg, yb):
    try:
        return {
            'rg_mean': np.mean(rg), 'rg_std': np.std(rg), 
            'yb_mean': np.mean(yb), 'yb_std': np.std(yb)
        }
    except Exception:
        return {'rg_mean': 0, 'rg_std': 0, 'yb_mean': 0, 'yb_std': 0}

def _uicm_improved(img):
    """img: [H, W, 3] float32 in range [0, 255]"""
    try:
        R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        rg = R - G
        yb = (R + G) / 2 - B
        
        stats_data = _calculate_color_distribution_features(rg, yb)
        
        try:
            color_correlation = np.corrcoef(rg.flatten(), yb.flatten())[0, 1]
            if np.isnan(color_correlation): color_correlation = 0
        except: color_correlation = 0
            
        uicm = (-0.0268 * np.sqrt(stats_data['rg_mean'] ** 2 + stats_data['yb_mean'] ** 2) +
                0.1586 * np.sqrt(stats_data['rg_std'] + stats_data['yb_std']) +
                0.05 * (1 - abs(color_correlation)))
        return uicm
    except Exception:
        return 0.0

def _uiconm_improved(img):
    """img: [H, W, 3] float32 in range [0, 255]"""
    try:
        if len(img.shape) == 3 and img.shape[2] == 3:
             gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
        else:
             gray = img

        std_contrast = np.std(gray)
        
        grad_x = filters.sobel_v(gray)
        grad_y = filters.sobel_h(gray)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        local_contrast = np.mean(grad_mag)

        return 0.5 * (std_contrast / 128.0) + 0.5 * (local_contrast / 10.0)
    except Exception:
        return 0.0

# ==================== 修复后的 UIQM 入口 ====================

def calculate_uiqm(image, return_components=False):
    """UIQM 计算入口。始终确保输入是 [0, 255] 范围"""
    # try:
    #     if image is None: return 0.0
    try:
        if image is None: 
            return (0.0, 0.0, 0.0, 0.0) if return_components else 0.0

        if isinstance(image, torch.Tensor):
            img_np = image.squeeze().detach().cpu().numpy()
            if img_np.ndim == 3 and img_np.shape[0] == 3: # CHW -> HWC
                img_np = img_np.transpose(1, 2, 0)
        elif isinstance(image, Image.Image):
            img_np = np.array(image)
        else:
            img_np = image

        img_np = img_np.astype(np.float32)
        if img_np.max() <= 1.1: 
            img_np = img_np * 255.0
            
        img_np = np.clip(img_np, 0, 255)

        uicm = _uicm_improved(img_np)
        uism = _uism_improved(img_np)
        uiconm = _uiconm_improved(img_np)

        uiqm = 0.0282 * uicm + 0.2953 * uism + 3.5753 * uiconm
        
        if return_components:
            return float(uiqm), float(uism), float(uicm), float(uiconm)
        else:
            return float(uiqm)
        
        # return float(uiqm)
    except Exception as e:
        logger.error(f"Error in calculate_uiqm: {e}")
        return 0.0

# def batch_uiqm(images):
#     try:
#         scores = [calculate_uiqm(img) for img in images]
#         return float(np.mean(scores))
#     except Exception:
#         return 0.0

def batch_uiqm(images, return_components=False):
    try:
        # 如果需要分量，计算每个图像的分量并求平均
        if return_components:
            results = [calculate_uiqm(img, return_components=True) for img in images]
            if not results: return 0.0, 0.0, 0.0, 0.0
            
            avg_uiqm = float(np.mean([r[0] for r in results]))
            avg_uism = float(np.mean([r[1] for r in results]))
            avg_uicm = float(np.mean([r[2] for r in results]))
            avg_uiconm = float(np.mean([r[3] for r in results]))
            return avg_uiqm, avg_uism, avg_uicm, avg_uiconm
        else:
            scores = [calculate_uiqm(img) for img in images]
            return float(np.mean(scores))
    except Exception:
        return (0.0, 0.0, 0.0, 0.0) if return_components else 0.0


def batch_uiqm_scores(images):
    try:
        return [calculate_uiqm(img) for img in images]
    except Exception:
        return []

# ==================== 修复后的 BRISQUE 计算 (无调试版) ====================

def batch_brisque(images, device='cuda', batch_size=32):
    """修复版：移除调试变量，防止报错"""
    try:
        scores = []
        processed_imgs = []
        for img in images:
            if isinstance(img, torch.Tensor):
                t = img.detach()
                if t.dim() == 4: t = t.squeeze(0) 
                if t.max() > 1.1: t = t / 255.0 # 如果意外传入 0-255，归一化到 0-1
                processed_imgs.append(t)
            elif isinstance(img, np.ndarray):
                t = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
                processed_imgs.append(t)
                
        for i in range(0, len(processed_imgs), batch_size):
            batch_list = processed_imgs[i:i + batch_size]
            if not batch_list: continue
            
            batch_tensor = torch.stack(batch_list).to(device)
            if batch_tensor.shape[1] == 1:
                batch_tensor = batch_tensor.repeat(1, 3, 1, 1)
            
            batch_tensor = torch.clamp(batch_tensor, 0, 1)

            with torch.no_grad():
                batch_scores = piq.brisque(batch_tensor, data_range=1.0, reduction='none')
            scores.extend(batch_scores.cpu().numpy().tolist())

        if device == 'cuda': torch.cuda.empty_cache()
        return float(np.mean(scores)) if scores else 0.0
    except Exception as e:
        logger.error(f"Error in batch_brisque: {e}")
        return 0.0

def batch_brisque_scores(images, device='cuda', batch_size=32):
    try:
        scores = []
        processed_imgs = []
        for img in images:
            if isinstance(img, torch.Tensor):
                t = img.detach()
                if t.dim() == 4: t = t.squeeze(0)
                if t.max() > 1.1: t = t / 255.0
                processed_imgs.append(t)
            elif isinstance(img, np.ndarray):
                t = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
                processed_imgs.append(t)
        
        for i in range(0, len(processed_imgs), batch_size):
            batch_list = processed_imgs[i:i + batch_size]
            if not batch_list: continue
            batch_tensor = torch.stack(batch_list).to(device)
            if batch_tensor.shape[1] == 1: batch_tensor = batch_tensor.repeat(1, 3, 1, 1)
            batch_tensor = torch.clamp(batch_tensor, 0, 1)
            with torch.no_grad():
                batch_scores = piq.brisque(batch_tensor, data_range=1.0, reduction='none')
            scores.extend(batch_scores.cpu().numpy().tolist())
            
        if device == 'cuda': torch.cuda.empty_cache()
        return scores
    except Exception:
        return []

# ========= 评估器类接口 =========
class ImageQualityEvaluator:
    def __init__(self, device='cuda'):
        self.device = device

    def calculate_batch_niqe(self, imgs):
        return batch_niqe_complete(imgs)

    def calculate_batch_brisque(self, images, batch_size=32):
        return batch_brisque(images, self.device, batch_size)

    def calculate_batch_uiqm(self, images, return_components=False):
        return batch_uiqm(images, return_components=return_components)
    
    def calculate_niqe(self, img): return calculate_niqe_complete(img)
    def calculate_brisque(self, img): return batch_brisque([img], self.device)
    def calculate_uiqm(self, img, return_components=False): return calculate_uiqm(img, return_components=return_components)
    
    def calculate_batch_niqe_scores(self, imgs): return batch_niqe_complete_scores(imgs)
    def calculate_batch_brisque_scores(self, imgs): return batch_brisque_scores(imgs, self.device)
    def calculate_batch_uiqm_scores(self, imgs): return batch_uiqm_scores(imgs)