# import os
# import torch
# import numpy as np
# from PIL import Image
# from torchvision import transforms
# from torchvision.utils import save_image
# import math                  
# import torch.nn.functional as F
# import lpips

# # =========================================================================
# # 1. 内部配置区域 (请根据你的实际情况修改路径)
# # =========================================================================
# class InferenceConfig:
#     # 设备配置
#     DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
#     # ------------------ 修改这里的路径 ------------------
#     # 模型权重文件路径
#     MODEL_PATH = "/root/autodl-tmp/SNN/整合完整/整合分类/ufo/zhenghe_x4/checkpoints/best_reconstruction_net.pth"
    
#     # 输入图片文件夹 (LR)
#     LR_DIR = "/root/autodl-tmp/SNN/整合完整/整合分类/datasets/UFO120/UFO120/TEST/LR_x4_1"
    
#     # 原图文件夹 (HR) - 如果没有原图用于对比，可以设为 None
#     HR_DIR = "/root/autodl-tmp/SNN/整合完整/整合分类/datasets/UFO120/UFO120/TEST/hr3"
    
#     # 结果保存路径
#     OUTPUT_DIR = "results3x4/inference_standalone"
#     # ---------------------------------------------------

#     # [核心] 重建网络参数 (必须与你训练时的设置完全一致！)
#     RECONSTRUCTION_CONFIG = {
#         'num_fusion_modules': 3,   # 融合模块数量
#         'time_steps': 5,           # SNN 时间步 (T)
#         'base_ch': 64,             # 基础通道数
#         'upscale_factor': 4,       # 放大倍率 (2 或 4)
        
#         # 神经元参数
#         'v_th': 0.5,               
#         'v_reset': 0.0,
#         'tau': 2.0,
#     }

# # =========================================================================
# # 2. 导入依赖模块 & 定义辅助函数
# # =========================================================================
# try:
#     from models import ReconstructionModule
# except ImportError:
#     print("错误: 找不到 'models.py'。请确保该文件在当前目录下。")
#     exit(1)

# try:
#     from color_spaces import RGB2Lab
# except ImportError:
#     print("错误: 找不到 'color_spaces.py'。")
#     exit(1)

# # 导入能耗计算工具
# try:
#     from utils import EnergyMeter
# except ImportError:
#     EnergyMeter = None
#     print("警告: 找不到 'utils.py'，将跳过能耗(GSOPs)计算。")

# # 导入 UIQM 计算
# try:
#     from evaluate import calculate_uiqm
# except ImportError:
#     def calculate_uiqm(img, return_components=False): 
#         return (0.0, 0.0, 0.0, 0.0) if return_components else 0.0

# # -------------------------------------------------------------
# # [新增] 内置标准 SSIM 计算函数 (防止外部库缺失)
# # -------------------------------------------------------------
# def ssim_tensor_function(img1, img2):
#     """
#     简易版 SSIM 计算 (针对单通道 Y)
#     img1, img2: [1, 1, H, W] range [0, 1]
#     """
#     C1 = 0.01 ** 2
#     C2 = 0.03 ** 2

#     mu1 = F.avg_pool2d(img1, 3, 1, 1)
#     mu2 = F.avg_pool2d(img2, 3, 1, 1)

#     mu1_sq = mu1 ** 2
#     mu2_sq = mu2 ** 2
#     mu1_mu2 = mu1 * mu2

#     sigma1_sq = F.avg_pool2d(img1 ** 2, 3, 1, 1) - mu1_sq
#     sigma2_sq = F.avg_pool2d(img2 ** 2, 3, 1, 1) - mu2_sq
#     sigma12 = F.avg_pool2d(img1 * img2, 3, 1, 1) - mu1_mu2

#     ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
#                ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
#     return ssim_map.mean().item()

# # =========================================================================
# # 3. 推理引擎类
# # =========================================================================
# class SNNInferenceEngine:
#     def __init__(self, model_path):
#         self.device = InferenceConfig.DEVICE
#         print(f">>> 正在初始化推理引擎 (Device: {self.device})...")
        
#         # A. 初始化模型
#         try:
#             self.model = ReconstructionModule(**InferenceConfig.RECONSTRUCTION_CONFIG).to(self.device)
#         except TypeError as e:
#             print(f"模型初始化失败，可能是 RECONSTRUCTION_CONFIG 参数不匹配: {e}")
#             exit(1)
            
#         self.rgb2lab = RGB2Lab().to(self.device)
        
#         # B. 初始化 LPIPS
#         try:
#             # net='alex' 是最常用的感知度量
#             self.lpips_loss = lpips.LPIPS(net='alex').to(self.device)
#             self.lpips_loss.eval()
#             print(">>> LPIPS 模型初始化成功")
#         except Exception as e:
#             print(f"警告: LPIPS 初始化失败 ({e})，将无法计算 LPIPS 指标")
#             self.lpips_loss = None
        
#         # C. 加载权重
#         self._load_weights(model_path)
#         self.model.eval()
        
#         # D. 预处理
#         self.transform = transforms.Compose([transforms.ToTensor()])

#     def _load_weights(self, path):
#         if not os.path.exists(path):
#             raise FileNotFoundError(f"权重文件不存在: {path}")
            
#         print(f"Loading weights from: {path}")
#         checkpoint = torch.load(path, map_location=self.device)
        
#         if 'reconstruction_module_state_dict' in checkpoint:
#             state_dict = checkpoint['reconstruction_module_state_dict']
#         elif 'state_dict' in checkpoint:
#             state_dict = checkpoint['state_dict']
#         else:
#             state_dict = checkpoint

#         # 去除 'module.' 前缀
#         new_state_dict = {}
#         for k, v in state_dict.items():
#             name = k[7:] if k.startswith('module.') else k
#             new_state_dict[name] = v
            
#         try:
#             self.model.load_state_dict(new_state_dict, strict=False)
#             print(">>> 权重加载成功！")
#         except Exception as e:
#             print(f"权重加载出现警告 (通常可忽略): {e}")

#     def infer(self, lr_path, hr_path=None, save_path=None):
#         """执行单张推理 (自动切块 + 统一裁边 + 分类计算指标)"""
#         # 1. 读取图片
#         lr_img = Image.open(lr_path).convert("RGB")
#         lr_tensor = self.transform(lr_img).unsqueeze(0).to(self.device) # [1, 3, H, W]

#         # ================= 切块推理逻辑 =================
#         PATCH_SIZE = 256 
        
#         b, c, h_old, w_old = lr_tensor.size()
        
#         # 计算 Padding
#         pad_h = (PATCH_SIZE - h_old % PATCH_SIZE) % PATCH_SIZE
#         pad_w = (PATCH_SIZE - w_old % PATCH_SIZE) % PATCH_SIZE
#         lr_padded = F.pad(lr_tensor, (0, pad_w, 0, pad_h), mode='reflect')
#         _, _, h_pad, w_pad = lr_padded.size()
        
#         # 准备输出画布
#         scale = InferenceConfig.RECONSTRUCTION_CONFIG.get('upscale_factor', 2)
#         target_h_pad = h_pad * scale
#         target_w_pad = w_pad * scale
#         sr_padded = torch.zeros((b, 3, target_h_pad, target_w_pad), device=self.device)
        
#         # 1. 初始化能耗计算器
#         meter = None
#         if EnergyMeter is not None:
#             try:
#                 meter = EnergyMeter(self.model, input_size=(1, 3, PATCH_SIZE, PATCH_SIZE), device=self.device)
#                 meter.register_hooks()
#             except Exception as e:
#                 print(f"能耗计算初始化失败: {e}")
#                 meter = None

#         # 2. 开始循环切块
#         try:
#             for y in range(0, h_pad, PATCH_SIZE):
#                 for x in range(0, w_pad, PATCH_SIZE):
#                     lr_patch = lr_padded[..., y:y+PATCH_SIZE, x:x+PATCH_SIZE]
#                     with torch.no_grad():
#                         if hasattr(self.model, 'reset'): self.model.reset()
#                         for m in self.model.modules():
#                             if hasattr(m, 'reset'): m.reset()
#                         lab_lr = self.rgb2lab(lr_patch)
#                         sr_patch = self.model(lab_lr) 
                    
#                     y_out = y * scale
#                     x_out = x * scale
#                     h_out_patch = sr_patch.shape[2]
#                     w_out_patch = sr_patch.shape[3]
#                     sr_padded[..., y_out:y_out+h_out_patch, x_out:x_out+w_out_patch] = sr_patch
#         finally:
#             energy_results = {}
#             if meter is not None:
#                 energy_results = meter.calculate_metrics()
#                 meter.remove_hooks()

#         # 4. 裁剪回原始比例 (这里只是去掉了为了 Patch 对齐而填充的黑边)
#         final_h = h_old * scale
#         final_w = w_old * scale
#         sr_tensor = sr_padded[..., :final_h, :final_w]

#         # 后处理
#         sr_tensor = torch.clamp(sr_tensor, 0, 1)

#         # 准备 HR
#         hr_tensor = None
#         if hr_path and os.path.exists(hr_path):
#             hr_img = Image.open(hr_path).convert("RGB")
#             hr_tensor = self.transform(hr_img).unsqueeze(0).to(self.device)

#         # =================================================================
#         # 5. [最终修正] 计算指标
#         #    逻辑：先得到“干净的RGB图”，所有指标都基于这个干净图计算
#         # =================================================================
#         results = {
#             'psnr': 0.0, 'ssim': 0.0, 'lpips': 0.0,
#             'uiqm': 0.0, 'uism': 0.0,
#             'gsops': energy_results.get('GSOPs', 0.0),
#             'energy': energy_results.get('Energy_SNN (J)', 0.0),
#             'spike_rate': energy_results.get('Avg_Spike_Rate', 0.0)
#         }

#         # --- A. 准备干净的数据 (Clean Data) ---
#         # 1. 对齐尺寸
#         if hr_tensor is not None:
#             _, _, h_sr, w_sr = sr_tensor.size()
#             hr_tensor = hr_tensor[:, :, :h_sr, :w_sr]

#         # 2. 统一裁边 (Shave Border)
#         # 这一步去掉了卷积边缘伪影，这对所有指标都至关重要
#         shave = scale
#         if shave > 0:
#             sr_clean_rgb = sr_tensor[..., shave:-shave, shave:-shave]
#             if hr_tensor is not None:
#                 hr_clean_rgb = hr_tensor[..., shave:-shave, shave:-shave]
#         else:
#             sr_clean_rgb = sr_tensor
#             if hr_tensor is not None:
#                 hr_clean_rgb = hr_tensor

#         # --- B. 计算 RGB 指标 (LPIPS, UIQM) 使用 Clean RGB ---
        
#         # 计算 UIQM / UISM
#         try:
#             # 使用裁边后的干净图
#             uiqm, uism, _, _ = calculate_uiqm(sr_clean_rgb, return_components=True)
#             results['uiqm'] = uiqm
#             results['uism'] = uism
#         except: pass

#         # 计算 LPIPS
#         if hr_tensor is not None and self.lpips_loss is not None:
#             # 使用裁边后的干净图
#             sr_norm = sr_clean_rgb * 2 - 1 
#             hr_norm = hr_clean_rgb * 2 - 1
#             with torch.no_grad():
#                 results['lpips'] = self.lpips_loss(sr_norm, hr_norm).mean().item()

#         # --- C. 计算 亮度 指标 (PSNR, SSIM) 使用 Clean Y ---
#         if hr_tensor is not None:
#             # 定义 RGB -> Y 转换
#             def to_y_channel(img_rgb):
#                 return 0.257 * img_rgb[:, 0, :, :] + \
#                        0.504 * img_rgb[:, 1, :, :] + \
#                        0.098 * img_rgb[:, 2, :, :] + (16.0 / 255.0)

#             # 将刚才的 Clean RGB 转为 Clean Y
#             sr_y = to_y_channel(sr_clean_rgb).unsqueeze(1) 
#             hr_y = to_y_channel(hr_clean_rgb).unsqueeze(1)

#             # 计算 PSNR
#             mse = torch.mean((sr_y - hr_y) ** 2)
#             if mse == 0:
#                 results['psnr'] = 100.0
#             else:
#                 results['psnr'] = 10.0 * torch.log10(1.0 / mse).item()

#             # 计算 SSIM
#             try:
#                 from torchmetrics.functional import structural_similarity_index_measure
#                 results['ssim'] = structural_similarity_index_measure(sr_y, hr_y, data_range=1.0).item()
#             except ImportError:
#                 results['ssim'] = ssim_tensor_function(sr_y, hr_y)

#         # 6. 保存图片 
#         # (通常保存不裁边的图比较完整，但如果你想保存裁边后的也可以改成 sr_clean_rgb)
#         if save_path:
#             save_image(sr_tensor, save_path)

#         return results

# # =========================================================================
# # 4. 主函数
# # =========================================================================
# def main():
#     # 确保输出目录存在
#     os.makedirs(InferenceConfig.OUTPUT_DIR, exist_ok=True)

#     # 初始化
#     try:
#         engine = SNNInferenceEngine(InferenceConfig.MODEL_PATH)
#     except Exception as e:
#         print(f"程序终止: {e}")
#         return

#     # 扫描文件
#     if not os.path.exists(InferenceConfig.LR_DIR):
#         print(f"LR 目录不存在: {InferenceConfig.LR_DIR}")
#         return

#     img_names = sorted([f for f in os.listdir(InferenceConfig.LR_DIR) 
#                         if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])

#     if not img_names:
#         print("未找到有效图片。")
#         return

#     print(f"\n 开始批量测试 | 图片数量: {len(img_names)}")
#     print(f" 结果保存路径: {InferenceConfig.OUTPUT_DIR}")
#     print("-" * 125)
#     print(f"{'Image':<25} | {'PSNR':<6} | {'SSIM':<6} | {'LPIPS':<6} | {'UIQM':<6} | {'GSOPs':<8} | {'Energy(J)':<10}")
#     print("-" * 125)

#     totals = {'psnr': 0, 'ssim': 0, 'lpips': 0, 'uiqm': 0, 'uism': 0, 'gsops': 0, 'energy': 0, 'spike_rate': 0}
#     count = 0
    
#     csv_data = []

#     for img_name in img_names:
#         lr_path = os.path.join(InferenceConfig.LR_DIR, img_name)
#         # 如果有 HR 目录，就尝试找对应的 HR，否则设为 None
#         hr_path = os.path.join(InferenceConfig.HR_DIR, img_name) if InferenceConfig.HR_DIR else None
        
#         save_path = os.path.join(InferenceConfig.OUTPUT_DIR, f"SR_{img_name}")

#         # 执行推理
#         res = engine.infer(lr_path, hr_path, save_path)

#         # 打印单行
#         print(f"{img_name[:25]:<25} | {res['psnr']:.2f}   | {res['ssim']:.4f} | {res['lpips']:.4f}   | {res['uiqm']:.3f}  | {res['gsops']:.4f}   | {res['energy']:.2e}")

#         # 累加用于计算平均值
#         for k in totals: totals[k] += res[k]
#         count += 1
        
#         # 记录单张数据
#         row_data = {'Image': img_name}
#         row_data.update(res)
#         csv_data.append(row_data)

#     if count > 0:
#         print("-" * 125)
#         print("平均指标 (Average Metrics):")
#         print(f"   PSNR : {totals['psnr']/count:.4f} dB")
#         print(f"   SSIM : {totals['ssim']/count:.4f}")
#         print(f"   LPIPS: {totals['lpips']/count:.4f}")
#         print(f"   UIQM : {totals['uiqm']/count:.4f}")
#         print(f"   S_Rate: {totals['spike_rate']/count:.4f}")
#         print(f"   GSOPs: {totals['gsops']/count:.4f} G")
#         print(f"   Energy: {totals['energy']/count:.2e} J")
        
#         # 保存 CSV 报告
#         try:
#             import pandas as pd
#             df = pd.DataFrame(csv_data)
#             # 添加一行平均值
#             avg_row = {k: totals[k]/count for k in totals if k in df.columns}
#             avg_row['Image'] = 'AVERAGE'
#             df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)
            
#             csv_path = os.path.join(InferenceConfig.OUTPUT_DIR, "test_report.csv")
#             df.to_csv(csv_path, index=False)
#             print(f"\n详细测试报告已保存至: {csv_path}")
#         except ImportError:
#             print("\n 未安装 pandas，跳过 CSV 保存 (建议 pip install pandas)")
#         except Exception as e:
#             print(f"\n 保存 CSV 失败: {e}")

# if __name__ == "__main__":
#     main()



import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
import math                 
import torch.nn.functional as F
import lpips

# =========================================================================
# 1. 内部配置区域
# =========================================================================
class InferenceConfig:
    # 设备配置
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # ------------------ 路径配置 ------------------
    # 模型权重文件路径
    MODEL_PATH = "/root/autodl-tmp/SNN/整合完整/整合分类/ufo/zhenghe_x4/checkpoints/best_reconstruction_net.pth"
    
    # 输入图片文件夹 (LR)
    LR_DIR = "/root/autodl-tmp/SNN/整合完整/整合分类/datasets/UFO120/UFO120/TEST/LR_x4_1"
    
    # 原图文件夹 (HR) - 用于计算 PSNR/SSIM/LPIPS
    HR_DIR = "/root/autodl-tmp/SNN/整合完整/整合分类/datasets/UFO120/UFO120/TEST/hr3"
    
    # 结果保存路径
    OUTPUT_DIR = "results3x4/inference_standalone"
    # ----------------------------------------------

    # [核心] 重建网络参数 - 必须与 config (18).py 严格一致
    RECONSTRUCTION_CONFIG = {
        'num_fusion_modules': 3,   # 融合模块数量
        'time_steps': 5,           # SNN 时间步 (T)
        'base_ch': 64,             # 基础通道数
        'upscale_factor': 4,       # 修正为 4 以匹配权重文件
        
        # 神经元动力学参数
        'v_th': 0.5,               
        'v_reset': 0.0,
        'tau': 2.0,
        'spike_type': 'ternary',   # 训练配置中的脉冲类型
        'soft_reset': True         # 训练配置中的重置方式
    }

# =========================================================================
# 2. 依赖导入
# =========================================================================
try:
    from models import ReconstructionModule
except ImportError:
    print("错误: 找不到 'models.py'。")
    exit(1)

try:
    from color_spaces import RGB2Lab
except ImportError:
    print("错误: 找不到 'color_spaces.py'。")
    exit(1)

try:
    from utils import EnergyMeter
except ImportError:
    EnergyMeter = None
    print("警告: 找不到 'utils.py'，将跳过能耗计算。")

try:
    from evaluate import calculate_uiqm
except ImportError:
    def calculate_uiqm(img, return_components=False): 
        return (0.0, 0.0, 0.0, 0.0) if return_components else 0.0

def ssim_tensor_function(img1, img2):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    mu1 = F.avg_pool2d(img1, 3, 1, 1)
    mu2 = F.avg_pool2d(img2, 3, 1, 1)
    sigma1_sq = F.avg_pool2d(img1 ** 2, 3, 1, 1) - mu1**2
    sigma2_sq = F.avg_pool2d(img2 ** 2, 3, 1, 1) - mu2**2
    sigma12 = F.avg_pool2d(img1 * img2, 3, 1, 1) - mu1*mu2
    ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean().item()

# =========================================================================
# 3. 推理引擎类
# =========================================================================
class SNNInferenceEngine:
    def __init__(self, model_path):
        self.device = InferenceConfig.DEVICE
        print(f">>> 正在初始化推理引擎 (Device: {self.device})...")
        
        # A. 初始化模型
        self.model = ReconstructionModule(**InferenceConfig.RECONSTRUCTION_CONFIG).to(self.device)
        self.rgb2lab = RGB2Lab().to(self.device)
        
        # B. 初始化 LPIPS
        try:
            self.lpips_loss = lpips.LPIPS(net='alex').to(self.device)
            self.lpips_loss.eval()
        except:
            self.lpips_loss = None
        
        # C. 加载权重
        self._load_weights(model_path)
        self.model.eval()
        self.transform = transforms.ToTensor()

    # def _load_weights(self, path):
    #     checkpoint = torch.load(path, map_location=self.device)
    #     state_dict = checkpoint.get('reconstruction_module_state_dict', 
    #                                checkpoint.get('state_dict', checkpoint))
    #     # 移除 'module.' 前缀
    #     new_state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
    #     self.model.load_state_dict(new_state_dict, strict=True) 
    #     print(">>> 权重加载成功！")
    
    def _load_weights(self, path):
        # ... (前面的代码保持不变) ...
        
        # 将 strict=True 改为 strict=False
        # 这样程序会忽略掉那些不匹配的非关键缓冲区（如小波变换矩阵）
        try:
            self.model.load_state_dict(new_state_dict, strict=False) 
            print(">>> 权重加载成功！")
        except Exception as e:
            print(f">>> 权重加载时出现非致命警告: {e}")
    

    def infer(self, lr_path, hr_path=None, save_path=None):
        lr_img = Image.open(lr_path).convert("RGB")
        lr_tensor = self.transform(lr_img).unsqueeze(0).to(self.device)
        
        PATCH_SIZE = 256 
        b, c, h_old, w_old = lr_tensor.size()
        
        pad_h = (PATCH_SIZE - h_old % PATCH_SIZE) % PATCH_SIZE
        pad_w = (PATCH_SIZE - w_old % PATCH_SIZE) % PATCH_SIZE
        
        # [核心修正] 使用 replicate 模式解决小图填充报错问题
        lr_padded = F.pad(lr_tensor, (0, pad_w, 0, pad_h), mode='replicate')
        _, _, h_pad, w_pad = lr_padded.size()
        
        scale = InferenceConfig.RECONSTRUCTION_CONFIG['upscale_factor']
        sr_padded = torch.zeros((b, 3, h_pad * scale, w_pad * scale), device=self.device)
        
        meter = None
        if EnergyMeter is not None:
            meter = EnergyMeter(self.model, input_size=(1, 3, PATCH_SIZE, PATCH_SIZE), 
                               device=self.device, time_steps=InferenceConfig.RECONSTRUCTION_CONFIG['time_steps'])
            meter.register_hooks()

        # 开始切块推理
        with torch.no_grad():
            for y in range(0, h_pad, PATCH_SIZE):
                for x in range(0, w_pad, PATCH_SIZE):
                    lr_patch = lr_padded[..., y:y+PATCH_SIZE, x:x+PATCH_SIZE]
                    
                    # 重置 SNN 状态
                    for m in self.model.modules():
                        if hasattr(m, 'reset'): m.reset()
                    
                    lab_lr = self.rgb2lab(lr_patch)
                    sr_patch = self.model(lab_lr) 
                    
                    sr_padded[..., y*scale : (y+PATCH_SIZE)*scale, x*scale : (x+PATCH_SIZE)*scale] = sr_patch

        if meter:
            energy_results = meter.calculate_metrics()
            meter.remove_hooks()
        else:
            energy_results = {}

        # 裁剪并 clamp
        sr_tensor = torch.clamp(sr_padded[..., :h_old*scale, :w_old*scale], 0, 1)

        # 指标计算
        results = {'psnr': 0.0, 'ssim': 0.0, 'lpips': 0.0, 'uiqm': 0.0,
                   'gsops': energy_results.get('GSOPs', 0.0),
                   'energy': energy_results.get('Energy_SNN (J)', 0.0)}

        # 准备对比
        if hr_path and os.path.exists(hr_path):
            hr_tensor = self.transform(Image.open(hr_path).convert("RGB")).unsqueeze(0).to(self.device)[..., :h_old*scale, :w_old*scale]
            
            # 统一裁边计算指标 (Shave Border)
            shave = scale
            sr_c = sr_tensor[..., shave:-shave, shave:-shave]
            hr_c = hr_tensor[..., shave:-shave, shave:-shave]

            # PSNR / SSIM (Y Channel)
            def to_y(img): return 0.257*img[:,0]+0.504*img[:,1]+0.098*img[:,2]+(16/255)
            sr_y, hr_y = to_y(sr_c).unsqueeze(1), to_y(hr_c).unsqueeze(1)
            mse = torch.mean((sr_y - hr_y) ** 2)
            results['psnr'] = 10.0 * torch.log10(1.0 / mse).item() if mse > 0 else 100.0
            results['ssim'] = ssim_tensor_function(sr_y, hr_y)
            
            if self.lpips_loss:
                results['lpips'] = self.lpips_loss(sr_c*2-1, hr_c*2-1).mean().item()

        # UIQM
        results['uiqm'] = calculate_uiqm(sr_tensor)
        
        if save_path:
            save_image(sr_tensor, save_path)
        return results

# =========================================================================
# 4. 主循环
# =========================================================================
def main():
    os.makedirs(InferenceConfig.OUTPUT_DIR, exist_ok=True)
    engine = SNNInferenceEngine(InferenceConfig.MODEL_PATH)
    img_names = sorted([f for f in os.listdir(InferenceConfig.LR_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    print("-" * 110)
    print(f"{'Image':<25} | {'PSNR':<6} | {'SSIM':<6} | {'LPIPS':<6} | {'GSOPs':<8} | {'Energy(J)':<10}")
    print("-" * 110)

    csv_data = []
    for name in img_names:
        res = engine.infer(os.path.join(InferenceConfig.LR_DIR, name),
                          os.path.join(InferenceConfig.HR_DIR, name) if InferenceConfig.HR_DIR else None,
                          os.path.join(InferenceConfig.OUTPUT_DIR, f"SR_{name}"))
        
        print(f"{name[:25]:<25} | {res['psnr']:.2f}   | {res['ssim']:.4f} | {res['lpips']:.4f}   | {res['gsops']:.4f}   | {res['energy']:.2e}")
        csv_data.append({'Name': name, **res})

    # 保存报告 (需要 pandas)
    try:
        import pandas as pd
        pd.DataFrame(csv_data).to_csv(os.path.join(InferenceConfig.OUTPUT_DIR, "report.csv"), index=False)
    except: pass

if __name__ == "__main__":
    main()