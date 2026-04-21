# import torch
# import torch.nn.functional as F
# import os
# import numpy as np
# from tqdm import tqdm
# from PIL import Image
# from torchvision import transforms
# import lpips
# from skimage.metrics import structural_similarity as ssim_func

# # =========================================================
# # 1. 尝试导入您项目中的计算函数 (用于 PSNR)
# # =========================================================
# try:
#     from losses1 import calculate_psnr
#     print(">>> 成功导入 losses1.py 中的 calculate_psnr")
# except ImportError:
#     print("警告：找不到 losses1.py，将使用内置的 PSNR 计算逻辑。")
#     # 定义一个简单的备用 PSNR 函数
#     def calculate_psnr(img1, img2):
#         mse = torch.mean((img1 - img2) ** 2)
#         if mse == 0: return 100
#         return 10 * torch.log10(1 / mse)

# # 2. 尝试导入 UIQM 评估器
# try:
#     from evaluate import ImageQualityEvaluator
#     HAS_UIQM = True
#     print(">>> 成功导入 evaluate.py 中的 ImageQualityEvaluator (UIQM)")
# except ImportError:
#     HAS_UIQM = False
#     print("警告：找不到 evaluate.py，将跳过 UIQM 计算。")

# # =========================================================
# # [核心修复] 安全的 SSIM 计算函数
# # 解决 "win_size exceeds image extent" 报错的关键
# # =========================================================
# def safe_calculate_ssim(img_tensor, gt_tensor):
#     """
#     输入: PyTorch Tensor [1, C, H, W] or [C, H, W], Range [0, 1]
#     功能: 自动转置维度，确保 skimage 能正确识别 (H, W, C)
#     """
#     # 1. 转为 Numpy 并移除 Batch 维度
#     img_np = img_tensor.squeeze().cpu().numpy() # 结果可能是 [C, H, W]
#     gt_np = gt_tensor.squeeze().cpu().numpy()
    
#     # 2. [关键] 如果是 CHW 格式 (例如 3, 128, 128)，强制转置为 HWC
#     if img_np.ndim == 3 and img_np.shape[0] == 3:
#         img_np = img_np.transpose(1, 2, 0) # 变成 [128, 128, 3]
#         gt_np = gt_np.transpose(1, 2, 0)
    
#     # 3. 计算 SSIM
#     try:
#         # 新版 skimage 参数
#         score = ssim_func(
#             gt_np, 
#             img_np, 
#             data_range=1.0, 
#             channel_axis=-1 
#         )
#     except TypeError:
#         # 旧版 skimage 参数
#         score = ssim_func(
#             gt_np, 
#             img_np, 
#             data_range=1.0, 
#             multichannel=True
#         )
#     return score

# # =========================================================
# # 主测试函数
# # =========================================================
# def run_bicubic_benchmark(hr_dir, upscale_factor=2, device='cuda'):
#     print(f"\n启动 Bicubic 基准测试 (Scale x{upscale_factor})...")
#     print(f"测试集路径: {hr_dir}")
    
#     device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
#     # 初始化 LPIPS
#     print("正在加载 LPIPS 模型...")
#     loss_fn_alex = lpips.LPIPS(net='alex').to(device)
#     loss_fn_alex.eval()
    
#     # 初始化 UIQM
#     if HAS_UIQM:
#         uiqm_evaluator = ImageQualityEvaluator(device=device)

#     # 读取文件
#     exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif')
#     # 过滤掉以 . 开头的隐藏文件 (如 .ipynb_checkpoints)
#     files = [f for f in os.listdir(hr_dir) if f.lower().endswith(exts) and not f.startswith('.')]
    
#     if len(files) == 0:
#         print("错误：目录下没有图片！")
#         return

#     metrics = {'psnr': [], 'ssim': [], 'lpips': [], 'uiqm': []}
#     to_tensor = transforms.ToTensor()

#     print(f"正在处理 {len(files)} 张图片...")
    
#     with torch.no_grad():
#         for filename in tqdm(files):
#             # A. 读取 HR
#             hr_path = os.path.join(hr_dir, filename)
#             try:
#                 hr_img = Image.open(hr_path).convert('RGB')
#             except Exception as e:
#                 print(f"读取失败: {filename}, {e}")
#                 continue
                
#             # B. 尺寸检查与裁剪
#             w, h = hr_img.size
            
#             # [新增] 跳过极小图片，防止 crash
#             if w < 8 or h < 8:
#                 print(f"跳过极小图片: {filename} ({w}x{h})")
#                 continue

#             new_w = w - (w % upscale_factor)
#             new_h = h - (h % upscale_factor)
            
#             if new_w != w or new_h != h:
#                 hr_img = hr_img.crop((0, 0, new_w, new_h))
            
#             # 转 Tensor [1, C, H, W]
#             hr_tensor = to_tensor(hr_img).unsqueeze(0).to(device)
            
#             # C. 模拟 Bicubic (先缩小再放大)
#             # Downsample
#             lr_tensor = F.interpolate(hr_tensor, scale_factor=1/upscale_factor, mode='bicubic', align_corners=False)
#             # Upsample (Baseline)
#             bicubic_tensor = F.interpolate(lr_tensor, size=(new_h, new_w), mode='bicubic', align_corners=False)
#             # Clamp 限制在 0-1
#             bicubic_tensor = torch.clamp(bicubic_tensor, 0, 1)

#             # D. 计算指标
#             try:
#                 # 1. PSNR
#                 cur_psnr = calculate_psnr(bicubic_tensor, hr_tensor)
#                 if isinstance(cur_psnr, torch.Tensor): cur_psnr = cur_psnr.item()
#                 metrics['psnr'].append(cur_psnr)

#                 # 2. SSIM (使用修复版函数)
#                 cur_ssim = safe_calculate_ssim(bicubic_tensor, hr_tensor)
#                 metrics['ssim'].append(cur_ssim)
                
#                 # 3. LPIPS
#                 cur_lpips = loss_fn_alex(bicubic_tensor, hr_tensor).item()
#                 metrics['lpips'].append(cur_lpips)
                
#                 # 4. UIQM
#                 if HAS_UIQM:
#                     res = uiqm_evaluator.calculate_metrics(bicubic_tensor)
#                     val = res.get('uiqm', 0.0) if isinstance(res, dict) else res
#                     metrics['uiqm'].append(val)
            
#             except Exception as e:
#                 print(f"计算指标出错 ({filename}): {e}")
#                 continue

#     # 5. 打印最终结果
#     print("\n" + "="*40)
#     print(f" Bicubic Baseline (Scale x{upscale_factor}) 最终结果")
#     print("="*40)
#     print(f" PSNR : {np.mean(metrics['psnr']):.4f}")
#     print(f" SSIM : {np.mean(metrics['ssim']):.4f}")
#     print(f" LPIPS: {np.mean(metrics['lpips']):.4f}")
#     if HAS_UIQM:
#         print(f" UIQM : {np.mean(metrics['uiqm']):.4f}")
#     else:
#         print(f" UIQM : 未计算")
#     print("="*40)

# if __name__ == "__main__":
#     # ==========================================
#     # 请确认此处路径指向的是【HR 原图】文件夹
#     # 比如: .../hrx2 或 .../hrx4
#     # ==========================================
#     TEST_HR_DIR = "/root/autodl-tmp/SNN/整合完整/整合分类/datasets/Test-206完整的数据/lrd" 
    
#     # 这里的 upscale_factor 要和你测试的倍率一致 (2, 3, or 4)
#     run_bicubic_benchmark(TEST_HR_DIR, upscale_factor=2)




# import torch
# import torch.nn as nn
# from thop import profile
# import os
# import sys

# # 导入您的项目模块 (确保这些文件在同一目录下)
# from config import config
# from models import ReconstructionModule

# def calculate_table5_metrics():
#     print("=" * 50)
#     print(">>> 正在计算 Table 5 所需指标 (Params, GFLOPs, Energy)")
#     print("=" * 50)

#     # 1. 加载模型
#     # 注意：这里我们只加载“重建网络”，因为它是推理时唯一工作的模块
#     device = config.DEVICE
#     print(f"正在初始化重建网络 (Device: {device})...")
    
#     try:
#         # 根据您的 config 初始化模型
#         model = ReconstructionModule(**config.RECONSTRUCTION_CONFIG).to(device)
#         model.eval()
#     except Exception as e:
#         print(f"模型初始化失败: {e}")
#         print("请检查 models.py 是否在当前目录下，以及 config 配置是否正确。")
#         return

#     # ==========================================
#     # 指标 1: Params (参数量)
#     # ==========================================
#     total_params = sum(p.numel() for p in model.parameters())
#     params_m = total_params / 1e6
#     print(f"\n[1] Params (参数量):")
#     print(f"    数值: {params_m:.4f} M  <---【请直接填入 Table 5】")

#     # ==========================================
#     # 指标 2: GFLOPs (计算复杂度)
#     # ==========================================
#     # 定义输入尺寸 (先算 512x512，再换算)
#     input_h, input_w = 512, 512
#     dummy_input = torch.randn(1, 3, input_h, input_w).to(device)
    
#     print(f"\n[2] GFLOPs (计算量):")
#     try:
#         # 使用 thop 计算 MACs (Multi-Adds)
#         # 注意：thop返回的是MACs，在超分论文中通常直接作为 GFLOPs 汇报
#         macs, _ = profile(model, inputs=(dummy_input, ), verbose=False)
#         gflops_512 = macs / 1e9
        
#         # 换算到 720p (1280x720)
#         scale_factor = (1280 * 720) / (512 * 512)
#         gflops_720p = gflops_512 * scale_factor
        
#         print(f"    基准 (512x512): {gflops_512:.4f} G")
#         print(f"    目标 (720p)   : {gflops_720p:.4f} G  <---【请直接填入 Table 5】")
        
#     except Exception as e:
#         print(f"    thop 计算失败: {e}")

#     # ==========================================
#     # 指标 3: Energy (能耗)
#     # ==========================================
#     print(f"\n[3] Energy (推理能耗):")
#     # 这里我们需要一个 ASR (平均脉冲率) 的估计值
#     # 如果您之前 CSV 里有记录 (Epoch 21)，直接用那个最好
#     # 假设 CSV 里的 val_energy_J 是在 512x512 下测得的
    
#     # 提示用户手动输入 CSV 里的值来自动换算
#     try:
#         val_energy_512 = float(input("    请输入 CSV 中 Epoch 21 的 val_energy_J 数值 (例如 0.01175): "))
        
#         energy_720p = val_energy_512 * scale_factor
#         print(f"    --------------------------------")
#         print(f"    基准 (512x512): {val_energy_512:.5f} J")
#         print(f"    目标 (720p)   : {energy_720p:.5f} J  <---【请直接填入 Table 5】")
        
#     except ValueError:
#         print("    输入无效，跳过能耗换算。请手动计算: Energy_720p = Energy_512 * 3.52")

#     print("\n" + "=" * 50)
#     print("计算完成！请将箭头指示的数值填入论文表格。")
#     print("=" * 50)

# if __name__ == "__main__":
#     calculate_table5_metrics()


import torch
from config import config
from models import ReconstructionModule

# 1. 加载模型
print("正在加载模型...")
try:
    model = ReconstructionModule(**config.RECONSTRUCTION_CONFIG)
    
    # 2. 计算并打印参数量
    total = sum(p.numel() for p in model.parameters())
    print("=" * 30)
    print(f"Params (参数量): {total / 1e6:.4f} M")  # <--- 结果就在这里
    print("=" * 30)
    
except Exception as e:
    print(f"出错啦: {e}")