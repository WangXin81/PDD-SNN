import os
import cv2
import glob
import sys

def make_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

def mod_crop(img, scale):
    """
    关键步骤：将 HR 裁剪为能被 scale 整除的尺寸。
    这保证了 LR 和 HR 的像素严格对齐。
    """
    h, w = img.shape[:2]
    h_rem = h % scale
    w_rem = w % scale
    if h_rem == 0 and w_rem == 0:
        return img
    return img[:h - h_rem, :w - w_rem]

def generate_sr_dataset(hr_path, output_root, scales=[2, 3, 4]):
    """
    纯超分任务标准流程：HR -> ModCrop -> Bicubic Downsample -> LR
    """
    
    # 1. 读取所有 HR (Y) 图像
    img_list = sorted(glob.glob(os.path.join(hr_path, '*')))
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    img_list = [x for x in img_list if os.path.splitext(x)[1].lower() in valid_extensions]

    if not img_list:
        print(f"错误：在 {hr_path} 未找到图片！请指向 UFO-120 的 'hr' 文件夹。")
        return

    print(f"模式：纯超分重建 (Super-Resolution)")
    print(f"源目录 (HR): {hr_path}")
    print(f"处理数量: {len(img_list)} 张")

    # 2. 为每个倍率创建文件夹
    for scale in scales:
        make_directory(os.path.join(output_root, f'LR_x{scale}'))

    # 3. 循环处理
    for i, img_path in enumerate(img_list):
        # 读取 HR (Y)
        hr_img = cv2.imread(img_path)
        if hr_img is None: continue

        img_name = os.path.basename(img_path)

        for scale in scales:
            # --- 步骤 A: Mod Crop (必须做，否则无法训练) ---
            # 比如 HR 是 640x480。
            # x2: 640%2=0, 完美。
            # x3: 640%3=1, 需裁剪为 639x480。
            hr_cropped = mod_crop(hr_img, scale)
            
            # --- 步骤 B: 计算 LR 尺寸 ---
            h_hr, w_hr = hr_cropped.shape[:2]
            h_lr = h_hr // scale
            w_lr = w_hr // scale
            
            # --- 步骤 C: Bicubic 下采样 ---
            # 生成纯净的 LR 图像
            lr_img = cv2.resize(hr_cropped, (w_lr, h_lr), interpolation=cv2.INTER_CUBIC)
            
            # --- 步骤 D: 保存 ---
            save_path = os.path.join(output_root, f'LR_x{scale}', img_name)
            cv2.imwrite(save_path, lr_img)

        sys.stdout.write(f'\r进度: {i+1}/{len(img_list)} - {img_name}')

    print("\n\n=== 生成完成！ ===")
    print("您现在拥有了标准的 SR 数据集：")
    print(f"HR: {hr_path}")
    print(f"LR: {output_root}/LR_x2, LR_x3, LR_x4")

if __name__ == '__main__':
    # ================= 配置区域 =================
    # 指向 UFO-120 的 'hr' 文件夹 (注意不是 lrd)
    input_hr_folder = r'/root/autodl-tmp/SNN/整合完整/整合分类/datasets/UFO120/UFO120/TEST/hr'   
    
    # 输出位置
    output_root = r'/root/autodl-tmp/SNN/整合完整/整合分类/datasets/UFO120/UFO120/TEST'
    
    # 生成 x2, x3, x4
    target_scales = [2, 3, 4]
    # ===========================================
    
    generate_sr_dataset(input_hr_folder, output_root, target_scales)