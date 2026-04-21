# import os
# import random
# from PIL import Image
# from torch.utils.data import Dataset, DataLoader
# from pathlib import Path
# import torch
# from torchvision import transforms

# class UnpairedAlignedDataset(Dataset):
#     """
#     HR / LR 数量一致时使用，严格一一对应
#     """

#     def __init__(self, hr_dir, lr_dir, hr_transform=None, lr_transform=None):
#         super().__init__()
#         self.hr_images = sorted(
#             [os.path.join(hr_dir, f) for f in os.listdir(hr_dir) if f.lower().endswith(('png', 'jpg', 'jpeg'))])
#         self.lr_images = sorted(
#             [os.path.join(lr_dir, f) for f in os.listdir(lr_dir) if f.lower().endswith(('png', 'jpg', 'jpeg'))])

#         assert len(self.hr_images) == len(self.lr_images), \
#             f"HR 数量 {len(self.hr_images)} 与 LR 数量 {len(self.lr_images)} 不一致！"

#         self.hr_transform = hr_transform
#         self.lr_transform = lr_transform

#     def __len__(self):
#         return len(self.hr_images)

#     def __getitem__(self, idx):
#         hr_path = self.hr_images[idx]
#         lr_path = self.lr_images[idx]

#         hr_img = Image.open(hr_path).convert("RGB")
#         lr_img = Image.open(lr_path).convert("RGB")

#         if self.hr_transform:
#             hr_img = self.hr_transform(hr_img)
#         if self.lr_transform:
#             lr_img = self.lr_transform(lr_img)

#         return {"hr": hr_img, "lr": lr_img}


# # class UnpairedUnalignedDataset(Dataset):
# #     """
# #     HR / LR 数量不一致时使用，采样时随机选择 HR 与 LR 配对
# #     """

# #     def __init__(self, hr_dir, lr_dir, hr_transform=None, lr_transform=None):
# #         super().__init__()
# #         self.hr_images = sorted(
# #             [os.path.join(hr_dir, f) for f in os.listdir(hr_dir) if f.lower().endswith(('png', 'jpg', 'jpeg', 'bmp'))])
# #         self.lr_images = sorted(
# #             [os.path.join(lr_dir, f) for f in os.listdir(lr_dir) if f.lower().endswith(('png', 'jpg', 'jpeg', 'bmp'))])

# #         self.hr_transform = hr_transform
# #         self.lr_transform = lr_transform

# #         self.hr_len = len(self.hr_images)
# #         self.lr_len = len(self.lr_images)

# #         # 数据集长度 = 较长的那个
# #         self.dataset_len = max(self.hr_len, self.lr_len)

# #     def __len__(self):
# #         return self.dataset_len

# #     def __getitem__(self, idx):
# #         # HR 图像：如果 idx 超过数量就随机采样
# #         if idx < self.hr_len:
# #             hr_path = self.hr_images[idx]
# #         else:
# #             hr_path = random.choice(self.hr_images)

# #         # LR 图像：如果 idx 超过数量就随机采样
# #         if idx < self.lr_len:
# #             lr_path = self.lr_images[idx]
# #         else:
# #             lr_path = random.choice(self.lr_images)

# #         hr_img = Image.open(hr_path).convert("RGB")
# #         lr_img = Image.open(lr_path).convert("RGB")

# #         if self.hr_transform:
# #             hr_img = self.hr_transform(hr_img)
# #         if self.lr_transform:
# #             lr_img = self.lr_transform(lr_img)

# #         return {"hr": hr_img, "lr": lr_img}


# class UnpairedUnalignedDataset(Dataset):
#     """
#     修改版：支持传入文件列表，以便在外部进行 Train/Val 划分
#     """
#     def __init__(self, hr_data, lr_data, hr_transform=None, lr_transform=None, is_train=True):
#         super().__init__()
        
#         # 1. 处理 HR 数据：如果是路径则读取，如果是列表则直接使用
#         if isinstance(hr_data, (str, Path)):
#             self.hr_images = sorted([os.path.join(hr_data, f) for f in os.listdir(hr_data) 
#                                      if f.lower().endswith(('png', 'jpg', 'jpeg', 'bmp'))])
#         else:
#             self.hr_images = hr_data

#         # 2. 处理 LR 数据
#         if isinstance(lr_data, (str, Path)):
#             self.lr_images = sorted([os.path.join(lr_data, f) for f in os.listdir(lr_data) 
#                                      if f.lower().endswith(('png', 'jpg', 'jpeg', 'bmp'))])
#         else:
#             self.lr_images = lr_data

#         self.hr_transform = hr_transform
#         self.lr_transform = lr_transform
#         self.is_train = is_train  # ✅ 修复：正确接收参数

#     def __len__(self):
#         return max(len(self.hr_images), len(self.lr_images))

#     def __getitem__(self, idx):
#         # HR 使用取模，保证确定性
#         hr_path = self.hr_images[idx % len(self.hr_images)]
        
#         if self.is_train:
#             # 训练模式：LR 随机采样
#             lr_path = random.choice(self.lr_images)
#         else:
#             # 验证模式：LR 确定性采样 (取模)
#             lr_path = self.lr_images[idx % len(self.lr_images)]

#         # 加载图像 (上下文管理器，安全)
#         with Image.open(hr_path) as img:
#             hr_img = img.convert("RGB")
#         with Image.open(lr_path) as img:
#             lr_img = img.convert("RGB")

#         if self.hr_transform:
#             hr_img = self.hr_transform(hr_img)
#         if self.lr_transform:
#             lr_img = self.lr_transform(lr_img)

#         return {"hr": hr_img, "lr": lr_img}



# class PairedReferenceHRDataset(Dataset):
#     def __init__(self, hr_dir, ref_hr_dir, transform=None):
#         super().__init__()
#         self.hr_dir = Path(hr_dir)
#         self.ref_hr_dir = Path(ref_hr_dir)
#         self.transform = transform
#         self.hr_files = []
#         for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp']:
#             self.hr_files.extend(list(self.hr_dir.glob(ext)))
#         if not self.hr_files:
#             raise ValueError(f"在 {hr_dir} 中没有找到HR图像文件")
#         self.pairs = []
#         missing_refs = []
#         for hr_file in self.hr_files:
#             ref_file = self.ref_hr_dir / hr_file.name
#             if ref_file.exists():
#                 self.pairs.append((hr_file, ref_file))
#             else:
#                 missing_refs.append(hr_file.name)
#         if not self.pairs:
#             raise ValueError(f"在 {ref_hr_dir} 中没有找到与HR图像对应的参考图像")
#         if missing_refs:
#             print(f"警告: 缺少 {len(missing_refs)} 个参考图像，例如: {missing_refs[:3]}")
#             print(f"将使用 {len(self.pairs)} 个配对图像进行计算")
#         print(f"找到 {len(self.pairs)} 个配对图像")

#     def __len__(self):
#         return len(self.pairs)

#     def __getitem__(self, idx):
#         hr_path, ref_path = self.pairs[idx]
#         hr_image = Image.open(hr_path)
#         ref_image = Image.open(ref_path)
#         if self.transform:
#             hr_image = self.transform(hr_image)
#             ref_image = self.transform(ref_image)
#         return {
#             'hr': hr_image,
#             'ref_hr': ref_image,
#             'hr_name': hr_path.name,
#             'ref_name': ref_path.name
#         }


# # def create_data_loaders(config):
# #     transform_hr = transforms.Compose([transforms.ToTensor()])
# #     transform_lr = transforms.Compose([transforms.ToTensor()])

# #     # 创建配对数据集（如果可用）
# #     try:
# #         paired_dataset = PairedReferenceHRDataset(
# #             hr_dir=config.HR_DIR,
# #             ref_hr_dir=config.REF_HR_DIR,
# #             transform=transform_hr
# #         )
# #         val_size = int(len(paired_dataset) * config.VAL_SPLIT)
# #         train_size = len(paired_dataset) - val_size

# #         from torch.utils.data import random_split
# #         train_paired_dataset, val_paired_dataset = random_split(paired_dataset, [train_size, val_size])
# #         train_paired_loader = DataLoader(
# #             train_paired_dataset,
# #             batch_size=config.BATCH_SIZE,
# #             shuffle=True,
# #             num_workers=4,
# #             pin_memory=True
# #         )
# #         val_paired_loader = DataLoader(
# #             val_paired_dataset,
# #             batch_size=config.BATCH_SIZE,
# #             shuffle=False,
# #             num_workers=2,
# #             pin_memory=True
# #         )
# #         PAIRED_DATASET_AVAILABLE = True
# #     except Exception as e:
# #         print(f"警告: 无法创建配对数据集: {e}")
# #         PAIRED_DATASET_AVAILABLE = False
# #         train_paired_loader = None
# #         val_paired_loader = None

# #     # 创建非配对数据集
# #     full_dataset = UnpairedUnalignedDataset(
# #         hr_dir=config.HR_DIR,
# #         lr_dir=config.LR_DIR,
# #         hr_transform=transform_hr,
# #         lr_transform=transform_lr
# #     )
# #     val_size = int(len(full_dataset) * config.VAL_SPLIT)
# #     train_size = len(full_dataset) - val_size

# #     from torch.utils.data import random_split
# #     train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# #     train_loader = DataLoader(
# #         train_dataset,
# #         batch_size=config.BATCH_SIZE,
# #         shuffle=True,
# #         num_workers=4,
# #         pin_memory=True
# #     )
# #     val_loader = DataLoader(
# #         val_dataset,
# #         batch_size=config.BATCH_SIZE,
# #         shuffle=False,
# #         num_workers=2,
# #         pin_memory=True
# #     )

# #     return {
# #         'train_loader': train_loader,
# #         'val_loader': val_loader,
# #         'train_paired_loader': train_paired_loader,
# #         'val_paired_loader': val_paired_loader,
# #         'paired_available': PAIRED_DATASET_AVAILABLE
# #     }


# def create_data_loaders(config):
#     # -------------------------------------------------------
#     # 1. 定义 Transforms (解决尺寸不一致问题)
#     # -------------------------------------------------------
#     # 获取裁剪尺寸，如果没有定义则默认为 256
#     crop_size = getattr(config, 'CROP_SIZE', 256) 
    
#     # 训练预处理：随机裁剪 + 随机翻转
#     base_transform = transforms.Compose([
#         transforms.ToTensor()
#     ])

#     # -------------------------------------------------------
#     # 2. 处理配对数据集 (这部分逻辑保持你原有的，未变动)
#     # -------------------------------------------------------
#     train_paired_loader = None
#     val_paired_loader = None
#     PAIRED_DATASET_AVAILABLE = False
    
#     try:
#         # 注意：这里也建议把 transform 换成上面的 transform_train/val
#         # 但为了最小化改动，暂且保留原逻辑，或者你可以手动替换进去
#         paired_dataset = PairedReferenceHRDataset(
#             hr_dir=config.HR_DIR,
#             ref_hr_dir=config.REF_HR_DIR,
#             transform=base_transform # 建议用这个
#         )
#         val_size = int(len(paired_dataset) * config.VAL_SPLIT)
#         train_size = len(paired_dataset) - val_size
        
#         from torch.utils.data import random_split
#         train_paired_dataset, val_paired_dataset = random_split(paired_dataset, [train_size, val_size])
        
#         train_paired_loader = DataLoader(train_paired_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
#         val_paired_loader = DataLoader(val_paired_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
#         PAIRED_DATASET_AVAILABLE = True
#     except Exception as e:
#         print(f"提示: 未启用配对数据集 ({e})")

#     # -------------------------------------------------------
#     # 3. 处理非配对数据集 (核心修改部分)
#     # -------------------------------------------------------
    
#     # A. 获取所有文件路径列表
#     hr_files = sorted([os.path.join(config.HR_DIR, f) for f in os.listdir(config.HR_DIR) 
#                        if f.lower().endswith(('png', 'jpg', 'jpeg', 'bmp'))])
#     lr_files = sorted([os.path.join(config.LR_DIR, f) for f in os.listdir(config.LR_DIR) 
#                        if f.lower().endswith(('png', 'jpg', 'jpeg', 'bmp'))])

#     # B. 计算切分点
#     val_split = config.VAL_SPLIT
    
#     # 切分 HR 文件列表
#     val_len_hr = int(len(hr_files) * val_split)
#     train_hr_files = hr_files[:-val_len_hr] # 训练列表
#     val_hr_files = hr_files[-val_len_hr:]   # 验证列表
    
#     # 切分 LR 文件列表
#     val_len_lr = int(len(lr_files) * val_split)
#     train_lr_files = lr_files[:-val_len_lr] # 训练列表
#     val_lr_files = lr_files[-val_len_lr:]   # 验证列表

#     # C. 实例化两个独立的数据集对象
#     # 训练集：is_train=True (开启随机采样)，使用 RandomCrop
#     train_dataset = UnpairedUnalignedDataset(
#         hr_data=train_hr_files,
#         lr_data=train_lr_files,
#         hr_transform=base_transform, 
#         lr_transform=base_transform,
#         is_train=True  
#     )

#     # 验证集：is_train=False (关闭随机采样)，使用 CenterCrop
#     val_dataset = UnpairedUnalignedDataset(
#         hr_data=val_hr_files,
#         lr_data=val_lr_files,
#         hr_transform=base_transform,   
#         lr_transform=base_transform,
#         is_train=False 
#     )

#     # D. 创建 DataLoader
#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=config.BATCH_SIZE,
#         shuffle=True, # DataLoader 层的 Shuffle，打乱 Batch 顺序
#         num_workers=4,
#         pin_memory=True
#     )
    
#     val_loader = DataLoader(
#         val_dataset,
#         batch_size=config.BATCH_SIZE,
#         shuffle=False, # 验证集不需要 Shuffle
#         num_workers=2,
#         pin_memory=True
#     )

#     return {
#         'train_loader': train_loader,
#         'val_loader': val_loader,
#         'train_paired_loader': train_paired_loader,
#         'val_paired_loader': val_paired_loader,
#         'paired_available': PAIRED_DATASET_AVAILABLE
#     }

import os
import random
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import torch
from torchvision import transforms
from torchvision.transforms import functional as F

# =========================================================
# 1. 辅助工具类
# =========================================================
def is_image_file(filename):
    return any(filename.lower().endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'])

# =========================================================
# 2. 数据集定义
# =========================================================

class PairedReferenceHRDataset(Dataset):
    """
    [修正版] 配对数据集加载器
    参数:
        hr_dir: 高清图文件夹
        lr_dir: 低清图文件夹 (对应 main_gan 中的参数名)
        upscale_factor: 放大倍数 (兼容 main_gan 传参，虽然这里可能用不到)
        transform: 预处理
    """
    def __init__(self, hr_dir, lr_dir, upscale_factor=4, transform=None):
        super().__init__()
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.upscale_factor = upscale_factor
        self.transform = transform

        # 1. 获取并排序文件列表，确保一一对应
        self.hr_filenames = sorted([x for x in os.listdir(hr_dir) if is_image_file(x)])
        
        if self.lr_dir and os.path.exists(self.lr_dir):
            self.lr_filenames = sorted([x for x in os.listdir(lr_dir) if is_image_file(x)])
            # 简单检查数量
            if len(self.hr_filenames) != len(self.lr_filenames):
                print(f"⚠️ 警告: HR数量 ({len(self.hr_filenames)}) 与 LR数量 ({len(self.lr_filenames)}) 不一致，可能导致匹配错误！")
        else:
            self.lr_filenames = []
            print(f"⚠️ 警告: 未找到有效的 LR 目录: {self.lr_dir}")

    def __getitem__(self, index):
        # 1. 读取 HR
        hr_name = self.hr_filenames[index]
        hr_path = os.path.join(self.hr_dir, hr_name)
        hr_img = Image.open(hr_path).convert('RGB')

        # 2. 读取 LR (假设文件名排序后是一一对应的)
        lr_img = None
        if index < len(self.lr_filenames):
            lr_name = self.lr_filenames[index]
            lr_path = os.path.join(self.lr_dir, lr_name)
            lr_img = Image.open(lr_path).convert('RGB')

        # 3. 应用变换
        if self.transform:
            hr_tensor = self.transform(hr_img)
            lr_tensor = self.transform(lr_img) if lr_img else None
        else:
            # 默认转 Tensor
            hr_tensor = F.to_tensor(hr_img)
            lr_tensor = F.to_tensor(lr_img) if lr_img else None

        # 4. 返回字典 (这是 main_gan 验证循环需要的格式)
        return {
            'hr': hr_tensor, 
            'lr': lr_tensor, 
            'hr_name': hr_name
        }

    def __len__(self):
        return len(self.hr_filenames)


class UnpairedUnalignedDataset(Dataset):
    """
    修改版：支持传入文件列表，以便在外部进行 Train/Val 划分
    """
    def __init__(self, hr_data, lr_data, hr_transform=None, lr_transform=None, is_train=True, upscale_factor=4):
        super().__init__()
        
        # 兼容 upscale_factor 参数传入（虽然这里可能不用，但为了防止报错）
        self.upscale_factor = upscale_factor

        # 1. 处理 HR 数据
        if isinstance(hr_data, (str, Path)):
            if os.path.exists(str(hr_data)):
                self.hr_images = sorted([os.path.join(hr_data, f) for f in os.listdir(hr_data) 
                                         if is_image_file(f)])
            else:
                self.hr_images = []
        else:
            self.hr_images = hr_data

        # 2. 处理 LR 数据
        if isinstance(lr_data, (str, Path)):
            if lr_data and os.path.exists(str(lr_data)):
                self.lr_images = sorted([os.path.join(lr_data, f) for f in os.listdir(lr_data) 
                                         if is_image_file(f)])
            else:
                self.lr_images = []
        else:
            self.lr_images = lr_data

        self.hr_transform = hr_transform
        self.lr_transform = lr_transform
        self.is_train = is_train

    def __len__(self):
        return max(len(self.hr_images), len(self.lr_images))

    def __getitem__(self, idx):
        # HR 使用取模，保证确定性
        if len(self.hr_images) > 0:
            hr_path = self.hr_images[idx % len(self.hr_images)]
        else:
            raise FileNotFoundError("HR 数据列表为空")
        
        if len(self.lr_images) > 0:
            if self.is_train:
                # 训练模式：LR 随机采样
                lr_path = random.choice(self.lr_images)
            else:
                # 验证模式：LR 确定性采样 (取模)
                lr_path = self.lr_images[idx % len(self.lr_images)]
        else:
            # 允许 LR 为空 (比如做模拟退化时)
            lr_path = None

        # 加载图像
        with Image.open(hr_path) as img:
            hr_img = img.convert("RGB")
        
        lr_img = None
        if lr_path:
            with Image.open(lr_path) as img:
                lr_img = img.convert("RGB")

        if self.hr_transform:
            hr_img = self.hr_transform(hr_img)
        if self.lr_transform and lr_img:
            lr_img = self.lr_transform(lr_img)
        elif lr_img: # 如果没有transform但也读到了图，默认转tensor
             lr_img = F.to_tensor(lr_img)

        return {"hr": hr_img, "lr": lr_img}

# =========================================================
# 3. 数据加载器创建函数
# =========================================================

def create_data_loaders(config):
    # 1. 定义 Transforms
    crop_size = getattr(config, 'CROP_SIZE', 256) 
    base_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # 2. 处理配对数据集 (训练用)
    train_paired_loader = None
    val_paired_loader = None
    PAIRED_DATASET_AVAILABLE = False
    
    try:
        # [关键修改] 这里对应修改了调用参数，把 ref_hr_dir 改为了 lr_dir
        paired_dataset = PairedReferenceHRDataset(
            hr_dir=config.HR_DIR,
            lr_dir=config.REF_HR_DIR,  # 传入 Ref 目录作为 LR 输入
            transform=base_transform
        )
        val_size = int(len(paired_dataset) * config.VAL_SPLIT)
        train_size = len(paired_dataset) - val_size
        
        from torch.utils.data import random_split
        train_paired_dataset, val_paired_dataset = random_split(paired_dataset, [train_size, val_size])
        
        train_paired_loader = DataLoader(train_paired_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
        val_paired_loader = DataLoader(val_paired_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
        PAIRED_DATASET_AVAILABLE = True
    except Exception as e:
        print(f"提示: 未启用配对数据集 ({e})")

    # 3. 处理非配对数据集 (Unpaired)
    # A. 获取文件列表
    if os.path.exists(config.HR_DIR):
        hr_files = sorted([os.path.join(config.HR_DIR, f) for f in os.listdir(config.HR_DIR) 
                           if is_image_file(f)])
    else:
        hr_files = []

    if os.path.exists(config.LR_DIR):
        lr_files = sorted([os.path.join(config.LR_DIR, f) for f in os.listdir(config.LR_DIR) 
                           if is_image_file(f)])
    else:
        lr_files = []

    # B. 切分 Train/Val
    val_split = config.VAL_SPLIT
    
    val_len_hr = int(len(hr_files) * val_split)
    train_hr_files = hr_files[:-val_len_hr]
    val_hr_files = hr_files[-val_len_hr:]
    
    val_len_lr = int(len(lr_files) * val_split)
    train_lr_files = lr_files[:-val_len_lr]
    val_lr_files = lr_files[-val_len_lr:]

    # C. 实例化 Dataset
    # 注意：这里也加上了 upscale_factor=4 参数，防止某些情况下报错
    train_dataset = UnpairedUnalignedDataset(
        hr_data=train_hr_files,
        lr_data=train_lr_files,
        hr_transform=base_transform, 
        lr_transform=base_transform,
        is_train=True,
        upscale_factor=getattr(config, 'UPSCALE_FACTOR', 4)
    )

    val_dataset = UnpairedUnalignedDataset(
        hr_data=val_hr_files,
        lr_data=val_lr_files,
        hr_transform=base_transform,   
        lr_transform=base_transform,
        is_train=False,
        upscale_factor=getattr(config, 'UPSCALE_FACTOR', 4)
    )

    # D. 创建 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'train_paired_loader': train_paired_loader,
        'val_paired_loader': val_paired_loader,
        'paired_available': PAIRED_DATASET_AVAILABLE
    }

