import torch

# 基础配置
class Config:
    # ==================== [数据路径] ====================
    # 确保 HR 和 LR 目录下的图片文件名是一一对应的
    HR_DIR = "/root/autodl-tmp/SNN/整合完整/整合分类/datasets/UFO120/UFO120/train_val/hr"
    LR_DIR = "/root/autodl-tmp/SNN/整合完整/整合分类/datasets/UFO120/UFO120/train_val/lrd"
    # REF_HR_DIR = "autodl-tmp/SNN/整合完整/整合分类/datasets/USR-248/hr"
    SAVE_DIR = "/root/autodl-tmp/SNN/整合完整/整合分类/ufo/zhenghe_x2_1/checkpoints"
    
    # 验证集路径
    VAL_HR_DIR = "/root/autodl-tmp/SNN/整合完整/整合分类/datasets/UFO120/UFO120/TEST/hr3"
    VAL_LR_DIR = "/root/autodl-tmp/SNN/整合完整/整合分类/datasets/UFO120/UFO120/TEST/lrd3"
    
    
    VAL_BATCH_SIZE = 1      
    NUM_WORKERS = 4
    LR_PATCH_SIZE = 48

    # ==================== [训练模式调整] ====================
    # [关键修改] 关闭退化模块预训练，直接进入重建训练
    PRETRAIN_DEGRADATION_ONLY = False
    PRETRAIN_EPOCHS = 0
    
    # [关键修改] 不使用模型生成的 LR，直接加载数据集里的 LR
    USE_PRETRAINED_RLGM = False       
    PRETRAINED_RLGM_PATH = ""

    # 训练参数
    BATCH_SIZE = 4
    EFFECTIVE_BATCH_SIZE = 8
    ACCUMULATION_STEPS = EFFECTIVE_BATCH_SIZE // BATCH_SIZE
    EPOCHS = 500
    
    # ==================== [学习率配置] ====================
    # 因为不训练退化模块和判别器，这两个学习率在 train_step 中将不再起作用
    # GENERATOR_LR = 2e-5
    # DISCRIMINATOR_LR = 1e-5
    # 重建模块的学习率（建议保持 1e-4 或根据实验微调）
    RECONSTRUCTION_LR = 2e-5

    # 设备配置
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_GPUS = torch.cuda.device_count()

    # 模型参数
    UPSCALE_FACTOR = 2
    # VAL_SPLIT = 0.1
    # VALIDATION_INTERVAL = 1
    TIME_STEPS = 5

    # 恢复训练
    RESUME_TRAINING = False
    RESUME_CHECKPOINT_PATH = "/root/autodl-tmp/SNN/整合完整/整合分类/ufo/zhenghe_x2_1/checkpoints/latest_checkpoint.pth"
    
    # 验证配置
    MAX_VAL_BATCHES = 5
    SAVE_ALL_VALIDATION_IMAGES = False

    # 学习率调度
    WARMUP_EPOCHS = 25
    DECAY_EPOCHS = 500

    # ==================== [损失权重优化] ====================
    # 针对配对数据的监督学习权重
    L1_WEIGHT = 10.0        # 像素级损失，保证色准
    PERCEPT_WEIGHT = 0.01   # 感知损失 (DINO/VGG)，保证结构 [已调高]
    LPIPS_WEIGHT = 0.1     # 感知相似度损失，保证观感
    GRAD_WEIGHT = 0.5      # 梯度损失，保证边缘锐度
    
    # 以下为退化模块相关权重，在纯重建模式下不生效
    # FM_WEIGHT = 0.2
    # MMD_WEIGHT = 0.2
    # ADV_WEIGHT = 0.01
    # LOWFREQ_WEIGHT = 1.0
    # CT_WEIGHT = 0.02

    # 数据增强
    AUGMENTATION_PROBABILITY = 0.5

    # AMP配置
    USE_AMP = True
    PRECISION_DTYPE = torch.float32

    # # ==================== [退化配置 - 仅保留参数] ====================
    # DEGRADATION_CONFIG = {
    #     'time_steps': TIME_STEPS,
    #     'downsample_factor': 3,
    #     'stats_path': "/root/autodl-tmp/SNN/整合完整/整合分类/KL1/scale3/degradation_stats.pth",
    #     'kernelgan_kernels_path': "/root/autodl-tmp/SNN/整合完整/整合分类/KL1/scale3/kernelgan_kernels.npy",
    #     'noise_patches_dir': "/root/autodl-tmp/SNN/整合完整/整合分类/KL1/scale3/noise_patches",
    #     'spike_type': 'ternary',
    #     'use_temporal': True
    # }

    # ==================== [重建配置 - 你的 SNN 核心] ====================
    RECONSTRUCTION_CONFIG = {
        'num_fusion_modules': 3,
        'time_steps': TIME_STEPS,
        'base_ch': 64,
        'upscale_factor': UPSCALE_FACTOR,
        'v_th': 0.5,        # 神经元阈值
        'v_reset': 0.0,
        'tau': 2.0,         # 时间常数
        'spike_type': 'ternary',
        'soft_reset': True 
    }
    
config = Config()