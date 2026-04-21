import torch


# 基础配置
class Config:
    # 数据路径
    HR_DIR = "/root/autodl-tmp/SNN/整合完整/Train-1360/hr1x2"
    LR_DIR = "/root/autodl-tmp/SNN/整合完整/Train-1360/LR"
    REF_HR_DIR = "/root/autodl-tmp/SNN/整合完整/Train-1360/hr2x2"
    SAVE_DIR = "/root/autodl-tmp/SNN/整合完整/整合分类/zhenghe2/checkpoints"
    VAL_HR_DIR = "/root/autodl-tmp/SNN/整合完整/整合分类/datasets/Test-206完整的数据/hrx2"
    VAL_LR_DIR = "/root/autodl-tmp/SNN/整合完整/整合分类/datasets/Test-206完整的数据/lrd"
    
    VAL_BATCH_SIZE = 1      # 验证集通常一张一张验证
    NUM_WORKERS = 4         # 数据加载线程数 (根据你的CPU核心数调整，4是通用值)

    # 训练参数
    BATCH_SIZE = 4
    EFFECTIVE_BATCH_SIZE = 8
    ACCUMULATION_STEPS = EFFECTIVE_BATCH_SIZE // BATCH_SIZE
    EPOCHS = 5
    PRETRAIN_DEGRADATION_ONLY = True
    PRETRAIN_EPOCHS = 5
    
    # [关键配置 2] 初始状态：不冻结 (因为第一阶段要训练它)
    USE_PRETRAINED_RLGM = False       

    # [关键配置 3] 路径留空即可 (代码会自动处理内存路径)
    PRETRAINED_RLGM_PATH = ""

    # 学习率
    GENERATOR_LR = 1e-4
    DISCRIMINATOR_LR = 1e-5
    RECONSTRUCTION_LR = 1e-4

    # 设备配置
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_GPUS = torch.cuda.device_count()

    # 模型参数
    UPSCALE_FACTOR = 2
    VAL_SPLIT = 0.1
    VALIDATION_INTERVAL = 1
    TIME_STEPS = 5

    # 恢复训练
    RESUME_TRAINING = False
    RESUME_CHECKPOINT_PATH = "/root/autodl-tmp/SNN/整合完整/整合分类/zhenghe2/checkpoints/latest_checkpoint.pth"
    
    
    # [修改 1] 必须手动设为 True！
    # USE_PRETRAINED_RLGM = True
    # PRETRAINED_RLGM_PATH = "/root/autodl-tmp/SNN/整合完整/整合分类/zhenghe9_4/checkpoints/degradation_module_epoch250.pth"

    # 验证配置
    MAX_VAL_BATCHES = 5
    SAVE_ALL_VALIDATION_IMAGES = False

    # 学习率调度
    WARMUP_EPOCHS = 10
    DECAY_EPOCHS = 300

    # 损失权重
    FM_WEIGHT = 0.2
    MMD_WEIGHT = 0.2
    ADV_WEIGHT = 0.01
    LOWFREQ_WEIGHT = 1.0
    PERCEPT_WEIGHT = 0.1
    CT_WEIGHT = 0.02

    # 数据增强
    AUGMENTATION_PROBABILITY = 0.5

    # AMP配置
    USE_AMP = False
    PRECISION_DTYPE = torch.float32

    # 退化配置
    DEGRADATION_CONFIG = {
        'time_steps': TIME_STEPS,
        'downsample_factor': 2,
        'stats_path': "/root/autodl-tmp/SNN/整合完整/整合分类/KL1/scale2/degradation_stats.pth",
        'kernelgan_kernels_path': "/root/autodl-tmp/SNN/整合完整/整合分类/KL1/scale2/kernelgan_kernels.npy",
        'noise_patches_dir': "/root/autodl-tmp/SNN/整合完整/整合分类/KL1/scale2/noise_patches",
        'spike_type': 'binary',
        'use_temporal': True
    }

    # 重建配置
    RECONSTRUCTION_CONFIG = {
        'num_fusion_modules': 3,
        'time_steps': TIME_STEPS,
        'base_ch': 64,
        'upscale_factor': UPSCALE_FACTOR,
        'v_th': 0.3,
        'v_reset': 0.0,
        'tau': 2.0
    }
    

config = Config()