from underwater_snn.config_types import EvalConfig, ExperimentConfig, ModelConfig, PathConfig, RuntimeConfig, TrainConfig


PATHS = PathConfig(
    train_hr_dir="data/Train-1360/train/hr",
    train_lr_dir="data/Train-1360/train/real_lr",
    ref_hr_dir="data/Train-1360/train/ref_hr",
    val_hr_dir="data/Test-206/hr",
    val_lr_dir="data/Test-206/lr_x2",
    save_dir="outputs/train1360_joint_x2",
    niqe_prior_path="outputs/train1360_joint_x2/niqe_water_params.mat",
    degradation_stats_path="data/priors/scale2/degradation_stats.pth",
    kernelgan_kernels_path="data/priors/scale2/kernelgan_kernels.npy",
    noise_patches_dir="data/priors/scale2/noise_patches",
)


MODEL = ModelConfig(
    upscale_factor=2,
    time_steps=5,
    generator_lr=2e-5,
    discriminator_lr=1e-5,
    reconstruction_lr=2e-5,
    fm_weight=0.2,
    mmd_weight=0.2,
    adv_weight=0.01,
    lowfreq_weight=1.0,
    percept_weight=0.1,
    ct_weight=0.02,
    degradation_config={
        "time_steps": 5,
        "downsample_factor": 2,
        "stats_path": PATHS.degradation_stats_path,
        "kernelgan_kernels_path": PATHS.kernelgan_kernels_path,
        "noise_patches_dir": PATHS.noise_patches_dir,
        "spike_type": "ternary",
        "use_temporal": True,
    },
    reconstruction_config={
        "num_fusion_modules": 3,
        "time_steps": 5,
        "base_ch": 64,
        "upscale_factor": 2,
        "v_th": 0.5,
        "v_reset": 0.0,
        "tau": 2.0,
        "spike_type": "ternary",
        "soft_reset": True,
    },
)


CONFIG = ExperimentConfig(
    name="train1360_joint_x2",
    description="Joint training template using Train-1360 as the real training source and Test-206 as the homologous validation subset at x2 scale.",
    mode="joint",
    paths=PATHS,
    runtime=RuntimeConfig(seed=42, device="auto", num_workers=4, use_amp=True, val_batch_size=1),
    train=TrainConfig(
        mode="joint",
        batch_size=4,
        effective_batch_size=8,
        epochs=500,
        val_split=0.1,
        validation_interval=1,
        resume_training=False,
        pretrain_degradation_only=False,
        pretrain_epochs=0,
        use_pretrained_rlgm=False,
        max_val_batches=5,
        warmup_epochs=25,
        decay_epochs=500,
        augmentation_probability=0.5,
        lr_patch_size=48,
    ),
    model=MODEL,
    evaluation=EvalConfig(max_val_batches=5, patch_size=256),
)
