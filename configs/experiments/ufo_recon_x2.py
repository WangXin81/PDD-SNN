from underwater_snn.config_types import EvalConfig, ExperimentConfig, ModelConfig, PathConfig, RuntimeConfig, TrainConfig


PATHS = PathConfig(
    train_hr_dir="data/UFO120/train/hr",
    train_lr_dir="data/UFO120/train/lr_x2",
    val_hr_dir="data/UFO120/val/hr",
    val_lr_dir="data/UFO120/val/lr_x2",
    save_dir="outputs/ufo_recon_x2",
    resume_checkpoint_path="",
)


MODEL = ModelConfig(
    upscale_factor=2,
    time_steps=5,
    reconstruction_lr=2e-5,
    recon_l1_weight=10.0,
    recon_percep_weight=0.01,
    recon_lpips_weight=0.1,
    recon_grad_weight=0.5,
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
    name="ufo_recon_x2",
    description="Supervised reconstruction training for the UFO dataset at x2 scale.",
    mode="reconstruction",
    paths=PATHS,
    runtime=RuntimeConfig(seed=42, device="auto", num_workers=4, use_amp=True, val_batch_size=1),
    train=TrainConfig(
        mode="reconstruction",
        batch_size=4,
        effective_batch_size=8,
        epochs=500,
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
