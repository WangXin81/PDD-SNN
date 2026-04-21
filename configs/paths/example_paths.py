from underwater_snn.config_types import PathConfig


EXAMPLE_PATHS = PathConfig(
    train_hr_dir="data/train/hr",
    train_lr_dir="data/train/lr",
    ref_hr_dir="data/train/ref_hr",
    val_hr_dir="data/val/hr",
    val_lr_dir="data/val/lr",
    save_dir="outputs/example_experiment",
    resume_checkpoint_path="",
    pretrained_rlgm_path="",
    niqe_prior_path="outputs/example_experiment/niqe_water_params.mat",
    degradation_stats_path="data/priors/degradation_stats.pth",
    kernelgan_kernels_path="data/priors/kernelgan_kernels.npy",
    noise_patches_dir="data/priors/noise_patches",
)
