from dataclasses import dataclass, field


@dataclass
class PathConfig:
    train_hr_dir: str = ""
    train_lr_dir: str = ""
    ref_hr_dir: str = ""
    val_hr_dir: str = ""
    val_lr_dir: str = ""
    save_dir: str = "outputs/default"
    resume_checkpoint_path: str = ""
    pretrained_rlgm_path: str = ""
    niqe_prior_path: str = ""
    degradation_stats_path: str = ""
    kernelgan_kernels_path: str = ""
    noise_patches_dir: str = ""


@dataclass
class RuntimeConfig:
    seed: int = 42
    device: str = "auto"
    num_workers: int = 4
    use_amp: bool = True
    precision_dtype: str = "float32"
    num_gpus: int = 0
    val_batch_size: int = 1


@dataclass
class TrainConfig:
    mode: str = "joint"
    batch_size: int = 4
    effective_batch_size: int = 8
    epochs: int = 1
    val_split: float = 0.1
    validation_interval: int = 1
    resume_training: bool = False
    pretrain_degradation_only: bool = False
    pretrain_epochs: int = 0
    use_pretrained_rlgm: bool = False
    save_all_validation_images: bool = False
    max_val_batches: int = 5
    lr_patch_size: int = 48
    warmup_epochs: int = 10
    decay_epochs: int = 200
    augmentation_probability: float = 0.5


@dataclass
class ModelConfig:
    upscale_factor: int = 2
    time_steps: int = 5
    generator_lr: float = 2e-5
    discriminator_lr: float = 1e-5
    reconstruction_lr: float = 1e-4
    fm_weight: float = 0.2
    mmd_weight: float = 0.2
    adv_weight: float = 0.01
    lowfreq_weight: float = 1.0
    percept_weight: float = 0.1
    ct_weight: float = 0.02
    recon_l1_weight: float = 10.0
    recon_percep_weight: float = 0.01
    recon_lpips_weight: float = 0.1
    recon_grad_weight: float = 0.5
    degradation_config: dict = field(default_factory=dict)
    reconstruction_config: dict = field(default_factory=dict)


@dataclass
class EvalConfig:
    save_all_validation_images: bool = False
    max_val_batches: int = 5
    patch_size: int = 256
    compute_uiqm: bool = True
    compute_brisque: bool = True
    compute_niqe: bool = True


@dataclass
class ExperimentConfig:
    name: str
    description: str
    mode: str
    paths: PathConfig
    runtime: RuntimeConfig
    train: TrainConfig
    model: ModelConfig
    evaluation: EvalConfig
    extras: dict = field(default_factory=dict)
