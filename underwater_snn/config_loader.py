import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch

from .config_runtime import set_active_config
from .config_types import ExperimentConfig


def _load_module_from_path(config_path: str):
    path = Path(config_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load config module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _to_legacy_namespace(exp: ExperimentConfig) -> SimpleNamespace:
    num_gpus = exp.runtime.num_gpus or torch.cuda.device_count()
    device = exp.runtime.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    ns = SimpleNamespace()
    ns.EXPERIMENT_NAME = exp.name
    ns.EXPERIMENT_DESCRIPTION = exp.description
    ns.MODE = exp.mode
    ns.SEED = exp.runtime.seed
    ns.HR_DIR = exp.paths.train_hr_dir
    ns.LR_DIR = exp.paths.train_lr_dir
    ns.REF_HR_DIR = exp.paths.ref_hr_dir
    ns.SAVE_DIR = exp.paths.save_dir
    ns.VAL_HR_DIR = exp.paths.val_hr_dir
    ns.VAL_LR_DIR = exp.paths.val_lr_dir
    ns.RESUME_CHECKPOINT_PATH = exp.paths.resume_checkpoint_path
    ns.PRETRAINED_RLGM_PATH = exp.paths.pretrained_rlgm_path
    ns.NIQE_PRIOR_PATH = exp.paths.niqe_prior_path
    ns.DEVICE = device
    ns.NUM_GPUS = num_gpus
    ns.NUM_WORKERS = exp.runtime.num_workers
    ns.VAL_BATCH_SIZE = exp.runtime.val_batch_size
    ns.USE_AMP = exp.runtime.use_amp
    ns.PRECISION_DTYPE = getattr(torch, exp.runtime.precision_dtype)
    ns.BATCH_SIZE = exp.train.batch_size
    ns.EFFECTIVE_BATCH_SIZE = exp.train.effective_batch_size
    ns.ACCUMULATION_STEPS = max(1, exp.train.effective_batch_size // exp.train.batch_size)
    ns.EPOCHS = exp.train.epochs
    ns.VAL_SPLIT = exp.train.val_split
    ns.VALIDATION_INTERVAL = exp.train.validation_interval
    ns.RESUME_TRAINING = exp.train.resume_training
    ns.PRETRAIN_DEGRADATION_ONLY = exp.train.pretrain_degradation_only
    ns.PRETRAIN_EPOCHS = exp.train.pretrain_epochs
    ns.USE_PRETRAINED_RLGM = exp.train.use_pretrained_rlgm
    ns.SAVE_ALL_VALIDATION_IMAGES = exp.train.save_all_validation_images or exp.evaluation.save_all_validation_images
    ns.MAX_VAL_BATCHES = exp.train.max_val_batches or exp.evaluation.max_val_batches
    ns.WARMUP_EPOCHS = exp.train.warmup_epochs
    ns.DECAY_EPOCHS = exp.train.decay_epochs
    ns.AUGMENTATION_PROBABILITY = exp.train.augmentation_probability
    ns.LR_PATCH_SIZE = exp.train.lr_patch_size
    ns.UPSCALE_FACTOR = exp.model.upscale_factor
    ns.TIME_STEPS = exp.model.time_steps
    ns.GENERATOR_LR = exp.model.generator_lr
    ns.DISCRIMINATOR_LR = exp.model.discriminator_lr
    ns.RECONSTRUCTION_LR = exp.model.reconstruction_lr
    ns.FM_WEIGHT = exp.model.fm_weight
    ns.MMD_WEIGHT = exp.model.mmd_weight
    ns.ADV_WEIGHT = exp.model.adv_weight
    ns.LOWFREQ_WEIGHT = exp.model.lowfreq_weight
    ns.PERCEPT_WEIGHT = exp.model.percept_weight
    ns.CT_WEIGHT = exp.model.ct_weight
    ns.RECON_L1_WEIGHT = exp.model.recon_l1_weight
    ns.RECON_PERCEP_WEIGHT = exp.model.recon_percep_weight
    ns.RECON_LPIPS_WEIGHT = exp.model.recon_lpips_weight
    ns.RECON_GRAD_WEIGHT = exp.model.recon_grad_weight
    ns.DEGRADATION_CONFIG = dict(exp.model.degradation_config)
    ns.RECONSTRUCTION_CONFIG = dict(exp.model.reconstruction_config)
    for key, value in exp.extras.items():
        setattr(ns, key, value)
    return ns


def load_experiment_config(config_path: str):
    module = _load_module_from_path(config_path)
    exp = getattr(module, "CONFIG", None)
    if not isinstance(exp, ExperimentConfig):
        raise TypeError(f"{config_path} must define CONFIG: ExperimentConfig")
    legacy = _to_legacy_namespace(exp)
    set_active_config(legacy)
    return exp, legacy
