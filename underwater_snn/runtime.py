import os
from pathlib import Path

from .config_runtime import get_active_config


def bootstrap_runtime():
    cfg = get_active_config()
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
    Path(cfg.SAVE_DIR).mkdir(parents=True, exist_ok=True)
    return cfg
