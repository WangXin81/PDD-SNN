"""Underwater SNN training and evaluation package."""

from .config_loader import load_experiment_config
from .config_runtime import config, set_active_config

__all__ = ["config", "load_experiment_config", "set_active_config"]
