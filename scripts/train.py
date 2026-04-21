import argparse
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from underwater_snn.config_loader import load_experiment_config
from underwater_snn.runtime import bootstrap_runtime


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Unified training entrypoint.")
    parser.add_argument("--config", required=True, help="Path to an experiment config file.")
    parser.add_argument("--device", default=None, help="Override runtime device, e.g. cpu or cuda:0.")
    parser.add_argument("--save-dir", default=None, help="Override experiment save directory.")
    parser.add_argument("--dry-run", action="store_true", help="Only validate config and model construction.")
    return parser.parse_args(argv)


def validate_paths(cfg):
    required = [("HR_DIR", cfg.HR_DIR), ("SAVE_DIR", cfg.SAVE_DIR)]
    if cfg.MODE == "joint":
        required.extend([("LR_DIR", cfg.LR_DIR), ("VAL_HR_DIR", cfg.VAL_HR_DIR)])
    else:
        required.extend([("LR_DIR", cfg.LR_DIR), ("VAL_LR_DIR", cfg.VAL_LR_DIR)])
    missing = [name for name, value in required if not value]
    if missing:
        raise ValueError(f"Missing required config paths: {', '.join(missing)}")
    invalid = [value for name, value in required if name != "SAVE_DIR" and value and not os.path.exists(value)]
    if invalid:
        raise FileNotFoundError(f"Configured paths do not exist: {invalid}")


def dry_run_models(cfg):
    from underwater_snn.models.networks import DegradationModule, DiscriminatorModule, ReconstructionModule

    if cfg.MODE == "joint":
        DegradationModule(cfg.DEGRADATION_CONFIG, device=cfg.DEVICE)
        DiscriminatorModule(cfg.DEGRADATION_CONFIG, device=cfg.DEVICE)
        ReconstructionModule(**cfg.RECONSTRUCTION_CONFIG, device=cfg.DEVICE)
    else:
        ReconstructionModule(**cfg.RECONSTRUCTION_CONFIG, device=cfg.DEVICE)


def main(argv=None):
    args = parse_args(argv)
    exp, legacy = load_experiment_config(args.config)
    if args.device:
        legacy.DEVICE = args.device
    if args.save_dir:
        legacy.SAVE_DIR = args.save_dir
    bootstrap_runtime()

    if args.dry_run:
        try:
            dry_run_models(legacy)
        except ModuleNotFoundError as exc:
            print(f"[dry-run] missing dependency: {exc.name}")
            return 1
        print(f"[dry-run] name={exp.name} mode={legacy.MODE} device={legacy.DEVICE} save_dir={legacy.SAVE_DIR}")
        return 0

    validate_paths(legacy)
    if legacy.MODE == "joint":
        from underwater_snn.training.run_joint import main as run_main
    elif legacy.MODE == "reconstruction":
        from underwater_snn.training.run_reconstruction import main as run_main
    else:
        raise ValueError(f"Unsupported training mode: {legacy.MODE}")
    run_main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
