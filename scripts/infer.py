import argparse
import csv
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from underwater_snn.config_loader import load_experiment_config


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Unified inference entrypoint.")
    parser.add_argument("--config", required=True, help="Path to an experiment config file.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path.")
    parser.add_argument("--input-dir", required=True, help="Directory with LR images.")
    parser.add_argument("--output-dir", required=True, help="Directory to save SR images and report.")
    parser.add_argument("--gt-dir", default=None, help="Optional HR directory for metrics.")
    parser.add_argument("--device", default=None, help="Override runtime device.")
    parser.add_argument("--dry-run", action="store_true", help="Only validate config and model construction.")
    return parser.parse_args(argv)


def image_names(input_dir):
    return sorted(
        name for name in os.listdir(input_dir)
        if name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))
    )


def main(argv=None):
    args = parse_args(argv)
    _, legacy = load_experiment_config(args.config)
    if args.device:
        legacy.DEVICE = args.device

    if args.dry_run:
        try:
            from underwater_snn.models.networks import ReconstructionModule

            ReconstructionModule(**legacy.RECONSTRUCTION_CONFIG, device=legacy.DEVICE)
        except ModuleNotFoundError as exc:
            print(f"[dry-run] missing dependency: {exc.name}")
            return 1
        print(f"[dry-run] inference config loaded for {args.config}")
        return 0

    if not os.path.exists(args.input_dir):
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    os.makedirs(args.output_dir, exist_ok=True)
    from underwater_snn.evaluation.inference import SNNInferenceEngine

    engine = SNNInferenceEngine(legacy, model_path=args.checkpoint, device=legacy.DEVICE)
    rows = []
    for name in image_names(args.input_dir):
        lr_path = os.path.join(args.input_dir, name)
        gt_path = os.path.join(args.gt_dir, name) if args.gt_dir else None
        save_path = os.path.join(args.output_dir, name)
        result = engine.infer(lr_path, hr_path=gt_path, save_path=save_path)
        rows.append({"image": name, **result})
        print(
            f"{name}: PSNR={result['psnr']:.2f}, SSIM={result['ssim']:.4f}, "
            f"LPIPS={result['lpips']:.4f}, UIQM={result['uiqm']:.4f}"
        )

    report_path = os.path.join(args.output_dir, "report.csv")
    with open(report_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image", "psnr", "ssim", "lpips", "uiqm", "gsops", "energy"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved inference report to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
