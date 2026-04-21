import argparse
import csv
import os
from pathlib import Path
import sys

import torch
from PIL import Image
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from underwater_snn.config_loader import load_experiment_config


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Unified evaluation entrypoint.")
    parser.add_argument("--config", required=True, help="Path to an experiment config file.")
    parser.add_argument("--pred-dir", required=True, help="Directory with predicted images.")
    parser.add_argument("--gt-dir", required=True, help="Directory with ground-truth images.")
    parser.add_argument("--device", default=None, help="Override runtime device.")
    parser.add_argument("--dry-run", action="store_true", help="Only validate evaluator initialization.")
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
        print(f"[dry-run] evaluation config loaded for {args.config}")
        return 0

    from underwater_snn.evaluation.metrics import ImageQualityEvaluator
    from underwater_snn.losses import calculate_psnr, calculate_ssim

    evaluator = ImageQualityEvaluator(device=legacy.DEVICE, prior_path=getattr(legacy, "NIQE_PRIOR_PATH", None))
    if not os.path.exists(args.pred_dir):
        raise FileNotFoundError(f"Prediction directory not found: {args.pred_dir}")
    if not os.path.exists(args.gt_dir):
        raise FileNotFoundError(f"GT directory not found: {args.gt_dir}")

    transform = transforms.ToTensor()
    device = legacy.DEVICE
    import lpips
    lpips_metric = lpips.LPIPS(net="alex").to(device)
    lpips_metric.eval()
    rows = []
    names = [name for name in image_names(args.pred_dir) if os.path.exists(os.path.join(args.gt_dir, name))]
    for name in names:
        pred = transform(Image.open(os.path.join(args.pred_dir, name)).convert("RGB")).to(device)
        gt = transform(Image.open(os.path.join(args.gt_dir, name)).convert("RGB")).to(device)
        psnr = calculate_psnr(pred, gt, crop_border=legacy.UPSCALE_FACTOR, test_y_channel=True)
        ssim = calculate_ssim(pred, gt, crop_border=legacy.UPSCALE_FACTOR, test_y_channel=True)
        lpips_score = lpips_metric(pred.unsqueeze(0) * 2 - 1, gt.unsqueeze(0) * 2 - 1).mean().item()
        pred_cpu = pred.detach().cpu()
        niqe = evaluator.calculate_batch_niqe_scores([pred_cpu])[0]
        brisque = evaluator.calculate_batch_brisque_scores([pred_cpu])[0]
        uiqm = evaluator.calculate_batch_uiqm_scores([pred_cpu])[0]
        row = {
            "image": name,
            "psnr": float(psnr),
            "ssim": float(ssim),
            "lpips": float(lpips_score),
            "niqe": float(niqe),
            "brisque": float(brisque),
            "uiqm": float(uiqm),
        }
        rows.append(row)
        print(
            f"{name}: PSNR={row['psnr']:.2f}, SSIM={row['ssim']:.4f}, "
            f"LPIPS={row['lpips']:.4f}, NIQE={row['niqe']:.4f}, BRISQUE={row['brisque']:.4f}, UIQM={row['uiqm']:.4f}"
        )

    report_path = os.path.join(args.pred_dir, "evaluation_report.csv")
    with open(report_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image", "psnr", "ssim", "lpips", "niqe", "brisque", "uiqm"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved evaluation report to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
