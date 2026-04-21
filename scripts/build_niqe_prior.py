import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from underwater_snn.evaluation.prior import build_niqe_prior_from_images


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Build NIQE prior from real LR images.")
    parser.add_argument("--lr-dir", required=True, help="Directory with LR images.")
    parser.add_argument("--output", required=True, help="Output .mat path.")
    parser.add_argument("--patch-size", type=int, default=32, help="Patch size used for NIQE prior features.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    build_niqe_prior_from_images(args.lr_dir, args.output, patch_size=args.patch_size)
    print(f"Saved NIQE prior to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
