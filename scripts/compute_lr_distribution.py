import argparse
import json
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from underwater_snn.evaluation.distribution import BatchLRComparator


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Compare real and generated LR distributions.")
    parser.add_argument("--real-dir", required=True, help="Directory with real LR images.")
    parser.add_argument("--generated-dir", required=True, help="Directory with generated LR images.")
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    comparator = BatchLRComparator(args.real_dir, args.generated_dir)
    results = comparator.compare_all_images()
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2, ensure_ascii=False)
        print(f"Saved LR distribution report to {args.output}")
    else:
        print(json.dumps(results, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
