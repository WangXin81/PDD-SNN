# PDD-SNN

[中文版说明](./README.zh-CN.md)

## Project Overview
PDD-SNN is a cleaned-up research codebase for underwater image restoration with spiking neural networks (SNNs). The project now exposes a single codebase with multiple experiment configs instead of maintaining separate duplicated copies for each dataset.

## Features
- Unified training entrypoint for joint degradation + reconstruction and supervised reconstruction-only workflows
- Unified inference and evaluation scripts
- Structured Python experiment configs under `configs/experiments/`
- Archived legacy scripts under `legacy/`

## Repository Structure
```text
.
├── assets/                     # Static resources such as NIQE priors
├── configs/
│   ├── experiments/           # Public experiment templates
│   └── paths/                 # Example local path template
├── legacy/                    # Original unrefactored snapshots
├── scripts/                   # Unified CLI entrypoints
├── underwater_snn/            # Main package
├── README.md
├── README.zh-CN.md
└── requirements.txt
```

## Environment Setup
Use Python 3.10+ in a dedicated virtual environment.

## Install PyTorch
Install `torch`, `torchvision`, and optionally `torchaudio` from the official PyTorch channel for your CUDA version.

## Install Remaining Dependencies
```bash
pip install -r requirements.txt
```
`requirements.txt` pins `numpy<2` to avoid binary compatibility issues with the current SciPy-based metric stack.

## Datasets
This repository does not redistribute training or test datasets. Please download the datasets from their original public sources and organize them under your local `data/` directory.

The experiments in this project use multiple public underwater datasets with different roles:

### Train-1360
- Name: `UIESR_dataset_Train-1360`
- Source: [Underwater-Lab-SHU/UIESR_dataset_Train-1360](https://github.com/Underwater-Lab-SHU/UIESR_dataset_Train-1360)
- Usage: the core real-world training data source for learning underwater degradation priors
- Note: according to our experimental design, this dataset is treated as real underwater data for degradation-aware training rather than a simple paired benchmark

### Test-206
- Name: `UIESR_dataset_Test-206`
- Source: [Underwater-Lab-SHU/UIESR_dataset_Test-206](https://github.com/Underwater-Lab-SHU/UIESR_dataset_Test-206)
- Usage: an internal evaluation and ablation subset homologous to the Train-1360 data distribution

### UFO-120
- Usage: the main benchmark dataset for quantitative comparison with prior underwater super-resolution methods
- Note: the repository currently provides an example reconstruction config at `configs/experiments/ufo_recon_x2.py`

### UIEB
- Usage: zero-shot cross-domain generalization evaluation on real underwater scenes
- Note: the repository currently provides an example joint-training template at `configs/experiments/uieb_joint_x2.py`

### Included Config Templates
- `configs/experiments/train1360_joint_x2.py`: Train-1360 to Test-206 joint training template for the homologous training and ablation setting described in the paper
- `configs/experiments/ufo_recon_x2.py`: paired reconstruction benchmark template on UFO-120
- `configs/experiments/uieb_joint_x2.py`: example template for cross-domain UIEB evaluation workflows

Please map each downloaded dataset to local directories that match the selected experiment config. The exact folder structure may differ between paired benchmarks, real-world training data, and internal ablation splits.

## Prepare Dataset Paths
Edit one of the files in `configs/experiments/` or copy `configs/paths/example_paths.py` into your own local config variant. Replace the placeholder `data/...` directories with your real dataset and output paths.

## Training
Joint training:
```bash
python scripts/train.py --config configs/experiments/joint_scale2.py
```

Baseline training:
```bash
python scripts/train.py --config configs/experiments/baseline_scale4.py
```

UFO reconstruction-only training:
```bash
python scripts/train.py --config configs/experiments/ufo_recon_x2.py
```

Dry-run config and model validation:
```bash
python scripts/train.py --config configs/experiments/joint_scale2.py --dry-run
```

## Inference
```bash
python scripts/infer.py \
  --config configs/experiments/ufo_recon_x2.py \
  --checkpoint outputs/ufo_recon_x2/best_reconstruction_net.pth \
  --input-dir data/UFO120/test/lr_x2 \
  --output-dir outputs/ufo_recon_x2/inference \
  --gt-dir data/UFO120/test/hr
```

## Evaluation
```bash
python scripts/evaluate.py \
  --config configs/experiments/ufo_recon_x2.py \
  --pred-dir outputs/ufo_recon_x2/inference \
  --gt-dir data/UFO120/test/hr
```

## Known Limitations
- Dataset paths and checkpoints are not shipped with the repository.
- Some legacy research components are archived under `legacy/` and are not part of the new primary workflow.
- The original training logic still depends on heavyweight research losses and third-party libraries; use `--dry-run` to validate structure before running a full experiment.
