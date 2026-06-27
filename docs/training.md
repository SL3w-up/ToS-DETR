# Training

This document describes the ToS-DETR training environment and four-stage training workflow.

## Verified Environment

The reference model training/inference environment used for this release is:

- Python `3.11.7`
- PyTorch `2.6.0+cu124`
- TorchVision `0.21.0+cu124`
- CUDA runtime reported by PyTorch: `12.4`
- cuDNN `90100`

Use this environment, or recreate an equivalent Python 3.11 environment with matching PyTorch/TorchVision CUDA builds. The CUDA extension under `models/ops` must be compiled in the target environment.

`requirements.txt` pins Python packages only. It does not install the NVIDIA driver, CUDA Toolkit, `nvcc`, or a C++ compiler. Running `pip install -r requirements.txt` in an existing environment can change package versions in that environment, so a dedicated ToS-DETR environment is recommended.

Before building the extension, confirm these system-level components are available:

- an NVIDIA GPU with a compatible driver,
- CUDA-enabled PyTorch/TorchVision wheels,
- CUDA Toolkit with `nvcc`,
- a compatible C++ compiler such as `g++`.

## Installation

```bash
python -m pip install -r requirements.txt
cd models/ops
sh make.sh
python test.py
cd ../..
```

`models/ops/make.sh` installs `MultiScaleDeformableAttention` into the active Python environment. Build it with the same Python executable used for training and inference.

## Four-stage Training Schedule

Default training uses `--training_stage auto_four_stage` and runs these stages:

| Stage | Epochs | Purpose |
| --- | ---: | --- |
| `stage1_heatmap` | 5 | Warm up backbone, input projection, slice embedding, and fracture heatmap. |
| `stage2_full` | 25 | Train full detector with box, center, heatmap, ROI, query coverage, slice memory, consistency, and contrastive losses. |
| `stage3_hard_negative` | 10 | Add dynamic hard-negative mining and hard-negative suppression loss. |
| `stage4_calibration` | 3 | Freeze the detector body and calibrate logit temperature/uncertainty. |

## Example Command

```bash
python main.py \
  --coco_path /path/to/data/coco \
  --output_dir outputs/tos_detr_four_stage \
  --backbone convnext_tiny \
  --num_feature_levels 3 \
  --training_stage auto_four_stage \
  --batch_size 8 \
  --num_workers 2
```

The convenience launcher applies the same ToS-DETR defaults and forwards additional arguments to `main.py`:

```bash
configs/tos_detr_convnext_tiny_four_stage.sh \
  --coco_path /path/to/data/coco \
  --output_dir outputs/tos_detr_four_stage
```

## Warm-start Weights

For inference, only the trained ToS-DETR checkpoint is needed. Original Deformable DETR or ConvNeXt warm-start weights are optional for retraining and should be passed explicitly if used:

```bash
python main.py \
  --pretrained_model /path/to/deformable_detr-r50_1.pth \
  --backbone_pretrained_path /path/to/convnext_tiny.in12k_ft_in1k.pytorch_model.bin
```

If warm-start weights are unavailable, set `--pretrained_model "" --no_backbone_pretrained` or use another initialization policy.

## Checkpoint Selection

Checkpoint selection uses medical metrics rather than validation loss. The selection rule prioritizes:

1. constraining negative-case primary FP/case,
2. constraining all-lesion FP/case,
3. maximizing primary lesion-track recall,
4. using FROC as a tie-breaker.
