# External Inference

This document describes external inference for fixed-size 2.5D images and same-name cartilage masks.

## Inputs

The inference script expects:

- an image directory containing RGB 2.5D CT images,
- a mask directory containing same-name RGB or grayscale cartilage masks,
- a trained ToS-DETR checkpoint such as `checkpoint_best_medical.pth`.

Input images are expected to follow the same preprocessing contract as training data: height `448`, width `224`, three 2.5D channels, and aligned masks with the same file name.

## Environment

Use a dedicated Python 3.11 environment for inference. `requirements.txt` pins Python packages only; it does not install the NVIDIA driver, CUDA Toolkit, `nvcc`, or a C++ compiler. Running `pip install -r requirements.txt` in an existing environment can still install, upgrade, or downgrade packages in that environment.

The MSDeformAttn operator must be compiled before inference:

```bash
python -m pip install -r requirements.txt
cd models/ops
sh make.sh
python test.py
cd ../..
```

`models/ops/make.sh` installs `MultiScaleDeformableAttention` into the active Python environment. Build it with the same Python executable used to run `infer_external_cases_tos_detr.py`.

## Checkpoint

The release checkpoint is hosted on Hugging Face:

- Model page: <https://huggingface.co/SL3w/ToS-DETR>
- Direct download: <https://huggingface.co/SL3w/ToS-DETR/resolve/main/checkpoint_best_medical.pth>

Download it into the ignored `weights/` directory:

```bash
scripts/download_checkpoint.sh --output weights/checkpoint_best_medical.pth
```

## Command

```bash
scripts/run_external_infer.sh \
  --checkpoint weights/checkpoint_best_medical.pth \
  --image-dir /path/to/25d_images \
  --mask-dir /path/to/25d_masks \
  --output-dir outputs/external_case
```

Useful environment overrides:

```bash
DEVICE=cuda:0 BATCH_SIZE=8 NUM_WORKERS=2 scripts/run_external_infer.sh ...
```

## Outputs

The output directory contains:

- `summary.json`: inference configuration, thresholds, case counts, and output summary.
- `case_predictions.json`: case-level prediction scores and predicted labels.
- `slice_summaries.jsonl`: per-slice detection summary.
- `positive_case_labels/`: YOLO-style normalized box text files for predicted-positive cases.
- `positive_case_images/`: predicted-positive case images with visualization boxes.
- `threshold_filtered_case_boxes.json`: retained boxes after score thresholding, including scores and normalized/pixel coordinates.

The exported text label format is:

```text
0 x_center y_center width height
```

Coordinates are normalized to the corresponding input image size. Empty text files represent positive-case slices without retained boxes.

## Checkpoint Safety

Only load checkpoints from trusted releases, model hubs, or controlled institutional storage. PyTorch checkpoints may execute code during deserialization.
