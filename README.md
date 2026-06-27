# ToS-DETR

**Structure-aware Deformable DETR for 2.5D costal cartilage fracture detection.**

ToS-DETR adapts Deformable DETR to ROI-constrained costal cartilage fracture detection on 2.5D CT slices.

This repository contains the model, training/evaluation code, external inference utility, and a notebook demo for a single external case-style inference workflow.


## Repository Layout

```text
ToS-DETR/
├── configs/                         # Training launch wrappers
├── datasets/                        # COCO-format 2.5D CT and mask dataset
├── docs/
│   ├── data_preparation.md          # Input preprocessing and annotation contract
│   ├── external_inference.md        # External inference usage and outputs
│   └── training.md                  # Environment and training workflow
├── models/                          # ToS-DETR model and MSDeformAttn operators
├── notebooks/
│   └── inference_demo.ipynb         # Step-by-step external inference demo
├── sample_data/
│   ├── images/                      # De-identified example 2.5D inputs
│   └── masks/                       # Same-name example cartilage masks
├── scripts/
│   ├── download_checkpoint.sh       # Optional trusted checkpoint downloader
│   └── run_external_infer.sh        # Generic external inference wrapper
├── util/                            # Fracture metrics, box ops, misc utils
├── infer_external_cases_tos_detr.py
├── main.py
├── requirements.txt
├── LICENSE
└── NOTICE
```

## Installation

Use a fresh Python 3.11 environment and install dependencies:

```bash
python -m pip install -r requirements.txt
```

`requirements.txt` pins the Python packages used by the release environment. Running this command in an existing Python environment can still install, upgrade, or downgrade packages in that environment. For reproducibility, create a dedicated environment for ToS-DETR instead of installing into a shared research environment.

The reference training/inference environment used for this release is:

- Python `3.11.7`
- PyTorch `2.6.0+cu124`
- TorchVision `0.21.0+cu124`
- CUDA runtime reported by PyTorch: `12.4`
- cuDNN `90100`

In addition to Python packages, ToS-DETR requires system-level GPU build/runtime components that are not installed by `requirements.txt`:

- an NVIDIA GPU and compatible NVIDIA driver,
- CUDA-enabled PyTorch/TorchVision wheels matching the driver,
- CUDA Toolkit with `nvcc`,
- a compatible C++ compiler such as `g++`.

Install a PyTorch/TorchVision build that matches your CUDA driver before building the CUDA extension. The CUDA Toolkit and compiler must already be available on the system.

Build the multi-scale deformable attention operator:

```bash
cd models/ops
sh make.sh
python test.py
cd ../..
```

The CUDA extension follows the original Deformable DETR operator implementation. `models/ops/make.sh` builds and installs `MultiScaleDeformableAttention` into the active Python environment, so run it inside the same environment that will execute training or inference.

## Model Weights

Do **not** commit model weights to git. GitHub has strict file-size limits, and checkpoints are better distributed through a model hub or release asset.

The trained ToS-DETR checkpoint for this release is hosted on Hugging Face:

- Model page: <https://huggingface.co/SL3w/ToS-DETR>
- Direct download: <https://huggingface.co/SL3w/ToS-DETR/resolve/main/checkpoint_best_medical.pth>

For normal users, only the trained ToS-DETR checkpoint is needed for inference, typically `checkpoint_best_medical.pth`. The original Deformable DETR transformer checkpoint and ConvNeXt backbone checkpoint are optional warm-start files for retraining and should not be uploaded into this repository. If used, pass them explicitly:

```bash
python main.py \
  --pretrained_model /path/to/deformable_detr-r50_1.pth \
  --backbone_pretrained_path /path/to/convnext_tiny.in12k_ft_in1k.pytorch_model.bin
```

Only load checkpoints from a trusted release, model hub, or institutional storage location. PyTorch checkpoints may execute code during deserialization.

Download the checkpoint into the ignored `weights/` directory:

```bash
scripts/download_checkpoint.sh \
  --output weights/checkpoint_best_medical.pth
```

## Notebook Demo

Open [notebooks/inference_demo.ipynb](notebooks/inference_demo.ipynb). It provides a step-by-step workflow for:

1. Installing dependencies.
2. Setting a checkpoint path.
3. Loading sample 2.5D images and masks.
4. Running ToS-DETR inference.
5. Visualizing positive predictions.

The sample images are de-identified example inputs intended to verify the inference pipeline and output format.

## Citation

If this repository is useful for your research, cite the upstream Deformable DETR work and this repository:

```bibtex
@article{zhu2020deformable,
  title={Deformable DETR: Deformable Transformers for End-to-End Object Detection},
  author={Zhu, Xizhou and Su, Weijie and Lu, Lewei and Li, Bin and Wang, Xiaogang and Dai, Jifeng},
  journal={arXiv preprint arXiv:2010.04159},
  year={2020}
}
```

## Acknowledgements

ToS-DETR is built on the open-source Deformable DETR implementation and retains its Apache-2.0 license notices. It also uses ConvNeXt through `timm` and can use cartilage masks generated by external segmentation tools such as TotalSegmentator or other institution-specific segmentation pipelines. Those tools are separate dependencies and are not redistributed in this repository.

## License

This project is released under the Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE). Clinical deployment, diagnosis, or treatment use requires independent validation and local regulatory review.
