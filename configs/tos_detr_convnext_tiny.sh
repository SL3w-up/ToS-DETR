#!/usr/bin/env bash
set -euo pipefail

EXP_DIR="${EXP_DIR:-outputs/tos_detr}"
PY_ARGS=("$@")

python -u main.py \
  --output_dir "$EXP_DIR" \
  --backbone convnext_tiny \
  --num_feature_levels 3 \
  "${PY_ARGS[@]}"
