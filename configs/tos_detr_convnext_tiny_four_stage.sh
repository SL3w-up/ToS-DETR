#!/usr/bin/env bash
set -euo pipefail

EXP_DIR="${EXP_DIR:-outputs/tos_detr_four_stage}"
PY_ARGS=("$@")

python -u main.py \
  --output_dir "$EXP_DIR" \
  --backbone convnext_tiny \
  --num_feature_levels 3 \
  --training_stage auto_four_stage \
  "${PY_ARGS[@]}"
