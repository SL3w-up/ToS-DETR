#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda:0}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-1}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-2}"

usage() {
  cat <<'USAGE'
Usage:
  scripts/run_external_infer.sh \
    --checkpoint weights/checkpoint_best_medical.pth \
    --image-dir /path/to/25d_images \
    --mask-dir /path/to/25d_masks \
    --output-dir /path/to/output

Environment overrides:
  PYTHON_BIN=python3
  DEVICE=cuda:0
  BATCH_SIZE=4
  NUM_WORKERS=1
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
USAGE
}

CHECKPOINT=""
IMAGE_DIR=""
MASK_DIR=""
OUTPUT_DIR=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint)
      CHECKPOINT="$2"
      shift 2
      ;;
    --image-dir)
      IMAGE_DIR="$2"
      shift 2
      ;;
    --mask-dir)
      MASK_DIR="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -z "$CHECKPOINT" || -z "$IMAGE_DIR" || -z "$MASK_DIR" || -z "$OUTPUT_DIR" ]]; then
  usage >&2
  exit 2
fi

export OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS NUMEXPR_NUM_THREADS
mkdir -p "$OUTPUT_DIR/logs"

cd "$REPO_DIR"
"$PYTHON_BIN" infer_external_cases_tos_detr.py \
  --checkpoint "$CHECKPOINT" \
  --image-dir "$IMAGE_DIR" \
  --mask-dir "$MASK_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --device "$DEVICE" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --save-positive-case-images \
  --save-positive-case-labels \
  "${EXTRA_ARGS[@]}" 2>&1 | tee "$OUTPUT_DIR/logs/infer.log"
