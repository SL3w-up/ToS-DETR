#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/download_checkpoint.sh [--url CHECKPOINT_URL] [--output weights/checkpoint_best_medical.pth]

The project does not store checkpoints in git. By default this helper downloads
the release checkpoint from:
  https://huggingface.co/SL3w/ToS-DETR/resolve/main/checkpoint_best_medical.pth

Use --url to override the source. Only download checkpoints from trusted sources.
USAGE
}

URL="https://huggingface.co/SL3w/ToS-DETR/resolve/main/checkpoint_best_medical.pth"
OUTPUT="weights/checkpoint_best_medical.pth"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --url)
      URL="$2"
      shift 2
      ;;
    --output)
      OUTPUT="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

mkdir -p "$(dirname "$OUTPUT")"
"${PYTHON_BIN:-python}" - "$URL" "$OUTPUT" <<'PYDOWNLOAD'
from pathlib import Path
from urllib.request import urlretrieve
import sys

url = sys.argv[1]
out = Path(sys.argv[2])
urlretrieve(url, out)
print(out)
PYDOWNLOAD
