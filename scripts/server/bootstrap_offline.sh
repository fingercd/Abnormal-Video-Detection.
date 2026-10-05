#!/usr/bin/env bash
set -euo pipefail

project_root="${VAD_PROJECT_ROOT:-/users/fotile/VAD}"
base_python="${VAD_BASE_PYTHON:-/users/fotile/miniconda3/envs/h3/bin/python}"
venv_root="$project_root/.venv"

if [[ ! -x "$base_python" ]]; then
  echo "base Python not found: $base_python" >&2
  exit 2
fi
if [[ ! -f "$project_root/pyproject.toml" ]]; then
  echo "project has not been synchronized to $project_root" >&2
  exit 2
fi

if [[ ! -x "$venv_root/bin/python" ]]; then
  "$base_python" -m venv --system-site-packages "$venv_root"
fi

"$venv_root/bin/python" -m pip install --no-index --find-links "$project_root/wheels" \
  accelerate==1.11.0 \
  easydict==1.13 \
  logzero==1.7.0 \
  opencv-python-headless==4.11.0.86 \
  timm==1.0.29
"$venv_root/bin/python" -m pip install --no-index --find-links "$project_root/wheels" \
  hatchling==1.27.0
"$venv_root/bin/python" -m pip install --no-deps --no-build-isolation "$project_root"

"$venv_root/bin/python" - <<'PY'
import json
import cv2
import torch
import transformers
import vadbench

print(json.dumps({
    "vadbench": vadbench.__version__,
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "transformers": transformers.__version__,
    "opencv": cv2.__version__,
}, sort_keys=True))
PY
