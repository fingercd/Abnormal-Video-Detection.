#!/usr/bin/env bash
set -euo pipefail

project_root="${VAD_PROJECT_ROOT:-/users/fotile/VAD}"
base_python="${VAD_BASE_PYTHON:-/users/fotile/miniconda3/envs/h3/bin/python}"
venv_root="$project_root/.venv-hermes"

if [[ ! -x "$venv_root/bin/python" ]]; then
  "$base_python" -m venv --system-site-packages "$venv_root"
fi

"$venv_root/bin/python" -m pip install --no-index --find-links "$project_root/wheels" \
  accelerate==1.11.0 \
  easydict==1.13 \
  logzero==1.7.0 \
  opencv-python-headless==4.11.0.86 \
  timm==1.0.29 \
  tokenizers==0.19.1 \
  transformers==4.45.0.dev0
"$venv_root/bin/python" -m pip install --no-index --find-links "$project_root/wheels" \
  hatchling==1.27.0
"$venv_root/bin/python" -m pip install --no-deps --no-build-isolation "$project_root"

export LD_LIBRARY_PATH="/users/fotile/miniconda3/envs/mllm-comp-platform/lib:${LD_LIBRARY_PATH:-}"
"$venv_root/bin/python" - <<'PY'
import json
import torch
import transformers
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
import vadbench

print(json.dumps({
    "vadbench": vadbench.__version__,
    "torch": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "transformers": transformers.__version__,
    "auto_processor": AutoProcessor.__name__,
    "llava_onevision": LlavaOnevisionForConditionalGeneration.__name__,
}, sort_keys=True))
PY
