#!/usr/bin/env bash
set -euo pipefail

echo "== host =="
hostname

echo "== conda environments =="
conda env list 2>/dev/null || true

echo "== candidate Python runtimes =="
while IFS= read -r python_bin; do
  printf '\n-- %s --\n' "$python_bin"
  "$python_bin" - <<'PY' || true
import importlib.util
import json
import sys

payload = {"python": sys.version.split()[0], "executable": sys.executable}
modules = (
    "accelerate", "av", "cv2", "decord", "flash_attn", "huggingface_hub",
    "logzero", "numpy", "pandas", "PIL", "psutil", "qwen_vl_utils",
    "safetensors", "sklearn", "tokenizers", "torch", "torchvision",
    "transformers", "yaml",
)
for name in modules:
    if importlib.util.find_spec(name) is None:
        payload[name] = None
        continue
    try:
        module = __import__(name)
        payload[name] = getattr(module, "__version__", "installed")
    except Exception as exc:
        payload[name] = f"import-error:{type(exc).__name__}"
if payload.get("torch") and not str(payload["torch"]).startswith("import-error"):
    import torch
    payload["cuda_available"] = torch.cuda.is_available()
    payload["torch_cuda"] = torch.version.cuda
print(json.dumps(payload, sort_keys=True))
PY
done < <(
  {
    command -v python || true
    find /users/fotile -maxdepth 5 -type f -path '*/bin/python' 2>/dev/null || true
  } | awk 'NF && !seen[$0]++'
)

echo "== disk =="
df -h /users

echo "== GPUs =="
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
