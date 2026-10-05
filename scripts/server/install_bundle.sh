#!/usr/bin/env bash
set -euo pipefail

project_root="${VAD_PROJECT_ROOT:-/users/fotile/VAD}"
incoming="$project_root/.incoming"
bundle="$incoming/vadbench.bundle"
branch="feat/video-encoder-benchmark-framework"

if [[ ! -f "$bundle" ]]; then
  echo "missing git bundle: $bundle" >&2
  exit 2
fi

cd "$project_root"
if [[ ! -d .git ]]; then
  git init
fi
git bundle verify "$bundle"
git fetch "$bundle" "$branch"
git checkout -B "$branch" FETCH_HEAD

# The historical repository contains a few CRLF blobs. An earlier broad text
# attribute may have rewritten only their line endings during first checkout.
# Restore them only when an ignore-EOL diff proves there is no content change.
legacy_eol_paths=(
  lab_anomaly/__init__.py
  lab_anomaly/configs/train_end2end.yaml
  lab_anomaly/data/__init__.py
  lab_anomaly/data/preclip_manifest.py
  lab_anomaly/infer/__init__.py
  lab_anomaly/models/__init__.py
  lab_anomaly/tool/precompute_clips.py
  lab_anomaly/train/__init__.py
  lab_anomaly/train/train_end2end.py
)
if ! git diff --quiet -- "${legacy_eol_paths[@]}"; then
  if git diff --ignore-space-at-eol --exit-code -- "${legacy_eol_paths[@]}" >/dev/null; then
    git restore --worktree -- "${legacy_eol_paths[@]}"
  else
    echo "refusing to overwrite non-EOL changes in legacy files" >&2
    exit 4
  fi
fi

if git remote get-url origin >/dev/null 2>&1; then
  git remote set-url origin 'https://github.com/fingercd/Abnormal-Video-Detection..git'
else
  git remote add origin 'https://github.com/fingercd/Abnormal-Video-Detection..git'
fi

mkdir -p external wheels data/smoke
tar -xzf "$incoming/vadbench-external.tar.gz" -C external
tar -xzf "$incoming/vadbench-wheels.tar.gz" -C wheels
install -m 0644 "$incoming/surveillance-smoke.mp4" data/smoke/surveillance-smoke.mp4

git rev-parse HEAD
git status --short --branch
