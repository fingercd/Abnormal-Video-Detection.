#!/usr/bin/env bash
set -euo pipefail

project_root="${VAD_PROJECT_ROOT:-/users/fotile/VAD}"
dataset_target="${1:-}"
link_path="$project_root/data/raw/ucf_crime"
legacy_link="$project_root/data/ucf_crime"

if [[ -z "$dataset_target" ]]; then
  echo "usage: $0 /absolute/path/to/UCF-Crime" >&2
  exit 2
fi
if [[ "$dataset_target" != /users/* ]] || [[ ! -d "$dataset_target" ]]; then
  echo "dataset target must be an existing directory below /users: $dataset_target" >&2
  exit 2
fi
if [[ -e "$link_path" || -L "$link_path" ]]; then
  echo "refusing to replace existing path: $link_path" >&2
  exit 3
fi

if [[ -e "$legacy_link" || -L "$legacy_link" ]]; then
  legacy_target="$(readlink -f "$legacy_link")"
  requested_target="$(readlink -f "$dataset_target")"
  if [[ "$legacy_target" != "$requested_target" ]]; then
    echo "legacy link points elsewhere: $legacy_link -> $legacy_target" >&2
    exit 4
  fi
fi

mkdir -p "$project_root/data/raw"
ln -s "$dataset_target" "$link_path"
readlink -f "$link_path"
