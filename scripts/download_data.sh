#!/usr/bin/env bash
# Download the two ann-benchmarks datasets (resumable; skips files that are already complete).
#   bash scripts/download_data.sh           # both
#   bash scripts/download_data.sh sift      # one
# Files land in <root>/data/raw/<name>/ (override the data root with ANNBENCH_DATA_DIR). The datasets are
# never committed or zipped.
set -euo pipefail

CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${ANNBENCH_DATA_DIR:-$CODE_DIR/../data}"
BASE_URL="${ANNBENCH_BASE_URL:-https://ann-benchmarks.com}"   # the assignment PDF links the same files over http

declare -A FILE=( [sift]="sift-128-euclidean.hdf5" [glove]="glove-100-angular.hdf5" )
names=("$@")
[ ${#names[@]} -gt 0 ] || names=(sift glove)

for name in "${names[@]}"; do
  if [ -z "${FILE[$name]:-}" ]; then
    echo "unknown dataset '$name' (expected: ${!FILE[*]})" >&2
    exit 2
  fi
  dest="$DATA_DIR/raw/$name/${FILE[$name]}"
  if [ -f "$dest" ]; then
    echo "[download] $name: already present ($dest)"
    continue
  fi
  echo "[download] $name: $BASE_URL/${FILE[$name]}"
  curl -fL --create-dirs --retry 5 --retry-delay 5 -C - --progress-bar -o "$dest.part" "$BASE_URL/${FILE[$name]}"
  mv "$dest.part" "$dest"
done
