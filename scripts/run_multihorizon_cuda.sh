#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 || ! "$1" =~ ^(check|train|evaluate)$ ]]; then
  echo 'Usage: bash scripts/run_multihorizon_cuda.sh {check|train|evaluate} DATA_DIR OUTPUT_DIR' >&2
  exit 2
fi

mode="$1"
data_dir="$2"
output_dir="$3"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
protocol='configs/forecasting/multihorizon_cuda.json'

if [[ "$mode" == check ]]; then
  python -m traffic_forecasting.check_multihorizon_cuda \
    --protocol "$protocol" --data-dir "$data_dir"
else
  python -m traffic_forecasting.multihorizon_cuda \
    --protocol "$protocol" --data-dir "$data_dir" \
    --output-dir "$output_dir" --mode "$mode"
fi
