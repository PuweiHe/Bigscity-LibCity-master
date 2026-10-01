#!/usr/bin/env bash
#SBATCH --job-name=traffic-search
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1-23:30:00
#SBATCH --output=slurm-%x-%j.log
set -euo pipefail
: "${TRAFFIC_REPO:?Set the absolute repository path}"
: "${TRAFFIC_DATA:?Set the absolute prepared-data path}"
: "${TRAFFIC_RUNS:?Set the persistent output path}"
: "${TRAFFIC_PYTHON:?Set the environment Python path}"
: "${TRAFFIC_DATASET:?Set METR_LA or PEMS_BAY}"
cd "$TRAFFIC_REPO"
export PYTHONPATH="$TRAFFIC_REPO${PYTHONPATH:+:$PYTHONPATH}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
mkdir -p "$TRAFFIC_RUNS/job_logs"
nvidia-smi > "$TRAFFIC_RUNS/job_logs/gpu-${SLURM_JOB_ID}.txt"
"$TRAFFIC_PYTHON" -m pip freeze > "$TRAFFIC_RUNS/job_logs/packages-${SLURM_JOB_ID}.txt"
srun "$TRAFFIC_PYTHON" -u -m traffic_forecasting.cuda_search \
  --data-dir "$TRAFFIC_DATA" --output-dir "$TRAFFIC_RUNS" \
  --dataset "$TRAFFIC_DATASET" --stage all
