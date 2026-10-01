# Run the multi-horizon study on an NVIDIA CUDA server

This is a separate CUDA replication of the [frozen multi-horizon protocol](MULTIHORIZON_STUDY.md). It trains the same seven preregistered configurations on each of METR-LA and PEMS-BAY with three seeds (42 runs total), using 12 five-minute input readings to predict the next 12 readings. The original CPU and Apple Silicon MPS runs remain independent. No CUDA accuracy or speed result is claimed before the server actually runs.

Use the public repository checkout, or transfer the local `traffic_forecasting_cuda_server.tar.gz` bundle and extract it with `tar -xzf traffic_forecasting_cuda_server.tar.gz && cd traffic-forecasting-cuda`. Run every command below from that repository/bundle root. The bundle contains code, configuration, tests, and instructions, but no traffic data.

## 1. Environment and public data

Use a Linux server with an NVIDIA GPU, compatible driver, and a Python 3.10 virtual environment. Install a **CUDA-enabled** PyTorch build compatible with that driver, then install `requirements-multihorizon-server.txt`. The old upstream `requirements.txt` pins CPU-era packages and is not the dependency set for this experiment. Verify `torch.cuda.is_available()` returns `True` before training.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-multihorizon-server.txt
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
```

Place the public input files outside Git at these paths (replace `/srv/traffic-data` with your location):

```text
/srv/traffic-data/METR_LA/METR_LA.npz
/srv/traffic-data/METR_LA/METR_LA_rn_adj.npy
/srv/traffic-data/PEMS_BAY/pems-bay.h5
/srv/traffic-data/PEMS_BAY/distances_bay_2017.csv
```

METR-LA source details and archive hash are in [the earlier study](METR_LA_STUDY.md); PEMS-BAY source links are in [the main study](MULTIHORIZON_STUDY.md). Convert PEMS-BAY once:

```bash
python -m traffic_forecasting.prepare_pems_bay --root /srv/traffic-data/PEMS_BAY
```

The loader verifies shapes, five-minute timestamps, graph weights, and file hashes. Normalization is fitted only on observed training readings. Windows remain within chronological 70/10/20 train/validation/test partitions. Do not commit either dataset or any private internship recording.

## 2. Check CUDA operations, then train

From the repository root, run the one-batch check on **training windows only** at the full configured batch size. It builds every configured model on both datasets and checks CUDA forward, loss, backward, finite gradients, optimizer update, and inference. This catches unsupported sparse or deterministic kernels and obvious memory limits before a long run. A CUDA build/driver mismatch or failed check must be fixed before training; do not silently fall back to CPU.

```bash
bash scripts/run_multihorizon_cuda.sh check /srv/traffic-data /srv/traffic-runs/multihorizon_cuda_v1
```

Run the study in a persistent terminal session such as `tmux` on a dedicated GPU server. On a scheduled HPC cluster, submit the same training command inside an allocated GPU job instead; do not train on a login node. The output directory must be dedicated to this CUDA protocol and on a filesystem with reliable POSIX file locking and atomic rename. Keep the exact code, Python/PyTorch environment, protocol, and data files unchanged until all runs finish.

```bash
tmux new -s traffic-cuda
bash scripts/run_multihorizon_cuda.sh train /srv/traffic-data /srv/traffic-runs/multihorizon_cuda_v1 \
  2>&1 | tee -a /srv/traffic-runs/multihorizon_cuda_v1.log
```

Detach from `tmux` with `Ctrl-b d`. Reattach with `tmux attach -t traffic-cuda`. If the process stops, first verify it is no longer running, inspect the last log lines, and run the **same** `train` command with the **same** output directory. A nonblocking file lock rejects a second writer. At each completed epoch, `progress.pt`, `epochs.json`, and `status.json` are saved; resume restores model, optimizer, scheduler, RNG, best validation checkpoint, and early-stopping state. A code/config/environment/data identity mismatch stops instead of mixing runs. A changed protocol requires a new output directory.

`TRAFFIC_CUDA_DEVICE=1` selects a GPU by local device index; `CUDA_VISIBLE_DEVICES=...` can instead restrict visible devices. `CUBLAS_WORKSPACE_CONFIG=:4096:8` is set by the script before Python starts so deterministic PyTorch algorithms can be requested. The code keeps cuDNN deterministic and disables TF32; this favors reproducibility over peak speed. If a deterministic operation is unsupported by the server's CUDA/PyTorch build, the check will fail explicitly; record the error and establish a new documented protocol rather than weakening this frozen run in place.

## 3. Locked evaluation and artifacts

Wait for `/srv/traffic-runs/multihorizon_cuda_v1/training_complete.json` and verify all 42 `result.json` files exist. The trainer selects each seed's checkpoint on **validation** MAE and selects a configuration using mean validation MAE across three seeds. Test scoring remains locked until every training run finishes. Then run once:

```bash
bash scripts/run_multihorizon_cuda.sh evaluate /srv/traffic-data /srv/traffic-runs/multihorizon_cuda_v1 \
  2>&1 | tee -a /srv/traffic-runs/multihorizon_cuda_v1.log
```

The final `evaluation_complete.json`, dataset `test_report.json`, per-run `test.json`, selected checkpoint `selected.pt`, epoch curves in `epochs.json`, and `frozen_protocol.json` provide the evidence. The report includes MAE/RMSE by 5–60 minute horizon and traffic-speed/missing-history strata, parameter counts, and synchronized CUDA inference timings. The earlier METR-LA test period was already explored; PEMS-BAY is the separate confirmation dataset, independently trained on its own training period. Compare CUDA and local results as separately identified runs, and do not choose models by repeatedly inspecting test scores.

Only source code, selected checkpoints, and aggregate reports should be published. Keep public raw archives/arrays and all private internship recordings outside Git. The server has not been available here, so CUDA execution and its speed remain **unverified** until the check and full run succeed there.
