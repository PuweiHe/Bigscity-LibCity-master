# METR-LA graph forecasting study

This September 2026 portfolio reconstruction uses a **public Los Angeles traffic-speed dataset** to exercise LibCity's STGCN and DCRNN model families. It is separate from the 2024 CASIA internship recordings. The task is to predict speed at all 207 loop detectors **5 minutes ahead** from the previous 12 readings (one hour). Speeds are in mph.

## Data and experimental design

- Dataset: METR-LA, 34,272 five-minute timestamps × 207 sensors. Source: [LibCity raw-data catalog](https://bigscity-libcity-docs.readthedocs.io/zh-cn/latest/user_guide/data/raw_data.html), downloaded as [the OpenCity METR_LA archive](https://huggingface.co/datasets/hkuds/OpenCity-dataset/blob/main/METR_LA.zip). Archive SHA-256: `6016db2487094461b2dcbd6bea0e560b64222db784195a38940336458d5bb34d`. The archive includes speed and a 207×207 weighted road-network adjacency matrix. The raw archive is not committed.
- Chronological timestamp cutoffs: 23,990 (70%) and 27,417 (80%). Every 12-history/1-target window is fully inside its partition. To bound CPU cost, training starts are sampled every 24 timestamps (2 hours), while validation and test starts are sampled every 12 timestamps (1 hour): **1,000 / 285 / 571 windows**. These are sampled-window metrics, not all-timestamp benchmark scores.
- Only positive train speeds determine normalization mean and standard deviation. Zero denotes unavailable sensor readings and is excluded from loss and metrics. No transformation is fitted to validation or test values.
- Validation MAE chooses among two predeclared candidates per model and the best epoch within eight. Seed 42, Adam, batch 8, gradient clipping at 5. Evaluation computes MAE and RMSE in original mph units. A persistence forecast (last observed speed) is the business baseline.

| Model selected by validation | Trainable parameters | Validation MAE, mph ↓ | Test MAE, mph ↓ | Test RMSE, mph ↓ | Reduction vs persistence |
|---|---:|---:|---:|---:|---:|
| Last observed speed | 0 | 2.623 | 2.815 | — | — |
| STGCN, 8/16 channels, 3-hop Chebyshev | 24,425 | 2.227 | **2.421** | **4.178** | **13.99%** |
| DCRNN, 16 hidden units, dual random walk | 5,009 | 2.251 | 2.451 | 4.257 | 12.95% |

The STGCN 16/32-channel residual candidate reached 2.227 mph validation MAE, marginally worse than the 8/16-channel version (2.2266). The DCRNN 32-unit two-step diffusion residual candidate reached 2.253 mph, worse than the 16-unit one-step version (2.251). The validation differences are tiny; no superiority claim between the graph models is justified. Full configurations, best epochs, and metrics are in [the JSON report](metr_la_report.json).

## Concrete model and engineering changes

1. **Graph convolution correctness:** DCRNN's sparse COO conversion sorted coordinates without sorting their corresponding weights. The conversion now reorders both together and coalesces the tensor; a synthetic non-symmetric graph regression test catches coordinate/value mismatches. Diffusion also restarts from the same input for each directed support, preventing the second random walk from inheriting the first support's polynomial state. We did not isolate a numerical performance gain for these correctness fixes.
2. **Task-aware tuning:** Compared Chebyshev STGCN widths/dropout and DCRNN diffusion depth/hidden width under identical data splits, rather than assuming the larger model wins. Added optional last-speed residual heads to both LibCity models, preserving their default behavior. These residual candidates did **not** win validation in the final run and are reported as negative ablations.
3. **Reproducibility:** The runner checks input shapes/nonfinite values, keeps windows inside chronological partitions, uses train-only statistics, saves both selected model weights and their configuration, and reports the persistence baseline on the identical test observations. Synthetic tests cover leakage boundaries, tensor shape/gradients, and sparse graph values.

## Reproduce

Extract `METR_LA.zip` under `raw_data/METR_LA/` so that `METR_LA.npz` and `METR_LA_rn_adj.npy` are present. Use a PyTorch/NumPy/SciPy environment, then run:

```bash
python -m traffic_forecasting.metr_la_study --epochs 8 --train-stride 24 --eval-stride 12 --models STGCN DCRNN
python -m unittest -q tests.forecasting.test_graph_models
```

The runner writes `outputs/metr_la/report.json` and checkpoints; published examples are in `artifacts/metr_la/`. Exact floats can vary across hardware and PyTorch versions. The code inherits LibCity STGCN and DCRNN architecture; this project adds the graph fixes, residual option, data protocol, search, evaluation, and checkpoints. It does not reproduce the original DCRNN paper's 15/30/60-minute benchmark or establish an online traffic-management outcome.

**Evaluation limitation:** A smaller pilot was inspected before the final configurations and full run. The published test comparison is therefore an exploratory retrospective result, not a sealed prospective holdout. The stated percentage improvements are measured offline MAE reductions versus persistence on these sampled windows; they are not reduced travel time, congestion, or operational cost.
