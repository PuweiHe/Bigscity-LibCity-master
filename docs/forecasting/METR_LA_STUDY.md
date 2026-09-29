# METR-LA graph forecasting study

This study reconstructs the author's 2024 internship-related graph-forecasting work using a **public Los Angeles traffic-speed dataset** to exercise LibCity's STGCN and DCRNN model families. METR-LA is distinct from the private CASIA internship recordings. The task is to predict speed at all 207 loop detectors **5 minutes ahead** from the previous 12 readings (one hour). Speeds are in mph. The measured numbers below come from the documented runs, not recovered original result files.

## Data and experimental design

- Dataset: METR-LA, 34,272 five-minute timestamps × 207 sensors. Source: [LibCity raw-data catalog](https://bigscity-libcity-docs.readthedocs.io/zh-cn/latest/user_guide/data/raw_data.html), downloaded as [the OpenCity METR_LA archive](https://huggingface.co/datasets/hkuds/OpenCity-dataset/blob/main/METR_LA.zip). Archive SHA-256: `6016db2487094461b2dcbd6bea0e560b64222db784195a38940336458d5bb34d`. The archive includes speed and a 207×207 weighted road-network adjacency matrix. The raw archive is not committed.
- Chronological timestamp cutoffs: 23,990 (70%) and 27,417 (80%). Every 12-history/1-target window is fully inside its partition. Training starts are sampled every four timestamps (20 minutes); validation and test cover every possible start: **5,995 / 3,415 / 6,843 windows**. The test metric covers 1,244,780 valid sensor targets. These are one-step, single-seed results, not the original multi-horizon paper benchmark.
- Only positive train speeds determine normalization mean and standard deviation. Zero denotes unavailable sensor readings and is excluded from loss and metrics. No transformation is fitted to validation or test values.
- Validation MAE chooses among two candidates per model and the best epoch within 30. Seed 42, Adam, batch 8, gradient clipping at 5. Evaluation computes MAE and RMSE in original mph units. The business baseline uses the most recent available speed within the 12-step history, falling back to the train mean if all 12 are missing. The training and validation phase saved checkpoints without scoring test; a separate command evaluated those locked checkpoints.

| Model selected by validation | Trainable parameters | Validation MAE, mph ↓ | Test MAE, mph ↓ | Test RMSE, mph ↓ | Reduction vs persistence |
|---|---:|---:|---:|---:|---:|
| Last available speed | 0 | 2.632 | 2.815 | — | — |
| STGCN, 16/32 channels, 3-hop Chebyshev, last-speed residual | 57,553 | **2.084** | **2.247** | **3.903** | **20.18%** |
| DCRNN, 32 hidden units, two-step dual random walk, last-speed residual | 31,905 | 2.095 | 2.263 | 3.915 | 19.61% |

The smaller non-residual STGCN reached 2.107 mph validation MAE; the smaller one-step DCRNN reached 2.135 mph. Wider residual candidates won validation, at epochs 29 and 27 respectively. Validation differences between the selected graph models remain small, so no general architecture superiority claim is justified. Full configurations, best epochs, and metrics are in [the JSON report](metr_la_report.json); [learning curves](metr_la_learning_curves.png) show all four candidates.

An [earlier 20-epoch sampled-window report](metr_la_sampled_report.json) used 1,000/285/571 windows and a mean-imputed last-reading baseline. It is retained for the tuning audit, but its test MAE is **not directly comparable** with this denser evaluation. Earlier test scores were visible before the dense study; the final comparison is retrospective, not a fresh blind test.

## Concrete model and engineering changes

1. **Graph convolution correctness:** DCRNN's sparse COO conversion sorted coordinates without sorting their corresponding weights. The conversion now reorders both together and coalesces the tensor; a synthetic non-symmetric graph regression test catches coordinate/value mismatches. Diffusion also restarts from the same input for each directed support, preventing the second random walk from inheriting the first support's polynomial state. We did not isolate a numerical performance gain for these correctness fixes.
2. **Task-aware tuning:** Compared Chebyshev STGCN widths/dropout and DCRNN diffusion depth/hidden width under identical data splits. Added optional last-speed residual heads to both LibCity models, preserving their default behavior. More training windows and a 30-epoch cap exposed the benefit of wider residual candidates on validation.
3. **Reproducibility:** The runner checks input shapes/nonfinite values, keeps windows inside chronological partitions, uses train-only statistics, saves selected model weights and configurations, and separates validation-only selection from checkpoint evaluation. A missing-aware last-available-speed baseline is scored on identical target observations. Tests cover split boundaries, tensor shape/gradients, sparse graph values, and baseline missingness.

## Reproduce

Extract `METR_LA.zip` under `raw_data/METR_LA/` so that `METR_LA.npz` and `METR_LA_rn_adj.npy` are present. Verify the archive hash above. With Python 3.12 and a PyTorch/NumPy/SciPy environment (the focused dependencies are in `requirements-reproduction.txt`), run:

```bash
python -m traffic_forecasting.metr_la_study --epochs 30 --train-stride 4 --eval-stride 1 --models STGCN DCRNN --validation-only
python -m traffic_forecasting.metr_la_study --epochs 30 --train-stride 4 --eval-stride 1 --models STGCN DCRNN --evaluation-only
python -m unittest -q tests.forecasting.test_graph_models
```

The runner writes `outputs/metr_la/report.json` and checkpoints; published examples are in `artifacts/metr_la/`. Use a fresh output directory for a new run. Exact floats can vary across hardware and PyTorch versions. The code inherits LibCity STGCN and DCRNN architecture; this project adds the graph fixes, residual option, data protocol, search, evaluation, and checkpoints. It does not reproduce the original DCRNN paper's 15/30/60-minute benchmark or establish an online traffic-management outcome.

**Evaluation limitation:** Earlier test scores on the same later chronological period were inspected before the 30-epoch dense study. This is therefore an exploratory retrospective result, not a sealed prospective holdout. The stated percentage improvements are offline MAE reductions versus last-available-speed persistence on these windows; they are not reduced travel time, congestion, or operational cost. A new time period or a fresh external dataset would be needed to validate generalization independently.
