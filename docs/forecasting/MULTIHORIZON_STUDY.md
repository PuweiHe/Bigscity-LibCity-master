# Multi-horizon speed forecasting: protocol and current status

**Status: training; final results are not yet available.** The existing one-step
results remain in [METR_LA_STUDY.md](METR_LA_STUDY.md). See the
[implementation audit](DEEP_AUDIT.md) for verified findings.

## Objective and data

Predict the next 12 five-minute speeds from the preceding 12 readings. The
operational motivation is earlier visibility into future road speeds; this is
an offline forecast study, not an evaluated traffic-control deployment.
METR-LA (207 sensors) is development/exploratory data. PEMS-BAY (325 sensors) is
an independent geographic/time replication: models are trained on Bay training
data, not transferred zero-shot from Los Angeles. The protocol is frozen before
Bay test scoring. Source data remain local, and only file hashes are published.

Each dataset uses chronological 70/10/20 partitions; every history and all 12
future targets stay inside its partition. All eligible training windows are used
(stride 1), compared with stride 4 in the older one-step study. Normalization is
fit on valid training observations only. Missing inputs become normalized zero,
but an explicit boolean mask retains the distinction from observed train-mean
speeds. Test masks/values are not used for tuning. METR-LA source files have no
retained timestamp vector, so only contiguous index timing is assumed there.
PEMS-BAY naive timestamps are interpreted in America/Los_Angeles and converted
to UTC; the spring DST jump is not treated as a physical missing hour.

## Preregistered comparisons

See [machine-readable configuration](../../configs/forecasting/multihorizon.json).
All graph/Transformer runs use masked MAE across all 12 horizons, batch 16,
Adam, gradient clipping 5, validation plateau LR reduction, at most 60 epochs,
minimum 20 epochs before early stopping, and patience 10. Seeds: 17, 42, 73.
All use the same chronological windows, observed targets and stopping rule;
training seconds and inference latency are reported because FLOP budgets differ.

| ID | Change relative to compact STGCN reference |
|---|---|
| stgcn_reference | Inherited two-block compact 8/16-channel architecture adapted with direct 12-step head, dropout 0, LR .003 |
| stgcn_reference_residual | Only add explicit-mask last-observed-speed residual |
| stgcn_hyperparameters | 16/32 channels, dropout .1, LR .001; no residual |
| stgcn_hyperparameters_residual | Changed recipe plus residual |
| stgcn_recursive_reference | Same compact encoder and one-step output, recursively rolled out and trained against all 12 future targets |
| dcrnn_reference | 32 recurrent units, two-step dual random walk, observed-speed residual, LR .001 |
| sttn_reference | 32 embedding dimensions, two layers, four heads, residual scale .5, LR .0005 |

The reference is a compact adaptation, **not the original paper's default model
or claimed paper-score reproduction**. The 2x2 design isolates the residual effect
at fixed recipe, and the recipe effect at fixed residual status. The two recipes
come from prior development; this is a bounded comparison, not an exhaustive
hyperparameter search. The recursive/direct comparison isolates the output-head
adaptation while keeping encoder width fixed. Source-fix effects are not conflated
with architectural accuracy improvements. Persistence and 12-reading moving mean
are evaluated on the same target observations.

## Selection and final analysis

All 42 runs must complete before the evaluation command can score any new test
window. Configurations are ranked by mean validation MAE across three seeds and
all 12 horizons. The selected epoch is based only on validation for each run.
All preregistered ablations are then reported, including failures to improve;
there is no test-driven reranking for model promotion. The selected checkpoint
for publication uses seed 42, fixed in advance rather than the best test seed.

Final tables will include mean ± sample standard deviation over seeds, 15/30/60
minute MAE/RMSE, all-horizon MAE, paired residual effects at each recipe, parameter
count, training seconds and CPU batch-1/batch-16 median/p95 inference latency.
Error analysis reports target-speed strata <30, 30–55 and >=55 mph, and histories
with >25% missing inputs. These are descriptive strata, not proven causal traffic
states. Counts represent forecast/target pairs, including overlapping windows.

## Reproduction

Install the pinned graph requirements plus `h5py==3.14.0` and `pandas==2.2.0` for
conversion. Download `pems-bay.h5` and `distances_bay_2017.csv` using links from the
[DCRNN author repository](https://github.com/liyaguang/DCRNN). Place them under a
local `data/PEMS_BAY` directory; retain METR_LA.npz and METR_LA_rn_adj.npy under
`data/METR_LA`.

```bash
python -m traffic_forecasting.prepare_pems_bay --root /path/to/data/PEMS_BAY
python -m traffic_forecasting.multihorizon_study --protocol configs/forecasting/multihorizon.json --data-dir /path/to/data --output-dir /path/to/runs --mode train
python -m traffic_forecasting.multihorizon_study --protocol configs/forecasting/multihorizon.json --data-dir /path/to/data --output-dir /path/to/runs --mode evaluate
```

The local convenience command is `bash ../model_training/run_multihorizon.sh all`.
A process lock prevents duplicate writers. Atomic epoch checkpoints restore model,
optimizer, scheduler, RNG state, early-stopping counter, and full epoch history.
Restart the same command after interruption. Code/config/data identity mismatches
stop the run rather than silently mix experiments. Use a new output directory
for a deliberately changed protocol. Final reports will be published only after
actual evaluation and verification.
