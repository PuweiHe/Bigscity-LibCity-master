# Validation search for 12-step speed forecasting

Status: implemented and locally tested; NVIDIA execution and all search results are pending.
No performance improvement is claimed. The existing CPU/MPS protocols are unchanged.

The task is 12 five-minute inputs to 12 future speeds in mph. Do not compare these
numbers with the earlier single-step experiment. The inherited architectures are
LibCity implementations; our contributions are explicit-mask evaluation,
resumable experiment engineering, controlled task adaptations, and measured
ablations when available.

## Search and attribution

`configs/forecasting/cuda_search_v1.json` declares every choice. A seeded sample
without replacement screens 32 STGCN, 8 DCRNN and 8 STTN configurations, plus
three fixed references (51 per dataset). STGCN searches Ks, Kt, bottleneck width,
dropout, learning rate, weight decay and batch size. Kt is limited to 2/3 because
12 - 4*(Kt-1) must remain positive. Two STConv blocks are fixed by the inherited
implementation; arbitrary depth is not exposed as a valid parameter.

Each model family's top three single-seed candidates are retrained with seeds
17, 42 and 73; the lowest mean validation objective selects its hyperparameters.
Screening and replication use the same cap of 180 epochs, minimum 40 and early
stopping patience 25. Adam, clip norm 5 and plateau scheduler (factor .5,
patience 5) are fixed. This is a finite search, not a claim of global optimality.
STGCN receives a larger search budget; comparisons must disclose that advantage.

The final cohort includes compact and tuned STGCN crossed with residual off/on
and full/top-8/top-16 road affinity graphs, all three seeds. Graph pruning retains
existing road edges only and symmetrizes for spectral convolution; it tests
whether limiting weak road connections improves useful spatial context. Union
symmetrization means final node degree may exceed k. Include a compact recursive
reference to isolate the direct multi-output head, plus fixed and tuned DCRNN/STTN.
Report residual effects within matched graph/hyperparameter settings and graph
effects within matched residual/hyperparameter settings, including negative effects.

Selection uses the equal-weight mean of twelve horizon MAEs. The previous runner
uses pooled valid-observation MAE; the new objective is explicitly versioned and
retains pooled MAE as a diagnostic. No configuration is chosen by test results.

## Execution and evidence

Use `python -m traffic_forecasting.cuda_search --data-dir DATA --output-dir RUNS
--dataset METR_LA --stage all` inside an allocated CUDA job. The identical command
resumes epoch checkpoints; unfinished epochs are repeated. `--stage screen`,
`replicate`, and `ablate` support staged operation. Dataset-specific locks allow
the two datasets in different jobs but prohibit duplicate writers per dataset.
Do not edit code, device architecture, package environment or plan after starting.
An incompatibility or OOM stops with an error log instead of silently changing
batch size, dropping a trial or falling back to CPU. Resolve failures in a new
version if they change the protocol.

Each candidate first executes a full configured training batch: forward, masked
loss, backward, finite gradients, optimizer step and inference, including a check
that changing future labels cannot change inference. FP32 is used; TF32 and AMP
are disabled. A future AMP throughput study must be separately registered.

Outputs contain frozen source hashes, runtime/hardware, data hashes, candidate
manifest, effective configurations, smoke checks, per-epoch metrics, atomic
model/optimizer/scheduler/RNG checkpoints, peak memory, errors, validation rankings
and a locked final selection. Selected publication seed is fixed at 42. Peak
memory after resume describes the current process segment. Wall-clock training
cost includes validation but excludes queueing and smoke checks; retain Slurm
accounting for total job cost.

## Holdout governance

This search runner intentionally has no test-scoring mode. METR-LA is exploratory.
No local multi-horizon test report existed at the preflight audit, but that is not
proof that PEMS-BAY has never been inspected. The old CPU wrapper automatically
evaluates when all 42 runs finish. Before final confirmation, audit local and server
test access and freeze the selection/evaluator. If PEMS-BAY test metrics informed
any new choice, label it exploratory and preregister a genuinely unused dataset or
time cohort; renaming a previously used partition does not create a holdout.

Final locked evaluation, scientific plots, paired seed effects, inference timings,
publication weights and resume claims remain pending real training outputs.
Uncertainty across seeds is not uncertainty across independent road systems.
Overlapping windows require temporal blocks, rather than treating windows as
independent observations, if estimating test confidence intervals.

## USC CARC

Use `scripts/slurm/carc_cuda_search.sbatch` via Slurm, not a login-node training
process. Supply your account with `sbatch --account=ACCOUNT ...`; the template
requests one A100, four CPUs and 32 GB RAM for 47.5 hours. Set `TRAFFIC_REPO`,
`TRAFFIC_DATA`, `TRAFFIC_RUNS`, `TRAFFIC_PYTHON`, and `TRAFFIC_DATASET` to absolute
server paths. Check availability/account limits before submission. Re-submit the
same command after a wall-time stop, preserving the output directory and GPU type.
No automatic infinite resubmission is installed. Back up checkpoints from scratch
storage to persistent project storage and mirror complete evidence to local
`Libcity/model_training`; retain job IDs, logs, package lists and hashes.

Official references:
- [CARC running jobs](https://www.carc.usc.edu/user-guides/hpc-systems/using-our-hpc-systems/running-jobs)
- [CARC Discovery login](https://www.carc.usc.edu/user-guides/hpc-systems/discovery/getting-started-discovery)
- [LibCity configuration](https://libcity.ai/Bigscity-LibCity-Docs/user_guide/config_settings.html)

LibCity's standard CLI/config precedence does not automatically apply to this
custom runner. Only keys explicitly consumed by this runner and model constructors
are effective; each trial records its resolved model and optimizer configuration.
