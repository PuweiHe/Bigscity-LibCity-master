# Parallel Apple Silicon STTN training

The frozen [multi-horizon study](MULTIHORIZON_STUDY.md) runs 42 CPU trials. An independent MPS run trains only its preregistered STTN configuration on METR-LA and PEMS-BAY, with the same chronological splits, 12-input/12-output windows, masking, normalization, model hyperparameters, 60-epoch cap, early-stopping rule and seeds 17/42/73. Its output directory and protocol fingerprint are separate. CPU training continues unchanged. No MPS validation or test improvement is claimed while training is in progress.

The MPS adapter transfers each batch to the GPU and returns predictions to CPU for double-precision metric accumulation. In STTN, PyTorch 2.10's MPS backward failed for a rank-5 `Linear` input. The adapter flattens only the leading dimensions into a 2-D matrix for `Linear` and restores the original shape; this is mathematically equivalent because `Linear` acts on the final dimension. A same-weight CPU/MPS smoke comparison had a maximum absolute prediction difference of `7.15e-7`. A checkpoint smoke test saved model and optimizer tensors on CPU, restored them to MPS with RNG state, and reproduced the next training step exactly. These checks establish execution compatibility, not final model accuracy.

In a concurrent local benchmark on METR-LA, warm-started MPS time for 100 batch-16 training steps was 9.32 s for reference STGCN, 27.47 s for reference DCRNN and 21.71 s for STTN. The CPU study's measured epochs are much shorter for STGCN and DCRNN but about 16 minutes for STTN. Because the GPU and CPU figures were not a controlled paired benchmark with identical warm-up and system load, they are used only to prioritize the expensive STTN run; they are not reported as a speedup claim. PEMS-BAY's 325-sensor STTN also passed a full batch-16 MPS forward/backward/optimizer smoke test.

To reproduce, prepare the public METR-LA and PEMS-BAY arrays as described in [the main study](MULTIHORIZON_STUDY.md), then run from the repository root in a macOS session where `torch.backends.mps.is_available()` is true:

```bash
PYTHONPATH=. python -m traffic_forecasting.multihorizon_mps \
  --protocol configs/forecasting/multihorizon_mps_sttn.json \
  --data-dir /path/to/prepared-public-data \
  --output-dir /path/to/separate-mps-output \
  --mode train
```

The same command resumes completed epochs from atomic checkpoints. It refuses to run if MPS is unavailable or the frozen source, environment, protocol, or data manifest changes. Use `--mode evaluate` only after all six training trials finish. Compare the MPS and CPU runs as separately identified replications; do not select a backend by repeated test-set inspection. Raw traffic archives are not published in this repository.
