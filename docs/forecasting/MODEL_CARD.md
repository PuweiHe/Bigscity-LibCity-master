# Compact mean-residual GRU

## Intended use

Offline one-minute mean-speed estimation from four complete observed minutes. The motivating workflow is construction-road traffic monitoring. No congestion-reduction, safety, incident-detection or live operational impact has been measured.

## Architecture and optimization

The base network is LibCity Seq2Seq with GRU encoder and decoder. Both baseline and tuned models receive `[batch, 4, 1, 4]` traffic inputs and predict one speed value. Baseline: 64 hidden units, direct prediction, random decoder start, MSE. Tuned: 8 hidden units, deterministic last-input decoder start, residual output around the four-minute speed mean, correction scale 0.25, Huber loss, weight decay 0.01. Train-only standardization, Adam at 0.003, gradient clipping 1.0, validation early stopping, up to 120 epochs.

Prediction in standardized units is `mean(past four speeds) + 0.25 * GRU(features)`. Reducing the residual's influence constrains the model around a strong smooth baseline. The comparison changes several ingredients together; it does not isolate each ingredient's causal contribution. The moving-mean row is the essential zero-correction ablation.

The tuned RF learns the same type of residual from lag and summary features, with 200 trees, depth 3, minimum leaf size 10 and correction scale 0.25. It improves on its unconstrained baseline but is effectively tied with the simple moving mean.

## Data and split

13 recordings contain 6,418 vehicle rows. A signed speed encodes direction; absolute magnitude is used. One duplicate row is removed, and one speed above 200 km/h is omitted from speed statistics while its vehicle count is retained. Each recording loses its first and last partial-minute bins. The feature `high_code_share` is the share of numeric vehicle codes >= 7; no unverified interpretation as a heavy-vehicle class is required.

Sessions, not individual rows, are sorted by time and split 9/2/2. Four historical minutes predict the next minute. Train/validation/test contain 132/82/112 windows. Windows never cross recordings; normalizers never fit validation/test. Test recordings occur later than validation and training and are not used to pick hyperparameters or seed.

Sessions are short and collected at different locations/times. One-minute labels average unequal numbers of vehicles. The study has only two independent held-out sessions, so 112 windows are not 112 independent deployment trials. No confidence interval for generalization is claimed.

## Results and uncertainty

Three independently trained seeds: 17, 42, 73. Reported neural/tree scores average each seed's metric, not predictions from an ensemble. The demo serves the predeclared seed 42 checkpoint; its individual GRU test MAE is 5.334 km/h, while the three-seed mean is 5.312 km/h.

GRU MAE: 6.067 → 5.312 km/h (12.44% lower). Compared with persistence: 26.23% lower. Compared with moving mean: 0.97% lower. Test MAE across-seed SD: 0.630 → 0.021 km/h. These quantify this experiment, not guaranteed future effects.

Parameter count: 26,369 → 609 (97.69% lower). This is parameter reduction through a separately trained smaller model, not compression of a trained baseline checkpoint. Warm CPU latency is machine-specific and excludes HTTP/network overhead; see the benchmark JSON.

## Serving and security

The service loads a trusted local artifact once per process, uses the saved training normalizer, disables gradients and validates shapes, feature ranges and consecutive minute timestamps. It does not accept model uploads. Joblib artifacts must never come from untrusted users. Default examples bind to localhost. Predictions are estimates, not calibrated confidence intervals; no physical clipping is applied after prediction.

## Reproducibility boundary

Source Excel files and per-example held-out predictions remain local. Public artifacts contain trained model parameters, aggregate evaluation, hyperparameter/seed histories and data hashes. Anyone can exercise the model API and run the unit/integration suite; exact retraining needs authorized access to the source recordings. This September 2026 reconstruction is linked to 2024 internship work and is not claimed to have been executed in 2024.
