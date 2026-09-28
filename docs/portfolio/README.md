# Traffic similarity optimization

This September 2026 portfolio extension reconstructs the station-pair loop used in the author's traffic-analysis notebooks. The inherited LibCity framework and its forecasting models belong to the upstream contributors; their presence does not establish an original model contribution or a completed training experiment.

## Problem and change

The original analysis filters each station's observations inside a nested loop and computes both `(i,j)` and `(j,i)`. The extension sorts and validates each series once, then evaluates only `i < j` and mirrors the result. For six series this reduces distance calls from 30 to 15. Both schedules remain quadratic in station count; this is a constant-factor improvement.

The runnable dependency-light benchmark uses **exact scalar DTW with absolute local cost** in BOTH schedules. Original notebooks use approximate FastDTW with scalar Euclidean cost. The benchmark therefore isolates pair scheduling; it does not claim exact DTW reproduces FastDTW's numerical output or is faster than FastDTW. Exact DTW costs O(TU) time and O(min(T,U)) working memory per pair. The resulting matrix costs O(N²) memory.

## Reproduce

From the repository root, with Python 3.12:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-portfolio.txt
python -m unittest discover -s tests/portfolio -p 'test_dtw.py' -v
python -m traffic_analysis --output outputs/synthetic --repeats 5
```

The default demo generates six synthetic 60-step signals using seed 42. No internship data is required. Committed measurements and the distance matrix are in [synthetic](synthetic/benchmark.json). Runtime is machine-dependent; use call counts as the stable resume metric.

For a local minute-level CSV:

```bash
python -m traffic_analysis --input /path/to/minute_flow.csv \
  --time-column 分钟 --station-column 站点 --value-column 交通流量 \
  --output outputs/local --repeats 5
```

Outputs: a labeled `dtw_matrix.csv` and `benchmark.json` containing timings, environment, sample counts, observation intervals and equivalence checks. Invalid, duplicate, nonfinite or irregular series fail explicitly. Missing intervals are not silently treated as zero traffic. Each station is sorted by timestamp. Unequal observation lengths are supported.

## Interpretation and limitations

DTW compares shape after temporal warping. Different stations in the available preconstruction CSV were observed at different times, so these distances are descriptive comparisons, not evidence of simultaneous spatial correlation, vehicle travel time, propagation direction or geographic distance. The implementation intentionally does not infer road distances from overlapping recording durations. Raw DTW distances also depend on sequence length and scale.

This is an analysis module, not a learned adjacency matrix integrated into forecasting. A future forecasting experiment must construct features from training data only, retain station/time alignment, split target timestamps without overlap, fit normalization on training data, compare against persistence, and report held-out metrics. The available short, nonsynchronous station series do not support the original resume's 8% accuracy claim.

## Contribution map

- `traffic_analysis/dtw.py`: rolling-memory exact DTW and unordered-pair computation.
- `traffic_analysis/__main__.py`: validated CSV loading, synthetic demo, benchmark and output artifacts.
- `tests/portfolio/test_dtw.py`: known answers, full-table reference, pair-call count, baseline equivalence and invalid-input regression tests.
- `.github/workflows/portfolio.yml`: automated tests and smoke demo.

Original notebooks and employer data remain outside this repository. The upstream Apache-2.0 license and README are retained. This directory was supplied as an archive without Git history, so its upstream revision and historical authorship cannot be inferred.

## Original FastDTW reproduction

The original notebook's FastDTW loop was rerun on the available preconstruction minute-flow CSV (six series, 93 station-minute rows). The diagnostic [result](fastdtw_reproduction.json) found a maximum difference of **1.0** when mirrored. The approximate algorithm is not symmetric on this input; therefore that shortcut was rejected for the original FastDTW backend. Only the exact-DTW benchmark demonstrates equivalent results with half the pair calls.

```bash
pip install -r requirements-reproduction.txt
PYTHONPATH=. python scripts/reproduce_fastdtw.py --input /path/to/minute_flow.csv --output outputs/fastdtw.json
```

## Seq2Seq inference repair

A forward-pass reproduction of the actual inherited Seq2Seq implementation identified two behaviors: repeated evaluation produced different outputs because the decoder starts from fresh random noise, and prediction raised `KeyError` without future `y` labels. [Before](seq2seq_before.json) and [after](seq2seq_after.json) reports use seed 42 and a synthetic input, not a trained forecasting benchmark.

Changes in `libcity/model/traffic_speed_prediction/Seq2Seq.py`:

- Added `decoder_start=last` (last observed target features) and `zero`. Default `random` preserves the inherited initialization behavior and checkpoint tensor shapes. Existing models should explicitly opt in and be revalidated; this is not a promised accuracy improvement.
- Read future labels only during training when teacher forcing is enabled. Evaluation accepts an X-only batch.
- Create recurrent states with the input dtype/device and normalize RNN-type capitalization, including lowercase LSTM.

```bash
pip install -r requirements-reproduction.txt
PYTHONPATH=. python scripts/reproduce_seq2seq.py --decoder-start last
python -m unittest discover -s tests/portfolio -v
```

Tests cover RNN/LSTM/GRU, float64, finite gradients, teacher forcing, legacy random mode, exact repeatability on CPU, label independence and tiny-batch overfitting. No GPU determinism, full LibCity training pipeline, STGCN experiment or held-out accuracy gain has been verified. The added `seq2seq_deterministic.json` illustrates the opt-in configuration. `scripts/reproduce_seq2seq.py` loads the actual module directly to avoid importing unrelated optional model dependencies.
