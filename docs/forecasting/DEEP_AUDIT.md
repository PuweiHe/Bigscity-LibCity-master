# Traffic forecasting implementation and evidence audit

## Verified existing evidence

The published one-step checkpoints were loaded and scored again on the documented
METR-LA windows. STGCN MAE = 2.2469141071, DCRNN = 2.2630795938, STTN = 2.4300647547
mph. Absolute differences from the published reports are below 1e-8 mph. The
current compatibility-preserving model changes therefore do not invalidate those
checkpoint predictions. See [checkpoint verification](audit/legacy_checkpoint_verification.json).
Shandong file hashes, 6,418 raw rows, and 132/82/112 chronological window counts
were independently regenerated and match the published audit; see
[source verification](audit/shandong_reverification.json).

These are reconstructed, reproducible measurements. Original lost result files
have not been recovered. Public METR-LA and private Shandong recordings remain
separate datasets, tasks, and units. The root Git commit already contains the
portfolio extensions, so Git history alone cannot reconstruct a pristine internship
folder or prove historical authorship. Inherited architectures are credited to LibCity.

## Findings and remedies

| Finding | Consequence | Action / evidence |
|---|---|---|
| Legacy masks infer missingness from normalized zero | An observed value equal to the training mean could be excluded | New dataset retains raw observation masks. Zero collisions were found in the old METR-LA array, so no numerical inflation is attributed to this edge case. |
| Legacy STGCN/DCRNN residual anchors use the last imputed value | A missing last reading makes the anchor the train mean | Opt-in last-observed anchor accepts an explicit mask; legacy defaults remain compatible. |
| STGCN forward emits one step even when output_window > 1 | A naive multi-step runner could silently broadcast a single prediction | Optional direct 12-step head; strict prediction-target shape checks; recursive full-horizon reference also trained. |
| STGCN graph polynomials were plain tensors | dtype/device moves do not move graph supports | Nonpersistent registered buffers preserve old checkpoint keys; tested with float64. |
| Recursive STGCN initially dropped explicit masks between rollout steps | Observed normalized zeros could become missing when residual mode is combined with recursion | Forward and roll the mask; predicted values become available inputs; regression test covers observed zero plus missing tail. |
| STGCN constructor mutates nested configuration lists | Caller-owned experiment settings can change | Copy block configuration before modification. |
| Original graph comparison changes width, dropout, learning rate and residual together | Cannot attribute its entire gain to residual design | A 2x2 recipe/residual factorial with identical seeds, plus recursive/direct comparison. |
| Old later METR-LA period was inspected during previous exploration | It is not an independent prospective test | Continue to label METR-LA exploratory; add independently trained PEMS-BAY replication with untouched test scores until all runs finish. |
| Original STGCN study supplies a directed graph to spectral Chebyshev convolution | The undirected spectral assumption is not explicit | New STGCN protocol uses max(A,A.T); road graph and sensor ordering are hashed. DCRNN retains directed supports. |
| One seed and 5-minute target dominate the previous graph report | Limited evidence of stability or long-horizon quality | Three seeds; 12 future readings; separate 15/30/60-minute and traffic-state metrics. |
| Only test report shape/statistics were checked against data | A changed source array could escape those checks | New artifacts bind source hashes, code hashes, configuration, environment and normalization. |
| PEMS-BAY naive timestamps skip 01:55→03:00 on March 12 | Treating wall-clock time as elapsed time creates a false outage | Declared America/Los_Angeles interpretation, convert to UTC; no physical five-minute gaps remain. |

The inherited STTN attention divides logits by sqrt(embed_dim), and shares its
per-head linear projections across heads. These differ from some Transformer
implementations but are retained as inherited architectural choices, not relabeled
as proven correctness bugs. Its graph branch uses instance-normalized adjacency
and a learned spatial embedding; it is not a random-walk probability matrix.
DCRNN curriculum learning is disabled in both old and new experiments; the new
runner nevertheless passes the actual optimizer step and restores all RNG states.
DCRNN's internal sparse supports remain CPU-oriented in this study; GPU portability
is not claimed.

## Interpretation limits

- STGCN's 20.18% historical improvement is versus persistence, not an isolated
  effect of a code fix. A correctness fix is not an accuracy gain without an ablation.
- The compact GRU's 12.44% improvement is versus the larger GRU; its improvement
  over the strong moving mean is only about 0.97%. Two held-out recordings do not
  establish broad road-level generalization.
- A seed standard deviation measures training variability, not uncertainty across
  new roads. Overlapping forecast windows are not independent observations.
- New direct-head results must not be compared numerically with old 5-minute
  scores as if they were the same target or data protocol.
- Nonpositive public speed readings are treated as missing, following the retained
  benchmark convention; genuine zero-speed events cannot be distinguished by this file.
- No travel-time, congestion, deployment, or financial improvement has been measured.
- No global optimum is promised; conclusions apply to the declared candidate set.

## Sources

- [LibCity documentation](https://bigscity-libcity-docs.readthedocs.io/zh-cn/latest/index.html)
- [Model list](https://bigscity-libcity-docs.readthedocs.io/zh-cn/latest/user_guide/model.html)
- [Evaluation](https://bigscity-libcity-docs.readthedocs.io/zh-cn/latest/user_guide/evaluator.html)
- [DCRNN authors' data, graph construction and multi-horizon benchmark](https://github.com/liyaguang/DCRNN)
- [STGCN paper](https://arxiv.org/abs/1709.04875)
