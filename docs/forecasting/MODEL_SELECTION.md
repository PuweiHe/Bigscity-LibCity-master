# Model selection from the LibCity catalog

The [LibCity reproduced-model list](https://bigscity-libcity-docs.readthedocs.io/zh-cn/latest/user_guide/model.html) groups SVR, FNN, and Seq2Seq among its time-series and traffic-state baselines; STGCN and DCRNN are graph-based traffic-speed models, while STTN uses spatial and temporal Transformer blocks. The retained internship data support a one-station next-minute speed task. A [separate public METR-LA study](METR_LA_STUDY.md) supports network-wide graph and Transformer forecasting.

| Catalog model | Repository implementation | Decision |
|---|---|---|
| SVR | `test/test_SVR.py` | Reproduced as a train-only-scaled, single-target sklearn pipeline in `traffic_forecasting/catalog_study.py`. The inherited script averages and repeats all future targets/features, so its METR-LA output contract cannot be used as-is for mean-speed prediction. |
| FNN | `libcity/model/traffic_speed_prediction/FNN.py` | Reproduced the original one-hidden-layer baseline; added optional second hidden layer and mean-speed residual correction without changing its defaults. |
| GRU Seq2Seq | `libcity/model/traffic_speed_prediction/Seq2Seq.py` | Reproduced and tuned in the primary study. Deterministic label-free inference and compact residual prediction are documented in the main model card. |
| STGCN / DCRNN | `libcity/model/traffic_speed_prediction/STGCN.py` / `DCRNN.py` | Excluded from the private-recording comparison because those 13 sessions have no synchronized sensor graph. Reproduced and tuned separately with the public METR-LA speed series and road adjacency matrix. |
| STTN | `libcity/model/traffic_speed_prediction/STTN.py` | Reproduced spatial-temporal attention on the same public METR-LA split; compared compact, residual, and wider/deeper variants. The validation-selected model did not beat STGCN on the later test period. |

The single-station models in the table below use the same 13 source Excel recordings and session-level chronological 9/2/2 split: 132 train, 82 validation, 112 later test windows. The original test sessions were opened in the earlier GRU/RF benchmark. **The SVR/FNN extension below is retrospective and exploratory, not a new blind test.** Preprocessing parameters and the standardizer fit train data only. The metric is MAE in km/h; the 4-minute moving mean is a strong nonlearned comparator. STGCN, DCRNN, and STTN use the separate public [METR-LA protocol](METR_LA_STUDY.md).

| Model | Validation MAE | Later-session MAE | Interpretation |
|---|---:|---:|---|
| Moving mean | 6.196 | 5.365 | Strong nonlearned comparator |
| Catalog SVR baseline, RBF | 6.902 | 5.497 | Original kernel/default-style setting, adapted to one target |
| Validation-selected linear SVR | 5.831 | 6.374 | Validation improved, later sessions worsened; rejected |
| Catalog FNN baseline, 128 hidden units | 6.861 | 6.435 | Direct speed prediction, MSE |
| Two-layer mean-residual FNN, 32/8 units | 6.118 | 5.480 | **14.84% lower MAE than FNN baseline**; still above moving mean |
| Previously selected compact GRU Seq2Seq | 6.111 | 5.312 | Best measured neural model in this small study; only 0.97% below moving mean on later sessions |

FNN candidate tuning covered first-layer sizes {8, 16, 32} and residual scales {0.1, 0.25}; all use an 8-unit second layer, Huber loss, weight decay 0.01, Adam 0.003, clipping 1.0, and validation early stopping. The baseline keeps the inherited single-layer FNN architecture, 128 units, direct output and MSE under the same 120-epoch cap and three seeds {17, 42, 73}. The selected FNN has 817 parameters versus 2,305 for the baseline. SVR candidates compare linear/RBF kernels, C and epsilon; each fits a `StandardScaler` on training windows only. Selection uses mean validation MAE (for FNN, over the three seeds). The checkpoint and data hashes are frozen before the test command.

The result supports model reproduction, correct task adaptation, tuning, and negative-result analysis. It does not show a measured congestion, safety, or operational benefit. Two test sessions are too few for a general claim that FNN or GRU will win on other roads. The original source recordings are not published; [candidate scores](catalog_selection.json), [test results](catalog_test_report.json), model weights, and hashes are published.

```bash
python -m traffic_forecasting.catalog_study select --data-root /path/to/山东高速数据分析-半幅封闭 --output outputs/catalog_study
python -m traffic_forecasting.catalog_study evaluate --data-root /path/to/山东高速数据分析-半幅封闭 --output outputs/catalog_study
python -m traffic_forecasting.inference --artifact artifacts/forecasting/fnn_6.joblib --family fnn --request docs/forecasting/example_request.json
```
