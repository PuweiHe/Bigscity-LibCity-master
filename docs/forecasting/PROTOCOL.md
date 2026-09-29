# Experiment protocol (fixed before test evaluation)

Business objective: offline one-minute-ahead mean-speed forecasting for short-term monitoring of expressway construction recordings. This is not a deployed alerting system or a measured reduction in congestion.

Source: 13 top-level Excel recordings, seven preconstruction and six construction. Do not reuse copied subdirectories or intermediate combined CSVs. Sort whole recording sessions chronologically: first nine train, next two validation, final two test. The test consists of later May 29 recordings and remains unopened for model selection. No random row split and no windows crossing recording boundaries. Recording durations are short; this is a small retrospective benchmark.

Drop exact duplicate vehicle rows and the first/last partial minute in each recording. Exclude speed magnitudes above a predeclared 200 km/h plausibility threshold from speed statistics (retain vehicle counts). Take signed-speed magnitude, then aggregate mean speed, observed count, speed standard deviation and high vehicle-code share (code >= 7, a numeric category proxy; not a verified heavy-vehicle semantic label). Each input is four consecutive complete minutes, target is next minute's mean speed. No interpolation or zero-filling outages. All normalization fits training observations only.

Baselines: persistence; four-minute moving mean; RandomForestRegressor (100 trees, default unconstrained depth); inherited LibCity GRU Seq2Seq (64 hidden units, random start, direct prediction, MSE).

Random forest search: eight predeclared configurations combining min_samples_leaf in {2,5}, max_depth in {4,8}, and direct versus residual prediction. Use 200 trees and engineered trailing-window summaries. Select lowest validation MAE. No test-driven retries.

GRU search: four predeclared configurations combining hidden dimension in {16,32} and residual scale in {0.25,1.0}, deterministic last-observation initialization, Huber loss, weight decay 0.001, gradient clipping 1.0. Adam learning rate 0.003; at most 120 epochs; validation early stopping patience 20. Baseline receives the same epoch limit and validation checkpoint selection. Seeds {17,42,73}; select by mean validation MAE across seeds. Publish all seed results, not the best seed.

Primary metric: MAE in km/h. Secondary: RMSE, per-recording MAE, CPU batch-one prediction latency. Quantify gains against the explicitly named untuned model and persistence separately. Freeze the selected configurations and checkpoints before running the test command. Test reports cannot be overwritten by the standard CLI.

Models are trained once on train only. Validation chooses hyperparameters and checkpoint epochs; no train+validation refit. Future observations in a test recording can become history for later one-step forecasts (rolling-origin evaluation), but model parameters remain frozen. Three-seed averages measure training randomness, not confidence across future deployments. Two held-out sessions are insufficient for broad claims about all roads or traffic conditions.

## Validation-only refinement (test still sealed)

First-round validation MAE: persistence 7.647, four-minute mean 6.196, baseline RF 6.772, baseline GRU 6.259 km/h. Last-observation residual variants did not improve on the best baselines. This suggests that anchoring on one noisy minute is unsuitable here.

A second and final validation search changes the residual anchor to the four-minute mean. RF: min_samples_leaf {3,10}, max_depth {3,6}, residual scale {0.25,0.5}, 200 trees. GRU: hidden units {8,16}, residual scale {0.1,0.25}, weight decay 0.01, Huber loss; all other training rules unchanged. Preserve the original search and select from *all* candidates including the untuned baseline. A failed tuning attempt must not force a worse model. No further selection will occur after test evaluation.

## Retrospective follow-up after the test was opened

The direct-GRU ablation in this reconstruction is explicitly exploratory. It compares hidden sizes {8,16,32} and residual scales {0.1,0.25} on the original validation split, selects by three-seed mean validation MAE, then records the result on the previously inspected test sessions once. It must not be characterized as the sealed test protocol above or used to revise the original benchmark claim.
