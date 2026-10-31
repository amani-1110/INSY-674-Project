# Presentation guide — 10 slides, maximum 10 minutes

Aim for about 55 seconds per slide, leaving a short transition buffer. Lead with the real-data question and distinguish telemetry imitation from simulator control throughout.

| Slide | Time | Story and evidence |
|---|---:|---|
| 1. Research question | 0:45 | Can historical telemetry predict throttle and braking? Audience: ML and motorsport researchers; define narrow scope |
| 2. Dataset and labels | 0:55 | 1,774 rows, three Monza qualifying laps, one driver/session; throttle continuous, brake binary |
| 3. Data quality | 1:00 | `figures/eda_distributions.png`; 13.0% brake-positive, no saved missing rows, irregular/interpolated signals, no start/recovery data |
| 4. Track context | 0:55 | `figures/eda_curvature_and_position.png`; relate current curvature, speed and recorded controls without causal claims |
| 5. Leakage-safe preparation | 0:55 | Whole-lap GroupKFold, transform fitted from training-lap positions, sklearn pipelines, two independent groups |
| 6. Throttle model results | 1:00 | `figures/model_comparison.png`; HistGradientBoosting MAE 0.0389 ± 0.0023; emphasize only two folds |
| 7. Brake model results | 1:00 | RF F1 0.8646 ± 0.0079, balanced accuracy 0.9292 ± 0.0083; show confusion matrix and weak MLP recall |
| 8. Tuning and ensemble | 0:55 | `hyperparameter_trials.csv`, `ensemble_cv_summary.csv`; throttle tuning promising, brake search/soft ensemble no improvement |
| 9. Simulator evidence | 1:00 | `figures/frozen_simulator_results.png`; Spa closed-loop results are not offline telemetry accuracy; simulated expert distinction |
| 10. Limitations and next step | 1:00 | No independent telemetry test; collect new sessions/drivers/circuits, freeze a protocol, then evaluate once |

## Visuals to use

- EDA: `output/rubric_analysis/figures/eda_distributions.png`, `eda_correlation_heatmap.png`, `eda_speed_controls.png`
- Models/errors: `model_comparison.png`, `brake_confusion_matrix.png`, `segment_error_analysis.png`, `throttle_oof_residuals.png`
- Simulator: `frozen_simulator_results.png`

Keep the limitation visible on results slides: only two training laps support the grouped model comparison. Do not describe lap 14 as untouched. Do not claim that the full-action simulator policy was trained from real steering labels. Rehearse out loud and time the presentation before submission; the prior slide deck rehearsal is not verified by these additions.
