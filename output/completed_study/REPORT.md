# Completed model study

This follow-up completes the tuning, explicit ensemble comparison and fresh third-circuit simulator evaluation that were missing from the earlier analysis.

Only Monza laps 5 and 11 select settings. The two-fold selection scores are optimistic because they are reused for tuning; Spa is the separate frozen-model driving test.

| Candidate | Throttle MAE | Brake MAE | Combined selection MAE |
|---|---:|---:|---:|
| Random Forest | 0.0452 | 0.0624 | 0.0538 |
| Support vector | 0.0511 | 0.0689 | 0.0600 |
| Neural network | 0.0952 | 0.0997 | 0.0975 |
| Equal RF and SVR ensemble | 0.0457 | 0.0632 | 0.0545 |

Selected before Spa: **Random Forest**. Equal ensemble weights were fixed at 0.5/0.5. No settings changed after final testing.

| Spa driver | Elapsed (s) | Departures | Valid completed lap |
|---|---:|---:|---|
| Planner only | 287.4 | 0 | True |
| Random Forest | 282.0 | 0 | True |
| Support vector | 281.9 | 0 | True |
| Neural network | 276.8 | 0 | True |
| Equal RF and SVR ensemble | 281.2 | 0 | True |
| Frozen full-action driver | 294.3 | 0 | True |

The tuned pedal candidates retain the same planner and steering rules. The frozen full-action driver is evaluated separately. All predefined results are included; final-test results do not select a replacement model.

This is one deterministic simulator condition. It is not independent real-telemetry prediction accuracy or evidence of statistical significance. More sessions and conditions remain useful future extensions.

Evidence: tuning.json (all 18 settings and fold scores), comparison.csv, final_test_protocol.json (recorded before fetching Spa), final_test.json (all runs and unchanged model hashes), data/spa.source.json (pinned geometry provenance).

Implementation: complete_model_study.py. Preserve this initial test record when reproducing the study.
