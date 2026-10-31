# Hyperparameter tuning

A seeded randomized candidate search was used instead of `RandomizedSearchCV`: fold-specific alignment must be refit from the training lap, so each candidate is scored manually against the same two GroupKFold partitions. HistGradientBoosting throttle minimizes MAE; binary brake Random Forest maximizes positive-class F1. No final test data was used. The small number of independent groups means these scores are exploratory.

- HistGradientBoosting throttle baseline: MAE 0.0389 (fold SD 0.0023); tuned 0.0374 (fold SD 0.0009).
- Random Forest brake baseline: F1 0.8646 (fold SD 0.0079); tuned 0.8636 (fold SD 0.0092).
