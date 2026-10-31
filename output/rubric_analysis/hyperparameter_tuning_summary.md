# Hyperparameter tuning

A seeded randomized candidate search was used instead of `RandomizedSearchCV`: fold-specific alignment must be refit from the training lap, so each candidate is scored manually against the same two GroupKFold partitions. HistGradientBoosting throttle minimizes MAE; binary brake Random Forest maximizes positive-class F1. No final test data was used. The small number of independent groups means these scores are exploratory.

- HistGradientBoosting throttle baseline: MAE 0.0394 (fold SD 0.0001); tuned 0.0377 (fold SD 0.0002).
- Random Forest brake baseline: F1 0.8660 (fold SD 0.0226); tuned 0.8570 (fold SD 0.0276).
