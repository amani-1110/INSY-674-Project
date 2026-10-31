# Model comparison interpretation

Throttle is treated as bounded regression and brake as binary classification. Whole laps are held out; scaling/imputation and coordinate alignment are fitted using training-lap data only. The two-fold variability describes only these two laps. Lap 14 is not an untouched test set.

Compare mean and fold standard deviation in `throttle_cv_summary.csv` and `brake_cv_summary.csv`. For braking, compare recall/F1/balanced accuracy with the most-frequent baseline; accuracy alone is misleading. See `error_analysis_by_segment.csv` for braking-zone, straight, and high-curvature errors. Random Forest impurity importances are descriptive and not causal; correlated curvature inputs can share importance.
