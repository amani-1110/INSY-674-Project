# EDA figure interpretations

- `figures/eda_distributions.png`: Speed is measured from fast qualifying laps, not standing starts; the binary brake-label boxplot compares speed conditional on a brake event; per-lap lateral-offset spread reflects both racing-line changes and alignment uncertainty.
- `figures/eda_correlation_heatmap.png`: Correlations summarize the three development laps; future curvature samples are related geometry features, so correlation is not causal evidence or independent feature contribution.
- `figures/eda_speed_controls.png`: Throttle is bounded and frequently high. Brake is an occurrence flag, not pressure; a classifier should report positive-class recall/F1 and balanced accuracy alongside the majority baseline.
- `figures/eda_curvature_and_position.png`: The relationship between current bend and speed is descriptive; the track-position plot locates pedal behavior around the circuit, but the samples remain serially dependent.
- `figures/eda_lap5_timeseries.png`: Irregular time intervals and piecewise/interpolated signals are visible; this should discourage treating each telemetry row as an independent observation.

No observations are removed based only on IQR flags. The flag table is diagnostic because large speed/position values can be genuine parts of a lap. The raw export has no missing values, but its source metadata records FastF1 interpolation across native streams and gaps; zero saved missing cells do not mean every signal value was natively measured.
