# Final evaluation status

**No independent real-telemetry final test is available.** The two training laps (5 and 11) were used for model selection, and lap 14 has been repeatedly inspected. It is not reported as an untouched test. The supplied frozen Spa result is a closed-loop simulator evaluation, not real-telemetry prediction accuracy. The procedure can be rerun on a future untouched session by extending the source-data manifest and freezing the selected artifacts before evaluation.

Group-CV predictions and residual plots are explicitly labeled exploratory. See `simulator_frozen_evaluation.csv` and `figures/frozen_simulator_results.png` for the historical simulator record; the original benchmark JSON and model hashes are preserved.
