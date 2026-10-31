# INSY 674 rubric compliance map

This evidence map distinguishes existing project work from the added real-telemetry workflow. “Partially satisfied” reflects substantive evidence with a material data limitation; no claim of complete Excellent-level compliance is made.

## Source-code and experiment criteria

| Criterion | Status | Evidence |
|---|---|---|
| Frame the problem, users, objective and learning task | Fully satisfied in repository evidence | [DOCUMENTATION.md](../DOCUMENTATION.md), [PROJECT_EVIDENCE.md](../PROJECT_EVIDENCE.md), [INSY674_ML_REPORT.md](INSY674_ML_REPORT.md) |
| Explore attributes, missingness, distributions, relationships and features | Fully satisfied for available data, with quality caveats | [ml_analysis/eda.py](../ml_analysis/eda.py), `output/rubric_analysis/eda_attribute_profile.csv`, `eda_integrity.csv`, `eda_sampling_intervals.csv`, `eda_correlations.csv`, and labeled `figures/eda_*.png` |
| Prepare data and avoid temporal/group leakage | Partially satisfied | [ml_analysis/preprocessing.py](../ml_analysis/preprocessing.py), [notebooks/02_Data_Preparation.ipynb](../notebooks/02_Data_Preparation.ipynb); lap-wise CV and fold-fitted transforms are implemented, but only two independent training laps exist |
| Compare multiple appropriate model families | Fully satisfied as exploratory screening | [ml_analysis/model_comparison.py](../ml_analysis/model_comparison.py), throttle/brake fold and summary CSVs, confusion matrix, feature importance and segment error tables |
| Report N-fold average/dispersion and suitable metrics | Partially satisfied | Two-fold grouped metrics are reported; two folds cannot characterize population variability reliably |
| Tune selected candidates without final-test use | Partially satisfied | [ml_analysis/hyperparameter_tuning.py](../ml_analysis/hyperparameter_tuning.py), `hyperparameter_trials.csv`; only six candidates per task and two groups |
| Evaluate ensembles and compare with individual candidates | Partially satisfied | [ml_analysis/ensembles.py](../ml_analysis/ensembles.py), `ensemble_cv_summary.csv`; regression blend is modestly better, classification ensemble is worse; stacking omitted for insufficient groups |
| Final held-out offline telemetry evaluation | **Still missing** | `output/rubric_analysis/final_evaluation.json` explicitly records no valid independent telemetry test. Lap 14 is not untouched |
| Distinguish offline results from closed-loop driving | Fully satisfied | [ml_analysis/final_evaluation.py](../ml_analysis/final_evaluation.py), preserved `output/completed_study/final_test.json` and `output/all_configurations/REPORT.md` |
| Reproducibility, integrity/split/artifact tests | Fully satisfied for supplemental workflow | [REPRODUCIBILITY.md](REPRODUCIBILITY.md), [ml_analysis/test_workflow.py](../ml_analysis/test_workflow.py) |
| Explain assumptions, failures, limitations and next steps | Fully satisfied | [INSY674_ML_REPORT.md](INSY674_ML_REPORT.md), EDA interpretations and experiment summaries |

## Presentation criteria

| Criterion | Status | Evidence |
|---|---|---|
| Coherent 9–11 slide / 10-minute story | Proposed | [PRESENTATION_GUIDE.md](PRESENTATION_GUIDE.md) |
| Presentation-ready labeled EDA/model/error figures | Fully satisfied as source assets | `output/rubric_analysis/figures/` |
| Final independent telemetry result suitable for presentation | **Still missing** | No untouched session exists |
| Existing slides fit a live 10-minute delivery | Partially satisfied / rehearsal needed | Existing 13-slide PDF is preserved; guide recommends a 10-slide talk and live timing/rehearsal |

## Overall conclusion

The repository now has a reproducible, additive workflow addressing the requested experiments and communication evidence. It does **not** justify claiming a fully generalizable final telemetry model: two training laps, one driver/session/circuit, a previously inspected third lap, and no independent real-telemetry test remain unresolved.
