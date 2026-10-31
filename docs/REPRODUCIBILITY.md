# Reproducibility and execution

Run commands from the repository root with the project's declared dependencies (`requirements.txt` already includes NumPy, pandas, scikit-learn, matplotlib, SciPy, joblib, pytest and simulator dependencies). No existing dependency manifest was changed.

```powershell
# EDA tables, interpretations and figures
python -m ml_analysis.eda

# Fold-aware GroupKFold comparison (throttle regression + brake classification)
python -m ml_analysis.model_comparison

# Seeded hyperparameter search (six candidates per target by default)
python -m ml_analysis.hyperparameter_tuning

# Voting ensemble comparison
python -m ml_analysis.ensembles

# Final evaluation status and plots (does not invent a telemetry test)
python -m ml_analysis.final_evaluation

# Run all supplemental experiments in sequence
python -m ml_analysis.run_all

# Supplemental tests and complete pre-existing suite
python -m pytest -q ml_analysis/test_workflow.py
python -m pytest -q
```

All additions read the saved FastF1 telemetry and existing simulator artifacts. New tables, JSON summaries and figures are written under `output/rubric_analysis/`; existing model artifacts and historical benchmark records are not overwritten.

Random seeds are 674 for new model/search routines. The model comparison and tuning use two folds by complete lap (laps 5 and 11). Each fold estimates the coordinate transform from training-lap positions and applies it to the validation lap. Pipeline imputers/scalers are fit within training folds. Lap 14 is intentionally excluded from final scoring because it has been inspected repeatedly.

The six Jupyter notebooks in `notebooks/` are executable entry points to the same functions. Python 3.10+ and a Jupyter kernel are required to run them interactively.

The existing full-action controller was trained from simulated expert demonstrations. It is separate from the real-telemetry throttle/brake experiments. The saved Spa test evaluates closed-loop simulated driving only.

## Validation record

- Supplemental tests: `python -m pytest -q ml_analysis/test_workflow.py` — **6 passed**.
- Full suite: **110 passed, 2 failed** when run with a process-local import alias for the existing `Alignment/` casing. The two failures are `test_track.py::test_source_provenance_hash` and `test_track.py::test_silverstone_source_geometry_and_periodicity`; both assert exact CSV byte hashes that differ under this Windows CRLF checkout. Existing project documentation already notes the line-ending/hash sensitivity. These source CSVs were not modified to avoid changing teammates' files.
- A direct full-suite collection initially also required the repository's declared FastF1 dependency and an import-casing alias for the existing `Alignment/` directory; neither was changed in the repository.
- Re-running the full suite with just those two hash-sensitive cases deselected gives **110 passed**. Loading the preserved trained artifact emits scikit-learn `InconsistentVersionWarning` because it was serialized with 1.8.0 while this environment uses 1.6.1; the artifact load/prediction test passes, but production reuse should pin the model's training version.
