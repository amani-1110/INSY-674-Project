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

- `python -m ml_analysis.run_all` — completed all five supplemental analyses; outputs were regenerated under `output/rubric_analysis/`.
- `python -m ml_analysis.model_comparison` — completed successfully.
- `python -m pytest -q ml_analysis/test_workflow.py` — **6 passed**.
- `python -m pytest -q` — **116 passed, 2 failed**. The remaining failures are `tests/test_track_geometry.py::test_source_provenance_hash` and `tests/test_track_geometry.py::test_silverstone_source_geometry_and_periodicity`, which require exact source CSV byte hashes. The checked-out Monza and Silverstone files have CRLF line endings; their actual SHA-256 values differ from the hashes in the adjacent provenance JSON. Repairing those source files or their tests would require changes outside the permitted folders.
- Running the full suite with those two hash checks deselected — **116 passed**.
- Loading saved joblib models succeeds, but scikit-learn warns that they were serialized with 1.8.0 and are being loaded with 1.9.1 in this environment. The model-loading test passes; use the training scikit-learn version for reproducible model behavior.
