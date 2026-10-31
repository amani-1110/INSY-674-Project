"""Seeded, group-aware searches using fold-specific coordinate calibration."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.metrics import f1_score, mean_absolute_error
from sklearn.model_selection import ParameterSampler
from threadpoolctl import threadpool_limits

from ml_analysis.model_comparison import _cv_folds
from ml_analysis.preprocessing import ROOT, clip_throttle_predictions, load_raw_laps, make_preprocessing_pipeline

OUTPUT = ROOT / "output" / "rubric_analysis"
SEED = 674


def _search(estimator, parameter_space, folds, *, target_index: int, classification: bool, n_iter: int):
    results = []
    candidates = list(ParameterSampler(parameter_space, n_iter=n_iter, random_state=SEED))
    with threadpool_limits(limits=1):
        for candidate_number, parameters in enumerate(candidates, 1):
            scores = []
            started = time.perf_counter()
            for fold in folds:
                model = make_preprocessing_pipeline(estimator, scale=False)
                model.set_params(**parameters)
                y_train = fold["y_train"][:, target_index]
                y_validation = fold["y_validation"][:, target_index]
                if classification:
                    y_train, y_validation = y_train.astype(int), y_validation.astype(int)
                model.fit(fold["X_train"], y_train)
                prediction = model.predict(fold["X_validation"])
                if classification:
                    scores.append(float(f1_score(y_validation, prediction, zero_division=0)))
                else:
                    scores.append(float(mean_absolute_error(y_validation, clip_throttle_predictions(prediction))))
            score_mean = float(np.mean(scores))
            results.append(
                {
                    "candidate": candidate_number,
                    "parameters": parameters,
                    "fold_scores": scores,
                    "mean_score": score_mean,
                    "std_score": float(np.std(scores, ddof=1)),
                    "elapsed_seconds": float(time.perf_counter() - started),
                }
            )
    maximize = classification
    best = (max if maximize else min)(results, key=lambda row: row["mean_score"])
    return results, best


def _reference_score(estimator, folds, *, target_index: int, classification: bool):
    scores = []
    elapsed = 0.0
    with threadpool_limits(limits=1):
        for fold in folds:
            model = make_preprocessing_pipeline(estimator, scale=False)
            y_train = fold["y_train"][:, target_index]
            y_validation = fold["y_validation"][:, target_index]
            if classification:
                y_train, y_validation = y_train.astype(int), y_validation.astype(int)
            started = time.perf_counter()
            model.fit(fold["X_train"], y_train)
            elapsed += time.perf_counter() - started
            prediction = model.predict(fold["X_validation"])
            if classification:
                scores.append(float(f1_score(y_validation, prediction, zero_division=0)))
            else:
                scores.append(float(mean_absolute_error(y_validation, clip_throttle_predictions(prediction))))
    return {
        "fold_scores": scores,
        "mean_score": float(np.mean(scores)),
        "std_score": float(np.std(scores, ddof=1)),
        "elapsed_seconds": float(elapsed),
    }


def run_tuning(output_dir: Path | None = None, *, n_iter: int = 6) -> dict:
    out = Path(output_dir or OUTPUT)
    out.mkdir(parents=True, exist_ok=True)
    folds = _cv_folds(load_raw_laps((5, 11)))

    throttle_reference_model = HistGradientBoostingRegressor(max_iter=100, random_state=SEED)
    brake_reference_model = RandomForestClassifier(
        n_estimators=160, min_samples_leaf=4, class_weight="balanced", random_state=SEED, n_jobs=1
    )
    throttle_reference = _reference_score(
        throttle_reference_model, folds, target_index=0, classification=False
    )
    brake_reference = _reference_score(
        brake_reference_model, folds, target_index=1, classification=True
    )
    throttle_trials, throttle_best = _search(
        throttle_reference_model,
        {
            "estimator__max_iter": [60, 100, 160],
            "estimator__learning_rate": [0.05, 0.1, 0.15],
            "estimator__max_leaf_nodes": [15, 31, 45],
            "estimator__min_samples_leaf": [10, 20, 30],
            "estimator__l2_regularization": [0.0, 0.1, 1.0],
        },
        folds,
        target_index=0,
        classification=False,
        n_iter=n_iter,
    )
    brake_trials, brake_best = _search(
        RandomForestClassifier(n_estimators=120, class_weight="balanced", random_state=SEED, n_jobs=1),
        {
            "estimator__n_estimators": [100, 160, 240],
            "estimator__max_depth": [8, 12, 16, None],
            "estimator__min_samples_leaf": [2, 4, 8],
            "estimator__max_features": ["sqrt", 0.75],
        },
        folds,
        target_index=1,
        classification=True,
        n_iter=n_iter,
    )
    result = {
        "method": "Seeded randomized candidate sampling; manually scored on fold-specific GroupKFold fits",
        "seed": SEED,
        "n_iter_per_target": n_iter,
        "training_laps": [5, 11],
        "folds": 2,
        "selection_metrics": {"throttle": "minimum mean MAE", "brake": "maximum mean positive-class F1"},
        "untuned_baselines": {
            "throttle_hist_gradient_boosting": throttle_reference,
            "brake_random_forest": brake_reference,
        },
        "final_test_used": False,
        "candidates": {
            "throttle_hist_gradient_boosting": throttle_trials,
            "brake_random_forest": brake_trials,
        },
        "best": {"throttle": throttle_best, "brake": brake_best},
        "tuned_vs_untuned": {
            "throttle_mae_reduction": float(throttle_reference["mean_score"] - throttle_best["mean_score"]),
            "brake_f1_change": float(brake_best["mean_score"] - brake_reference["mean_score"]),
        },
        "limitation": "Two independent laps give only two validation folds; tuned scores are model-selection estimates, not an independent test result.",
    }
    (out / "hyperparameter_tuning.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    flat = []
    for task, trials in (("throttle", throttle_trials), ("brake", brake_trials)):
        for trial in trials:
            flat.append({"task": task, **trial, "parameters": json.dumps(trial["parameters"], sort_keys=True),
                         "fold_scores": json.dumps(trial["fold_scores"])})
    pd.DataFrame(flat).to_csv(out / "hyperparameter_trials.csv", index=False)
    (out / "hyperparameter_tuning_summary.md").write_text(
        "# Hyperparameter tuning\n\n"
        "A seeded randomized candidate search was used instead of `RandomizedSearchCV`: fold-specific alignment "
        "must be refit from the training lap, so each candidate is scored manually against the same two GroupKFold "
        "partitions. HistGradientBoosting throttle minimizes MAE; binary brake Random Forest maximizes positive-class F1. "
        "No final test data was used. The small number of independent groups means these scores are exploratory.\n\n"
        f"- HistGradientBoosting throttle baseline: MAE {throttle_reference['mean_score']:.4f} "
        f"(fold SD {throttle_reference['std_score']:.4f}); tuned {throttle_best['mean_score']:.4f} "
        f"(fold SD {throttle_best['std_score']:.4f}).\n"
        f"- Random Forest brake baseline: F1 {brake_reference['mean_score']:.4f} "
        f"(fold SD {brake_reference['std_score']:.4f}); tuned {brake_best['mean_score']:.4f} "
        f"(fold SD {brake_best['std_score']:.4f}).\n",
        encoding="utf-8",
    )
    return result


if __name__ == "__main__":
    print(json.dumps(run_tuning(), indent=2))
