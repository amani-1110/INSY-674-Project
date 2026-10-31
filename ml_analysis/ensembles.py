"""Compare small voting ensembles to their individual models on identical lap folds."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    VotingClassifier,
    VotingRegressor,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    recall_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from threadpoolctl import threadpool_limits

from ml_analysis.model_comparison import _cv_folds
from ml_analysis.preprocessing import ROOT, clip_throttle_predictions, load_raw_laps, make_preprocessing_pipeline

OUTPUT = ROOT / "output" / "rubric_analysis"
SEED = 674


def run_ensembles(output_dir: Path | None = None) -> dict:
    out = Path(output_dir or OUTPUT)
    out.mkdir(parents=True, exist_ok=True)
    folds = _cv_folds(load_raw_laps((5, 11)))
    regression_models = {
        "Random Forest": RandomForestRegressor(n_estimators=160, min_samples_leaf=4, random_state=SEED, n_jobs=1),
        "SVR": make_pipeline(StandardScaler(), SVR(C=5.0, epsilon=0.05)),
        "Equal RF-SVR VotingRegressor": VotingRegressor(
            [
                ("rf", RandomForestRegressor(n_estimators=160, min_samples_leaf=4, random_state=SEED, n_jobs=1)),
                ("svr", make_pipeline(StandardScaler(), SVR(C=5.0, epsilon=0.05))),
            ],
            weights=[1, 1],
        ),
    }
    classification_models = {
        "Random Forest": RandomForestClassifier(n_estimators=160, min_samples_leaf=4, class_weight="balanced", random_state=SEED, n_jobs=1),
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED)),
        "Soft RF-Logistic VotingClassifier": VotingClassifier(
            [
                ("rf", RandomForestClassifier(n_estimators=160, min_samples_leaf=4, class_weight="balanced", random_state=SEED, n_jobs=1)),
                ("logistic", make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED))),
            ],
            voting="soft",
            weights=[1, 1],
        ),
    }
    rows = []
    with threadpool_limits(limits=1):
        for fold in folds:
            for name, estimator in regression_models.items():
                model = make_preprocessing_pipeline(estimator, scale=False)
                model.fit(fold["X_train"], fold["y_train"][:, 0])
                prediction = clip_throttle_predictions(model.predict(fold["X_validation"]))
                rows.append(
                    {"task": "throttle", "model": name, "fold": fold["fold"],
                     "mae": float(mean_absolute_error(fold["y_validation"][:, 0], prediction))}
                )
            for name, estimator in classification_models.items():
                model = make_preprocessing_pipeline(estimator, scale=False)
                y_train = fold["y_train"][:, 1].astype(int)
                y_true = fold["y_validation"][:, 1].astype(int)
                model.fit(fold["X_train"], y_train)
                prediction = model.predict(fold["X_validation"]).astype(int)
                rows.append(
                    {"task": "brake", "model": name, "fold": fold["fold"],
                     "precision": float(precision_score(y_true, prediction, zero_division=0)),
                     "recall": float(recall_score(y_true, prediction, zero_division=0)),
                     "f1": float(f1_score(y_true, prediction, zero_division=0)),
                     "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction))}
                )
    results = pd.DataFrame(rows)
    results.to_csv(out / "ensemble_cv_folds.csv", index=False)
    metric_names = [name for name in ("mae", "precision", "recall", "f1", "balanced_accuracy") if name in results]
    summary = results.groupby(["task", "model"], sort=False)[metric_names].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(out / "ensemble_cv_summary.csv", index=False)

    report_rows = [
        {key: (None if pd.isna(value) else value) for key, value in row.items()}
        for row in summary.to_dict(orient="records")
    ]
    report = {
        "folds": 2,
        "method": "Same fold-specific alignment, GroupKFold laps 5 and 11",
        "ensembles": ["Equal RF-SVR VotingRegressor", "Soft RF-Logistic VotingClassifier"],
        "stacking": "Not attempted: only two independent training groups are insufficient for a credible leakage-safe meta-learner.",
        "summary": report_rows,
        "interpretation": "Compare each ensemble with its individual constituent models; no ensemble is presumed superior.",
    }
    (out / "ensemble_comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (out / "ensemble_interpretation.md").write_text(
        "# Ensemble interpretation\n\n"
        "The equal-weight RF/SVR regressor and soft-voting RF/logistic brake classifier are compared against "
        "their component estimators using the same two whole-lap folds. Stacking was deliberately excluded: "
        "two independent laps cannot support a reliable out-of-fold meta-training layer. Read the measured "
        "mean and fold SD from `ensemble_cv_summary.csv`; an ensemble that does not beat its constituent "
        "models is reported as non-improving rather than selected by assumption.\n",
        encoding="utf-8",
    )
    return report


if __name__ == "__main__":
    print(json.dumps(run_ensembles(), indent=2))
