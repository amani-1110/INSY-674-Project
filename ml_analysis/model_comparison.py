"""Fold-wise model comparison with train-lap-only telemetry alignment."""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR
from threadpoolctl import threadpool_limits

from ml_analysis.preprocessing import (
    ROOT,
    clip_throttle_predictions,
    fold_dataset,
    group_splits,
    load_raw_laps,
    make_preprocessing_pipeline,
)

SEED = 674
OUTPUT = ROOT / "output" / "rubric_analysis"


def _families():
    throttle = {
        "Dummy mean": (DummyRegressor(strategy="mean"), False),
        "Ridge": (Ridge(alpha=1.0), True),
        "Random Forest": (RandomForestRegressor(n_estimators=160, min_samples_leaf=4, random_state=SEED, n_jobs=1), False),
        "HistGradientBoosting": (HistGradientBoostingRegressor(max_iter=100, random_state=SEED), False),
        "SVR": (SVR(C=5.0, epsilon=0.05), True),
        "MLP": (MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=500, early_stopping=True, random_state=SEED), True),
    }
    brake = {
        "Dummy most-frequent": (DummyClassifier(strategy="most_frequent"), False),
        "Logistic Regression": (LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED), True),
        "Random Forest": (RandomForestClassifier(n_estimators=160, min_samples_leaf=4, class_weight="balanced", random_state=SEED, n_jobs=1), False),
        "Gradient Boosting": (GradientBoostingClassifier(n_estimators=100, random_state=SEED), False),
        "SVC": (SVC(C=1.0, class_weight="balanced"), True),
        "MLP": (MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=500, early_stopping=True, random_state=SEED), True),
    }
    return throttle, brake


def _cv_folds(raw_laps):
    lap_ids = np.asarray(sorted(raw_laps))
    splits = group_splits(lap_ids, n_splits=2)
    folds = []
    for fold_number, (train_idx, validation_idx) in enumerate(splits, 1):
        train_laps = tuple(int(v) for v in lap_ids[train_idx])
        validation_laps = tuple(int(v) for v in lap_ids[validation_idx])
        folds.append(
            fold_dataset(raw_laps, train_laps, validation_laps)
        )
        folds[-1]["fold"] = fold_number
    return folds


def _regression_metrics(y_true, prediction):
    return {
        "mae": float(mean_absolute_error(y_true, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, prediction))),
        "r2": float(r2_score(y_true, prediction)),
    }


def _classification_metrics(y_true, prediction):
    return {
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "recall": float(recall_score(y_true, prediction, zero_division=0)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
    }


def _save_comparison_plot(regression, classification, target: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    axes[0].bar(regression["model"], regression["mae_mean"], yerr=regression["mae_std"], capsize=3)
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].set(ylabel="MAE (0-1 throttle scale)", title="Throttle CV MAE (fold mean +/- SD)")
    axes[1].bar(classification["model"], classification["balanced_accuracy_mean"],
                yerr=classification["balanced_accuracy_std"], capsize=3)
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].set(ylabel="Balanced accuracy", ylim=(0, 1), title="Brake CV balanced accuracy (fold mean +/- SD)")
    fig.savefig(target, dpi=160, bbox_inches="tight")
    plt.close(fig)


def run_model_comparison(output_dir: Path | None = None) -> dict:
    """Run two-fold lap-grouped CV on laps 5 and 11; lap 14 is not a test set."""
    out = Path(output_dir or OUTPUT)
    out.mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    folds = _cv_folds(load_raw_laps((5, 11)))
    throttle_models, brake_models = _families()
    throttle_rows, brake_rows = [], []
    throttle_predictions, brake_predictions, importance_rows = [], [], []

    with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        for fold in folds:
            x_train, x_val = fold["X_train"], fold["X_validation"]
            y_train, y_val = fold["y_train"], fold["y_validation"]
            fold_number = fold["fold"]
            for name, (base, scale) in throttle_models.items():
                model = make_preprocessing_pipeline(base, scale=scale)
                started = time.perf_counter()
                model.fit(x_train, y_train[:, 0])
                elapsed = time.perf_counter() - started
                prediction = clip_throttle_predictions(model.predict(x_val))
                metrics = _regression_metrics(y_val[:, 0], prediction)
                throttle_rows.append(
                    {"model": name, "fold": fold_number, "validation_laps": ",".join(map(str, np.unique(fold["groups_validation"]))),
                     "fit_seconds": elapsed, **metrics}
                )
                if name == "Random Forest":
                    estimator = model.named_steps["estimator"]
                    importance_rows.extend(
                        {"task": "throttle", "fold": fold_number, "feature": feature, "importance": float(value)}
                        for feature, value in zip(fold["feature_names"], estimator.feature_importances_)
                    )
                if name in ("Dummy mean", "Random Forest", "SVR"):
                    throttle_predictions.extend(
                        {"model": name, "fold": fold_number, "lap_id": int(group), "throttle_true": float(actual),
                         "throttle_predicted": float(pred)}
                        for group, actual, pred in zip(fold["groups_validation"], y_val[:, 0], prediction)
                    )

            for name, (base, scale) in brake_models.items():
                model = make_preprocessing_pipeline(base, scale=scale)
                started = time.perf_counter()
                model.fit(x_train, y_train[:, 1].astype(int))
                elapsed = time.perf_counter() - started
                prediction = model.predict(x_val).astype(int)
                metrics = _classification_metrics(y_val[:, 1].astype(int), prediction)
                brake_rows.append(
                    {"model": name, "fold": fold_number, "validation_laps": ",".join(map(str, np.unique(fold["groups_validation"]))),
                     "fit_seconds": elapsed, **metrics}
                )
                if name == "Random Forest":
                    estimator = model.named_steps["estimator"]
                    importance_rows.extend(
                        {"task": "brake", "fold": fold_number, "feature": feature, "importance": float(value)}
                        for feature, value in zip(fold["feature_names"], estimator.feature_importances_)
                    )
                if name in ("Dummy most-frequent", "Random Forest"):
                    brake_predictions.extend(
                        {"model": name, "fold": fold_number, "lap_id": int(group), "brake_true": int(actual),
                         "brake_predicted": int(pred)}
                        for group, actual, pred in zip(fold["groups_validation"], y_val[:, 1], prediction)
                    )

    def summarize(rows, metric_names):
        data = pd.DataFrame(rows)
        summary = data.groupby("model", sort=False)[metric_names].agg(["mean", "std"])
        summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
        summary = summary.reset_index()
        return data, summary

    regression_folds, regression_summary = summarize(throttle_rows, ["mae", "rmse", "r2"])
    brake_folds, brake_summary = summarize(brake_rows, ["precision", "recall", "f1", "balanced_accuracy"])
    regression_folds.to_csv(out / "throttle_cv_folds.csv", index=False)
    regression_summary.to_csv(out / "throttle_cv_summary.csv", index=False)
    brake_folds.to_csv(out / "brake_cv_folds.csv", index=False)
    brake_summary.to_csv(out / "brake_cv_summary.csv", index=False)
    pd.DataFrame(throttle_predictions).to_csv(out / "throttle_oof_predictions.csv", index=False)
    pd.DataFrame(brake_predictions).to_csv(out / "brake_oof_predictions.csv", index=False)
    pd.DataFrame(importance_rows).to_csv(out / "random_forest_feature_importance.csv", index=False)
    _save_comparison_plot(regression_summary, brake_summary, out / "figures" / "model_comparison.png")

    # Confusion matrix is accumulated on out-of-fold predictions from the RF classifier.
    # Accumulate out-of-fold predictions from the Random Forest classifier.
    cm = [[0, 0], [0, 0]]
    segment_rows = []
    for fold in folds:
        model = make_preprocessing_pipeline(brake_models["Random Forest"][0], scale=False)
        model.fit(fold["X_train"], fold["y_train"][:, 1].astype(int))
        brake_prediction = model.predict(fold["X_validation"]).astype(int)
        y_val = fold["y_validation"]
        fold_cm = confusion_matrix(y_val[:, 1].astype(int), brake_prediction, labels=[0, 1])
        cm = (np.asarray(cm) + fold_cm).tolist()

        reg = make_preprocessing_pipeline(throttle_models["Random Forest"][0], scale=False)
        reg.fit(fold["X_train"], fold["y_train"][:, 0])
        throttle_prediction = clip_throttle_predictions(reg.predict(fold["X_validation"]))
        current_curvature = np.abs(fold["X_train"][:, 3])
        straight_cut = float(np.quantile(current_curvature, 0.25))
        corner_cut = float(np.quantile(current_curvature, 0.75))
        val_curvature = np.abs(fold["X_validation"][:, 3])
        masks = {
            "braking_zone": y_val[:, 1].astype(int) == 1,
            "straight": val_curvature <= straight_cut,
            "high_curvature": val_curvature >= corner_cut,
        }
        for segment, mask in masks.items():
            if mask.any():
                segment_rows.append(
                    {"fold": fold["fold"], "validation_laps": ",".join(map(str, np.unique(fold["groups_validation"]))),
                     "segment": segment, "samples": int(mask.sum()),
                     "throttle_mae": float(mean_absolute_error(y_val[mask, 0], throttle_prediction[mask])),
                     "brake_f1": float(f1_score(y_val[mask, 1].astype(int), brake_prediction[mask], zero_division=0))}
                )
    pd.DataFrame(segment_rows).to_csv(out / "error_analysis_by_segment.csv", index=False)
    _write_cm = {"labels": ["brake off", "brake on"], "matrix": cm}
    (out / "brake_confusion_matrix.json").write_text(json.dumps(_write_cm, indent=2) + "\n", encoding="utf-8")
    fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
    image = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["Predicted off", "Predicted on"])
    ax.set_yticks([0, 1], ["Actual off", "Actual on"])
    for row in range(2):
        for col in range(2):
            ax.text(col, row, str(cm[row][col]), ha="center", va="center")
    fig.colorbar(image, ax=ax, label="Out-of-fold observations")
    ax.set_title("Random Forest brake classification")
    fig.savefig(out / "figures" / "brake_confusion_matrix.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    result = {
        "training_laps": [5, 11],
        "independent_training_groups": 2,
        "folds": 2,
        "validation_protocol": "GroupKFold by complete lap; each fold refits the coordinate transform using training-lap positions only",
        "prior_calibration caveat": "The original saved aligned outputs used lap 5 for a shared transform; this experiment recomputes transforms inside each fold.",
        "lap_14": "Excluded: repeatedly inspected during development; not an untouched telemetry test",
        "throttle": regression_summary.to_dict(orient="records"),
        "brake": brake_summary.to_dict(orient="records"),
        "random_forest_brake_confusion_matrix": cm,
        "convergence_warnings": sum(issubclass(w.category, ConvergenceWarning) for w in caught),
        "screening": "Model comparison is exploratory due to only two independent training laps; fold dispersion is not a population confidence interval.",
    }
    (out / "model_comparison.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result


def _write_summary(out: Path, result: dict) -> None:
    (out / "model_comparison_summary.md").write_text(
        "# Model comparison interpretation\n\n"
        "Throttle is treated as bounded regression and brake as binary classification. "
        "Whole laps are held out; scaling/imputation and coordinate alignment are fitted using training-lap data only. "
        "The two-fold variability describes only these two laps. Lap 14 is not an untouched test set.\n\n"
        "Compare mean and fold standard deviation in `throttle_cv_summary.csv` and `brake_cv_summary.csv`. "
        "For braking, compare recall/F1/balanced accuracy with the most-frequent baseline; accuracy alone is misleading. "
        "See `error_analysis_by_segment.csv` for braking-zone, straight, and high-curvature errors. "
        "Random Forest impurity importances are descriptive and not causal; correlated curvature inputs can share importance.\n",
        encoding="utf-8",
    )


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    result = run_model_comparison()
    _write_summary(OUTPUT, result)
    print(json.dumps(result, indent=2))
