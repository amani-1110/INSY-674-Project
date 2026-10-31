"""Exploratory analysis of the real, saved 2025 Monza qualifying telemetry."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ml_analysis.preprocessing import ROOT, TELEMETRY_DIR, raw_feature_matrix

RAW_COLUMNS = ("time", "x", "y", "speed", "throttle", "brake")
ALIGNED_COLUMNS = (
    "time", "s", "lateral_offset", "heading_error", "speed", "throttle", "brake", "lap_id"
)


def _read_sources():
    raw_frames = []
    aligned_frames = []
    for lap_number in (5, 11, 14):
        raw_path = TELEMETRY_DIR / f"2025_Monza_Q_VER_lap{lap_number:02d}.csv"
        aligned_path = ROOT / "output" / "pipeline_500m" / f"lap{lap_number:02d}_track_relative.csv"
        raw = pd.read_csv(raw_path)
        aligned = pd.read_csv(aligned_path)
        raw["lap_id"] = lap_number
        aligned["lap_number"] = lap_number
        raw_frames.append(raw)
        aligned_frames.append(aligned)
    return pd.concat(raw_frames, ignore_index=True), pd.concat(aligned_frames, ignore_index=True)


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _save(fig, target: Path) -> None:
    fig.savefig(target, dpi=160, bbox_inches="tight")
    plt.close(fig)


def run_eda(output_dir: Path | None = None) -> dict:
    """Write reusable EDA tables, interpreted figures, and a compact summary."""
    out = Path(output_dir or ROOT / "output" / "rubric_analysis")
    out.mkdir(parents=True, exist_ok=True)
    figure_dir = out / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    raw, aligned = _read_sources()

    manifest = json.loads((TELEMETRY_DIR / "manifest.json").read_text(encoding="utf-8"))
    raw_missing = raw.isna().sum()
    raw_profile = pd.DataFrame(
        {
            "attribute": raw.columns,
            "dtype": [str(raw[c].dtype) for c in raw.columns],
            "unit": ["s", "source-frame m", "source-frame m", "m/s", "fraction 0-1", "binary 0/1", "lap ID"],
            "role": ["timestamp/grouped", "source coordinate", "source coordinate", "feature", "regression target", "classification target", "split group"],
            "missing_count": [int(raw_missing[c]) for c in raw.columns],
            "missing_percent": [float(raw_missing[c] / len(raw) * 100) for c in raw.columns],
            "unique_values": [int(raw[c].nunique()) for c in raw.columns],
            "minimum": [float(raw[c].min()) for c in raw.columns],
            "median": [float(raw[c].median()) for c in raw.columns],
            "maximum": [float(raw[c].max()) for c in raw.columns],
            "std": [float(raw[c].std()) for c in raw.columns],
        }
    )
    raw_profile.to_csv(out / "eda_attribute_profile.csv", index=False)
    raw.describe(include="all").T.to_csv(out / "eda_descriptive_statistics.csv")

    missing_rows = []
    duplicate_rows = []
    interval_rows = []
    outlier_rows = []
    for lap_id, frame in raw.groupby("lap_id", sort=True):
        deltas = frame["time"].diff().dropna()
        missing_rows.append(
            {
                "lap": int(lap_id),
                "rows": len(frame),
                "missing_cells": int(frame.isna().sum().sum()),
                "duplicate_records": int(frame.duplicated().sum()),
                "duplicate_timestamps": int(frame["time"].duplicated().sum()),
            }
        )
        duplicate_rows.append(int(frame.duplicated().sum()))
        interval_rows.append(
            {
                "lap": int(lap_id),
                "interval_count": len(deltas),
                "median_interval_s": float(deltas.median()),
                "mean_interval_s": float(deltas.mean()),
                "min_interval_s": float(deltas.min()),
                "max_interval_s": float(deltas.max()),
                "interval_cv": float(deltas.std() / deltas.mean()),
            }
        )
        for column in ("speed", "throttle", "x", "y"):
            q1, q3 = frame[column].quantile([0.25, 0.75])
            iqr = q3 - q1
            count = int(((frame[column] < q1 - 1.5 * iqr) | (frame[column] > q3 + 1.5 * iqr)).sum())
            outlier_rows.append(
                {"lap": int(lap_id), "attribute": column, "iqr_flagged_rows": count}
            )
    pd.DataFrame(missing_rows).to_csv(out / "eda_integrity.csv", index=False)
    pd.DataFrame(interval_rows).to_csv(out / "eda_sampling_intervals.csv", index=False)
    pd.DataFrame(outlier_rows).to_csv(out / "eda_iqr_outliers.csv", index=False)

    features, targets, names = raw_feature_matrix(aligned.to_dict("records"))
    model_data = pd.DataFrame(features, columns=names)
    model_data["throttle"] = targets[:, 0]
    model_data["brake"] = targets[:, 1].astype(int)
    model_data["lap_number"] = aligned["lap_number"].to_numpy()
    curvature_name = "curv_0"
    model_data.to_csv(out / "eda_modeling_table.csv", index=False)
    correlation_features = ["speed", "lateral_offset", "heading_error", curvature_name, "throttle", "brake"]
    model_data[correlation_features].corr().to_csv(out / "eda_correlations.csv")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes[0, 0].hist(raw["speed"] * 3.6, bins=30, color="#2864a5")
    axes[0, 0].set(xlabel="Speed (km/h)", ylabel="Samples", title="Speed distribution")
    axes[0, 1].hist(raw["throttle"], bins=24, color="#25966f")
    axes[0, 1].set(xlabel="Throttle (0-1)", ylabel="Samples", title="Throttle label distribution")
    axes[1, 0].boxplot([raw.loc[raw.brake == 0, "speed"] * 3.6, raw.loc[raw.brake == 1, "speed"] * 3.6],
                       tick_labels=["Brake off", "Brake on"])
    axes[1, 0].set(ylabel="Speed (km/h)", title="Speed by binary brake label")
    axes[1, 1].boxplot([aligned.loc[aligned.lap_number == lap, "lateral_offset"] for lap in (5, 11, 14)],
                       tick_labels=["Lap 5", "Lap 11", "Lap 14"])
    axes[1, 1].set(ylabel="Lateral offset (m; left positive)", title="Track-relative position by lap")
    _save(fig, figure_dir / "eda_distributions.png")

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    corr = model_data[correlation_features].corr()
    image = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(corr.columns)), corr.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(corr.index)), corr.index)
    for row in range(len(corr.index)):
        for column in range(len(corr.columns)):
            ax.text(column, row, f"{corr.iloc[row, column]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label="Pearson correlation")
    ax.set_title("Track-relative inputs and targets (descriptive; not causal)")
    _save(fig, figure_dir / "eda_correlation_heatmap.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    axes[0].scatter(raw["speed"] * 3.6, raw["throttle"], s=8, alpha=0.35)
    axes[0].set(xlabel="Speed (km/h)", ylabel="Throttle (0-1)", title="Speed versus throttle")
    axes[1].boxplot([raw.loc[raw.brake == label, "speed"] * 3.6 for label in (0, 1)],
                    tick_labels=["Brake off", "Brake on"])
    axes[1].set(ylabel="Speed (km/h)", title="Speed versus braking occurrence")
    _save(fig, figure_dir / "eda_speed_controls.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    axes[0].scatter(model_data[curvature_name], model_data["speed"] * 3.6, s=8, alpha=0.3)
    axes[0].set(xlabel="Current track curvature (1/m)", ylabel="Speed (km/h)", title="Curvature versus speed")
    axes[1].scatter(aligned["s"], aligned["throttle"], s=7, alpha=0.25, label="Throttle")
    brake_points = aligned.loc[aligned["brake"] == 1]
    axes[1].scatter(brake_points["s"], np.full(len(brake_points), 1.04), s=8, alpha=0.5,
                    color="#c13b34", label="Brake on")
    axes[1].set(xlabel="Distance from track start (m)", ylabel="Throttle / brake marker",
                ylim=(-0.05, 1.12), title="Pedals versus track position")
    axes[1].legend()
    _save(fig, figure_dir / "eda_curvature_and_position.png")

    lap = raw[raw["lap_id"] == 5]
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
    axes[0].plot(lap["time"], lap["speed"] * 3.6)
    axes[0].set(ylabel="Speed (km/h)", title="Telemetry over time: lap 5")
    axes[1].plot(lap["time"], lap["throttle"], color="#25966f")
    axes[1].set(ylabel="Throttle (0-1)")
    axes[2].step(lap["time"], lap["brake"], where="post", color="#c13b34")
    axes[2].set(xlabel="Lap-relative time (s)", ylabel="Brake (0/1)")
    _save(fig, figure_dir / "eda_lap5_timeseries.png")

    class_counts = raw["brake"].value_counts().sort_index()
    per_lap = raw.groupby("lap_id").agg(
        n=("time", "size"), speed_mean_mps=("speed", "mean"),
        throttle_mean=("throttle", "mean"), brake_fraction=("brake", "mean")
    )
    per_lap.to_csv(out / "eda_lap_comparison.csv")
    session_count = len(manifest["laps"])
    summary = {
        "source": "Saved FastF1 telemetry; 2025 Italian GP qualifying; VER; Monza",
        "session_count": 1,
        "lap_count": session_count,
        "rows": int(len(raw)),
        "raw_fields": list(RAW_COLUMNS),
        "brake_counts": {str(int(k)): int(v) for k, v in class_counts.items()},
        "brake_positive_fraction": float(raw["brake"].mean()),
        "unique_timestamps": int(raw.groupby("lap_id")["time"].nunique().sum()),
        "duplicate_records": int(sum(duplicate_rows)),
        "missing_values": int(raw.isna().sum().sum()),
        "max_native_gap_policy_s": float(manifest["max_native_gap_seconds"]),
        "sampling_interval_s": {
            "median": float(pd.DataFrame(interval_rows)["median_interval_s"].median()),
            "max_observed": float(pd.DataFrame(interval_rows)["max_interval_s"].max()),
        },
        "lap_group_ids": [5, 11, 14],
        "training_group_ids": [5, 11],
        "lap_14_role": "Repeatedly inspected during development; exploratory only, not an untouched test set",
        "units": {"time": "s", "coordinates": "m", "speed": "m/s", "throttle": "0-1", "brake": "binary 0/1"},
    }
    _write_json(out / "eda_summary.json", summary)

    interpretations = """# EDA figure interpretations

- `figures/eda_distributions.png`: Speed is measured from fast qualifying laps, not standing starts; the binary brake-label boxplot compares speed conditional on a brake event; per-lap lateral-offset spread reflects both racing-line changes and alignment uncertainty.
- `figures/eda_correlation_heatmap.png`: Correlations summarize the three development laps; future curvature samples are related geometry features, so correlation is not causal evidence or independent feature contribution.
- `figures/eda_speed_controls.png`: Throttle is bounded and frequently high. Brake is an occurrence flag, not pressure; a classifier should report positive-class recall/F1 and balanced accuracy alongside the majority baseline.
- `figures/eda_curvature_and_position.png`: The relationship between current bend and speed is descriptive; the track-position plot locates pedal behavior around the circuit, but the samples remain serially dependent.
- `figures/eda_lap5_timeseries.png`: Irregular time intervals and piecewise/interpolated signals are visible; this should discourage treating each telemetry row as an independent observation.

No observations are removed based only on IQR flags. The flag table is diagnostic because large speed/position values can be genuine parts of a lap. The raw export has no missing values, but its source metadata records FastF1 interpolation across native streams and gaps; zero saved missing cells do not mean every signal value was natively measured.
"""
    (out / "eda_interpretations.md").write_text(interpretations, encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(run_eda(), indent=2))
