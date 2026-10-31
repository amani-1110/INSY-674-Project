"""Fold-aware loading and feature preparation for the saved Monza telemetry."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import GroupKFold

from f1_simulator.core.imitation import (
    _targets,
    _to_matrix,
    feature_names,
    offsets_for_horizon,
    prepare_training_data,
)
from f1_simulator.core.track import Track

ROOT = Path(__file__).resolve().parents[1]
TELEMETRY_DIR = ROOT / "data" / "telemetry" / "monza_three_new_laps"
TRAINING_LAPS = (5, 11)
KNOWN_LAPS = (5, 11, 14)

from f1_simulator.alignment.alignment import (
    align_coordinates,
    convert_to_track_coordinates,
    estimate_transform,
    load_lap_csv,
)


def load_raw_laps(lap_numbers: Iterable[int] = KNOWN_LAPS) -> dict[int, list[dict]]:
    """Load verified CSVs without changing source files or resampling timestamps."""
    laps: dict[int, list[dict]] = {}
    for lap_number in lap_numbers:
        if lap_number not in KNOWN_LAPS:
            raise ValueError(f"Unknown saved lap: {lap_number}")
        path = TELEMETRY_DIR / f"2025_Monza_Q_VER_lap{lap_number:02d}.csv"
        laps[lap_number] = load_lap_csv(path)
    if not laps:
        raise ValueError("At least one telemetry lap is required")
    return laps


def group_splits(groups: Iterable[object], n_splits: int | None = None):
    """Create reproducible folds that keep each complete lap in one partition."""
    groups_array = np.asarray(list(groups))
    unique_groups = np.unique(groups_array)
    folds = len(unique_groups) if n_splits is None else n_splits
    if folds < 2 or folds > len(unique_groups):
        raise ValueError("GroupKFold needs between 2 and the number of unique groups")
    indices = np.arange(len(groups_array))
    return list(GroupKFold(n_splits=folds).split(indices, groups=groups_array))


def assert_disjoint_groups(train_groups: Iterable[object], validation_groups: Iterable[object]) -> None:
    overlap = set(train_groups) & set(validation_groups)
    if overlap:
        raise ValueError(f"Training and validation groups overlap: {sorted(overlap)}")


def clip_throttle_predictions(predictions) -> np.ndarray:
    """Keep regression predictions within the physical throttle label range."""
    values = np.asarray(predictions, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Throttle predictions must be finite")
    return np.clip(values, 0.0, 1.0)


def fold_dataset(
    raw_laps: dict[int, list[dict]],
    train_lap_numbers: Iterable[int],
    validation_lap_numbers: Iterable[int],
    *,
    track: Track | None = None,
    lookahead: int = 500,
) -> dict:
    """Align the validation lap with a transform fitted only on training-lap positions.

    The historical saved aligned tables use a shared transform fitted on lap 5.
    This function instead estimates a transform from a training lap within each fold,
    then reuses that transform for both partitions of that fold.
    """
    train_ids = tuple(train_lap_numbers)
    validation_ids = tuple(validation_lap_numbers)
    assert_disjoint_groups(train_ids, validation_ids)
    if not train_ids or not validation_ids:
        raise ValueError("Both training and validation laps are required")
    missing = (set(train_ids) | set(validation_ids)) - set(raw_laps)
    if missing:
        raise ValueError(f"Missing raw telemetry laps: {sorted(missing)}")

    circuit = track or Track("monza")
    offsets = offsets_for_horizon(lookahead)
    transform = estimate_transform(raw_laps[train_ids[0]], circuit)

    def prepare(lap_ids: tuple[int, ...]):
        records = []
        for lap_number in lap_ids:
            aligned = align_coordinates(raw_laps[lap_number], circuit, transform=transform)
            relative = convert_to_track_coordinates(aligned, circuit)
            for record in relative:
                record["lap_id"] = lap_number
            records.extend(relative)
        observations, targets = prepare_training_data(records, circuit, offsets)
        return (
            _to_matrix(observations, offsets),
            _targets(targets),
            np.asarray([record["lap_id"] for record in records]),
            records,
        )

    x_train, y_train, train_groups, train_records = prepare(train_ids)
    x_validation, y_validation, validation_groups, validation_records = prepare(validation_ids)
    assert_disjoint_groups(train_groups, validation_groups)
    return {
        "X_train": x_train,
        "y_train": y_train,
        "groups_train": train_groups,
        "train_records": train_records,
        "X_validation": x_validation,
        "y_validation": y_validation,
        "groups_validation": validation_groups,
        "validation_records": validation_records,
        "feature_names": feature_names(offsets),
        "lookahead_offsets": offsets,
        "transform_fitted_laps": train_ids,
    }


def make_preprocessing_pipeline(estimator, *, scale: bool):
    """Return a sklearn pipeline whose imputation/scaling fit only on each train fold."""
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("estimator", clone(estimator)))
    return Pipeline(steps)


def raw_feature_matrix(records: list[dict], *, lookahead: int = 500, track: Track | None = None):
    """Feature helper for exploratory full-dataset summaries; not a test split."""
    circuit = track or Track("monza")
    offsets = offsets_for_horizon(lookahead)
    observations, targets = prepare_training_data(records, circuit, offsets)
    return _to_matrix(observations, offsets), _targets(targets), feature_names(offsets)
