"""Focused tests for the supplemental INSY 674 analysis workflow."""

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor

from f1_simulator.core.imitation import load_model, predict
from ml_analysis.preprocessing import (
    ROOT,
    assert_disjoint_groups,
    clip_throttle_predictions,
    group_splits,
    load_raw_laps,
    make_preprocessing_pipeline,
)


def test_saved_telemetry_integrity_and_independent_group_count():
    laps = load_raw_laps()
    assert sorted(laps) == [5, 11, 14]
    assert len(laps) == 3
    for lap_id, records in laps.items():
        frame = pd.DataFrame(records)
        assert list(frame.columns) == ["time", "x", "y", "speed", "throttle", "brake"]
        assert frame.notna().all().all()
        assert frame["time"].is_monotonic_increasing
        assert frame["time"].is_unique
        assert frame["brake"].isin([0, 1]).all()
        assert frame["throttle"].between(0, 1).all()
        assert lap_id in (5, 11, 14)


def test_group_splits_keep_each_lap_whole_and_cover_every_group():
    groups = np.repeat([5, 11], [590, 591])
    splits = group_splits(groups, n_splits=2)
    repeat_splits = group_splits(groups, n_splits=2)
    assert all(
        np.array_equal(train, repeated_train) and np.array_equal(validation, repeated_validation)
        for (train, validation), (repeated_train, repeated_validation) in zip(splits, repeat_splits)
    )
    observed_validation = set()
    for train_index, validation_index in splits:
        train_groups = set(groups[train_index])
        validation_groups = set(groups[validation_index])
        assert_disjoint_groups(train_groups, validation_groups)
        assert len(validation_groups) == 1
        observed_validation.update(validation_groups)
    assert observed_validation == {5, 11}


def test_overlapping_groups_are_rejected():
    with pytest.raises(ValueError, match="overlap"):
        assert_disjoint_groups([5, 11], [11])


def test_pipeline_scaling_is_fitted_on_training_rows_only():
    train_x = np.asarray([[1.0], [3.0]])
    validation_x = np.asarray([[1000.0]])
    model = make_preprocessing_pipeline(DummyRegressor(), scale=True)
    model.fit(train_x, [0.0, 1.0])
    assert model.named_steps["scaler"].mean_[0] == pytest.approx(2.0)
    assert model.named_steps["scaler"].mean_[0] != validation_x.mean()


def test_throttle_predictions_are_bounded_and_nonfinite_values_rejected():
    np.testing.assert_array_equal(clip_throttle_predictions([-0.2, 0.4, 1.3]), [0.0, 0.4, 1.0])
    with pytest.raises(ValueError, match="finite"):
        clip_throttle_predictions([np.nan])


def test_existing_model_bundle_loads_and_returns_bounded_controls():
    model_path = ROOT / "output" / "pipeline_500m" / "imitation_model.joblib"
    model = load_model(str(model_path))
    observation = {
        "speed": 30.0,
        "lateral_offset": 0.0,
        "heading_error": 0.0,
        "curvature_ahead": [0.0] * len(model.lookahead_offsets_),
    }
    prediction = predict(model, observation, brake_threshold=0.5)
    assert 0.0 <= prediction["throttle"] <= 1.0
    assert prediction["brake"] in (0.0, 1.0)
