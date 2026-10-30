
from __future__ import annotations

import math

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold

LOOKAHEAD_OFFSETS = (0.0, 10.0, 20.0)  # metres
FEATURE_NAMES = [
    "speed", "lateral_offset", "heading_error",
    "curv_0", "curv_10", "curv_20",
]
LONG_LOOKAHEAD_OFFSETS = (0., 10., 20., 40., 60., 80., 100., 150., 200., 250.)

TARGET_NAMES = ["throttle", "brake"]


# ----------------------------------------------------------------------------
# Feature handling
# ----------------------------------------------------------------------------
def offsets_for_horizon(horizon):
    """Preserve near-field samples and add progressively more distant curvature."""
    choices = {
        20: LOOKAHEAD_OFFSETS,
        250: LONG_LOOKAHEAD_OFFSETS,
        500: LONG_LOOKAHEAD_OFFSETS + (300., 350., 400., 450., 500.),
        1000: LONG_LOOKAHEAD_OFFSETS + (300., 350., 400., 450., 500.,
                                      600., 700., 800., 900., 1000.),
    }
    if horizon not in choices:
        raise ValueError("lookahead must be 20, 250, 500 or 1000 metres")
    return choices[horizon]


def validate_offsets(offsets):
    offsets = tuple(float(v) for v in offsets)
    if (not offsets or offsets[0] != 0 or not all(math.isfinite(v) for v in offsets)
            or any(b <= a for a, b in zip(offsets, offsets[1:]))):
        raise ValueError("Lookahead offsets must start at zero and increase, with finite values")
    return offsets


def feature_names(offsets):
    return ["speed", "lateral_offset", "heading_error"] + [f"curv_{v:g}" for v in offsets]


def _wrap(angle: float) -> float:
    """Wrap an angle to [-pi, pi)."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


def _features(obs: dict, offsets=LOOKAHEAD_OFFSETS) -> list[float]:
    curv = list(obs["curvature_ahead"])
    if len(curv) != len(offsets):
        raise ValueError(f"curvature_ahead must have {len(offsets)} values")
    return [float(obs["speed"]), float(obs["lateral_offset"]),
            _wrap(float(obs["heading_error"])), *map(float, curv)]


def _to_matrix(X, offsets=LOOKAHEAD_OFFSETS) -> np.ndarray:
    if isinstance(X, np.ndarray):
        return X
    return np.asarray([_features(o, offsets) for o in X], dtype=float)


def _targets(y) -> np.ndarray:
    if isinstance(y, np.ndarray):
        return y
    return np.asarray([[float(a["throttle"]), float(a["brake"])] for a in y], dtype=float)


# ----------------------------------------------------------------------------
# Data preparation
# ----------------------------------------------------------------------------
def prepare_training_data(aligned_laps: list[dict], track, lookahead_offsets=LOOKAHEAD_OFFSETS):
    """Return (X, y).
 
    X: list of observation dicts {speed, lateral_offset, heading_error,
       curvature_ahead=[k(s+offset) for offset in lookahead_offsets]}
    y: list of {"throttle", "brake"} labels.
    Optional "lap_id" values are kept in a side list on the returned lists'
    `.lap_ids` attribute-free way: use `lap_ids_of(aligned_laps)` to get them.
    """
    X, y = [], []
    for rec in aligned_laps:
        s = float(rec["s"])
        length = getattr(track, "length", None)
        curv = []
        for off in lookahead_offsets:
            q = s + off
            if length:
                q %= length
            curv.append(float(track.curvature(q)))
        X.append({
            "speed": float(rec["speed"]),
            "lateral_offset": float(rec["lateral_offset"]),
            "heading_error": _wrap(float(rec["heading_error"])),
            "curvature_ahead": curv,
        })
        y.append({
            "throttle": float(np.clip(rec["throttle"], 0.0, 1.0)),
            "brake": float(np.clip(rec["brake"], 0.0, 1.0)),
        })
    return X, y


def lap_ids_of(aligned_laps: list[dict]) -> np.ndarray:
    """Group labels for lap-wise splitting (0 for all if 'lap_id' is absent)."""
    return np.asarray([rec.get("lap_id", 0) for rec in aligned_laps])


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------
def train(X, y, n_estimators: int = 200, min_samples_leaf: int = 5,
                random_state: int = 0, lookahead_offsets=LOOKAHEAD_OFFSETS):
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        n_jobs=-1,
        random_state=random_state,
    )
    offsets = validate_offsets(lookahead_offsets)
    model.fit(_to_matrix(X, offsets), _targets(y))
    model.lookahead_offsets_ = offsets
    return model


def predict(model, observation: dict, brake_threshold: float | None = None) -> dict:
    """Return {'throttle': float, 'brake': float}, both clipped to [0, 1].
 
    brake_threshold: if given, brake is returned as 0.0 / 1.0 (hard decision);
    otherwise it is the model's brake probability-like value.
    """
    pred = model.predict(np.asarray([_features(observation, getattr(model, "lookahead_offsets_", LOOKAHEAD_OFFSETS))], dtype=float))[0]
    throttle = float(np.clip(pred[0], 0.0, 1.0))
    brake = float(np.clip(pred[1], 0.0, 1.0))
    if brake_threshold is not None:
        brake = 1.0 if brake >= brake_threshold else 0.0
    return {"throttle": throttle, "brake": brake}


def save_model(model, path: str = "imitation_model.joblib") -> None:
    offsets = validate_offsets(getattr(model, "lookahead_offsets_", LOOKAHEAD_OFFSETS))
    joblib.dump({"model": model, "features": feature_names(offsets),
                 "targets": TARGET_NAMES, "lookahead": offsets}, path)


def load_model(path: str = "imitation_model.joblib"):
    bundle = joblib.load(path)
    model = bundle["model"]
    offsets = validate_offsets(bundle["lookahead"])
    if bundle["features"] != feature_names(offsets) or bundle["targets"] != TARGET_NAMES:
        raise ValueError("Saved model feature/target schema is inconsistent")
    if model.n_features_in_ != 3 + len(offsets):
        raise ValueError("Saved model feature count does not match its lookahead metadata")
    model.lookahead_offsets_ = offsets
    return model


# ----------------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------------
def validate_by_lap(aligned_laps: list[dict], track, n_splits: int = 5, lookahead_offsets=LOOKAHEAD_OFFSETS) -> dict:
    """Grouped K-fold by lap_id (use circuit id as lap_id for leave-circuit-out).
 
    Compares the model against a constant-mean baseline on held-out laps.
    """
    X, y = prepare_training_data(aligned_laps, track, lookahead_offsets)
    Xm, ym = _to_matrix(X, lookahead_offsets), _targets(y)
    groups = lap_ids_of(aligned_laps)
    n_splits = min(n_splits, len(np.unique(groups)))
    if n_splits < 2:
        raise ValueError("Need at least 2 distinct lap_id values for validation.")
 
    rows = []
    for train_idx, test_idx in GroupKFold(n_splits=n_splits).split(Xm, ym, groups):
        m = train(Xm[train_idx], ym[train_idx], lookahead_offsets=lookahead_offsets)
        p = np.clip(m.predict(Xm[test_idx]), 0, 1)
        mean_pred = ym[train_idx].mean(axis=0)
        t = ym[test_idx]
        hard = (p[:, 1] >= 0.5).astype(float)
        rows.append({
            "throttle_mae": float(np.abs(p[:, 0] - t[:, 0]).mean()),
            "brake_mae": float(np.abs(p[:, 1] - t[:, 1]).mean()),
            "brake_accuracy": float((hard == t[:, 1]).mean()),
            "baseline_throttle_mae": float(np.abs(mean_pred[0] - t[:, 0]).mean()),
            "baseline_brake_mae": float(np.abs(mean_pred[1] - t[:, 1]).mean()),
            "baseline_brake_accuracy": float(((mean_pred[1] >= 0.5) == t[:, 1]).mean()),
        })
    keys = rows[0].keys()
    return {
        "n_samples": int(len(Xm)),
        "n_groups": int(len(np.unique(groups))),
        "n_folds": n_splits,
        "mean": {k: float(np.mean([r[k] for r in rows])) for k in keys},
        "folds": rows,
    }


# Shared handoff contract names; retain existing callers.
train_model = train
predict_action = predict
