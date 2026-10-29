"""Task 4: FastF1 -> clean lap records for Person 5. Read START_HERE.md first."""

import argparse
import csv
import json
import math
import os
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

FIELDS = ["time", "x", "y", "speed", "throttle", "brake"]
ROOT = Path(__file__).resolve().parent
MAX_GAP = 1.0  # Project quality policy, seconds; not a FastF1 guarantee.
EDGE_TOLERANCE = 0.75


def example_records() -> list[dict]:
    """The PDF's two illustrative rows. NOT a real lap or training dataset."""
    return [
        {"time": 0.0, "x": 1200.0, "y": 800.0,
         "speed": 20.0, "throttle": 0.8, "brake": 0.0},
        {"time": 0.1, "x": 1200.0, "y": 802.0,
         "speed": 20.4, "throttle": 0.8, "brake": 0.0},
    ]


def validate_records(records: list[dict]) -> None:
    """Check the shared schema, finite numbers, units' ranges and ordering."""
    if len(records) < 2:
        raise ValueError("Need at least two usable samples.")
    previous = -math.inf
    for index, row in enumerate(records):
        if set(row) != set(FIELDS):
            raise ValueError(f"Row {index}: expected exactly {FIELDS}.")
        if not all(math.isfinite(float(row[key])) for key in FIELDS):
            raise ValueError(f"Row {index}: missing or infinite value.")
        if row["time"] < 0 or row["time"] <= previous:
            raise ValueError(f"Row {index}: times must increase strictly from >= 0.")
        if row["speed"] < 0:
            raise ValueError(f"Row {index}: negative speed.")
        if not 0 <= row["throttle"] <= 1:
            raise ValueError(f"Row {index}: throttle outside [0, 1].")
        if row["brake"] not in (0.0, 1.0):
            raise ValueError(f"Row {index}: brake must be 0.0 or 1.0.")
        previous = row["time"]


@lru_cache(maxsize=4)
def _load_session(year, circuit, session_type):
    import fastf1

    cache = Path(os.environ.get("F1_TELEMETRY_CACHE", ROOT / "cache"))
    cache.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.configure(cache_dir=str(cache))
    session = fastf1.get_session(year, circuit, session_type)
    try:
        session.load(telemetry=True, weather=False, messages=True)
        if session.laps.empty:
            raise ValueError("Session has no lap records.")
    except (fastf1.exceptions.DataNotLoadedError, fastf1.exceptions.F1AuthError) as error:
        raise RuntimeError(
            "F1 data was not available. Check your internet connection and run "
            "'python -m fastf1 auth f1tv --authenticate' in your own terminal. "
            "See START_HERE.md for the HTTP 401 explanation."
        ) from error
    return session


def _native_quality(data, lap_start, duration, channels):
    """Check original channels before merged telemetry can hide long gaps."""
    import pandas as pd

    if data.empty:
        raise ValueError("Native telemetry is empty.")
    required = ["SessionTime", *channels]
    missing = set(required) - set(data.columns)
    if missing:
        raise ValueError(f"Missing native columns: {sorted(missing)}")
    missing_counts = {key: int(data[key].isna().sum()) for key in required}
    if any(missing_counts.values()):
        raise ValueError(f"Native channels contain missing values: {missing_counts}")
    for key in channels:
        values = pd.to_numeric(data[key], errors="coerce")
        if not values.map(math.isfinite).all():
            raise ValueError(f"Native channel {key} contains non-numeric values.")
    times = (data["SessionTime"] - lap_start).dt.total_seconds()
    times = times[(times >= 0) & (times <= duration)].sort_values().drop_duplicates()
    if len(times) < 20:
        raise ValueError("Too few native samples for a complete real lap.")
    largest_gap = float(times.diff().max())
    if times.iloc[0] > EDGE_TOLERANCE or duration - times.iloc[-1] > EDGE_TOLERANCE:
        raise ValueError("Native data does not cover both lap boundaries closely enough.")
    if largest_gap > MAX_GAP:
        raise ValueError(f"Native data gap {largest_gap:.3f}s exceeds {MAX_GAP}s.")
    return {
        "samples": int(len(times)), "missing_values": missing_counts,
        "first_sample_seconds": float(times.iloc[0]),
        "last_sample_seconds": float(times.iloc[-1]),
        "max_gap_seconds": largest_gap,
    }


def _clean_telemetry(telemetry, lap_start, duration):
    import pandas as pd

    required = ["SessionTime", "X", "Y", "Speed", "Throttle", "Brake"]
    missing = set(required) - set(telemetry.columns)
    if missing:
        raise ValueError(f"Missing merged columns: {sorted(missing)}")
    # SessionTime is tied to the lap start, never to the first surviving row.
    clean = pd.DataFrame({
        "time": (telemetry["SessionTime"] - lap_start).dt.total_seconds(),
        "x": pd.to_numeric(telemetry["X"], errors="coerce") / 10.0,
        "y": pd.to_numeric(telemetry["Y"], errors="coerce") / 10.0,
        "speed": pd.to_numeric(telemetry["Speed"], errors="coerce") / 3.6,
        "throttle": pd.to_numeric(telemetry["Throttle"], errors="coerce") / 100.0,
        "brake": pd.to_numeric(telemetry["Brake"], errors="coerce"),
    }).astype(float)
    original_count = len(clean)
    clean = clean.replace([math.inf, -math.inf], float("nan")).dropna()
    missing_rows = original_count - len(clean)
    in_lap = clean["time"].between(-1e-6, duration + 1e-6)
    outside_rows = int((~in_lap).sum())
    clean = clean.loc[in_lap].copy()
    clean.loc[clean["time"].abs() < 1e-6, "time"] = 0.0
    clean.loc[(clean["time"] - duration).abs() < 1e-6, "time"] = duration
    clean = clean.sort_values("time", kind="stable")
    duplicates = int(clean["time"].duplicated().sum())
    clean = clean.drop_duplicates("time", keep="first")
    records = clean[FIELDS].to_dict(orient="records")
    validate_records(records)
    if records[0]["time"] > EDGE_TOLERANCE or duration - records[-1]["time"] > EDGE_TOLERANCE:
        raise ValueError("Clean data is missing a substantial part of the lap.")
    gaps = clean["time"].diff()
    if float(gaps.max()) > MAX_GAP:
        raise ValueError("A gap remains after cleaning; this lap is rejected.")
    quality = {
        "merged_rows": original_count, "exported_rows": len(records),
        "rows_dropped_missing_or_infinite": missing_rows,
        "rows_dropped_outside_lap": outside_rows,
        "duplicate_timestamps_removed": duplicates,
        "first_time_seconds": records[0]["time"],
        "last_time_seconds": records[-1]["time"],
        "max_gap_seconds": float(gaps.max()),
    }
    if "Source" in telemetry:
        quality["merged_source_counts_before_cleaning"] = {
            str(key): int(value) for key, value in telemetry["Source"].value_counts().items()
        }
    return records, quality


def extract_laps(year: int, circuit: str, session_type: str,
                 driver: str, max_laps: int = 3) -> list[dict]:
    """Extra helper: return up to max_laps bundles, fastest accepted first.

    A bundle contains records plus metadata. The required extract_lap below
    still returns only list[dict], preserving the team's exact interface.
    """
    import fastf1
    import pandas as pd

    if max_laps < 1:
        raise ValueError("max_laps must be at least 1.")
    session = _load_session(year, circuit, session_type)
    laps = session.laps.pick_drivers(driver).pick_accurate()
    needed = ["LapTime", "LapStartTime", "PitInTime", "PitOutTime", "Deleted", "TrackStatus"]
    if not set(needed).issubset(laps.columns):
        raise ValueError("Lap quality fields unavailable; cannot verify clean laps.")
    laps = laps.loc[
        laps["LapTime"].notna() & laps["LapStartTime"].notna()
        & laps["PitInTime"].isna() & laps["PitOutTime"].isna()
        & laps["Deleted"].eq(False) & laps["TrackStatus"].eq("1")
    ].sort_values("LapTime")
    if laps.empty:
        raise ValueError(f"No eligible clean laps for {driver} at {circuit} {year}.")
    bundles, rejected = [], []
    for _, lap in laps.iterlaps():
        lap_number = int(lap["LapNumber"])
        duration = float(lap["LapTime"].total_seconds())
        start = lap["LapStartTime"]
        try:
            native = {
                "car": _native_quality(lap.get_car_data(), start, duration,
                                       ["Speed", "Throttle", "Brake"]),
                "position": _native_quality(lap.get_pos_data(), start, duration, ["X", "Y"]),
            }
            # FastF1 synchronizes car/position channels, with interpolated edges.
            telemetry = lap.get_telemetry(frequency="original")
            records, quality = _clean_telemetry(telemetry, start, duration)
        except ValueError as error:
            rejected.append({"lap_number": lap_number, "reason": str(error)})
            continue
        metadata = {
            "source": "FastF1", "synthetic": False,
            "fastf1_version": fastf1.__version__, "pandas_version": pd.__version__,
            "exported_at_utc": datetime.now(timezone.utc).isoformat(),
            "year": year, "circuit": circuit, "event_name": session.event["EventName"],
            "session_type": session_type, "driver": driver,
            "lap_number": lap_number, "lap_time_seconds": duration,
            "compound": str(lap.get("Compound", "unknown")),
            "tyre_life_laps": None if pd.isna(lap.get("TyreLife")) else float(lap["TyreLife"]),
            "track_status": str(lap["TrackStatus"]),
            "lap_start_session_seconds": float(start.total_seconds()),
            "units": {"time": "s", "x": "m", "y": "m", "speed": "m/s",
                      "throttle": "fraction 0..1", "brake": "binary 0 or 1"},
            "coordinate_frame": "F1 position frame, converted from 1/10 m; NOT Track world frame",
            "selection": "IsAccurate; not deleted; no pit in/out; green TrackStatus 1; fastest accepted first",
            "quality_policy": {"max_gap_seconds": MAX_GAP,
                               "max_boundary_gap_seconds": EDGE_TOLERANCE},
            "native_quality": native, "cleaning": quality,
            "interpolation": "FastF1 get_telemetry(original) merges native car/position timestamps and interpolates channels and lap edges. No extra fixed-rate resampling or filling here. Source counts are row origins, not per-channel proof of measurement.",
            "distance_plot": "Trapezoidal integral of speed over time, relative to first exported sample; estimated travelled distance, not track centreline s.",
        }
        bundles.append({"records": records, "metadata": metadata})
        if len(bundles) >= max_laps:
            break
    if not bundles:
        raise ValueError(f"No lap passed telemetry checks. Rejections: {rejected}")
    for bundle in bundles:
        bundle["metadata"]["rejected_laps_checked"] = rejected.copy()
        bundle["metadata"]["requested_laps"] = max_laps
        bundle["metadata"]["exported_laps"] = len(bundles)
    return bundles


def extract_lap(year: int, circuit: str, session_type: str, driver: str) -> list[dict]:
    """Required team function: return one complete accepted lap as records."""
    return extract_laps(year, circuit, session_type, driver, max_laps=1)[0]["records"]


def save_bundle(bundle, folder, stem):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    records, metadata = bundle["records"], bundle["metadata"]
    validate_records(records)
    csv_path = folder / f"{stem}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(records)
    (folder / f"{stem}.metadata.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return csv_path


def plot_records(records, folder, stem):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    distance = [0.0]
    for before, after in zip(records, records[1:]):
        step = (before["speed"] + after["speed"]) / 2 * (after["time"] - before["time"])
        distance.append(distance[-1] + step)
    figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(distance, [row["speed"] for row in records], color="#145DA0")
    axes[0].set_ylabel("Speed (m/s)")
    axes[0].set_title(stem)
    axes[1].plot(distance, [row["throttle"] for row in records], label="Throttle", color="#16855B")
    axes[1].step(distance, [row["brake"] for row in records], where="post", label="Brake on/off", color="#C23B3B")
    axes[1].set(xlabel="Estimated travelled distance (m)", ylabel="Control (0 to 1)", ylim=(-0.05, 1.05))
    axes[1].legend()
    for ax in axes:
        ax.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(Path(folder) / f"{stem}.png", dpi=160)
    plt.close(figure)
    figure, ax = plt.subplots(figsize=(8, 7))
    ax.plot([row["x"] for row in records], [row["y"] for row in records], linewidth=1)
    ax.set(xlabel="Telemetry X (m)", ylabel="Telemetry Y (m)", title=f"{stem} trajectory")
    ax.set_aspect("equal", adjustable="datalim")
    figure.tight_layout()
    figure.savefig(Path(folder) / f"{stem}_trajectory.png", dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true", help="Only write the two synthetic PDF rows; no network or packages needed")
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--circuit", default="Monza")
    parser.add_argument("--session", default="Q")
    parser.add_argument("--driver", default="VER")
    parser.add_argument("--max-laps", type=int, default=1)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    if args.demo:
        bundle = {"records": example_records(), "metadata": {
            "synthetic": True, "source": "Two illustrative rows from team PDF",
            "warning": "Interface demo only. Not a complete lap. Never use as real training data."
        }}
        path = save_bundle(bundle, args.out or ROOT / "data" / "example_only", "example_only")
        print(f"SYNTHETIC EXAMPLE: wrote 2 rows to {path}")
        return
    folder = args.out or ROOT / "data" / "real"
    bundles = extract_laps(args.year, args.circuit, args.session, args.driver, args.max_laps)
    safe_circuit = "".join(c if c.isalnum() else "_" for c in args.circuit)
    for bundle in bundles:
        number = bundle["metadata"]["lap_number"]
        stem = f"{args.year}_{safe_circuit}_{args.session}_{args.driver}_lap{number:02d}"
        path = save_bundle(bundle, folder, stem)
        plot_records(bundle["records"], folder, stem)
        print(f"REAL LAP: {len(bundle['records'])} rows, {bundle['metadata']['lap_time_seconds']:.3f}s -> {path}")
    print(f"Exported {len(bundles)} of up to {args.max_laps} requested laps. Read each metadata JSON and inspect both plots.")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, ImportError, RuntimeError) as error:
        raise SystemExit(f"STOP: {error}") from error
