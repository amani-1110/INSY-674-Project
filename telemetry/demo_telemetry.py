"""Offline handoff demo. Run: python demo_telemetry.py. No packages/account needed."""

import csv
import hashlib
import json
from pathlib import Path

if __package__:
    from .telemetry import FIELDS, validate_records
else:
    from telemetry import FIELDS, validate_records

DATA = Path(__file__).resolve().parent / "data" / "monza_three_new_laps"


def load_lap_csv(path: str | Path) -> list[dict]:
    """Load the exact six-field records expected by Person 5, without FastF1."""
    with Path(path).open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != FIELDS:
            raise ValueError(f"Expected CSV header {FIELDS}; got {reader.fieldnames}")
        records = [{key: float(value) for key, value in row.items()} for row in reader]
    validate_records(records)
    return records


def inspect_saved_laps() -> list[dict]:
    """Validate CSVs against their existing manifest and metadata."""
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    summaries = []
    seen = set()
    expected = {5, 11, 14}
    for item in sorted(manifest["laps"], key=lambda row: row["lap_number"]):
        number = item["lap_number"]
        if number in seen or number not in expected:
            raise ValueError("Unexpected or duplicate lap identity in manifest.")
        seen.add(number)
        path = DATA / f"2025_Monza_Q_VER_lap{number:02d}.csv"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != item["csv_sha256"]:
            raise ValueError(f"CSV hash does not match manifest: {path.name}")
        records = load_lap_csv(path)
        metadata = json.loads(path.with_suffix(".metadata.json").read_text(encoding="utf-8"))
        if metadata["synthetic"] is not False or metadata["source"] != "FastF1":
            raise ValueError(f"Unexpected source label: {path.name}")
        identity = (metadata["year"], metadata["circuit"], metadata["session_type"],
                    metadata["driver"], metadata["lap_number"])
        if identity != (2025, "Monza", "Q", "VER", number):
            raise ValueError(f"Lap identity mismatch: {path.name}")
        if len(records) != item["rows"] or len(records) != metadata["cleaning"]["exported_rows"]:
            raise ValueError(f"Row count mismatch: {path.name}")
        duration = metadata["lap_time_seconds"]
        if abs(duration - item["lap_time_seconds"]) > 1e-6:
            raise ValueError(f"Lap duration mismatch: {path.name}")
        if abs(records[0]["time"]) > 1e-6 or abs(records[-1]["time"] - duration) > 1e-6:
            raise ValueError(f"Lap coverage mismatch: {path.name}")
        gap = max(b["time"] - a["time"] for a, b in zip(records, records[1:]))
        policy = metadata["quality_policy"]
        if gap > policy["max_gap_seconds"]:
            raise ValueError(f"CSV gap exceeds its documented policy: {path.name}")
        native_gaps = []
        for stream in metadata["native_quality"].values():
            native_gaps.append(stream["max_gap_seconds"])
            if stream["max_gap_seconds"] > policy["max_gap_seconds"]:
                raise ValueError(f"Reported native gap exceeds export policy: {path.name}")
            if any(stream["missing_values"].values()):
                raise ValueError(f"Native metadata reports missing values: {path.name}")
        summaries.append({
            "lap": number, "rows": len(records), "lap_time_seconds": duration,
            "max_csv_gap_seconds": gap, "max_reported_native_gap_seconds": max(native_gaps),
            "export_gap_limit_seconds": policy["max_gap_seconds"],
            "csv_sha256": digest,
        })
    if seen != expected:
        raise ValueError("Manifest must contain laps 5, 11 and 14.")
    return summaries


def main():
    summaries = inspect_saved_laps()
    print("PASS: three saved FastF1 lap exports; schema, ranges, times, identities and hashes checked.")
    print("Driver VER | 2025 Monza qualifying | coordinates still need Person 5 alignment")
    print("Lap | Rows | Lap time (s) | Largest reported native gap (s)")
    for row in summaries:
        print(f"{row['lap']:3d} | {row['rows']:4d} | {row['lap_time_seconds']:12.3f} | {row['max_reported_native_gap_seconds']:.3f}")
    print("Saved exports use a documented 1.3s gap limit; telemetry.py keeps its original 1.0s limit.")
    print("One circuit only. These checks do not independently certify upstream provenance or positional accuracy.")


if __name__ == "__main__":
    main()
