"""Run: python check_csv.py data/real/YOUR_FILE.csv"""
import csv
import json
import sys
from pathlib import Path

from f1_simulator.telemetry.telemetry import FIELDS, validate_records

if len(sys.argv) != 2:
    raise SystemExit("Usage: python check_csv.py path/to/lap.csv")
path = Path(sys.argv[1])
with path.open(newline="", encoding="utf-8") as file:
    reader = csv.DictReader(file)
    if reader.fieldnames != FIELDS:
        raise SystemExit(f"Expected header {FIELDS}; got {reader.fieldnames}")
    records = [{key: float(value) for key, value in row.items()} for row in reader]
validate_records(records)
print(f"PASS: {len(records)} rows; exact schema, finite values, strictly increasing times, valid control ranges.")
print(f"Time range: {records[0]['time']:.3f} to {records[-1]['time']:.3f} seconds")
print(f"Speed range: {min(r['speed'] for r in records):.3f} to {max(r['speed'] for r in records):.3f} m/s")
metadata_path = path.with_suffix(".metadata.json")
if metadata_path.exists():
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("synthetic"):
        print("SYNTHETIC EXAMPLE ONLY: schema checks do not make this a real lap.")
    else:
        print(f"Source: {metadata.get('source')}; driver: {metadata.get('driver')}; lap: {metadata.get('lap_number')}")
        print("Still inspect both plots and the metadata; schema validation cannot certify physical accuracy.")
