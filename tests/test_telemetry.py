"""Offline checks of the data handoff and unit conversion, not ML tests."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
from fastf1.core import Lap, Laps

from f1_simulator.telemetry import telemetry as module


class TelemetryChecks(unittest.TestCase):
    def setUp(self):
        self.start = pd.Timedelta(seconds=100)
        self.native = pd.DataFrame({
            "SessionTime": self.start + pd.to_timedelta([i / 10 for i in range(51)], unit="s"),
            "X": [12000.0] * 51, "Y": [8000 + 100 * i for i in range(51)],
            "Speed": [72.0] * 51, "Throttle": [80.0] * 51,
            "Brake": [False] * 51, "Source": ["car"] * 51,
        })

    def test_units_and_lap_clock(self):
        records, _ = module._clean_telemetry(self.native, self.start, 5.0)
        self.assertEqual(records[0], {"time": 0.0, "x": 1200.0, "y": 800.0,
                                      "speed": 20.0, "throttle": 0.8, "brake": 0.0})
        self.assertEqual(records[-1]["time"], 5.0)

    def test_missing_row_and_duplicate_report(self):
        data = self.native.copy()
        data.loc[10, "Speed"] = float("nan")
        data = pd.concat([data, data.iloc[[3]]], ignore_index=True)
        records, quality = module._clean_telemetry(data, self.start, 5.0)
        self.assertEqual(quality["rows_dropped_missing_or_infinite"], 1)
        self.assertEqual(quality["duplicate_timestamps_removed"], 1)
        self.assertEqual(len(records), 50)

    def test_no_clock_reset_to_hide_missing_start(self):
        records, _ = module._clean_telemetry(self.native.iloc[2:], self.start, 5.0)
        self.assertAlmostEqual(records[0]["time"], 0.2)
        with self.assertRaises(ValueError):
            module._clean_telemetry(self.native.iloc[10:], self.start, 5.0)

    def test_reject_long_native_gap(self):
        data = self.native.drop(index=range(15, 30))
        with self.assertRaisesRegex(ValueError, "gap"):
            module._native_quality(data, self.start, 5.0, ["X", "Y"])

    def test_reject_invalid_throttle_instead_of_silent_clipping(self):
        data = self.native.copy()
        data.loc[5, "Throttle"] = 104.0
        with self.assertRaisesRegex(ValueError, "throttle"):
            module._clean_telemetry(data, self.start, 5.0)

    def test_brake_and_order(self):
        records = module.example_records()
        records[1]["brake"] = 0.5
        with self.assertRaises(ValueError):
            module.validate_records(records)
        records[1]["brake"] = 1.0
        records[1]["time"] = 0.0
        with self.assertRaises(ValueError):
            module.validate_records(records)

    def test_export_keeps_six_columns(self):
        with tempfile.TemporaryDirectory() as folder:
            path = module.save_bundle({"records": module.example_records(),
                                       "metadata": {"synthetic": True}}, folder, "example")
            self.assertEqual(path.read_text().splitlines()[0], ",".join(module.FIELDS))
            self.assertTrue(Path(folder, "example.metadata.json").exists())

    def test_team_function_with_mocked_session(self):
        # Use real FastF1 Lap/Laps containers, but no remote data.
        rows = []
        for number, deleted, pit_out in [(1, True, None), (2, False, self.start), (3, False, None)]:
            rows.append({"Driver": "VER", "DriverNumber": "1", "LapNumber": float(number),
                         "LapTime": pd.Timedelta(seconds=5), "LapStartTime": self.start,
                         "PitInTime": pd.NaT, "PitOutTime": pit_out or pd.NaT,
                         "Deleted": deleted, "TrackStatus": "1", "IsAccurate": True,
                         "Compound": "SOFT", "TyreLife": 2.0})
        session = SimpleNamespace(laps=Laps(rows), event={"EventName": "Offline fixture"})
        with patch.object(module, "_load_session", return_value=session), \
                patch.object(Lap, "get_car_data", return_value=self.native), \
                patch.object(Lap, "get_pos_data", return_value=self.native), \
                patch.object(Lap, "get_telemetry", return_value=self.native):
            bundle = module.extract_laps(2025, "Monza", "Q", "VER", max_laps=3)[0]
            self.assertEqual(bundle["metadata"]["lap_number"], 3)
            self.assertEqual(bundle["metadata"]["exported_laps"], 1)
            records = module.extract_lap(2025, "Monza", "Q", "VER")
            self.assertIsInstance(records, list)
            self.assertEqual(set(records[0]), set(module.FIELDS))


if __name__ == "__main__":
    unittest.main()
