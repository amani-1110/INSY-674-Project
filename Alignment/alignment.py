"""Compatibility imports for the packaged telemetry alignment pipeline."""

if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from f1_simulator.data.alignment import (
    DemoTrack,
    align_coordinates,
    alignment_report,
    convert_to_track_coordinates,
    demo_track,
    estimate_transform,
    load_lap_csv,
    plot_alignment,
)

__all__ = [
    "align_coordinates",
    "alignment_report",
    "convert_to_track_coordinates",
    "DemoTrack",
    "demo_track",
    "estimate_transform",
    "load_lap_csv",
    "plot_alignment",
]


if __name__ == "__main__":
    from f1_simulator.data.alignment import _self_test

    import sys

    sys.exit(0 if _self_test() else 1)
