"""Offline telemetry -> alignment -> imitation -> simulator benchmark."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from track import Track
from alignment.alignment import (estimate_transform, align_coordinates,
    convert_to_track_coordinates, alignment_report, plot_alignment)
from telemetry.demo_telemetry import inspect_saved_laps, load_lap_csv
from f1_simulator.core.imitation import (prepare_training_data, train_model,
    predict_action, save_model, load_model, validate_by_lap, offsets_for_horizon)
from data_eval.baseline import baseline_action
from data_eval.ai_controller import make_ai_controller, PEDAL_POLICY
from data_eval.evaluation import evaluate_controller, compare_controllers

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "telemetry/data/monza_three_new_laps"


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, records):
    if not records:
        raise ValueError("Cannot export an empty dataset")
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def prediction_metrics(model, records, track, training_records):
    X, y = prepare_training_data(records, track, model.lookahead_offsets_)
    predictions = np.asarray([[p["throttle"], p["brake"]]
                             for p in (predict_action(model, obs) for obs in X)])
    labels = np.asarray([[row["throttle"], row["brake"]] for row in y])
    mean = np.mean([[r["throttle"], r["brake"]] for r in training_records], axis=0)
    return dict(n_samples=len(records),
                throttle_mae=float(np.abs(predictions[:, 0]-labels[:, 0]).mean()),
                brake_mae=float(np.abs(predictions[:, 1]-labels[:, 1]).mean()),
                brake_accuracy=float(((predictions[:, 1] >= .5) == labels[:, 1]).mean()),
                baseline_throttle_mae=float(np.abs(mean[0]-labels[:, 0]).mean()),
                baseline_brake_mae=float(np.abs(mean[1]-labels[:, 1]).mean()))


def run_pipeline(output=None, dt=.1, max_time=600., held_out_lap=14, lookahead=250):
    """Run the supplied Monza session; the test lap never trains the model.

    One transform is fitted on a training lap and reused for the entire session.
    This is a held-out-lap experiment, not an unseen-circuit experiment.
    """
    import math
    if not math.isfinite(dt) or not 0 < dt <= 1:
        raise ValueError("dt must be finite and between 0 and 1")
    if not math.isfinite(max_time) or max_time <= 0:
        raise ValueError("max_time must be finite and positive")
    if held_out_lap not in (5, 11, 14):
        raise ValueError("held_out_lap must be one of the supplied laps: 5, 11, 14")
    offsets = offsets_for_horizon(lookahead)
    output = Path(output if output is not None else ROOT / f"output/pipeline_{lookahead}m").resolve()
    output.mkdir(parents=True, exist_ok=True)
    print("1/5 Validate saved telemetry and load Monza", flush=True)
    source_checks = inspect_saved_laps()
    track = Track("monza")
    laps = {n: load_lap_csv(DATA / f"2025_Monza_Q_VER_lap{n:02d}.csv") for n in (5, 11, 14)}
    training_laps = sorted(set(laps)-{held_out_lap})
    print("2/5 Fit training-lap transform and align all laps", flush=True)
    transform = estimate_transform(laps[training_laps[0]], track)
    records, reports = {}, {}
    for number, lap in laps.items():
        aligned = align_coordinates(lap, track, transform=transform)
        relative = convert_to_track_coordinates(aligned, track)
        report = alignment_report(relative, track)
        if (not .95 <= report["progress_fraction_of_lap"] <= 1.05 or
                report["backward_steps"] > 5 or report["share_within_track_widths"] < .8):
            raise ValueError(f"Lap {number} failed alignment checks: {report}")
        # Alignment's six/eight-field public contracts remain unchanged;
        # attach experiment identity here, before joining the separate laps.
        for row in relative:
            row["lap_id"] = f"2025_Monza_Q_VER_{number}"
        records[number] = relative
        reports[str(number)] = report
        write_csv(output / f"lap{number:02d}_aligned.csv", aligned)
        write_csv(output / f"lap{number:02d}_track_relative.csv", relative)
        plot_alignment(aligned, track, str(output / f"lap{number:02d}_overlay.png"),
                       f"Monza qualifying lap {number}: telemetry aligned to TUM track")
        print(f"  Lap {number}: {report['share_within_track_widths']:.1%} inside widths", flush=True)
    serial_transform = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                        for k, v in transform.items()}
    write_json(output / "alignment.json", dict(fitted_on_lap=training_laps[0],
               transform=serial_transform, laps=reports))
    train_records = [r for n in training_laps for r in records[n]]
    print("3/5 Validate by training lap, train, save and reload model", flush=True)
    validation = dict(training_laps=training_laps, held_out_lap=held_out_lap,
                      grouped_training_validation=validate_by_lap(train_records, track, lookahead_offsets=offsets))
    X, y = prepare_training_data(train_records, track, offsets)
    model = train_model(X, y, lookahead_offsets=offsets)
    save_model(model, str(output / "imitation_model.joblib"))
    model = load_model(str(output / "imitation_model.joblib"))
    # Avoid a thread pool per single-observation simulator prediction.
    model.n_jobs = 1
    validation["held_out_metrics"] = prediction_metrics(model, records[held_out_lap], track, train_records)
    write_json(output / "validation.json", validation)
    print("4/5 Run baseline and learned controls with identical steering", flush=True)
    baseline = evaluate_controller(baseline_action, track, dt=dt,
                                   controller_name="baseline", max_time=max_time)
    print(f"  Baseline: {baseline}", flush=True)
    ai = evaluate_controller(make_ai_controller(model, predict_action), track, dt=dt,
                             controller_name="imitation", max_time=max_time)
    print(f"  Imitation: {ai}", flush=True)
    comparison = compare_controllers(baseline, ai)
    # An off-track completion must never count as a valid performance gain.
    comparison["valid_time_improvement"] = (comparison["time_improvement"]
                                            if baseline["valid_lap"] and ai["valid_lap"] else None)
    benchmark = dict(pedal_policy=PEDAL_POLICY, lookahead_offsets_m=offsets, dt=dt, max_time=max_time, start_speed=0.,
                     steering="data_eval.baseline.baseline_action (identical for both)",
                     baseline=baseline, imitation=ai, comparison=comparison)
    write_json(output / "benchmark.json", benchmark)
    write_csv(output / "benchmark.csv", [baseline, ai])
    provenance = dict(lookahead_offsets_m=offsets, circuit="monza", track_length_m=track.length,
        track_source=json.loads((ROOT / "data/monza.source.json").read_text()),
        telemetry=source_checks, training_laps=training_laps, held_out_lap=held_out_lap,
        model_sha256=hashlib.sha256((output / "imitation_model.joblib").read_bytes()).hexdigest(),
        experiment="Held-out lap from one driver and one session on one circuit")
    write_json(output / "run_manifest.json", provenance)
    summary = f"""# Monza pipeline results

Model curvature offsets (metres): {offsets}.

Training laps: {training_laps}. Held-out lap: {held_out_lap}.
Track length: {track.length:.3f} m. Timestep: {dt} s. Timeout: {max_time} s.

| Controller | Completed | Valid (no off-track events) | Lap time (s) | Off-track events |
|---|---|---|---|---|
| Baseline | {baseline['completed_lap']} | {baseline['valid_lap']} | {baseline['lap_time']} | {baseline['off_track_events']} |
| Imitation | {ai['completed_lap']} | {ai['valid_lap']} | {ai['lap_time']} | {ai['off_track_events']} |

Held-out throttle MAE: {validation['held_out_metrics']['throttle_mae']:.4f}.
Held-out brake MAE: {validation['held_out_metrics']['brake_mae']:.4f}.

Both drivers use the same steering function, physics, start and timestep.
The AI uses exclusive_v1 pedal arbitration (brake threshold 0.5) and a modest
launch input below 5 m/s near the centerline on gentle road.
Any off-track event makes the lap invalid for performance claims; failed laps
have no lap time. A timeout is reported explicitly. No artificial recovery or
teleport is used. The saved model was trained only on the training laps.

Limitations: these laps share a session and circuit. This does not test unseen
circuits or establish real-world driving performance. Source positions are
approximate; some aligned samples fall outside the estimated road. Public brake
labels are binary, not measured brake pressure. Regression outputs are continuous
control commands. The simulator starts at rest, outside qualifying-speed training
conditions. The physics has no combined friction circle, tyre wear or calibrated
car-specific dynamics. Steering is unchanged; a longer pedal-model horizon alone
does not guarantee a valid lap.

See alignment.json, validation.json, benchmark.json, run_manifest.json and the
per-lap CSVs/overlays for the underlying results. Load joblib models only from
trusted sources.
"""
    (output / "REPORT.md").write_text(summary, encoding="utf-8")
    print(f"5/5 Saved results to {output}", flush=True)
    return dict(alignment=reports, validation=validation, benchmark=benchmark)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--dt", type=float, default=.1)
    parser.add_argument("--max-time", type=float, default=600.)
    parser.add_argument("--held-out-lap", type=int, choices=[5, 11, 14], default=14)
    parser.add_argument("--lookahead", type=int, choices=[20, 250, 500, 1000], default=250)
    args = parser.parse_args()
    run_pipeline(args.output, args.dt, args.max_time, args.held_out_lap, args.lookahead)


if __name__ == "__main__":
    main()
