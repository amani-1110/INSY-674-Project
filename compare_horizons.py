"""Summarize saved, matched-condition lookahead experiments."""
import json
from pathlib import Path
from f1_simulator.core.imitation import load_model

ROOT = Path(__file__).resolve().parent


def main():
    rows = []
    expected = None
    for horizon in (20, 250, 500, 1000):
        folder = ROOT / ("output/pipeline" if horizon == 20 else f"output/pipeline_{horizon}m")
        benchmark = json.loads((folder / "benchmark.json").read_text())
        manifest = json.loads((folder / "run_manifest.json").read_text())
        validation = json.loads((folder / "validation.json").read_text())
        conditions = dict(pedal_policy=benchmark.get("pedal_policy", {"name": "raw"}), dt=benchmark["dt"], max_time=benchmark["max_time"],
                          start_speed=benchmark["start_speed"], steering=benchmark["steering"],
                          track=manifest["track_source"]["sha256"],
                          training_laps=manifest["training_laps"], held_out_lap=manifest["held_out_lap"],
                          telemetry=[r["csv_sha256"] for r in manifest["telemetry"]])
        if expected is not None and conditions != expected:
            raise ValueError(f"Experiment {horizon} has different comparison conditions")
        expected = conditions
        model = load_model(folder / "imitation_model.joblib")
        if model.lookahead_offsets_[-1] != horizon:
            raise ValueError(f"Experiment {horizon} has unexpected model offsets")
        rows.append(dict(horizon_m=horizon, offsets_m=model.lookahead_offsets_,
                         **benchmark["imitation"], held_out_metrics=validation["held_out_metrics"]))
    output = ROOT / "output/lookahead_comparison.json"
    output.write_text(json.dumps(dict(conditions=expected, results=rows), indent=2)+"\n")
    report = ["# Curvature lookahead comparison", "",
              "Same Monza data, training laps 5/11, held-out lap 14, steering, physics, start and timestep.", "",
              "| Horizon | Time (s) | Off-track events | Valid lap | Throttle MAE | Brake MAE |",
              "|---|---:|---:|---|---:|---:|"]
    for r in rows:
        time = f"{r['lap_time']:.1f}" if r['lap_time'] is not None else "timeout"
        m = r["held_out_metrics"]
        report.append(f"| {r['horizon_m']} m | {time} | {r['off_track_events']} | {r['valid_lap']} | {m['throttle_mae']:.4f} | {m['brake_mae']:.4f} |")
    report += ["", "A completed circuit with off-track events is invalid. Lap times from invalid laps do not establish a performance gain.",
               "This is an exploratory comparison on one circuit/session, not evidence of unseen-circuit performance.",
               "Near-field curvature samples are retained and additional far-field samples are added; feature count changes with horizon.",
               "The steering controller is unchanged, so this tests extended throttle/brake-model features rather than a new driving planner."]
    text = "\n".join(report)+"\n"
    (ROOT / "output/lookahead_comparison.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
