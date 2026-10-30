"""Re-evaluate a saved model with exclusive pedals, preserving raw-run results."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

from track import Track
from f1_simulator.core.imitation import load_model, predict_action
from data_eval.ai_controller import make_ai_controller, PEDAL_POLICY
from data_eval.evaluation import evaluate_controller

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon", type=int, choices=[250,500,1000], default=1000)
    args = parser.parse_args()
    source = ROOT / f"output/pipeline_{args.horizon}m"
    output = ROOT / f"output/pedal_fix_{args.horizon}m"
    output.mkdir(parents=True, exist_ok=True)
    old = json.loads((source/"benchmark.json").read_text())
    if old.get("pedal_policy", {"name":"raw"}) != {"name":"raw"}:
        raise ValueError("Source benchmark must be the preserved raw-pedal experiment")
    model_path = source/"imitation_model.joblib"
    model = load_model(model_path)
    model.n_jobs = 1
    manifest = json.loads((source/"run_manifest.json").read_text())
    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if digest != manifest["model_sha256"]:
        raise ValueError("Model no longer matches original benchmark")
    dt = old["dt"]
    trace, last_prediction = [], {}

    def predict_logged(model, obs):
        result = predict_action(model, obs)
        last_prediction.update(result)
        return result

    controller = make_ai_controller(model, predict_logged)

    def logged_controller(obs):
        action = controller(obs)
        elapsed = len(trace)*dt
        trace.append(dict(time=elapsed, speed_mps=obs["speed"],
                          lateral_offset_m=obs["lateral_offset"], heading_error_rad=obs["heading_error"],
                          raw_throttle=last_prediction["throttle"], raw_brake=last_prediction["brake"], **action))
        if len(trace) == 1 or len(trace) % 600 == 0:
            print(f"t={elapsed:.1f}s speed={obs['speed']*3.6:.1f}km/h pedals={action}", flush=True)
        return action

    logged_controller.lookahead_offsets = controller.lookahead_offsets
    print(f"Benchmarking saved {args.horizon} m model with {PEDAL_POLICY}", flush=True)
    result = evaluate_controller(logged_controller, Track("monza"), dt=dt,
                                max_time=old["max_time"], controller_name="imitation_exclusive")
    if any(r["throttle"]*r["brake"] != 0 for r in trace):
        raise AssertionError("Pedals overlapped")
    report = dict(horizon_m=args.horizon, model_sha256=digest, model_retrained=False,
                  dt=dt, max_time=old["max_time"], pedal_policy=PEDAL_POLICY,
                  steering_unchanged=True, original=old["imitation"], corrected=result)
    (output/"benchmark.json").write_text(json.dumps(report, indent=2)+"\n")
    with (output/"controls.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=trace[0].keys())
        writer.writeheader()
        writer.writerows(trace)
    rows = ["# Pedal arbitration benchmark", "",
            f"Saved {args.horizon} m model; model weights, steering, track, physics and timestep unchanged.", "",
            "| Controller | Lap time (s) | Completed | Off-track events | Valid lap |",
            "|---|---:|---|---:|---|"]
    for name, r in [("Original raw pedals", old["imitation"]), ("Exclusive pedals + launch", result)]:
        lap = f"{r['lap_time']:.1f}" if r['lap_time'] is not None else "timeout"
        rows.append(f"| {name} | {lap} | {r['completed_lap']} | {r['off_track_events']} | {r['valid_lap']} |")
    rows += ["", "This fixes conflicting pedals, not steering or speed planning. An off-track completion remains invalid.",
             "Below 5 m/s, near the centerline on gentle road, throttle is at least 0.4 and brake is zero.",
             "Otherwise brake predictions below 0.5 are ignored; predictions at/above 0.5 select braking with zero throttle.",
             "The brake value is a regression of binary labels, not calibrated brake pressure. Threshold 0.5 is a controller rule.",
             "controls.csv records model outputs and executed actions at every timestep. No model retraining was needed."]
    (output/"REPORT.md").write_text("\n".join(rows)+"\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
