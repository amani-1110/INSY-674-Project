"""Measure the contribution of learned pedals with a matched planner-only ablation."""
from f1_simulator.paths import PROJECT_ROOT
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from f1_simulator.core.track import Track
from f1_simulator.controllers.speed_planner import PlannerConfig,make_hybrid_controller,make_planner_controller
from f1_simulator.core.imitation import load_model,predict_action
from scripts.benchmark_speed_planner import run_attempt

ROOT = PROJECT_ROOT


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon",type=int,choices=[250,500,1000],default=500)
    parser.add_argument("--dt",type=float,default=.1)
    parser.add_argument("--track",default="monza")
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    if not 0 < args.dt <= 1:
        parser.error("--dt must be between 0 and 1")
    path=(ROOT/"models/original_pedals.joblib" if args.horizon == 500 else ROOT/f"output/pipeline_{args.horizon}m/imitation_model.joblib")
    before_hash=hashlib.sha256(path.read_bytes()).hexdigest()
    training_manifest = (json.loads((ROOT/"models/manifest.json").read_text())["models"]["original_pedals.joblib"]["training"]
                         if args.horizon == 500 else json.loads((path.parent/"run_manifest.json").read_text()))
    if training_manifest["model_sha256"] != before_hash:
        raise ValueError("Model differs from its recorded training artifact")
    training_circuit=training_manifest["circuit"].lower()
    unseen=args.track.lower() != training_circuit
    model=load_model(path);model.n_jobs=1
    track=Track(args.track);config=PlannerConfig()
    suffix="" if args.track.lower()=="monza" else args.track.lower()+"_"
    folder=args.output or ROOT/f"output/planner_ablation_{suffix}{args.horizon}m"
    folder.mkdir(parents=True,exist_ok=True)
    source=json.loads((ROOT/f"data/{args.track.lower()}.source.json").read_text())
    if hashlib.sha256((ROOT/f"data/{args.track.lower()}.csv").read_bytes()).hexdigest() != source["sha256"]:
        raise ValueError("Circuit data differs from its source metadata")
    print(f"Training circuit: {training_circuit}; evaluation: {args.track}; unseen to model: {unseen}",flush=True)
    controllers={"planner_only":make_planner_controller(config,track,model.lookahead_offsets_),
                 "hybrid":make_hybrid_controller(model,predict_action,config,track)}
    results={}
    for name,controller in controllers.items():
        print("Running",name,flush=True)
        results[name]=run_attempt(controller,track,name,folder,args.dt,600.)
    after_hash=hashlib.sha256(path.read_bytes()).hexdigest()
    if after_hash != before_hash:
        raise AssertionError("Model artifact changed during evaluation")
    report=dict(dt=args.dt,max_time=600.,horizon_m=args.horizon,
                training_circuit=training_circuit,test_circuit=args.track,
                unseen_to_model=unseen,model_retrained=False,track_source=source,
                track_length_m=track.length,model_unchanged_during_test=True,
                source_code_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                    for name in ["f1_simulator/controllers/speed_planner.py","f1_simulator/controllers/baseline.py",
                                 "f1_simulator/controllers/ai_controller.py","f1_simulator/physics/physics.py"]},
                model_sha256=after_hash,
                planner=asdict(config),steering_unchanged=True,
                only_difference="learned pedal proposals versus neutral proposals before the same speed-tracking rules",
                **results)
    both_valid=all(r["valid_lap"] for r in results.values())
    report["valid_hybrid_time_gain_s"]=(results["planner_only"]["lap_time"]-results["hybrid"]["lap_time"]
                                        if both_valid else None)
    (folder/"benchmark.json").write_text(json.dumps(report,indent=2)+"\n")
    lines=["# Does the learned pedal model help?","",
           f"Matched {args.track.title()} ablation: same track, start at rest, physics, steering, curvature observations, planner and speed-tracking rules.",
           f"Model trained on {training_circuit.title()} only. Test circuit unseen to model: {unseen}. No retraining or planner tuning.",
           f"Timestep: {args.dt} s. Model horizon: {args.horizon} m. Planner horizon: 500 m.","",
           "| Controller | Time (s) | Off-track events | Valid lap |","|---|---:|---:|---|"]
    for name,r in results.items():
        time=f"{r['lap_time']:.1f}" if r['lap_time'] is not None else "timeout"
        lines.append(f"| {name} | {time} | {r['off_track_events']} | {r['valid_lap']} |")
    lines += ["", "Planner-only never calls a trained predictor: neutral pedal proposals leave all acceleration and braking to the shared rule-based target-speed tracker.",
              "The hybrid adds the saved model's pedal suggestions before those same overrides. No model was retrained."]
    if both_valid:
        gain=report["valid_hybrid_time_gain_s"]
        if gain > .1:
            lines.append(f"The hybrid was {gain:.1f} s faster on this matched run. This is a limited simulator result, not proof of generalization.")
        elif gain < -.1:
            lines.append(f"The planner-only controller was {-gain:.1f} s faster. This experiment does not show a lap-time benefit from learned pedals.")
        else:
            lines.append("The lap times were effectively equal at this timestep; this experiment does not establish a meaningful benefit from learned pedals.")
    else:
        lines.append("At least one controller failed to finish a valid lap, so an invalid time is not counted as an improvement.")
    lines += ["", "These are deterministic single-condition tests, not statistical confidence estimates or broad evidence of generalization.",
              "The planner can query the evaluation circuit's geometry; unseen means excluded from model training, not hidden from the planner.",
              "No real telemetry from the evaluation circuit is used. This is a closed-loop simulator test, not held-out real-driver prediction accuracy.",
              "The clean laps demonstrate the full controller's behavior, not independently learned corner speeds.",
              "Next learning experiment: imitate successful simulator target speeds/accelerations, with separate rollout/circuit evaluation. Teacher imitation is not independent discovery."]
    text="\n".join(lines)+"\n"
    (folder/"REPORT.md").write_text(text)
    print(text,flush=True)


if __name__ == "__main__":
    main()
