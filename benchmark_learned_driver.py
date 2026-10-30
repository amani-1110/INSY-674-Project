"""Benchmark a frozen full-action learned policy with all driving helpers disabled."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

from track import Track
from f1_simulator.core.full_policy import load_learned_driver
from data_eval.evaluation import evaluate_controller

ROOT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track",default="monza")
    parser.add_argument("--dt",type=float,default=.1)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    model_path=ROOT/"output/learned_driver/policy.joblib"
    before=hashlib.sha256(model_path.read_bytes()).hexdigest()
    track=Track(args.track);policy=load_learned_driver(model_path,track)
    folder=args.output or ROOT/f"output/learned_driver/{args.track}"
    folder.mkdir(parents=True,exist_ok=True)
    trace=[]
    def logged(obs):
        action=policy(obs)
        trace.append(dict(time=len(trace)*args.dt,speed_mps=obs["speed"],target_speed_mps=None,
                          lateral_offset_m=obs["lateral_offset"],heading_error_rad=obs["heading_error"],**action))
        return action
    logged.observe=policy.observe
    logged.lookahead_offsets=policy.lookahead_offsets
    # Fail loudly if any forbidden driving helper is accidentally invoked.
    import data_eval.baseline as baseline
    import data_eval.speed_planner as planner
    import data_eval.ai_controller as pedals
    def forbidden(*args,**kwargs):
        raise AssertionError("A hand-written driving helper was called during learned-policy evaluation")
    original=(baseline.baseline_action,planner.target_speed,pedals.resolve_pedals)
    baseline.baseline_action=planner.target_speed=pedals.resolve_pedals=forbidden
    try:
        result=evaluate_controller(logged,track,dt=args.dt,max_time=600.,controller_name="full_action_learned")
    finally:
        baseline.baseline_action,planner.target_speed,pedals.resolve_pedals=original
    if hashlib.sha256(model_path.read_bytes()).hexdigest()!=before:
        raise AssertionError("Policy changed during testing")
    with (folder/"controls.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=trace[0]);writer.writeheader();writer.writerows(trace)
    report=dict(track=args.track,dt=args.dt,max_time=600.,model_sha256=before,
                training_circuits=policy.metadata["training_circuits"],
                unseen_to_training=args.track not in policy.metadata["training_circuits"],
                helpers_disabled_and_guarded=True,runtime_speed_planner=False,runtime_rule_steering=False,
                runtime_launch_rule=False,training_method=policy.metadata["method"],result=result)
    (folder/"benchmark.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2),flush=True)


if __name__ == "__main__":
    main()
