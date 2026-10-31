"""Compare corrected learned pedals and hybrid planning with the same saved model."""
from f1_simulator.paths import PROJECT_ROOT
import argparse
import csv
import hashlib
import json
from pathlib import Path

from f1_simulator.core.track import Track
from f1_simulator.core.imitation import load_model, predict_action
from f1_simulator.controllers.ai_controller import make_ai_controller
from f1_simulator.controllers.speed_planner import make_hybrid_controller
from f1_simulator.controllers.evaluation import evaluate_controller

ROOT = PROJECT_ROOT


REPORT_INTERPRETATION = "\n## What the AI learned, and what the planner supplied\n\nThe Random Forest learned to predict throttle and brake labels from recorded\nqualifying telemetry. It did not learn the hybrid controller's target corner\nspeeds. Those targets are calculated by a hand-written curvature/braking planner\nwith a conservative lateral-acceleration limit of 4 m/s^2 and a 45 m/s top-speed\ncap. For sufficiently tight corners, that rule can set a target near 30 km/h.\nA separate rule-based controller supplies steering. Pedal arbitration and the\nlow-speed launch rule are also hand-written.\n\nThe planner can override either learned pedal prediction. Therefore, the valid\n205.4 s lap demonstrates the performance of the combined hybrid controller in\nthis simulator; it is not evidence that the Random Forest independently learned\nsafe cornering or that the learned component caused the improvement. The\ncorrected learned-pedal controller with the same steering recorded 13 departures.\nThe smaller-timestep check supports consistency between these two tested\nsettings, not a general guarantee of reliability.\n\nA matched planner-only ablation is reported separately in\n../planner_ablation_500m/REPORT.md. In the initial matched Monza run, both\ncontrollers completed valid laps: planner-only took 208.7 s and the hybrid\n205.4 s. The learned pedal proposals provided a modest 3.3 s (about 1.6%) lap-time\nbenefit in that condition. This does not establish independently learned corner\nspeeds or unseen-circuit generalization; the rule-based planner supplied the\nsame speed limits to both drivers.\n\n## Could a model learn cornering speeds?\n\nThe original pedal model did not independently learn safe cornering. A follow-up full-action policy now learns steering and pedals from simulated expert labels and completes both tracks without runtime driving helpers; see ../learned_driver/REPORT.md. This is imitation of an expert, not independent discovery of optimal cornering. One\nnext experiment is to train a model to predict target speed or acceleration\nfrom curvature, speed and vehicle state using successful simulator driving as\nteacher data. Learning from this planner would be imitation of a rule-based\nteacher, not independent discovery of its speed limits. Closed-loop rollout\nand recovery-state examples would help address the mismatch between recorded\nqualifying states and the states visited by the simulator.\n\nA later reinforcement-learning experiment could optimize simulator actions\nusing forward progress, lap completion and track-departure penalties. It would\nrequire careful evaluation against shortcuts and other reward exploits; success\nis not assured. More real sessions and circuits are also needed to distinguish\ntrack memorization from transferable driving behavior. Separate circuits must\nbe reserved for evaluation, and valid lap rate and speed should both be reported.\n"

def run_attempt(controller, track, name, folder, dt, max_time):
    trace = []
    def logged(obs):
        action = controller(obs)
        trace.append(dict(time=len(trace)*dt, speed_mps=obs["speed"],
                          target_speed_mps=getattr(controller,"last_target_speed",None),
                          lateral_offset_m=obs["lateral_offset"],
                          heading_error_rad=obs["heading_error"], **action))
        return action
    logged.lookahead_offsets = controller.lookahead_offsets
    if hasattr(controller,"observe"):
        logged.observe = controller.observe
    result = evaluate_controller(logged,track,dt=dt,max_time=max_time,controller_name=name)
    with (folder/f"{name}_controls.csv").open("w", newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=trace[0].keys());writer.writeheader();writer.writerows(trace)
    print(name, result, flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon",type=int,choices=[250,500,1000],default=500)
    parser.add_argument("--dt",type=float,default=.1)
    args=parser.parse_args()
    model_path=(ROOT/"models/original_pedals.joblib" if args.horizon == 500 else ROOT/f"output/pipeline_{args.horizon}m/imitation_model.joblib")
    model=load_model(model_path);model.n_jobs=1
    folder=ROOT/f"output/speed_planner_{args.horizon}m";folder.mkdir(parents=True,exist_ok=True)
    track=Track("monza")
    corrected=make_ai_controller(model,predict_action)
    hybrid=make_hybrid_controller(model,predict_action,track=track)
    results={}
    for name, controller in [("corrected_imitation",corrected),("hybrid",hybrid)]:
        print(f"Running {name}",flush=True)
        results[name]=run_attempt(controller,track,name,folder,args.dt,600.)
    report=dict(horizon_m=args.horizon, model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
                dt=args.dt,max_time=600.,start_speed=0.,steering_unchanged=True,
                planner=hybrid.planner_config,**results)
    (folder/"benchmark.json").write_text(json.dumps(report,indent=2)+"\n")
    lines=["# Hybrid speed planner benchmark","", "Same model weights, exclusive pedal policy, steering, track, physics, start and timestep.","",
           "| Controller | Lap time (s) | Off-track events | Valid lap |","|---|---:|---:|---|"]
    for name,r in results.items():
        time=f"{r['lap_time']:.1f}" if r['lap_time'] is not None else "timeout"
        lines.append(f"| {name} | {time} | {r['off_track_events']} | {r['valid_lap']} |")
    lines += ["", "The hybrid uses dense 5 m curvature samples up to 500 m, a lateral-acceleration speed limit and backward braking constraints.",
              "Planner limits override learned pedals; acceleration toward the target can override learned braking. Deviations trigger a 12 m/s cap.",
              "This is a hybrid controller, not a retrained model. Both runs retain the original steering.",
              "An off-track completion is invalid. These are exploratory simulator results on Monza, not unseen-circuit validation."]
    (folder/"REPORT.md").write_text("\n".join(lines)+"\n"+REPORT_INTERPRETATION, encoding="utf-8")


if __name__ == "__main__":
    main()
