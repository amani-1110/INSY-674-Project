"""Full-action expert imitation with dataset aggregation on Monza only."""
from f1_simulator.paths import PROJECT_ROOT
import argparse
import json
from pathlib import Path
import warnings

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from f1_simulator.core.track import Track
from f1_simulator.controllers.speed_planner import make_planner_controller
from f1_simulator.core.full_policy import OFFSETS,LearnedDriver,model_features
from f1_simulator.core import simulator as sim

ROOT = PROJECT_ROOT
OUT=ROOT/"output/learned_driver"


def fit_models(Xs,Xp,ys,yp,iteration):
    print(f"Fit iteration {iteration}: {len(ys)} expert-labelled states",flush=True)
    steering=make_pipeline(StandardScaler(),MLPRegressor(hidden_layer_sizes=(32,32),activation="tanh",
                  early_stopping=True,max_iter=400,n_iter_no_change=20,tol=1e-7,random_state=0,alpha=1e-5))
    pedals=ExtraTreesRegressor(n_estimators=100,min_samples_leaf=2,n_jobs=4,random_state=0)
    with threadpool_limits(limits=1):
        steering.fit(Xs,ys)
        pedals.fit(Xp,yp)
    pedals.n_jobs=1
    bundle=dict(schema="full_action_v1",offsets=OFFSETS,steering_model=steering,pedal_model=pedals,
                metadata=dict(training_circuits=["monza"],method="full-action behavior cloning" if iteration==0 else "behavior cloning + dataset aggregation",
                              teacher="rule-based planner-only driver",iteration=iteration,
                              runtime_planner=False,runtime_rule_steering=False,runtime_launch_rule=False))
    joblib.dump(bundle,OUT/f"policy_iteration_{iteration}.joblib",compress=3)
    return bundle


def teacher_action(teacher,obs):
    return teacher(obs)


def generate_dataset(track,teacher,count,seed=0):
    rng=np.random.default_rng(seed)
    s_grid=np.linspace(0,track.length,int(track.length/2)+1)
    k_grid=np.asarray([track.curvature(s) for s in s_grid])
    stations=rng.uniform(0,track.length,count)
    curvatures=np.stack([np.interp(np.mod(stations+d,track.length),s_grid,k_grid) for d in OFFSETS],axis=1)
    # Perturbations supply recovery and launch examples absent from qualifying laps.
    speed=rng.uniform(0,50,count)
    lateral=rng.uniform(-3.5,3.5,count)
    heading=rng.uniform(-.45,.45,count)
    # Many centered states improve precision around successful driving trajectories.
    centered=np.arange(count)%2==0
    lateral[centered]=rng.normal(0,.65,centered.sum())
    heading[centered]=rng.normal(0,.08,centered.sum())
    Xs=[];Xp=[];ys=[];yp=[]
    for i in range(count):
        obs=dict(speed=speed[i],lateral_offset=lateral[i],heading_error=heading[i],curvature_ahead=curvatures[i].tolist())
        action=teacher_action(teacher,obs)
        a,b=model_features(obs);Xs.append(a);Xp.append(b)
        ys.append(action["steering"]/.4);yp.append(action["throttle"]-action["brake"])
    return np.asarray(Xs),np.asarray(Xp),np.asarray(ys),np.asarray(yp)


def collect_rollout(bundle,track,teacher,iteration):
    policy=LearnedDriver(bundle,track)
    state,info=sim.reset_simulation(track)
    extra=[[],[],[],[]];trace=[]
    for step in range(4500):
        obs=policy.observe(state,track)
        expert=teacher_action(teacher,obs)
        a,b=model_features(obs)
        extra[0].append(a);extra[1].append(b)
        extra[2].append(expert["steering"]/.4);extra[3].append(expert["throttle"]-expert["brake"])
        action=policy(obs)
        trace.append(dict(time=step*.1,speed_mps=state["speed"],lateral_offset_m=obs["lateral_offset"],
                          heading_error_rad=obs["heading_error"],**action))
        state,info=sim.simulate_step(state,action,.1,track)
        if info["off_track"] or info["lap_complete"]:
            break
    result=dict(iteration=iteration,elapsed_time=sim._session["time"],
                completed_lap=info["lap_complete"],off_track=info["off_track"],
                off_track_events=sim._session["off_events"],progress_m=sim._session["progress"],
                speed_mps=state["speed"])
    import pandas as pd
    pd.DataFrame(trace).to_csv(OUT/f"training_rollout_{iteration}.csv",index=False)
    print("Training rollout",result,flush=True)
    return tuple(np.asarray(v) for v in extra),result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples",type=int,default=40000)
    parser.add_argument("--iterations",type=int,default=4)
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    track=Track("monza")
    # Dense observations are shared between expert labels and student features.
    teacher=make_planner_controller(lookahead_offsets=OFFSETS)
    print("Generate Monza expert labels",flush=True)
    Xs,Xp,ys,yp=generate_dataset(track,teacher,args.samples)
    history=[]
    for iteration in range(args.iterations):
        bundle=fit_models(Xs,Xp,ys,yp,iteration)
        extra,result=collect_rollout(bundle,track,teacher,iteration)
        history.append(result)
        if result["completed_lap"] and not result["off_track"]:
            joblib.dump(bundle,ROOT/"models/full_control.joblib",compress=3)
            break
        # Reweight visited-state corrections to address covariate shift.
        Xs=np.concatenate([Xs,np.repeat(extra[0],8,axis=0)])
        Xp=np.concatenate([Xp,np.repeat(extra[1],8,axis=0)])
        ys=np.concatenate([ys,np.repeat(extra[2],8,axis=0)])
        yp=np.concatenate([yp,np.repeat(extra[3],8,axis=0)])
    else:
        joblib.dump(bundle,ROOT/"models/full_control.joblib",compress=3)
    (OUT/"training.json").write_text(json.dumps(dict(method="Full-action behavior cloning" if len(history)==1 else "Full-action cloning + dataset aggregation",
        aggregation_refits=len(history)-1,
        teacher="Planner-only controller; hand-written expert used only for training labels",
        training_circuits=["monza"],samples_initial=args.samples,history=history,
        steering="StandardScaler + tanh MLP (32,32)",pedals="100 ExtraTrees; signed throttle/brake",
        reinforcement_learning=False,model_selected_using="Monza training rollouts only",
        runtime_assistance="No planner, steering rule, launch override or teacher calls"),indent=2)+"\n")
    print("Saved",ROOT/"models/full_control.joblib",flush=True)


if __name__ == "__main__":
    main()
