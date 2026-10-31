"""Run every saved driving configuration on every circuit without retraining.

This is an expanded development benchmark, not a new untouched final test.
Prepare once, then run independent circuit processes with --track.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, json
import joblib
from threadpoolctl import threadpool_limits
from track import Track
from f1_simulator.core.imitation import load_model, predict_action, offsets_for_horizon
from f1_simulator.core.full_policy import load_learned_driver
from data_eval.ai_controller import make_ai_controller
from data_eval.speed_planner import make_hybrid_controller, make_planner_controller
from benchmark_speed_planner import run_attempt

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'output/all_configurations'
TRACKS=('monza','silverstone','spa')
MODELS={
    'original_pedals':ROOT/'output/pipeline_500m/imitation_model.joblib',
    'learned_full_control':ROOT/'output/learned_driver/policy.joblib',
    'tuned_rf':ROOT/'output/completed_study/Random_Forest.joblib',
    'tuned_svr':ROOT/'output/completed_study/Support_vector.joblib',
    'tuned_mlp':ROOT/'output/completed_study/Neural_network.joblib',
    'combined_rf_svr':ROOT/'output/completed_study/Equal_RF_and_SVR_ensemble.joblib'}
CONFIGS=[
    dict(id='rules_only',name='Rules only',learns='Nothing',assistance='Rule steering, speed planner, launch and pedal cleanup',training='No fitted model'),
    dict(id='original_pedals_no_planner',name='Original RF pedals, no speed planner',learns='Throttle and brake',assistance='Rule steering, launch and pedal cleanup',training='Real Monza telemetry, laps 5 and 11'),
    dict(id='original_pedals_with_planner',name='Original RF pedals + speed planner',learns='Throttle and brake',assistance='Rule steering, speed planner, launch and pedal cleanup',training='Same original RF as no-planner driver'),
    dict(id='learned_full_control',name='Learned steering and pedals',learns='Steering, throttle and brake',assistance='Actuator limits only',training='40,000 simulated Monza teacher-labelled situations'),
    dict(id='tuned_rf',name='Tuned RF pedals + speed planner',learns='Throttle and brake',assistance='Rule steering, speed planner, launch and pedal cleanup',training='Real Monza laps 5 and 11; training-only tuning'),
    dict(id='tuned_svr',name='Tuned SVR pedals + speed planner',learns='Throttle and brake',assistance='Rule steering, speed planner, launch and pedal cleanup',training='Real Monza laps 5 and 11; training-only tuning'),
    dict(id='tuned_mlp',name='Tuned neural pedals + speed planner',learns='Throttle and brake',assistance='Rule steering, speed planner, launch and pedal cleanup',training='Real Monza laps 5 and 11; training-only tuning'),
    dict(id='combined_rf_svr',name='50/50 RF + SVR pedals + speed planner',learns='Throttle and brake',assistance='Rule steering, speed planner, launch and pedal cleanup',training='Real Monza laps 5 and 11; fixed blend weights')]

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def verify(protocol):
    for name,path in MODELS.items(): assert sha(path)==protocol['model_hashes'][name],name
    for name in TRACKS: assert sha(ROOT/f'data/{name}.csv')==protocol['track_hashes'][name],name
    for name,digest in protocol['code_hashes'].items(): assert sha(ROOT/name)==digest,name

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepare',action='store_true')
    p.add_argument('--track',choices=TRACKS)
    args=p.parse_args();OUT.mkdir(exist_ok=True)
    protocol_path=OUT/'protocol.json'
    if args.prepare:
        if protocol_path.exists(): raise SystemExit('Protocol already exists; preserve it.')
        code=['physics/physics.py','f1_simulator/core/simulator.py','f1_simulator/core/full_policy.py','f1_simulator/core/imitation.py','data_eval/speed_planner.py','data_eval/ai_controller.py','data_eval/baseline.py','data_eval/evaluation.py']
        protocol=dict(created_utc=datetime.now(timezone.utc).isoformat(),purpose='Expanded development benchmark of eight complete driving configurations on all three circuits',tracks=TRACKS,configurations=CONFIGS,dt=.1,max_time=600.,start='Standard circuit start, heading aligned with road, speed zero',model_hashes={n:sha(f) for n,f in MODELS.items()},track_hashes={n:sha(ROOT/f'data/{n}.csv') for n in TRACKS},code_hashes={n:sha(ROOT/n) for n in code},retraining=False,success='Completed lap with zero departure episodes',interpretation='Compare whole-system driving performance; only the planner-only versus learned-pedal hybrids share all rule-based controls. Full-control imitation changes the training source and steering. Offline pedal error is a different metric. These circuits have now been inspected; no claim of a new untouched final test.')
        protocol_path.write_text(json.dumps(protocol,indent=2)+'\n');print('Prepared 8 x 3 = 24 attempts.',flush=True);return
    if not args.track: p.error('Choose --prepare or --track.')
    protocol=json.loads(protocol_path.read_text());verify(protocol)
    track=Track(args.track);folder=OUT/args.track;folder.mkdir(exist_ok=True)
    offsets=offsets_for_horizon(500)
    with threadpool_limits(limits=1):
        for cfg in CONFIGS:
            dest=folder/(cfg['id']+'.json')
            if dest.exists():
                old=json.loads(dest.read_text());assert old['protocol_sha256']==sha(protocol_path)
                print(args.track,cfg['id'],'already measured',flush=True);continue
            name=cfg['id']
            if name=='rules_only':controller=make_planner_controller(track=track,lookahead_offsets=offsets)
            elif name=='learned_full_control':controller=load_learned_driver(MODELS[name],track)
            elif name.startswith('original_pedals_'):
                model=load_model(MODELS['original_pedals']);model.n_jobs=1
                controller=(make_ai_controller(model,predict_action) if name.endswith('no_planner') else make_hybrid_controller(model,predict_action,track=track))
            else:
                model=joblib.load(MODELS[name])
                controller=make_hybrid_controller(model,predict_action,track=track)
            print('RUN',args.track,name,flush=True)
            # A full-action policy must not silently use any driving-rule helper.
            if name=='learned_full_control':
                import data_eval.baseline as baseline
                import data_eval.speed_planner as planner
                import data_eval.ai_controller as pedals
                def forbidden(*a,**kw): raise AssertionError('Full-control policy called a driving helper')
                previous=(baseline.baseline_action,planner.target_speed,pedals.resolve_pedals)
                baseline.baseline_action=planner.target_speed=pedals.resolve_pedals=forbidden
                try: result=run_attempt(controller,track,name,folder,.1,600.)
                finally: baseline.baseline_action,planner.target_speed,pedals.resolve_pedals=previous
            else:result=run_attempt(controller,track,name,folder,.1,600.)
            verify(protocol)
            dest.write_text(json.dumps(dict(track=args.track,configuration=cfg,protocol_sha256=sha(protocol_path),model_files_unchanged=True,helpers_guarded=name=='learned_full_control',result=result),indent=2)+'\n')
    print('COMPLETE',args.track,flush=True)

if __name__=='__main__': main()
