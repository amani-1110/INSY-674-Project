"""Grouped tuning and ensemble comparison; frozen third-circuit evaluation."""
from pathlib import Path
import json, hashlib, warnings, urllib.request, argparse
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import joblib
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import GroupKFold, ParameterSampler
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from track import Track
from f1_simulator.core.imitation import prepare_training_data, offsets_for_horizon, _to_matrix, _targets, predict_action
from data_eval.speed_planner import make_hybrid_controller, make_planner_controller
from data_eval.evaluation import evaluate_controller
from f1_simulator.core.full_policy import load_learned_driver

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'output/completed_study'

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    global OUT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUT,
                        help='Use a new directory for reproductions; existing final-test records are preserved.')
    OUT=parser.parse_args().output.resolve()
    OUT.mkdir(exist_ok=True)
    if (OUT/'final_test.json').exists():
        raise SystemExit('Completed results already exist. Preserve the original protocol and final test; use a separate output directory for a reproduction.')
    records=[]
    for lap in (5,11):
        frame=pd.read_csv(ROOT/f'output/pipeline_500m/lap{lap:02d}_track_relative.csv')
        frame['lap_id']=str(lap)
        records.extend(frame.to_dict('records'))
    offsets=offsets_for_horizon(500)
    x,y=prepare_training_data(records,Track('monza'),offsets)
    x,y=_to_matrix(x,offsets),_targets(y)
    groups=np.array([r['lap_id'] for r in records])
    folds=list(GroupKFold(2).split(x,y,groups))
    # Small seeded searches avoid spending a large budget on only two independent laps.
    families={
        'Random Forest':(RandomForestRegressor(n_estimators=200,random_state=0,n_jobs=1),{'min_samples_leaf':[2,5,10],'max_features':[0.7,1.0],'max_depth':[None,16]}),
        'Support vector':(make_pipeline(StandardScaler(),MultiOutputRegressor(SVR())),{'multioutputregressor__estimator__C':[0.5,1.,5.,10.],'multioutputregressor__estimator__epsilon':[.02,.05,.1],'multioutputregressor__estimator__gamma':['scale',.03,.1]}),
        'Neural network':(make_pipeline(StandardScaler(),MLPRegressor(max_iter=1500,random_state=0)),{'mlpregressor__hidden_layer_sizes':[(32,16),(32,32),(64,32)],'mlpregressor__alpha':[.0001,.01,.1],'mlpregressor__learning_rate_init':[.0003,.001]})}
    rows=[]; best={}; summaries=[]
    def score(est):
        values=[]; warning_count=0
        for tr,va in folds:
            assert not set(groups[tr]) & set(groups[va])
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always',ConvergenceWarning)
                m=clone(est).fit(x[tr],y[tr])
            warning_count+=sum(issubclass(w.category,ConvergenceWarning) for w in caught)
            pred=np.clip(m.predict(x[va]),0,1)
            values.append(np.abs(pred-y[va]).mean(axis=0))
        return np.array(values),warning_count
    with threadpool_limits(limits=1):
        for name,(base,space) in families.items():
            trials=[]
            for i,params in enumerate(ParameterSampler(space,n_iter=6,random_state=0)):
                est=clone(base).set_params(**params); errors,nwarn=score(est)
                row=dict(model=name,candidate=i,params=params,combined_mae=float(errors.mean()),throttle_mae=float(errors[:,0].mean()),brake_mae=float(errors[:,1].mean()),fold_errors=errors.tolist(),convergence_warnings=nwarn)
                rows.append(row); trials.append((row,est))
                print(name,i,row['combined_mae'],flush=True)
            selected,est=min(trials,key=lambda pair:pair[0]['combined_mae'])
            best[name]=est; summaries.append(selected)
        # Equal blend weights fixed in advance. Each output receives its own RF/SVR blend.
        svr=best['Support vector'].get_params()['multioutputregressor__estimator']
        blend=MultiOutputRegressor(VotingRegressor([('rf',clone(best['Random Forest'])),('svr',make_pipeline(StandardScaler(),clone(svr)))]))
        errors,nwarn=score(blend)
        summaries.append(dict(model='Equal RF and SVR ensemble',combined_mae=float(errors.mean()),throttle_mae=float(errors[:,0].mean()),brake_mae=float(errors[:,1].mean()),fold_errors=errors.tolist(),convergence_warnings=nwarn,params={'weights':[0.5,0.5]}))
        best['Equal RF and SVR ensemble']=blend
        winner=min(summaries,key=lambda r:r['combined_mae'])['model']
        for name,est in best.items():
            est.fit(x,y); est.lookahead_offsets_=offsets
            joblib.dump(est,OUT/(name.replace(' ','_')+'.joblib'))
    report=dict(training_laps=[5,11],excluded_from_selection=[14],selection_metric='mean throttle/brake MAE; clipped to [0,1]',search='6 seeded random candidates per family; 2-fold whole-lap GroupKFold',rationale='Tree leaf/depth/feature limits vary smoothness; SVR C/epsilon/gamma vary flexibility and tolerance; network width/regularization/learning rate vary capacity and fitting. Budget limited by two independent training laps.',caveat='Selection scores reuse tuning folds and are optimistic, not independent test estimates; fresh circuit evaluates driving, not real telemetry prediction.',trials=rows,summary=summaries,selected=winner)
    (OUT/'tuning.json').write_text(json.dumps(report,indent=2)+'\n')
    pd.DataFrame([{k:v for k,v in r.items() if k!='fold_errors'} for r in summaries]).to_csv(OUT/'comparison.csv',index=False)
    # Record the complete protocol before retrieving or evaluating Spa geometry.
    models={name:OUT/(name.replace(' ','_')+'.joblib') for name in best}
    full=ROOT/'output/learned_driver/policy.joblib'
    protocol=dict(created_utc=datetime.now(timezone.utc).isoformat(),circuit='spa',evaluation_kind='reproduction on an already inspected circuit' if (ROOT/'data/spa.csv').exists() else 'initial fresh-circuit evaluation',source_commit='e59595d1f3573b30d1ded6a08984935b957688e0',source_path='tracks/Spa.csv',dt=.1,max_time=600,start='default start from rest',selection=winner,model_hashes={n:sha(p) for n,p in models.items()},full_action_hash=sha(full),primary_comparison='training-CV selected candidate versus planner-only; frozen full-action policy separately',all_candidates_reported=True,no_post_test_tuning=True,success='lap complete and zero departures; compare time only when both valid')
    (OUT/'final_test_protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    path=ROOT/'data/spa.csv'
    if not path.exists():
        url=f"https://raw.githubusercontent.com/TUMFTM/racetrack-database/{protocol['source_commit']}/tracks/Spa.csv"
        path.write_bytes(urllib.request.urlopen(url,timeout=45).read())
    source=dict(repository='https://github.com/TUMFTM/racetrack-database',commit=protocol['source_commit'],source_path='tracks/Spa.csv',sha256=sha(path),license='LGPL-3.0; see TUM_LICENSE',modifications='None; upstream bytes preserved')
    (ROOT/'data/spa.source.json').write_text(json.dumps(source,indent=2)+'\n')
    track=Track('spa'); results={}
    controllers={'Planner only':make_planner_controller(track=track,lookahead_offsets=offsets)}
    for name,p in models.items():
        controllers[name]=make_hybrid_controller(joblib.load(p),predict_action,track=track)
    controllers['Frozen full-action driver']=load_learned_driver(full,track)
    with threadpool_limits(limits=1):
        for name,controller in controllers.items():
            print('Circuit evaluation:',name,flush=True)
            results[name]=evaluate_controller(controller,track,dt=.1,max_time=600,controller_name=name)
            (OUT/'final_test_partial.json').write_text(json.dumps(results,indent=2)+'\n')
    for n,p in models.items():
        assert sha(p)==protocol['model_hashes'][n]
    assert sha(full)==protocol['full_action_hash']
    (OUT/'final_test.json').write_text(json.dumps(dict(protocol=protocol,track_source=source,results=results,models_unchanged=True),indent=2)+'\n')
    print(json.dumps(results,indent=2),flush=True)

if __name__=='__main__':
    main()
