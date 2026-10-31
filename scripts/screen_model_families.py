"""Exploratory model-family comparison using only the two training laps."""
from f1_simulator.paths import PROJECT_ROOT
import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from threadpoolctl import threadpool_limits

from f1_simulator.core.track import Track
from f1_simulator.core.imitation import prepare_training_data, offsets_for_horizon, _to_matrix, _targets

ROOT = PROJECT_ROOT
OUT=ROOT/"output/analysis_pack"


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if not (ROOT / "output/pipeline_500m/lap05_track_relative.csv").exists():
        raise SystemExit("Prepare data first: python -m scripts.run_pipeline --lookahead 500")
    records=[]
    for lap in (5,11):
        frame=pd.read_csv(ROOT/f"output/pipeline_500m/lap{lap:02d}_track_relative.csv")
        frame["lap_id"]=str(lap)
        records.extend(frame.to_dict("records"))
    track=Track("monza"); offsets=offsets_for_horizon(500)
    X,y=prepare_training_data(records,track,offsets)
    X,y=_to_matrix(X,offsets),_targets(y)
    groups=np.asarray([r["lap_id"] for r in records])
    models={
        "Mean baseline":DummyRegressor(strategy="mean"),
        "Ridge regression":make_pipeline(StandardScaler(),Ridge(alpha=1.)),
        "Random Forest":RandomForestRegressor(n_estimators=200,min_samples_leaf=5,random_state=0,n_jobs=1),
        "RBF support vector":make_pipeline(StandardScaler(),MultiOutputRegressor(SVR(C=1.,epsilon=.05))),
        "Neural network":make_pipeline(StandardScaler(),MLPRegressor(hidden_layer_sizes=(32,16),max_iter=1000,random_state=0)),
    }
    rows=[]
    with threadpool_limits(limits=1):
        for name,estimator in models.items():
            for fold,(train,test) in enumerate(GroupKFold(2).split(X,y,groups),1):
                model=clone(estimator)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always",ConvergenceWarning)
                    model.fit(X[train],y[train])
                p=np.clip(model.predict(X[test]),0,1)
                rows.append(dict(model=name,fold=fold,training_laps=",".join(sorted(set(groups[train]))),
                    validation_laps=",".join(sorted(set(groups[test]))),
                    throttle_mae=float(np.abs(p[:,0]-y[test,0]).mean()),
                    brake_mae=float(np.abs(p[:,1]-y[test,1]).mean()),
                    brake_accuracy=float(((p[:,1]>=.5)==y[test,1]).mean()),
                    convergence_warnings=sum(issubclass(w.category,ConvergenceWarning) for w in caught)))
            print(name,rows[-1],flush=True)
    pd.DataFrame(rows).to_csv(OUT/"model_family_cv_folds.csv",index=False)
    summaries=[]
    for name in models:
        folds=[r for r in rows if r["model"]==name]
        item=dict(model=name,n_folds=2,convergence_warnings=sum(r["convergence_warnings"] for r in folds))
        for metric in ("throttle_mae","brake_mae","brake_accuracy"):
            values=[r[metric] for r in folds]
            item[metric+"_mean"]=float(np.mean(values))
            item[metric+"_std"]=float(np.std(values,ddof=1))
        summaries.append(item)
    pd.DataFrame(summaries).to_csv(OUT/"model_family_cv_summary.csv",index=False)
    (OUT/"model_family_cv.json").write_text(json.dumps(dict(
        scope="Exploratory screening, not hyperparameter tuning or final model selection",
        training_laps=[5,11],validation_method="2-fold GroupKFold by lap; one complete lap per validation fold",
        excluded_lap=14,preprocessing="StandardScaler inside folds for Ridge/SVR/MLP; no scaling needed for trees",
        predictions="All predictions clipped to [0,1] before evaluation",folds=rows,summary=summaries),indent=2)+"\n")


if __name__ == "__main__":
    main()
