"""Create reproducible report visuals from saved measurements and control replays."""
import csv
import hashlib
import json
import math
import os
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER","dummy")
os.environ.setdefault("SDL_AUDIODRIVER","dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT","1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pygame
from PIL import Image
from scipy.signal import find_peaks

from track import Track
from f1_simulator.core import simulator as sim
from f1_simulator.core.imitation import load_model,prepare_training_data,_to_matrix,feature_names
from f1_simulator.ui import renderer

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"output/analysis_pack"
FIG=OUT/"figures"
MEDIA=OUT/"media"
MANIFEST={"method":"Recorded-control replay, not invented paths or fresh model predictions", "runs":{},"media":[]}
COLORS={"blue":"#2464a0","red":"#c54a43","green":"#218673","gray":"#8995a3"}
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":11,"axes.spines.top":False,"axes.spines.right":False})


def save_figure(fig,name):
    path=FIG/name
    fig.savefig(path,dpi=180,bbox_inches="tight",facecolor="white")
    plt.close(fig)
    return path


def replay(track,trace_path,expected,key):
    records=list(csv.DictReader(trace_path.open()))
    state,info=sim.reset_simulation(track)
    samples=[];max_speed_error=0.
    for row in records:
        error=abs(state["speed"]-float(row["speed_mps"]))
        max_speed_error=max(max_speed_error,error)
        action={k:float(row[k]) for k in ("steering","throttle","brake")}
        sample=dict(time=sim._session["time"],state=state.copy(),session=sim._session.copy(),
                    s=info["distance"],off_track=info["off_track"],action=action,
                    target=float(row["target_speed_mps"]) if row.get("target_speed_mps") else None)
        samples.append(sample)
        state,info=sim.simulate_step(state,action,.1,track)
    samples.append(dict(time=sim._session["time"],state=state.copy(),session=sim._session.copy(),
                        s=info["distance"],off_track=info["off_track"],action=dict(steering=0.,throttle=0.,brake=0.),target=None))
    if max_speed_error>1e-7:
        raise AssertionError(f"Replay diverged from recorded speeds: {key}, {max_speed_error}")
    if sim._session["off_events"] != expected["off_track_events"] or bool(info["lap_complete"]) != expected["completed_lap"]:
        raise AssertionError(f"Replay race results differ: {key}")
    if abs(samples[-1]["time"]-expected["elapsed_time"])>1e-7:
        raise AssertionError(f"Replay duration differs: {key}")
    MANIFEST["runs"][key]=dict(source=str(trace_path.relative_to(ROOT)),
        sha256=hashlib.sha256(trace_path.read_bytes()).hexdigest(),n_steps=len(records),
        speed_replay_max_error_mps=max_speed_error,expected_result=expected)
    frame=pd.DataFrame([dict(time=s["time"],x=s["state"]["x"],y=s["state"]["y"],
                            heading=s["state"]["heading"],speed_mps=s["state"]["speed"],
                            s=s["s"],off_track=s["off_track"],target_speed_mps=s["target"]) for s in samples])
    frame.to_csv(OUT/f"{key}_replayed_trajectory.csv",index=False)
    print(f"Verified {key}: {len(records)} steps, max speed error {max_speed_error:.2g}",flush=True)
    return samples


def native_screenshot(track,sample,name,label):
    sim._session.clear();sim._session.update(sample["session"])
    renderer.render(track,sample["state"],controller_name=label,zoom=6.,action=sample["action"])
    pygame.image.save(pygame.display.get_surface(),str(MEDIA/name))
    MANIFEST["media"].append(dict(file=name,type="native simulator screenshot",time_s=sample["time"],circuit=track.name))


def annotated_frame(track,context,sample,title,view="overview",center=None,zoom=5.,playback="1x"):
    surface=pygame.Surface((1100,750))
    state=sample["state"]
    if view=="overview":
        center,zoom=context["center"],context["scale"]*.88
        if "export_background" not in context:
            background=pygame.Surface((1100,750));background.fill((30,85,55))
            renderer._draw_road(background,context["boundary"],context["centerline"],center,zoom,False)
            context["export_background"]=background
        surface.blit(context["export_background"],(0,0))
    else:
        surface.fill((30,85,55))
        renderer._draw_road(surface,context["boundary"],context["centerline"],center,zoom,True)
    point=np.array([state["x"],state["y"]])
    pixel=renderer._pixels(point,center,zoom)
    sprite=context["cars"][0]
    sprite=pygame.transform.smoothscale(sprite,(max(7,round(2*zoom)),max(18,round(6*zoom))))
    sprite=pygame.transform.rotate(sprite,math.degrees(state["heading"])-90)
    surface.blit(sprite,sprite.get_rect(center=pixel))
    if view=="overview":
        pygame.draw.circle(surface,(255,211,61),pixel,6,2)
    pygame.draw.rect(surface,(17,38,35),(0,0,1100,58))
    font=pygame.font.SysFont("consolas",24,bold=True)
    small=pygame.font.SysFont("consolas",20)
    surface.blit(font.render(title,True,(255,255,255)),(20,15))
    pygame.draw.rect(surface,(17,38,35),(12,72,325,145),border_radius=7)
    target="n/a" if sample["target"] is None else f"{sample['target']*3.6:.0f} km/h"
    lines=[f"Time: {sample['time']:.1f} s | {playback} playback",
           f"Speed: {state['speed']*3.6:.1f} km/h",
           f"Planner target: {target}",
           f"Off track: {sample['off_track']}",
           f"Departures so far: {sample['session']['off_events']}"]
    for i,text in enumerate(lines):
        color=(255,169,145) if sample["off_track"] else (237,245,241)
        surface.blit(small.render(text,True,color),(24,83+24*i))
    pygame.draw.rect(surface,(17,38,35),(0,683,1100,67))
    action=sample["action"]
    surface.blit(small.render(f"Recorded controls replay | throttle {action['throttle']:.0%} | brake {action['brake']:.0%}",True,(255,255,255)),(20,695))
    explanation=("Learned steering + pedals; NO planner, steering formula or launch override."
                 if "full-action" in title.lower() else
                 "Learned pedals + arbitration; NO speed planner; steering is rule-based."
                 if "without speed planner" in title else
                 "Model proposes pedals; planner sets speed limits; steering is rule-based.")
    surface.blit(small.render(explanation,True,(210,223,216)),(20,720))
    return Image.frombytes("RGB",surface.get_size(),pygame.image.tostring(surface,"RGB")).resize((880,600),Image.Resampling.LANCZOS)


def save_gif(frames,name,duration,metadata):
    # One palette across the entire clip avoids frame-to-frame color flicker.
    sheet=Image.new("RGB",(frames[0].width*3,frames[0].height))
    for i,j in enumerate([0,len(frames)//2,len(frames)-1]):sheet.paste(frames[j],(i*frames[0].width,0))
    palette=sheet.quantize(colors=128)
    indexed=[f.quantize(palette=palette,dither=Image.Dither.NONE) for f in frames]
    path=MEDIA/name
    indexed[0].save(path,save_all=True,append_images=indexed[1:],duration=duration,loop=0,disposal=2,optimize=False)
    MANIFEST["media"].append(dict(file=name,frames=len(frames),frame_duration_ms=duration,**metadata))
    # Contact sheet is a QA artifact, not a claim of additional observations.
    preview=Image.new("RGB",(880,600),(255,255,255))
    for n,j in enumerate(np.linspace(0,len(frames)-1,4,dtype=int)):
        preview.paste(frames[j].resize((440,300)),((n%2)*440,(n//2)*300))
    preview.save(MEDIA/(path.stem+"_contact.png"))
    print(f"Saved {name}: {len(frames)} frames, {path.stat().st_size/1e6:.1f} MB",flush=True)


def closest_index(samples,s,length):
    values=np.array([sample["s"] for sample in samples])
    distance=np.abs((values-s+length/2)%length-length/2)
    return int(distance.argmin())


def export_circuit(track,samples):
    name=track.name
    role="training circuit" if name=="monza" else "unseen to trained model"
    native_screenshot(track,samples[len(samples)//3],f"{name}_follow_screenshot.png","hybrid")
    context=renderer._renderer
    times=np.array([s["time"] for s in samples])
    indices=np.unique(np.r_[np.searchsorted(times,np.arange(0,times[-1],2.)),len(samples)-1])
    frames=[annotated_frame(track,context,samples[i],f"{name.title()} | hybrid | {role}",playback="10x") for i in indices]
    save_gif(frames,f"{name}_full_lap_10x.gif",200,dict(circuit=name,playback_rate="10x; 2 simulated seconds per 0.2 s frame",source="verified hybrid replay"))
    del frames
    ss=np.linspace(0,track.length,2000,endpoint=False)
    curvature=np.abs([track.curvature(s) for s in ss])
    peaks,_=find_peaks(curvature,distance=80)
    selected=[]
    for peak in sorted(peaks,key=lambda i:curvature[i],reverse=True):
        if all(abs((ss[peak]-s+track.length/2)%track.length-track.length/2)>500 for s in selected):
            selected.append(float(ss[peak]))
        if len(selected)==2:break
    for number,s in enumerate(selected,1):
        apex=closest_index(samples,s,track.length)
        start,end=max(0,apex-40),min(len(samples),apex+41)
        center=np.array(track.position(s))
        frames=[annotated_frame(track,context,samples[i],
                    f"{name.title()} | turn {number} near s={s:.0f} m | hybrid",view="fixed",center=center,zoom=5.)
                for i in range(start,end)]
        save_gif(frames,f"{name}_turn_{number}_1x.gif",100,dict(circuit=name,playback_rate="1x",centerline_station_m=s,
                   selection="largest absolute-curvature peaks separated by at least 500 m",source="verified hybrid replay"))
        frames[len(frames)//2].save(MEDIA/f"{name}_turn_{number}.png")
        native_screenshot(track,samples[apex],f"{name}_turn_{number}_screenshot.png","hybrid")
        del frames
    renderer.close_renderer()
    return selected


def maps_plot(tracks):
    fig,axes=plt.subplots(1,2,figsize=(12,8),layout="constrained")
    for ax,track in zip(axes,tracks):
        ss=np.linspace(0,track.length,1800)
        c=np.array([track.position(s) for s in ss]);b=np.array([track.boundaries(s) for s in ss])
        ax.plot(*b[:,0].T,color=COLORS["green"],lw=.7,label="Left boundary")
        ax.plot(*b[:,1].T,color=COLORS["red"],lw=.7,label="Right boundary")
        ax.plot(*c.T,color=COLORS["blue"],lw=.8,label="Centerline")
        ax.scatter(*track.position(0),marker="*",color="#182f40",s=80,label="Dataset start")
        for s in np.linspace(0,track.length,14,endpoint=False):
            x,y=track.position(s);h=track.heading(s)
            ax.arrow(x,y,45*np.cos(h),45*np.sin(h),width=2,head_width=14,color="#253e50",length_includes_head=True)
        role="Training circuit" if track.name=="monza" else "Unseen to model; geometry available to planner"
        ax.set(title=f"{track.name.title()} | {track.length:,.1f} m\n{role}",xlabel="Local x (m)",ylabel="Local y (m)")
        ax.set_aspect("equal");ax.grid(alpha=.15)
    axes[0].legend(loc="best",fontsize=8)
    fig.suptitle("Same Monza-trained model; two distinct track geometries",fontsize=16)
    save_figure(fig,"01_track_maps.png")


def results_plots():
    transfer=json.loads((ROOT/"output/transfer_comparison.json").read_text())
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout="constrained")
    for ax,r in zip(axes,transfer["results"]):
        values=[r["planner_only"]["lap_time"],r["hybrid"]["lap_time"]]
        bars=ax.bar(["Planner only","Planner + learned pedals"],values,color=[COLORS["gray"],COLORS["blue"]],width=.6)
        for bar,value in zip(bars,values):ax.text(bar.get_x()+bar.get_width()/2,value+3,f"{value:.1f} s\n0 departures",ha="center",fontsize=10)
        ax.set(ylim=(0,max(values)*1.25),ylabel="Valid lap time (s); lower is better",
               title=f"{r['circuit']} | gain {r['gain_s']:.1f} s ({r['gain_percent']:.1f}%)")
    fig.suptitle("Ablation: learned pedals add a small benefit beyond the shared planner",fontsize=14)
    save_figure(fig,"02_valid_lap_ablation.png")
    source=json.loads((ROOT/"output/lookahead_comparison.json").read_text())
    rows=source["results"]
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout="constrained")
    labels=[f"{r['horizon_m']} m" for r in rows]
    values=[r["lap_time"] or 0 for r in rows]
    axes[0].bar(labels,values,color=[COLORS["red"]]*3+[COLORS["gray"]])
    for i,r in enumerate(rows):
        text=f"{r['lap_time']:.1f} s\nINVALID" if r["lap_time"] is not None else "TIMEOUT\n883 m / 600 s"
        axes[0].text(i,values[i]+4,text,ha="center",fontsize=9)
    axes[0].set(ylim=(0,155),ylabel="Circuit completion time (s)",title="Original raw-pedal models; none valid")
    axes[1].bar(labels,[r["off_track_events"] for r in rows],color=COLORS["red"])
    axes[1].set(ylabel="Off-track events",title="Zero departures alone is insufficient: 1,000 m stalled")
    fig.suptitle("Looking farther ahead did not solve closed-loop driving",fontsize=15)
    save_figure(fig,"03_horizon_failures.png")


def data_analysis():
    raw=[];aligned=[]
    for lap in (5,11,14):
        frame=pd.read_csv(ROOT/f"telemetry/data/monza_three_new_laps/2025_Monza_Q_VER_lap{lap:02d}.csv")
        frame["lap_id"]=lap;raw.append(frame)
        data=pd.read_csv(ROOT/f"output/pipeline_500m/lap{lap:02d}_track_relative.csv")
        data["lap_id"]=lap;aligned.append(data)
    raw=pd.concat(raw,ignore_index=True);aligned=pd.concat(aligned,ignore_index=True)
    profile=[]
    for data,stage in [(raw,"clean saved telemetry"),(aligned,"aligned records")]:
        for column in data.columns:
            values=data[column]
            profile.append(dict(stage=stage,attribute=column,type=str(values.dtype),n=len(values),
                missing_percent=float(values.isna().mean()*100),unique_values=int(values.nunique()),
                minimum=float(values.min()),median=float(values.median()),maximum=float(values.max()),
                std=float(values.std())))
    pd.DataFrame(profile).to_csv(OUT/"attribute_profile.csv",index=False)
    fig,axes=plt.subplots(2,2,figsize=(11,8),layout="constrained")
    axes[0,0].hist(raw.speed*3.6,bins=35,color=COLORS["blue"]);axes[0,0].set(xlabel="Speed (km/h)",ylabel="Samples",title="Qualifying states rarely resemble a standing start")
    axes[0,1].hist(raw.throttle,bins=25,color=COLORS["green"]);axes[0,1].set(xlabel="Throttle (0-1)",ylabel="Samples",title="Bounded, non-Gaussian control labels")
    axes[1,0].bar(["Brake off","Brake on"],[(raw.brake==0).sum(),(raw.brake==1).sum()],color=[COLORS["gray"],COLORS["red"]]);axes[1,0].set(ylabel="Samples",title=f"Brake-on share: {raw.brake.mean():.1%}; accuracy needs a baseline")
    axes[1,1].hist(aligned.lateral_offset,bins=35,color=COLORS["blue"]);axes[1,1].set(xlabel="Aligned lateral offset (m)",ylabel="Samples",title="Racing lines differ from the centerline")
    fig.suptitle(f"EDA: {len(raw):,} cleaned samples from 3 Monza laps (one session)",fontsize=15)
    save_figure(fig,"04_data_distributions.png")
    model=load_model(ROOT/"output/pipeline_500m/imitation_model.joblib");model.n_jobs=1
    train=aligned[aligned.lap_id.isin([5,11])].to_dict("records")
    X,y=prepare_training_data(train,Track("monza"),model.lookahead_offsets_)
    matrix=_to_matrix(X,model.lookahead_offsets_);names=feature_names(model.lookahead_offsets_)
    data=pd.DataFrame(matrix,columns=names);data["throttle"]=[r["throttle"] for r in y];data["brake"]=[r["brake"] for r in y]
    selected=["speed","lateral_offset","heading_error","curv_0","curv_20","curv_100","curv_250","curv_500","throttle","brake"]
    correlation=data[selected].corr();correlation.to_csv(OUT/"training_correlations.csv")
    fig,ax=plt.subplots(figsize=(10,8),layout="constrained")
    im=ax.imshow(correlation,vmin=-1,vmax=1,cmap="RdBu_r")
    ax.set_xticks(range(len(selected)),selected,rotation=60,ha="right");ax.set_yticks(range(len(selected)),selected)
    for i in range(len(selected)):
        for j in range(len(selected)):
            v=correlation.iloc[i,j];ax.text(j,i,f"{v:.2f}",ha="center",va="center",fontsize=8,color="white" if abs(v)>.6 else "black")
    fig.colorbar(im,ax=ax,label="Pearson correlation")
    ax.set_title("Training laps 5 and 11 only | correlation is not causation")
    save_figure(fig,"05_training_correlations.png")
    importance=pd.DataFrame(dict(feature=names,importance=model.feature_importances_)).sort_values("importance",ascending=False)
    importance.to_csv(OUT/"rf_feature_importance.csv",index=False)
    fig,ax=plt.subplots(figsize=(10,6),layout="constrained")
    shown=importance.head(10).iloc[::-1]
    ax.barh(shown.feature,shown.importance,color=COLORS["blue"])
    ax.set(xlabel="Random Forest impurity importance (not causal)",title="What the saved pedal model uses | correlated features share importance")
    save_figure(fig,"06_feature_importance.png")
    held=aligned[aligned.lap_id==14]
    X,y=prepare_training_data(held.to_dict("records"),Track("monza"),model.lookahead_offsets_)
    pred=np.clip(model.predict(_to_matrix(X,model.lookahead_offsets_)),0,1)
    fig,axes=plt.subplots(2,1,figsize=(12,6),sharex=True,layout="constrained")
    for dim,label in enumerate(["throttle","brake"]):
        axes[dim].plot(held.time,held[label],label="Recorded driver label",color=COLORS["gray"],lw=1.3)
        axes[dim].plot(held.time,pred[:,dim],label="Model prediction",color=COLORS["blue"],lw=1)
        axes[dim].set(ylabel=label.title()+" (0-1)",ylim=(-.05,1.05));axes[dim].legend(loc="upper right",fontsize=8)
    axes[1].set_xlabel("Lap-relative time (s)")
    fig.suptitle("Lap 14: offline prediction error is different from closed-loop driving",fontsize=14)
    save_figure(fig,"07_offline_prediction_errors.png")
    return dict(samples=len(raw),brake_on_fraction=float(raw.brake.mean()),min_speed_kmh=float(raw.speed.min()*3.6),
                largest_importance=importance.iloc[0].to_dict(),saved_csv_missing_percent=float(raw.isna().mean().mean()*100))


def cv_plot():
    path=OUT/"model_family_cv_summary.csv"
    if not path.exists():raise FileNotFoundError("Run screen_model_families.py first")
    data=pd.read_csv(path)
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout="constrained")
    for ax,metric in zip(axes,["throttle_mae","brake_mae"]):
        ax.bar(data.model,data[metric+"_mean"],yerr=data[metric+"_std"],capsize=4,
               color=[COLORS["gray"]]+[COLORS["blue"]]*4)
        ax.tick_params(axis="x",rotation=30);ax.set(ylabel="Mean absolute error (0-1)",title=metric.replace("_"," ").title())
    fig.suptitle("Model-family screening | 2-fold lap-grouped CV | bars: mean, error bars: sample SD",fontsize=13)
    save_figure(fig,"08_model_family_cv.png")


def motion_plots(track,samples):
    frame=pd.DataFrame([dict(time=s["time"],s=s["s"],speed=s["state"]["speed"]*3.6,
                            target=np.nan if s["target"] is None else s["target"]*3.6,
                            throttle=s["action"]["throttle"],brake=s["action"]["brake"]) for s in samples])
    fig,axes=plt.subplots(2,1,figsize=(12,6),sharex=True,layout="constrained")
    axes[0].plot(frame.s,frame.target,color=COLORS["red"],lw=1,label="Rule-based target")
    axes[0].plot(frame.s,frame.speed,color=COLORS["blue"],lw=1,label="Actual simulated speed")
    axes[0].set_ylabel("Speed (km/h)");axes[0].legend(loc="upper right")
    axes[1].plot(frame.s,frame.throttle,color=COLORS["green"],label="Executed throttle")
    axes[1].plot(frame.s,frame.brake,color=COLORS["red"],label="Executed brake")
    axes[1].set(xlabel="Centerline distance from dataset start (m)",ylabel="Command (0-1)");axes[1].legend(loc="upper right")
    fig.suptitle(f"{track.name.title()} hybrid: corner speed comes from the planner",fontsize=15)
    save_figure(fig,f"09_{track.name}_speed_and_pedals.png")


def main():
    for path in (OUT,FIG,MEDIA):path.mkdir(parents=True,exist_ok=True)
    tracks=[Track("monza"),Track("silverstone")]
    maps_plot(tracks);results_plots();eda=data_analysis();cv_plot()
    runs={}
    for track in tracks:
        source=ROOT/("output/planner_ablation_500m" if track.name=="monza" else "output/planner_ablation_silverstone_500m")
        benchmark=json.loads((source/"benchmark.json").read_text())
        samples=replay(track,source/"hybrid_controls.csv",benchmark["hybrid"],track.name+"_hybrid")
        runs[track.name]=samples
        motion_plots(track,samples)
        export_circuit(track,samples)
    track=tracks[0]
    source=ROOT/"output/speed_planner_500m"
    expected=json.loads((source/"benchmark.json").read_text())["corrected_imitation"]
    failed=replay(track,source/"corrected_imitation_controls.csv",expected,"monza_model_without_planner")
    first=next(i for i,s in enumerate(failed) if s["off_track"])
    sample=failed[first];s=sample["s"]
    native_screenshot(track,sample,"monza_first_departure_screenshot.png","imitation")
    context=renderer._renderer;center=np.array(track.position(s))
    frames=[annotated_frame(track,context,failed[i],"Monza | first departure | model without speed planner",
                view="fixed",center=center,zoom=4.) for i in range(max(0,first-10),min(len(failed),first+21))]
    save_gif(frames,"monza_first_departure_1x.gif",100,dict(circuit="monza",playback_rate="1x",
             selection="first off-track sample in the saved corrected-model run",source="verified recorded-action replay"))
    hybrid=runs["monza"];apex=closest_index(hybrid,s,track.length)
    frames=[annotated_frame(track,context,hybrid[i],"Monza | same location | hybrid with speed planner",
                 view="fixed",center=center,zoom=4.) for i in range(max(0,apex-30),min(len(hybrid),apex+31))]
    save_gif(frames,"monza_same_turn_with_planner_1x.gif",100,dict(circuit="monza",playback_rate="1x",source="verified hybrid replay",centerline_station_m=s))
    fig,ax=plt.subplots(figsize=(9,7),layout="constrained")
    ss=np.mod(s+np.linspace(-100,100,400),track.length);bounds=np.array([track.boundaries(q) for q in ss])
    for side in (0,1):ax.plot(*bounds[:,side].T,color=COLORS["gray"],lw=2)
    for label,samples,index,color in [("Without planner: first departure",failed,first,COLORS["red"]),("With planner: same location",hybrid,apex,COLORS["blue"])]:
        nearby=[sample for sample in samples if abs((sample["s"]-s+track.length/2)%track.length-track.length/2)<100]
        points=np.array([[sample["state"]["x"],sample["state"]["y"]] for sample in nearby])
        ax.plot(*points.T,color=color,label=label,lw=2)
    ax.scatter(sample["state"]["x"],sample["state"]["y"],color=COLORS["red"],s=35,zorder=5)
    ax.set_aspect("equal",adjustable="datalim");ax.legend(loc="upper left",fontsize=9)
    ax.set(xlabel="Local x (m)",ylabel="Local y (m)",title="Error analysis: fast completion can hide a track departure")
    save_figure(fig,"10_first_departure_comparison.png")
    renderer.close_renderer()
    MANIFEST["eda"]=eda
    (OUT/"manifest.json").write_text(json.dumps(MANIFEST,indent=2)+"\n")
    print("Analysis visuals exported",flush=True)


if __name__ == "__main__":
    main()
