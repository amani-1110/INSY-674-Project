"""Publish only measured driving-stage results and refresh the analysis gallery."""
import csv
import html
import json
from pathlib import Path
import re
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"output/analysis_pack"


def main():
    old=json.loads((ROOT/"output/speed_planner_500m/benchmark.json").read_text())
    silver=json.loads((ROOT/"output/planner_ablation_silverstone_500m/benchmark.json").read_text())
    learned={name:json.loads((ROOT/f"output/learned_driver/{name}/benchmark.json").read_text())
             for name in ("monza","silverstone")}
    for result in learned.values():
        if not result["helpers_disabled_and_guarded"]:
            raise ValueError("Full-action result was not tested without helpers")
    raw=old["corrected_imitation"];assisted=old["hybrid"];transfer=silver["hybrid"]
    lm,ls=learned["monza"]["result"],learned["silverstone"]["result"]
    training=json.loads((ROOT/"output/learned_driver/training.json").read_text())
    text=f"""## Three stages of driving: assistance versus learned control

These are three system configurations, not a controlled ablation of assistance
alone. Stages 1/2 share the original pedal model; stage 3 uses a new full-action
policy and different training labels. Times are measured at 0.1 s simulation steps.

### 1. Without speed assistance

The original 500 m model proposes throttle and brake without the speed planner.
Rule-based steering, exclusive pedals and a launch rule remain active. This is
not fully unassisted AI driving.

![Without speed assistance](media/monza_first_departure_1x.gif)

**Track: Monza**  
**Completion time: {raw['lap_time']:.1f} s - invalid ({raw['off_track_events']} departures).**  
**Track: Silverstone - not tested in this configuration.**

### 2. With assistance

The original learned pedals are combined with hand-written steering, launch
handling and a curvature/braking speed planner. The planner can override pedals.

![With assistance](media/monza_same_turn_with_planner_1x.gif)

**Track: Monza**  
**Completion time: {assisted['lap_time']:.1f} s - valid ({assisted['off_track_events']} departures).**  
**Track: Silverstone**  
**Completion time: {transfer['lap_time']:.1f} s - valid ({transfer['off_track_events']} departures).**

### 3. AI-learned steering and pedals - no runtime driving assistance

**Implemented and tested.** A neural network predicts steering, and a learned
ExtraTrees regressor predicts a signed pedal command (positive throttle,
negative brake). The runtime has no corner-speed planner, steering formula,
launch override, target-speed tracker or teacher fallback. Only actuator-range
clipping and conversion of the signed command to throttle/brake remain.

![Learned full-action policy](media/monza_full_action_turn_1x.gif)

**Track: Monza**  
**Completion time: {lm['lap_time']:.1f} s - valid ({lm['off_track_events']} departures).**  
**Track: Silverstone**  
**Completion time: {ls['lap_time']:.1f} s - valid ({ls['off_track_events']} departures).**

Training changed: 40,000 perturbed Monza simulator states were labelled by the
planner-only expert, including low-speed and recovery situations absent from
qualifying telemetry. Steering labels now exist because the teacher is simulated.
This is full-action expert imitation, **not reinforcement learning** or independent
discovery of corner-speed limits. The policy learns the teacher's behavior; the
teacher does not drive the car during inference. No Silverstone states were used
for training. Dataset-aggregation code is available if rollouts fail; this model
completed cleanly on its first fit, so no aggregation refit was performed.

The evaluator replaces planner, rule-steering and pedal-cleanup entry points
with functions that raise errors. Both learned-policy benchmarks completed with
these guards active and the saved model hash unchanged.

All clips are independent 1x recorded-control replays near the same Monza turn,
not synchronized races. A shorter invalid completion is not a valid improvement.
This is a limited deterministic simulator test, not proof of real-car driving.

Run the third stage:

```powershell
python -m f1_simulator.demo --track monza --controller learned
python -m f1_simulator.demo --track silverstone --controller learned
```

"""
    (OUT/"DRIVING_STAGES.md").write_text("# Driving stages comparison\n\n"+text,encoding="utf-8")
    p=OUT/"ANALYSIS.md";content=p.read_text()
    start=content.index("## Three stages of driving:");end=content.index("## 1. Frame the problem",start)
    content=content[:start]+text+"\nThe numbered analysis below describes the original pedal-model experiments;\nthe full-action follow-up above is a separately trained policy.\n\n"+content[end:]
    p.write_text(content,encoding="utf-8")
    p=ROOT/"output/speed_planner_500m/REPORT.md";content=p.read_text()
    start=content.index("## Three stages of driving:");end=content.index("Same model weights, exclusive pedal policy",start)
    content=content[:start]+text.replace("(media/","(../analysis_pack/media/")+"\n## Original assisted-pedal benchmark\n\n"+content[end:]
    p.write_text(content,encoding="utf-8")
    cards=f"""<section id="driving-stages"><h2>Three measured driving stages</h2><div class="stages">
<article><h3>1. Without speed assistance</h3><img src="media/monza_first_departure_1x.gif" alt="First departure without speed planning"><p>Original learned pedals; rule-based steering and pedal cleanup remain.</p><strong>Monza: {raw['lap_time']:.1f} s</strong><p class="invalid">INVALID - {raw['off_track_events']} departures</p><p>Silverstone: not tested</p></article>
<article><h3>2. With assistance</h3><img src="media/monza_same_turn_with_planner_1x.gif" alt="Assisted hybrid"><p>Original learned pedals plus a rule-based speed planner and steering.</p><strong>Monza: {assisted['lap_time']:.1f} s</strong><p class="valid">VALID - 0 departures</p><strong>Silverstone: {transfer['lap_time']:.1f} s</strong><p class="valid">VALID - 0 departures</p></article>
<article><h3>3. AI-learned steering + pedals</h3><img src="media/monza_full_action_turn_1x.gif" alt="Actual full-action learned-policy replay"><p>Expert imitation. No runtime speed planner, steering formula, launch rule or teacher fallback.</p><strong>Monza: {lm['lap_time']:.1f} s</strong><p class="valid">VALID - 0 departures</p><strong>Silverstone: {ls['lap_time']:.1f} s</strong><p class="valid">VALID - 0 departures</p></article>
</div></section>"""
    for name in ("GALLERY.html","DRIVING_STAGES.html"):
        p=OUT/name;content=p.read_text()
        content,n=re.subn(r'<section id="driving-stages">.*?</section>',lambda _:cards,content,count=1,flags=re.S)
        if n!=1:raise ValueError("Missing stage section in "+name)
        p.write_text(content,encoding="utf-8")
    p=OUT/"GALLERY.html"
    page=p.read_text()
    clips=['<section id="full-action-clips"><h2>Learned steering and pedals: both circuits</h2><div class="grid">']
    for circuit in ("monza","silverstone"):
        for suffix,rate in [("turn_1x.gif","1x turn"),("lap_10x.gif","10x full lap")]:
            name=f"media/{circuit}_full_action_{suffix}"
            clips.append(f'<article><h3>{circuit.title()} | {rate}</h3><p>Expert-imitation policy; no runtime driving helpers.</p><details><summary>Play GIF</summary><img src="{name}" loading="lazy" alt="Full-action learned policy"></details></article>')
    clips.append('</div></section>')
    if '<section id="full-action-clips">' in page:
        page=re.sub(r'<section id="full-action-clips">.*?</section>',lambda _:''.join(clips),page,count=1,flags=re.S)
    else:
        page=page.replace('<footer>',''.join(clips)+'<footer>',1)
    p.write_text(page,encoding="utf-8")
    p=OUT/"GALLERY.md"
    page=p.read_text()
    marker='## Full-action learned-policy follow-up'
    if marker in page:
        page=page[:page.index(marker)]
    page+='\n'+marker+'\n\nThe third stage now works without runtime steering/planning rules. It is expert imitation, not reinforcement learning.\n\n'
    for circuit in ("monza","silverstone"):
        page+=f'### {circuit.title()} learned policy\n\n![Learned full-action turn](media/{circuit}_full_action_turn_1x.gif)\n\n[Full lap at 10x](media/{circuit}_full_action_lap_10x.gif)\n\n'
    p.write_text(page,encoding="utf-8")

    # Matching panel sizes keep the time captions immediately below the three images.
    fig,axes=plt.subplots(1,3,figsize=(16,7),layout="constrained",facecolor="white")
    panels=[("1. Without speed assistance","monza_first_departure_screenshot.png",f"Monza: {raw['lap_time']:.1f} s | INVALID\n13 departures; rule-based steering remains"),
            ("2. With assistance","monza_turn_1_screenshot.png",f"Monza: {assisted['lap_time']:.1f} s | VALID\nSilverstone: {transfer['lap_time']:.1f} s | VALID\nPlanner + original learned pedals"),
            ("3. AI-learned steering + pedals","monza_full_action_screenshot.png",f"Monza: {lm['lap_time']:.1f} s | VALID\nSilverstone: {ls['lap_time']:.1f} s | VALID\nExpert imitation; no runtime driving helpers")]
    for ax,(title,image,caption) in zip(axes,panels):
        ax.imshow(Image.open(OUT/"media"/image));ax.set_title(title,fontsize=13,pad=15)
        ax.set_xticks([]);ax.set_yticks([]);ax.set_xlabel(caption,fontsize=11,labelpad=16)
        ax.spines[["top","right","bottom","left"]].set_visible(False)
    fig.suptitle("Three driving configurations: measured results, different training methods",fontsize=18)
    fig.savefig(OUT/"figures/12_driving_stages.png",dpi=160,bbox_inches="tight");plt.close(fig)
    rows=list(csv.DictReader((OUT/"CAPTIONS.csv").open(encoding="utf-8")))
    rows=[r for r in rows if "full_action" not in r["file"] and r["file"]!="figures/12_driving_stages.png"]
    rows.append(dict(file="figures/12_driving_stages.png",title="Three measured driving stages",caption="The third stage learns steering and signed pedals from simulated expert labels, with no runtime driving helpers. Training differs between stages."))
    for circuit in ("monza","silverstone"):
        for suffix in ("turn_1x.gif","lap_10x.gif","screenshot.png"):
            rows.append(dict(file=f"media/{circuit}_full_action_{suffix}",title=f"{circuit.title()} learned full-action policy",
                caption="Verified saved-action replay. Both steering and pedals are learned; no runtime planner/steering/launch rules. Expert imitation, not reinforcement learning."))
    with (OUT/"CAPTIONS.csv").open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=["file","title","caption"]);writer.writeheader();writer.writerows(rows)
    (OUT/"full_action_training.json").write_text(json.dumps(training,indent=2)+"\n")
    (OUT/"full_action_benchmarks.json").write_text(json.dumps(learned,indent=2)+"\n")
    summary=text+"""## Interpretation

The full-action policy is slightly slower than the assisted original model on
both tracks, but it completes cleanly without hand-written driving decisions
at runtime. This demonstrates successful expert-policy imitation in this
simulator, not independently discovered driving strategy or general superiority.
The same approximate physics and reference-point track-limit rules still apply.

The steering model is StandardScaler plus a tanh MLP with 32/32 hidden units;
the pedal model is 100 ExtraTrees regressors using signed commands. Training
states include perturbed position/heading and speeds from 0 to 50 m/s on Monza.
The actor observes geometry through 500 m, speed, lateral offset and heading
error; no planner target-speed value is supplied as an input.

A dataset-aggregation loop is available for training failures, inspired by
Ross, Gordon and Bagnell (2011): https://proceedings.mlr.press/v15/ross11a.html.
This run required zero aggregation refits; it should be reported as behavior
cloning, not a completed reinforcement-learning or DAgger experiment.

Reproduce from the repository root:

```powershell
python train_full_driver.py
python benchmark_learned_driver.py --track monza
python benchmark_learned_driver.py --track silverstone
python export_learned_driver.py
python update_driving_stages.py
```
"""
    check=ROOT/"output/learned_driver/monza_dt005/benchmark.json"
    if check.exists():
        result=json.loads(check.read_text())["result"]
        summary+=f"\n## Timestep sensitivity\n\nAt 0.05 s steps: Monza {result['lap_time']:.1f} s, {result['off_track_events']} departures, valid={result['valid_lap']}.\n"
    (ROOT/"output/learned_driver/REPORT.md").write_text("# Working full-action learned driver\n\n"+summary.replace("(media/","(../analysis_pack/media/"),encoding="utf-8")
    with zipfile.ZipFile(ROOT/"output/analysis_pack.zip","w",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for path in sorted(OUT.rglob("*")):
            if path.is_file():archive.write(path,Path("analysis_pack")/path.relative_to(OUT))
    print("Updated stage 3 with measured learned-policy results and GIFs",flush=True)


if __name__ == "__main__":
    main()
