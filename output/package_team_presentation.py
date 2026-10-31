from pathlib import Path
import shutil,csv,zipfile,html,hashlib
from PIL import Image
from pypdf import PdfReader
r=Path.cwd();p=r/'output/F1_Team_Presentation_Package';media=r/'output/analysis_pack/media';current=r/'output/pdf/presentation/media'
shutil.copy2(r/'output/pdf/presentation/F1_Project_Presentation.pdf',p/'01_Presentation/F1_Team_Presentation.pdf')
records=[]
def add(src,folder,name,use):
 dest=p/folder/name;shutil.copy2(src,dest);records.append(dict(file=str(dest.relative_to(p)).replace('\\','/'),use=use,source=str(src.relative_to(r)).replace('\\','/')))
add(media/'monza_full_action_lap_10x.gif','02_GIFs','Slide_01_Monza_LearnedDriver_FullLap_10x.gif','Slide 1: optional opening demo. Nominal 10x playback, not real time.')
add(media/'monza_first_departure_1x.gif','02_GIFs','Slides_04_07_Stage1_Monza_NoSpeedPlanner_1x.gif','Slides 4 and 7: naive learned pedals, first road departure. Rule steering and pedal cleanup remain.')
add(media/'monza_same_turn_with_planner_1x.gif','02_GIFs','Slides_04_08_Stage2_Monza_WithSpeedPlanner_1x.gif','Slides 4 and 8: assisted learned pedals near the same turn. Planner can override pedals.')
add(media/'monza_full_action_turn_1x.gif','02_GIFs','Slides_04_09_Stage3_Monza_LearnedSteeringAndPedals_1x.gif','Slides 4 and 9: learned full control, no runtime driving-rule assistance; actuator limits remain.')
add(media/'silverstone_full_action_turn_1x.gif','02_GIFs','Slides_11_13_Stage3_Silverstone_TransferTurn_1x.gif','Slides 11 and 13: actual Silverstone learned-driver replay. Not Spa footage.')
add(media/'silverstone_full_action_lap_10x.gif','02_GIFs','Optional_Stage3_Silverstone_FullLap_10x.gif','Optional full-lap transfer demo; nominal 10x playback.')
for file,name,use in [
 ('monza_full_action_screenshot.png','Slides_01_04_09_Stage3_Monza_LearnedFullControl.png','Slides 1, 4, 9: full-control driver at Monza.'),
 ('monza_follow_screenshot.png','Slide_03_Monza_Simulator_FollowCamera.png','Slide 3: simulator with timing, progress and road-status display.'),
 ('monza_first_departure_screenshot.png','Slides_04_07_Stage1_Monza_FirstDeparture.png','Slides 4 and 7: original pedal driver departure.'),
 ('monza_turn_1_screenshot.png','Slides_04_08_Stage2_Monza_AssistedTurn.png','Slides 4 and 8: hybrid assisted turn.'),
 ('silverstone_full_action_screenshot.png','Slides_11_13_Stage3_Silverstone_LearnedFullControl.png','Slides 11 and 13: actual Silverstone replay.')]:add(media/file,'03_Screenshots',name,use)
for key,name,use in [('monza','Slides_02_11_Monza_TrainingCircuit.png','Slides 2 and 11: training circuit map, Italy.'),('silverstone','Slide_11_Silverstone_TransferCircuit.png','Slide 11: first transfer circuit, United Kingdom.'),('spa','Slide_11_SpaFrancorchamps_FinalTestCircuit_Belgium.png','Slide 11: Spa-Francorchamps, Belgium; fresh frozen-model test.')]:add(current/f'{key}_map.png','04_Track_Maps',name,use)
add(current/'physics_acceleration_braking.png','05_Charts_and_Diagrams','Slide_02_Physics_AccelerationAndBraking_Demonstration.png','Slide 2: controlled run of shared physics, not a racing benchmark.')
add(current/'lap5_recorded_speed.png','05_Charts_and_Diagrams','Slide_05_Monza_Lap5_RecordedSpeed_590Moments.png','Slide 5: speed over one saved lap; explains what recorded moments represent.')
add(r/'output/pipeline_500m/lap05_overlay.png','05_Charts_and_Diagrams','Slide_06_Monza_Lap5_TelemetryToTrack_Alignment.png','Slide 6: actual training-lap alignment output.')
add(r/'output/analysis_pack/figures/02_valid_lap_ablation.png','05_Charts_and_Diagrams','Optional_OriginalModel_PlannerOnly_vs_LearnedPedals.png','Optional historical planner-only versus original hybrid comparison; not tuned-model Spa results.')
for src,name,use in [('comparison.csv','Slide_10_TunedModels_SelectionErrors.csv','Tuned candidate selection means; two training-lap folds.'),('tuning.json','Slide_10_Tuning_All18Trials_AndFoldErrors.json','All settings, per-fold errors and selected candidate.'),('final_test.json','Slides_11_12_Spa_FinalBenchmarks_AllSixDrivers.json','All six final Spa results, frozen model hashes and protocol.')]:add(r/'output/completed_study'/src,'07_Benchmark_Data',name,use)
# Include a spreadsheet-friendly copy of the exact master table, with validity explicit.
import json
complete=json.loads((r/'output/all_configurations/all_24_attempts.json').read_text())
rows=[]
for cfg in complete['protocol']['configurations']:
    cells=[]
    for track in ['monza','silverstone','spa']:
        result=complete['results'][cfg['id']][track]
        time=f"{result['elapsed_time']:.1f} s" if result['completed_lap'] else 'Timeout at 600 s'
        status='Clean lap' if result['valid_lap'] else f"{result['off_track_events']} departures; invalid"
        cells.append(time+'; '+status)
    rows.append([cfg['name'],cfg['learns'],cfg['assistance'],*cells])
with (p/'07_Benchmark_Data/Slide_12_MasterComparison_LapTimes_Seconds.csv').open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.writer(f);w.writerow(['Driving configuration','Learned controls','Driving rules','Monza','Silverstone','Spa']);w.writerows(rows)
for folder in ['08_Complete_Driving_Benchmark','09_Canva_Videos']:(p/folder).mkdir(exist_ok=True)
for src in (r/'output/all_configurations').rglob('*'):
    if src.is_file() and src.suffix in ['.json','.csv','.md']:
        relative=src.relative_to(r/'output/all_configurations');dest=p/'08_Complete_Driving_Benchmark'/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
for src in (r/'output/canva_video').glob('*.mp4'):shutil.copy2(src,p/'09_Canva_Videos'/src.name)
headings=['Project_Goal','Track_And_Physics','Simulator','ThreeDrivingSystems_Teaser','Recorded_Driving_Examples','Coordinate_Alignment','Stage1_LearnedPedals','Stage2_SpeedAssistance','Stage3_LearnedFullControl','ModelChecks_And_Tuning','NewCircuits_And_FinalTest','Benchmarks_And_Performance','What_We_Built']
for i,title in enumerate(headings,1):
 old=p/f'06_Slide_Previews/Slide-{i:02d}.png';new=p/f'06_Slide_Previews/Slide_{i:02d}_{title}.png'
 if old.exists():old.replace(new)
readme=['# Start here','', 'Open **01_Presentation/F1_Team_Presentation.pdf**. This is the latest 13-slide version with full-page visuals, no commentary strips and the three-system teaser.','',
'## Folders','', '1. **01_Presentation** — the current PDF.','2. **02_GIFs** — six replay clips. Names start with the slide(s), stage, circuit and playback speed.','3. **03_Screenshots** — original-resolution stills used in the PDF.','4. **04_Track_Maps** — Monza, Silverstone and Spa-Francorchamps.','5. **05_Charts_and_Diagrams** — physics demonstration, actual recorded-speed chart, alignment plot and one optional historical benchmark graphic.','6. **06_Slide_Previews** — every slide as a named PNG, for quick reference.','7. **07_Benchmark_Data** — master comparison CSV and measured tuning/final-test data.','',
'8. **08_Complete_Driving_Benchmark** — all 24 attempts, protocol, results and per-step traces.','9. **09_Canva_Videos** — six MP4 replays successfully accepted as Canva video assets; use these for Canva playback.','', '## Using the media','', '- The PDF shows still images. Insert the matching GIF into PowerPoint or Google Slides to show motion.','- Open **MEDIA_PREVIEW.html** for an animated gallery; keep it beside the folders after extracting the ZIP.','- **1x** means real-time replay; **10x** means nominal accelerated playback. Clips are independent recorded-control simulator replays, not synchronized races or real onboard video.','- Spa is Spa-Francorchamps in Belgium. We have a geometry map and benchmark data here, not Spa replay footage. Do not label the Silverstone clip as Spa.','- The physics chart is a controlled code demonstration, not a racing benchmark.','- The three development configurations are original RF without speed planning, original RF with speed planning, and learned full control. Tuned pedal predictors are supporting experiments. All eight complete driving configurations are now tested on all three circuits.','', '## Slide-to-file index','', '| Asset | Where to use it |','|---|---|']
for rec in records:readme.append(f"| [{rec['file']}]({rec['file']}) | {rec['use']} |")
(p/'00_START_HERE.md').write_text('\n'.join(readme)+'\n',encoding='utf-8')
with (p/'00_ASSET_INDEX.csv').open('w',newline='',encoding='utf-8-sig') as f:w=csv.DictWriter(f,fieldnames=['file','use','source']);w.writeheader();w.writerows(records)
gallery=['<!doctype html><html lang="en"><meta charset="utf-8"><title>F1 Presentation Assets</title><style>body{font:17px/1.5 system-ui;color:#1d2d3e;background:#f3f5f7;margin:35px auto;max-width:1150px;padding:20px}article{background:white;padding:24px;margin:24px 0;border-radius:14px}img{max-width:100%;max-height:580px}h1{font-size:38px}a{color:#235f88}</style><h1>Presentation media</h1><p><a href="01_Presentation/F1_Team_Presentation.pdf">Open the latest PDF</a> · <a href="00_START_HERE.md">Slide-to-file guide</a></p>']
for rec in records:
 if rec['file'].endswith(('.gif','.png')):gallery += [f"<article><h2>{html.escape(Path(rec['file']).name)}</h2><p>{html.escape(rec['use'])}</p><img src='{rec['file']}' alt='{html.escape(rec['use'],quote=True)}'><p><a href='{rec['file']}'>Open original file</a></p></article>"]
(p/'MEDIA_PREVIEW.html').write_text('\n'.join(gallery)+'</html>',encoding='utf-8')
assert hashlib.sha256((p/'01_Presentation/F1_Team_Presentation.pdf').read_bytes()).digest()==hashlib.sha256((r/'output/pdf/presentation/F1_Project_Presentation.pdf').read_bytes()).digest()
assert len(PdfReader(p/'01_Presentation/F1_Team_Presentation.pdf').pages)==13
assert len(list((p/'06_Slide_Previews').glob('*.png')))==13
for rec in records:
 assert (p/rec['file']).exists()
 if rec['file'].endswith('.gif'):assert Image.open(p/rec['file']).n_frames>1
archive=r/'output/F1_Team_Presentation_Package.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
 for file in p.rglob('*'):
  if file.is_file():z.write(file,Path(p.name)/file.relative_to(p))
with zipfile.ZipFile(archive) as z:assert z.testzip() is None;count=len(z.namelist())
print(f'Packaged {count} files; {archive.stat().st_size/1024/1024:.1f} MB. Verified latest PDF, 13 slide previews and six animated GIFs.')
