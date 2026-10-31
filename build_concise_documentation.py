"""Build a concise visual document and matching presentation plan from saved evidence."""
from pathlib import Path
import json, html, shutil, zipfile

ROOT=Path(__file__).resolve().parent
PACK=ROOT/'output/analysis_pack'
OVER=ROOT/'output/overleaf_documentation'
study=json.loads((ROOT/'output/completed_study/tuning.json').read_text())
final=json.loads((ROOT/'output/completed_study/final_test.json').read_text())
if not (ROOT/'PROJECT_EVIDENCE.md').exists():
    shutil.copy2(ROOT/'DOCUMENTATION.md',ROOT/'PROJECT_EVIDENCE.md')
if not (OVER/'detailed.tex').exists():
    shutil.copy2(OVER/'main.tex',OVER/'detailed.tex')

stages=[
    ('Without speed assistance','Original learned pedals; hand-written steering, pedal cleanup and launch handling remain.','91.8 s','13 departures — invalid','Not tested','monza_first_departure_1x.gif'),
    ('With assistance','The same learned pedals, now with a speed planner that can override them. Hand-written steering remains.','205.4 s','0 departures — valid','256.5 s · 0 departures','monza_same_turn_with_planner_1x.gif'),
    ('Learned steering + pedals','A new driver copies a simulated teacher. No driving-rule assistance during evaluation; actuator limits remain.','209.1 s','0 departures — valid','266.3 s · 0 departures','monza_full_action_turn_1x.gif')]

# Each block is a slide-sized claim, with plain-language evidence and one takeaway.
blocks=[
    ('01','Can a learned driver finish a clean lap?', 'Frame the Problem and Look at the Big Picture',
     ['We built a 2D racing simulator for students and researchers to compare driving approaches.',
      'Success means finishing with zero road departures. Compare speed only after a lap is valid.'],
     'Clean driving comes first; speed comes second.', '0:00–0:45', 'Track map: figures/01_track_maps.png', 'Open with the question and explain what counts as success.'),
    ('02','One table: three stages and supporting comparisons', 'Process and insights',
     ['First, learned pedals without a speed planner left the road.',
      'Adding a planner produced clean laps. A separately trained driver then learned both steering and pedals.'],
     'The stages show how the system developed—not a comparison changing only assistance.', '0:45–2:30', 'Master comparison table: highlight the three stage rows', 'Explain each card. Stage 1 still has steering and pedal rules; stage 3 uses different examples and models.'),
    ('03','The examples did not cover every driving situation', 'Explore the Data',
     ['1,774 samples from three Verstappen qualifying laps at Monza; one driver and one session.',
      'The slowest recorded speed was 73 km/h, but the simulator starts at rest. Only about 13% of samples contain braking.',
      'Saved cleaned tables have no missing values; original channels were interpolated. Real measured steering labels are unavailable.'],
     'Good qualifying examples do not automatically teach starts or recovery.', '2:30–3:30', 'figures/04_data_distributions.png', 'Explain the data gap before introducing any model terminology.'),
    ('04','Keep whole laps separate', 'Prepare the Data',
     ['Match driver positions to the circuit, standardize units and describe the road ahead.',
      'Use laps 5 and 11 for training. Train on one whole lap and check on the other, then swap.',
      'Keep scaling inside each training split. Lap 14 was inspected during development, so its scores are exploratory.'],
     'Nearby moments from the same lap should not be mixed between training and checking.', '3:30–4:30', 'Telemetry → alignment → road features → learned controls → simulator', 'Introduce grouped cross-validation only after explaining the whole-lap split.'),
    ('05','Test settings—and test a combined model', 'Measure and Compare Their Performance / Fine-Tune the System',
     ['Screen five prediction approaches, then test 18 settings across Random Forest, support-vector and neural-network models.',
      'Compare the tuned models with a fixed 50/50 Random Forest + support-vector combination.',
      'Random Forest has the lowest selection error. The combination does not improve that error. Selection errors and fold variability are in the supporting evidence.'],
     'Choose using training laps, before looking at the fresh circuit.', '4:30–5:45', 'Master table: highlight the four tuned candidates', 'MAE means average prediction error. These two-fold selection scores are not final-test accuracy.'),
    ('06','Check whether the learned pedals actually help', 'Measure and Compare Their Performance',
     ['Keep steering, planner, physics and starting conditions the same; remove only the learned pedal proposals.',
      'Monza: planner-only 208.7 s → assisted learned pedals 205.4 s.',
      'Silverstone: planner-only 258.3 s → assisted learned pedals 256.5 s. All four laps are valid.'],
     'The original learned pedals add a small measured benefit beyond the shared planner.', '5:45–6:45', 'Two paired results; figures/02_valid_lap_ablation.png', 'These are historical original-model comparisons, separate from the later tuned-model Spa test.'),
    ('07','Freeze the models, then test a new circuit', 'Final evaluation',
     ['Save the models and test protocol before retrieving Spa geometry or running the test.',
      'The preselected Random Forest hybrid finishes in 282.0 s; planner-only takes 287.4 s. Both have zero departures.',
      'The separate stage-3 learned driver finishes in 294.3 s, also with zero departures. All six predefined drivers complete cleanly.'],
     'The selected hybrid is 5.4 seconds faster here; the learned full-action driver also transfers cleanly.', '6:45–8:15', 'Master table: highlight the Spa column', 'All results are reported. The neural hybrid is faster despite worse prediction errors; do not reselect the model from final-test results.'),
    ('08','What we learned—and where it stops', 'Assumptions and Limitations',
     ['Matching recorded pedals and driving well are different tasks. Starts, recovery and the complete controller matter.',
      'Stage 3 learns a simulated teacher’s behaviour; it does not independently discover a racing strategy.',
      'Approximate physics, estimated track widths and reference-point road limits differ from real F1. The final test covers one deterministic starting condition.'],
     'We demonstrated clean simulated driving—not real-car performance or broad reliability.', '8:15–9:30', 'One closing takeaway with three short limitations', 'Close with the result. Keep the remaining 30 seconds as a timing buffer.')]

def esc(s):
    s=s.replace('—','--').replace('–','--').replace('→', 'to').replace('·',';').replace('’',"'")
    return ''.join({'\\':r'\textbackslash{}','&':r'\&','%':r'\%','$':r'\$','#':r'\#','_':r'\_','{':r'\{','}':r'\}'}.get(c,c) for c in s)

md=['# F1 Racing Simulator: the project story','',
    '[GitHub repository](https://github.com/amani-1110/INSY-674-Project) · [Visual document](output/analysis_pack/PROJECT_STORY.html) · [Presentation plan](output/analysis_pack/PRESENTATION_STORYBOARD.md) · [Detailed evidence](PROJECT_EVIDENCE.md)','']
page=['<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>F1 simulator · Project story</title><style>',
'''*{box-sizing:border-box}body{font:16px/1.55 system-ui;background:#f3f5f7;color:#1d2d3e;margin:0;padding:32px}main{max-width:1240px;margin:auto}header{padding:22px 0 30px}h1{font-size:44px;line-height:1.12;margin:12px 0}h2{font-size:28px;line-height:1.2;margin:6px 0 20px}h3{line-height:1.25}a{color:#235f88}section{margin:0 0 24px;padding:26px;background:white;border:1px solid #dce2e8;border-radius:12px}.eyebrow{font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:.10em;color:#536b7f}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:18px}.card{padding:18px;border:1px solid #dce2e8;border-radius:12px;background:#f8fafb}.card img{width:100%;height:auto;border-radius:6px}.metric{font-size:29px;font-weight:750;margin-top:12px}.valid{color:#17664f;font-weight:650}.invalid{color:#ad302a;font-weight:650}.takeaway{padding:14px 18px;background:#edf2f6;border-left:4px solid #235f88;border-radius:4px;margin-top:20px;font-weight:650}ul{padding-left:22px}li{margin:8px 0}table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums;font-size:14px}th,td{padding:12px;text-align:left;border-bottom:1px solid #dce2e8}th{background:#edf2f6}.small{font-size:13px;color:#536b7f}details{margin-top:14px}summary{cursor:pointer;font-weight:650}.tag{display:inline-block;background:#edf2f6;padding:4px 10px;border-radius:20px;margin-right:8px;font-size:13px}.tablewrap{overflow-x:auto}@media(max-width:850px){.grid{grid-template-columns:1fr}body{padding:16px}h1{font-size:34px}}@media print{@page{size:A4;margin:14mm}body{padding:0;background:white;font-size:10pt}header{padding:0 0 12px}h1{font-size:28pt}h2{font-size:18pt}section{padding:15px;break-inside:avoid;margin-bottom:14px}.grid{grid-template-columns:repeat(3,minmax(0,1fr));gap:8px}.card{padding:9px}.card h3{font-size:12pt}.card p{font-size:9pt}.metric{font-size:18pt}details{display:none}a{color:inherit}table{font-size:9pt}th,td{padding:7px}}
''','</style></head><body><main><header><div class="eyebrow">INSY 674 · Project documentation</div><h1>From recorded pedals<br>to a learned driver.</h1><p>A clean lap first. A faster valid lap second.</p><a href="https://github.com/amani-1110/INSY-674-Project">GitHub repository</a> · <a href="PRESENTATION_STORYBOARD.md">Presentation plan</a> · <a href="../../PROJECT_EVIDENCE.md">Detailed evidence</a></header>']
tex=[r'''\documentclass[10pt,a4paper]{article}
\usepackage[margin=17mm]{geometry}
\usepackage[T1]{fontenc}
\usepackage{lmodern,graphicx,xcolor,booktabs,hyperref,enumitem,tabularx,pdflscape}
\usepackage[most]{tcolorbox}
\definecolor{ink}{HTML}{1D2D3E}
\definecolor{paperblue}{HTML}{EDF2F6}
\hypersetup{colorlinks=true,urlcolor=blue}
\setlength{\parindent}{0pt}
\setlength{\parskip}{5pt}
\setlist[itemize]{leftmargin=*,nosep,topsep=5pt}
\raggedbottom
\newcommand{\takeaway}[1]{\begin{tcolorbox}[colback=paperblue,colframe=paperblue,boxrule=0pt]#1\end{tcolorbox}}
\begin{document}
{\LARGE\bfseries From recorded pedals to a learned driver.}\par
INSY 674 --- Concise project documentation\par
\href{https://github.com/amani-1110/INSY-674-Project}{GitHub repository}\quad |\quad A clean lap first. A faster valid lap second.
''']
story=['# Presentation plan · 9 minutes 30 seconds','', 'Use the same eight sections as the visual document. One takeaway per slide; leave 30 seconds as a buffer. Use one master table throughout: highlight the relevant rows or circuit column instead of introducing more tables.','', '| Time | Slide headline | Visual | Speaker cue |','|---|---|---|---|']


def system_table():
    matrix=json.loads((ROOT/'output/all_configurations/all_24_attempts.json').read_text())['results']
    labels=[('rules_only','Rules only','Rule','Rule','Yes'),('original_pedals_no_planner','Original RF, no speed planner','Rule','Original RF','No'),('original_pedals_with_planner','Original RF + speed planner','Rule','Original RF','Yes'),('learned_full_control','Learned steering + pedals','Neural network','ExtraTrees','No'),('tuned_rf','Tuned RF + planner','Rule','Tuned RF','Yes'),('tuned_svr','Tuned SVR + planner','Rule','Tuned SVR','Yes'),('tuned_mlp','Tuned neural pedals + planner','Rule','Tuned neural net','Yes'),('combined_rf_svr','50/50 RF + SVR + planner','Rule','RF + SVR blend','Yes')]
    rows=[]
    for key,name,steer,pedals,planner in labels:
        cells=[]
        for track in ['monza','silverstone','spa']:
            v=matrix[key][track]
            time=f"{v['elapsed_time']:.1f} s" if v['completed_lap'] else 'Timeout (600 s)'
            status='clean lap' if v['valid_lap'] else f"{v['off_track_events']} departures; invalid"
            cells.append(time+'; '+status)
        rows.append((name,steer,pedals,planner,*cells))
    return ['Driving configuration','Steering','Pedals','Speed planner','Monza','Silverstone','Spa'],rows

def tuning_table():
    rows=[]
    for r in study['summary']:
        name=r['model'].replace('Equal RF and SVR ensemble','50/50 combined model')
        rows.append((name,f"{r['throttle_mae']:.4f}",f"{r['brake_mae']:.4f}"))
    return ['Tuned candidate','Throttle error','Brake error'],rows

def final_table():
    rows=[]
    for name,r in final['results'].items():
        name=name.replace('Equal RF and SVR ensemble','50/50 combined hybrid').replace('Frozen full-action driver','Stage 3: learned steering + pedals')
        if name in ('Random Forest','Support vector','Neural network'): name+=' hybrid'
        rows.append((name,f"{r['elapsed_time']:.1f}",str(r['off_track_events']),'Yes' if r['valid_lap'] else 'No'))
    return ['Spa driver','Time (s)','Departures','Valid lap'],rows

for num,title,rubric,bullets,takeaway,time,visual,cue in blocks:
    md+=['## '+num+' · '+title,'','**Rubric:** '+rubric,'']+['- '+b for b in bullets]+['','**Takeaway:** '+takeaway,'']
    page+=[f'<section id="section-{num}"><div class="eyebrow">{num} · {html.escape(rubric)}</div><h2>{html.escape(title)}</h2>']
    tex+=[r'\section*{'+esc(num+' | '+title)+'}',r'{\small\textit{'+esc(rubric)+'}}']
    if num=='02':
        headers,rows=system_table()
        md+=['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(row)+' |' for row in rows]+['']
        page+=['<div class="tablewrap"><table><thead><tr>'+''.join('<th>'+html.escape(v)+'</th>' for v in headers)+'</tr></thead><tbody>']+['<tr>'+''.join('<td>'+html.escape(v)+'</td>' for v in row)+'</tr>' for row in rows]+['</tbody></table></div>']
        tex += [r'\begin{landscape}\section*{All driving systems and model comparisons}',r'\small\begin{tabularx}{\linewidth}{p{27mm}p{32mm}p{25mm}Xp{22mm}p{22mm}p{22mm}}\toprule',' & '.join(esc(v) for v in headers)+r'\\\midrule']+[' & '.join(esc(v) for v in row)+r'\\\addlinespace' for row in rows]+[r'\bottomrule\end{tabularx}']
        note='All 24 attempts use the same start, physics, 0.1-second timestep and 600-second limit. Clean means completed with zero departures. This compares complete driving configurations, not isolated algorithms. The original and tuned RF fits differ; full-control imitation also changes steering and training data. These are expanded development tests on already inspected circuits. See output/all_configurations/REPORT.md for the protocol and all outcomes.'
        md+=[note,''];page+=['<p class="small">'+html.escape(note)+'</p>'];tex+=[esc(note),r'\end{landscape}']
    else:
        page+=['<ul>']+[f'<li>{html.escape(b)}</li>' for b in bullets]+['</ul>']
        tex+=[r'\begin{itemize}']+[r'\item '+esc(b) for b in bullets]+[r'\end{itemize}']
    if False:  # All model identities and driving results are consolidated above.
        headers,rows=tuning_table() if num=='05' else final_table()
        md+=['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(row)+' |' for row in rows]+['']
        page+=['<div class="tablewrap"><table><thead><tr>'+''.join('<th>'+html.escape(v)+'</th>' for v in headers)+'</tr></thead><tbody>']+['<tr>'+''.join('<td>'+html.escape(v)+'</td>' for v in row)+'</tr>' for row in rows]+['</tbody></table></div>']
        tex += [r'\begin{center}\small\begin{tabular}{'+'l'+'r'*(len(headers)-1)+r'}\toprule',' & '.join(esc(v) for v in headers)+r'\\\midrule']+[' & '.join(esc(v) for v in row)+r'\\' for row in rows]+[r'\bottomrule\end{tabular}\end{center}']
        note=('Mean absolute error on a 0–1 pedal scale; lower is better. Mean across two whole-lap selection folds, reused for tuning—not independent test errors. Full fold variability and settings are in the evidence document.' if num=='05' else 'The four tuned pedal candidates retain the same planner and steering. All Spa results were reported without post-test tuning. One deterministic start; no statistical-significance claim.')
        md+=[note,'']; page+=['<p class="small">'+html.escape(note)+'</p>']; tex+=[r'{\small '+esc(note)+'}']
    page+=[f'<div class="takeaway">{html.escape(takeaway)}</div><details><summary>Build slide {num} · {time}</summary><p><b>Visual:</b> {html.escape(visual)}</p><p>{html.escape(cue)}</p></details></section>']
    tex += [r'\takeaway{\textbf{Takeaway:} '+esc(takeaway)+'}']
    if num in ('02','04','06'): tex+=[r'\clearpage']
    story += [f'| {time} | {title} | {visual} | {cue} |']

md+=['## Evidence and reproduction','', '[Detailed methods, attributes, fold variability and limitations](PROJECT_EVIDENCE.md) · [Completed study](output/completed_study/REPORT.md) · [Saved protocol](output/completed_study/final_test_protocol.json)','',
     'Run `python build_concise_documentation.py` to rebuild the concise document, presentation outline and Overleaf ZIP from saved results. This does not retrain or rerun the final test. The experiment implementation is `complete_model_study.py`; preserve the original final-test outputs.','']
story+=['','## Backup slides only','',
        '- Full attribute profile and labelled data distributions; correlations and feature importance stay in the evidence appendix. No heatmap is needed in the main presentation.',
        '- Model settings, all 18 trials, mean ± fold standard deviation, convergence checks.',
        '- Original failed preview-distance experiments and pedal-conflict diagnosis.',
        '- All six Spa results, frozen hashes and protocol; timestep sensitivity.',
        '', 'Use the visual document’s “Build slide” controls for speaker cues. Keep code paths and settings in backup material, not the main spoken story.',
        '', 'Technical definitions: supervised learning = learning from labelled examples; grouped cross-validation = checking on whole laps excluded from fitting; MAE = average size of a prediction error; expert imitation = copying a teacher’s decisions.']
page+=['<footer class="small"><p>Evidence: <a href="../../PROJECT_EVIDENCE.md">detailed project documentation</a> · <a href="../completed_study/REPORT.md">completed study</a> · <a href="../completed_study/final_test_protocol.json">frozen test protocol</a></p><p>Print this page for a static copy. Use the Overleaf ZIP for the LaTeX version. Motion is available in the HTML clips; printed pages use still images.</p></footer></main></body></html>']
tex += [r'\section*{Evidence}', r'The repository includes PROJECT\_EVIDENCE.md for detailed methods, attribute profiles, initial model comparisons, fold variability, settings, assumptions and reproduction instructions. The completed study folder contains all 18 trials, the frozen protocol and all six final results. The matching presentation plan uses these same eight sections.',r'\end{document}']
(ROOT/'DOCUMENTATION.md').write_text('\n'.join(md),encoding='utf-8')
(PACK/'PROJECT_STORY.html').write_text('\n'.join(page),encoding='utf-8')
(PACK/'PRESENTATION_STORYBOARD.md').write_text('\n'.join(story)+'\n',encoding='utf-8')
(OVER/'main.tex').write_text('\n'.join(tex),encoding='utf-8')
shutil.copy2(PACK/'PRESENTATION_STORYBOARD.md',OVER/'PRESENTATION_PLAN.md')
shutil.copy2(ROOT/'PROJECT_EVIDENCE.md',OVER/'PROJECT_EVIDENCE.md')
notes=OVER/'README.md'
text=notes.read_text(encoding='utf-8')
text+='\n\n## Concise presentation structure\n\nmain.tex is the concise, print-friendly version organized around eight slide-sized sections. detailed.tex retains the longer supporting document. Set the desired file as the Overleaf main document. PRESENTATION_PLAN.md gives timing, visuals and speaker cues; PROJECT_EVIDENCE.md holds full supporting detail. Rebuild this version with python build_concise_documentation.py from the repository root.\n'
# Keep repeated rebuilds stable.
text=text.split('\n\n## Concise presentation structure')[0]+'\n\n## Concise presentation structure'+text.split('\n\n## Concise presentation structure')[-1]
notes.write_text(text,encoding='utf-8')
with zipfile.ZipFile(ROOT/'output/F1_Project_Overleaf.zip','w',zipfile.ZIP_DEFLATED) as z:
    for p in OVER.rglob('*'):
        if p.is_file(): z.write(p,p.relative_to(OVER))
print('Built concise Markdown, visual HTML, matching presentation plan and Overleaf ZIP; detailed evidence preserved.')
