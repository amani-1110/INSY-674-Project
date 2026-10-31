"""Validate the full 8-by-3 driving benchmark and rebuild its presentation table."""
from pathlib import Path
import sys, json, csv, hashlib
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph,Table,TableStyle
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.colors import HexColor,white
from pypdf import PdfReader,PdfWriter
from benchmark_all_configurations import OUT,TRACKS,CONFIGS,verify

root=Path(__file__).resolve().parent.parent
protocol=json.loads((OUT/'protocol.json').read_text());verify(protocol)
report={};flat=[]
for cfg in CONFIGS:
    report[cfg['id']]={}
    for track in TRACKS:
        path=OUT/track/(cfg['id']+'.json')
        if not path.exists():raise SystemExit(f'Incomplete: waiting for {track} / {cfg["id"]}')
        item=json.loads(path.read_text());v=item['result']
        assert item['protocol_sha256']==hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest()
        assert v['valid_lap']==(v['completed_lap'] and v['off_track_events']==0)
        assert v['elapsed_time']<=600.000001
        if cfg['id']=='learned_full_control':assert item['helpers_guarded']
        report[cfg['id']][track]=v
        flat.append(dict(configuration=cfg['name'],track=track,learns=cfg['learns'],assistance=cfg['assistance'],training=cfg['training'],**v))
with (OUT/'all_24_attempts.csv').open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
(OUT/'all_24_attempts.json').write_text(json.dumps(dict(protocol=protocol,results=report),indent=2)+'\n')
labels={
    'rules_only':('Rules only','Rule','Rule','Yes'),
    'original_pedals_no_planner':('Original RF, no speed planner','Rule','Original RF','No'),
    'original_pedals_with_planner':('Original RF + speed planner','Rule','Original RF','Yes'),
    'learned_full_control':('Learned steering + pedals','Neural network','ExtraTrees','No'),
    'tuned_rf':('Tuned RF + speed planner','Rule','Tuned RF','Yes'),
    'tuned_svr':('Tuned SVR + speed planner','Rule','Tuned SVR','Yes'),
    'tuned_mlp':('Tuned neural pedals + planner','Rule','Tuned neural net','Yes'),
    'combined_rf_svr':('50/50 RF + SVR + planner','Rule','RF + SVR blend','Yes')}
lines=['# All configurations on all circuits','',
       'This table measures whole driving systems: lap completion, road-departure episodes and elapsed time. It is not an isolated ranking of machine-learning algorithms. Every configuration uses the same circuit geometry, physics, aligned starting heading, standing start, 0.1-second timestep and 600-second limit. No model was retrained.',
       '', 'Original RF and tuned RF are different saved fits. Full-control imitation uses simulated-teacher training and learned steering; its results measure the complete policy, not the isolated effect of replacing a pedal algorithm.',
       '', 'Comparisons among planner-equipped pedal candidates hold rule steering and speed assistance fixed. Comparing the same original RF with/without the speed planner isolates the added planner. Rules-only versus planner-equipped RF isolates the marginal contribution of learned pedal proposals.',
       '', 'These are expanded development/reproduction tests on already inspected circuits. The earlier frozen Spa protocol is preserved; this run is not claimed as another untouched final test.',
       '', '| Driving configuration | Steering | Pedals | Speed planner | Monza | Silverstone | Spa |', '|---|---|---|---|---|---|---|']
style=ParagraphStyle('cell',fontName='Helvetica',fontSize=13,leading=16,textColor=HexColor('#1d2d3e'))
head=ParagraphStyle('head',parent=style,fontName='Helvetica-Bold',fontSize=14)
headers=['Driving configuration','Steering','Pedals','Speed planner','Monza','Silverstone','Spa']
data=[[Paragraph(v,head) for v in headers]]
for cfg in CONFIGS:
    lab=labels[cfg['id']];cells=[];plain=[]
    for track in TRACKS:
        v=report[cfg['id']][track]
        time=f"{v['elapsed_time']:.1f} s" if v['completed_lap'] else 'Timeout (600 s)'
        status='Clean lap' if v['valid_lap'] else f"{v['off_track_events']} departures; invalid"
        color='#17664f' if v['valid_lap'] else '#ad302a'
        cells.append(Paragraph(f'<b>{time}</b><br/><font color="{color}">{status}</font>',style))
        plain.append(time+'; '+status)
    data.append([Paragraph(v,style) for v in lab]+cells)
    lines.append('| '+' | '.join([*lab,*plain])+' |')
lines+=['','Clean lap means completed with zero departures. An invalid completion is not a time improvement. Each cell is one predefined deterministic attempt, not a statistical reliability estimate. Per-step control traces and all model/geometry/code hashes are preserved.']
(OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
page=OUT/'complete_benchmark_slide.pdf';c=canvas.Canvas(str(page),pagesize=(1280,720))
def text(s,x,y,w,size=20,bold=False):
    st=ParagraphStyle('p',fontName='Helvetica-Bold' if bold else 'Helvetica',fontSize=size,leading=size*1.25,textColor=HexColor('#1d2d3e'))
    p=Paragraph(s,st);_,h=p.wrap(w,900);p.drawOn(c,x,720-y-h)
text('Benchmarks and performance',50,39,1160,39,True)
text('Same start, physics and time step. Eight complete driving configurations; all three circuits.',50,111,1180,21)
t=Table(data,colWidths=[250,120,190,105,170,170,175],rowHeights=[44]+[56]*8)
t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),HexColor('#dfe8ef')),('ROWBACKGROUNDS',(0,1),(-1,-1),[white,HexColor('#edf2f6')]),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('LEFTPADDING',(0,0),(-1,-1),8),('RIGHTPADDING',(0,0),(-1,-1),8),('LINEBELOW',(0,0),(-1,-1),.4,HexColor('#dce2e8'))]))
t.wrapOn(c,1180,492);t.drawOn(c,50,720-160-492)
text('Clean = completed, zero departures. Other completions are invalid; timeout = no completed lap. Only compare speed for clean laps.',50,666,1180,13)
text('All attempts: output/all_configurations. Pedal-prediction error is a separate metric. Full-control imitation also changes training and steering.',50,690,1180,10)
c.save()
for name in ['F1_Project_Presentation.pdf','F1_Presentation_With_Optional_Visuals.pdf']:
    original=root/'output/pdf/presentation'/name
    if not original.exists():continue
    reader=PdfReader(original);writer=PdfWriter()
    for i,p in enumerate(reader.pages):writer.add_page(PdfReader(page).pages[0] if i==11 else p)
    temp=original.with_suffix('.updated.pdf')
    with temp.open('wb') as f:writer.write(f)
    temp.replace(original)
print('Verified all 24 attempts, frozen inputs, validity rules, and updated both presentation PDFs.')
