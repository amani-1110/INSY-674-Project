from pathlib import Path
import json,math
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor,white
from reportlab.platypus import Paragraph,Table,TableStyle
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
R=Path.cwd();O=R/'output/pdf/presentation';W,H=1280,720
INK='#1d2d3e';BLUE='#235f88';GREEN='#17664f';RED='#ad302a';LINE='#dce2e8';BG='#ffffff'
c=canvas.Canvas(str(O/'F1_Project_Presentation.pdf'),pagesize=(W,H));c.setTitle('F1 Racing Simulator - Visual Story Outline')
def para(text,x,y,w,size=22,color=INK,bold=False):
 st=ParagraphStyle('p',fontName='Helvetica-Bold' if bold else 'Helvetica',fontSize=size,leading=size*1.23,textColor=HexColor(color));p=Paragraph(text,st);_,h=p.wrap(w,1200);p.drawOn(c,x,H-y-h);return h

def image(path,x,y,w,h):
 im=Image.open(path);iw,ih=im.size;q=min(w/iw,h/ih);ww,hh=iw*q,ih*q;c.drawImage(ImageReader(im),x+(w-ww)/2,H-y+(h-hh)/2-h,width=ww,height=hh)
def rule(x,y,w,col=LINE):
 c.setStrokeColor(HexColor(col));c.setLineWidth(1);c.line(x,H-y,x+w,H-y)
def begin(n,title,source='',dark=False):
 c.setFillColor(HexColor('#132e26' if dark else BG));c.rect(0,0,W,H,fill=1,stroke=0)
 para(title,50,39,1160,39,'#ffffff' if dark else INK,True)
 para(f'{n:02d}',1190,47,45,15,'#a8c2b7' if dark else BLUE)
 if source:para(source,50,688,1180,10,'#a8c2b7' if dark else '#536b7f')
def end():c.showPage()
def photo(name,x,y,w,h):image(R/'output/analysis_pack/media'/name,x,y,w,h)
def big(value,label,x,y,w=350,col=BLUE):
 para(value,x,y,w,60,col,True);para(label,x,y+83,w,21)

begin(1,'Project goal','Actual simulator replay at Monza. Optional clip: 04_monza_full_lap.gif.')
para('Teach a computer<br/>to finish a<br/>clean lap.',50,167,430,48,bold=True)
para('Then give it a different road.',50,450,430,27,BLUE)
para('A completed lap with zero road departures comes before a faster time.',50,541,405,22)
photo('monza_full_action_screenshot.png',505,126,725,520);end()

begin(2,'Track geometry and vehicle physics','Road map: bundled Monza geometry. Physics plot: controlled 10 s acceleration, then 0.6 braking, 0.1 s steps; not a racing benchmark.')
para('THE ROAD',50,136,500,14,BLUE,True)
image(O/'media/monza_map.png',50,182,475,387)
para('Shape, bends and boundaries',50,604,500,27,bold=True)
rule(570,147,1)
para('THE CAR',650,136,570,14,BLUE,True)
image(O/'media/physics_acceleration_braking.png',620,204,610,283)
para('Steering and pedals change<br/>position, direction and speed.',650,532,550,28,bold=True)
end()

begin(3,'Simulator','Track + shared physics + session state. Actual follow-camera screenshot; map is not a real onboard image.')
photo('monza_follow_screenshot.png',50,112,1180,499)
for x,title,detail in [(50,'Road limits','Is the car on the road?'),(460,'Lap progress','How far has it travelled?'),(870,'Time','Has it completed a lap?')]:
 para(title,x,617,360,23,bold=True);para(detail,x,652,360,17)
end()

begin(4,'Three driving systems','GIFs 01-03: independent replays near the same Monza turn, not a synchronized race. Times use 0.1-second steps.')
items=[('01','Naive learned pedals','monza_first_departure_screenshot.png','Random Forest pedals','Rules still steer and handle launch.','91.8 s','13 departures',RED,'01_no_speed_planner.gif'),('02','Assisted learned pedals','monza_turn_1_screenshot.png','Same Random Forest','Add a rule-based speed planner.','205.4 s','0 departures',GREEN,'02_with_speed_planner.gif'),('03','Learned full control','monza_full_action_screenshot.png','Neural steering + ExtraTrees pedals','New simulated-teacher examples.','209.1 s','0 departures',GREEN,'03_learned_full_control.gif')]
for i,(num,title,img,model,fact,time,status,col,ref) in enumerate(items):
 x=50+i*400;para(num,x,133,370,17,BLUE,True);para(title,x,168,370,27,bold=True)
 photo(img,x,224,370,255);para(model,x,493,370,18,bold=True);para(fact,x,526,370,17)
 para(time,x,574,370,43,col,True);para(status,x,630,370,18,col,True)
 para('GIF '+ref,x,660,370,10,BLUE)
end()

begin(5,'Recorded driving examples','Actual saved lap 5. Public channels were synchronized/interpolated; saved moments are not independent trials.')
para('One row records one moment of a lap.',50,134,1180,29,bold=True)
image(O/'media/lap5_recorded_speed.png',30,204,865,381)
para('590 moments',934,219,295,35,BLUE,True);para('in lap 5 alone',934,270,295,23)
para('1,774 moments',934,343,295,35,BLUE,True);para('across laps 5, 11 and 14',934,394,295,22)
rule(50,608,1180)
para('At 0.154 s',50,628,220,24,bold=True);para('323.0 km/h',340,628,230,24);para('100% throttle',650,628,260,24);para('Brake off',1000,628,230,24)
end()

begin(6,'Coordinate alignment','Actual lap-5 alignment output. A full training trajectory estimates the transform; a single point cannot determine it.')
image(R/'output/pipeline_500m/lap05_overlay.png',20,128,760,525)
para('Same moment.<br/>A useful road reference.',825,167,405,38,bold=True)
para('Before: position in the recording\'s coordinates.',825,315,390,24)
para('After: 49.1 m along our road,<br/>2.26 m right of its centre.',825,421,390,27,BLUE,True)
para('Keep speed and recorded pedals attached to that road position.',825,545,390,22)
end()

begin(7,'What if the model chose the pedals?','Stage 1: original Random Forest, 500 m preview. GIF: 01_no_speed_planner.gif. Invalid completion; not a time improvement.')
photo('monza_first_departure_screenshot.png',25,143,780,492)
para('Random Forest',852,154,378,30,bold=True)
para('Learns throttle and brake<br/>from matched examples.',852,214,378,25)
para('Rules still handle steering,<br/>launch and pedal cleanup.',852,306,378,22)
big('91.8 s','Monza completion',852,407,378,RED)
para('13 road departures',852,548,378,29,RED,True)
end()

begin(8,'What if we added speed assistance?','Stage 2: same original learned-pedal model. GIF: 02_with_speed_planner.gif. The speed planner can override its proposals.')
big('205.4 s','Monza: valid lap',50,153,450,GREEN)
para('0 departures',50,312,450,34,GREEN,True)
para('Keep the model.<br/>Add a speed planner<br/>before the corners.',50,404,435,35,bold=True)
para('Rules still steer.',50,588,430,23)
photo('monza_turn_1_screenshot.png',510,137,720,508)
end()

begin(9,'What if the models chose all the controls?','Stage 3: expert imitation, not reinforcement learning. GIF: 03_learned_full_control.gif. Teacher labels are used during training only.')
para('40,000 simulated situations',50,137,1180,29,BLUE,True)
para('Starts, corners and recovery, labelled by a rule-based teacher.',50,185,1180,23)
photo('monza_full_action_screenshot.png',554,248,676,399)
para('Neural network',50,293,460,33,bold=True);para('Chooses steering',50,342,460,23)
para('ExtraTrees',50,409,460,33,bold=True);para('Chooses throttle or brake',50,458,460,23)
para('209.1 s  /  0 departures',50,567,460,31,GREEN,True)
para('No runtime driving rules; actuator limits remain.',50,625,460,18)
end()

begin(10,'Model checks and tuning','Pedal prediction MAE on a 0-1 scale; lower is better. Mean +/- sample SD over two whole-lap selection folds, reused for tuning; not final-test accuracy.')
study=json.loads((R/'output/completed_study/tuning.json').read_text())
para('Train on one whole lap.<br/>Check on the other. Swap.',50,144,420,33,bold=True)
para('18 settings across three model families.<br/>A fixed 50/50 combined predictor.<br/>31 controller/evaluation tests passed.',50,346,420,23)
labels={'Random Forest':'Random Forest','Support vector':'Support-vector model','Neural network':'Neural network','Equal RF and SVR ensemble':'50/50 combined model'}
for i,row in enumerate(sorted(study['summary'],key=lambda r:r['combined_mae'])):
 y=165+i*110;name=labels[row['model']];values=[sum(v)/2 for v in row['fold_errors']];mean=sum(values)/2;sd=abs(values[0]-values[1])/math.sqrt(2)
 para(name,550,y,650,23,bold=True)
 c.setFillColor(HexColor(BLUE if i==0 else '#95acba'));c.rect(550,H-y-68,420*mean/.1,23,fill=1,stroke=0)
 para(f'{mean:.4f} +/- {sd:.4f}',1000,y+36,230,20)
para('Random Forest selected before Spa testing.',50,618,1180,26,BLUE,True)
end()

begin(11,'New circuits and the final test','All shown stage-3 laps completed with zero departures. Circuit maps use bundled geometry; Spa identity: formula1.com/en/racing/2026/belgium.')
for i,(name,title,role,time,desc) in enumerate([('monza','Monza, Italy','TRAINING ROAD','209.1 s','Examples come from this circuit.'),('silverstone','Silverstone, UK','FIRST TRANSFER CHECK','266.3 s','Excluded from model training.'),('spa','Spa-Francorchamps','FRESH FINAL TEST','294.3 s','Belgium. Frozen models; no test tuning.')]):
 x=50+i*400;para(title,x,137,370,27,bold=True);image(O/'media'/f'{name}_map.png',x,205,370,278);para(role,x,512,370,13,BLUE,True);para(time,x,547,370,48,GREEN,True);para(desc,x,616,370,20)
end()

begin(12,'Benchmarks and performance','Same standing start, physics, 0.1 s steps and 600 s limit. Complete driving systems, not an isolated algorithm ranking. All circuits already inspected.')
benchmark=json.loads((R/'output/all_configurations/all_24_attempts.json').read_text())['results']
labels=[('rules_only','Rules only','Rule','Rule','Yes'),('original_pedals_no_planner','Original RF, no speed planner','Rule','Original RF','No'),('original_pedals_with_planner','Original RF + speed planner','Rule','Original RF','Yes'),('learned_full_control','Learned steering + pedals','Neural network','ExtraTrees','No'),('tuned_rf','Tuned RF + speed planner','Rule','Tuned RF','Yes'),('tuned_svr','Tuned SVR + speed planner','Rule','Tuned SVR','Yes'),('tuned_mlp','Tuned neural pedals + planner','Rule','Tuned neural net','Yes'),('combined_rf_svr','50/50 RF + SVR + planner','Rule','RF + SVR blend','Yes')]
st=ParagraphStyle('cell',fontName='Helvetica',fontSize=13,leading=16,textColor=HexColor(INK));hd=ParagraphStyle('head',parent=st,fontName='Helvetica-Bold',fontSize=14)
headers=['Driving configuration','Steering','Pedals','Speed planner','Monza','Silverstone','Spa'];data=[[Paragraph(v,hd) for v in headers]]
for key,name,steer,pedals,planner in labels:
 row=[Paragraph(v,st) for v in (name,steer,pedals,planner)]
 for track in ['monza','silverstone','spa']:
  v=benchmark[key][track];time=f"{v['elapsed_time']:.1f} s" if v['completed_lap'] else 'Timeout (600 s)';status='Clean lap' if v['valid_lap'] else f"{v['off_track_events']} departures; invalid";col=GREEN if v['valid_lap'] else RED
  row.append(Paragraph(f'<b>{time}</b><br/><font color="{col}">{status}</font>',st))
 data.append(row)
t=Table(data,colWidths=[250,120,190,105,170,170,175],rowHeights=[44]+[56]*8)
t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),HexColor('#dfe8ef')),('ROWBACKGROUNDS',(0,1),(-1,-1),[white,HexColor('#edf2f6')]),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('LEFTPADDING',(0,0),(-1,-1),8),('RIGHTPADDING',(0,0),(-1,-1),8),('LINEBELOW',(0,0),(-1,-1),.4,HexColor(LINE))]))
t.wrapOn(c,1180,492);t.drawOn(c,50,H-160-492)
para('Clean = completed with zero departures. Other completions are invalid. Full-control imitation changes both training data and steering.',50,666,1180,13)
end()

begin(13,'What we built','Approximate physics and track limits; a simulated teacher; one deterministic final-test start. Real-car performance and broad reliability remain unestablished.')
photo('silverstone_full_action_screenshot.png',35,132,775,516)
para('Seven components.<br/>One working simulator.',852,153,378,36,bold=True)
para('Recorded examples became<br/>learned controls.',852,333,378,27)
para('The full learned driver<br/>completed clean laps<br/>on new circuits.',852,467,378,29,GREEN,True)
end()
c.save();print(O/'F1_Project_Presentation.pdf')
