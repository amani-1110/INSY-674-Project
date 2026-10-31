from pathlib import Path
from reportlab.pdfgen import canvas
r=Path.cwd();s=(r/'build_presentation_pdf.py').read_text(encoding='utf-8')
# Reuse the same native PDF drawing functions; export progressive diagram frames.
head=s[:s.index("start(1,")]
helpers=s[s.index('def local_image'):s.index("start(0,'Meet the circuits")]
ns={};exec(head+'\n'+helpers,ns)
tmp=r/'tmp/pdfs/pipeline_build';tmp.mkdir(parents=True,exist_ok=True)
ns['c']=canvas.Canvas(str(tmp/'pipeline_build.pdf'),pagesize=(1280,720))
entries=[(50,189,240,126,'1','Track geometry','Road shape, bends, widths'),(330,189,240,126,'2','Vehicle physics','Controls become movement'),(650,189,260,126,'3','Simulator','State, time, progress, view'),(50,414,240,126,'4','F1 telemetry','Recorded positions and pedals'),(360,414,240,126,'5','Alignment','Match recordings to the road'),(680,414,240,126,'6','Learning','Examples become controls'),(990,189,240,126,'7','Evaluation','Rules, baselines, results')]
for step in range(1,8):
 ns['start'](0,'Seven components become one driving system','',f'Build {step}/7. Actual project components; numbers match the original team brief.')
 for entry in entries[:step]:ns['node'](*entry)
 if step>=3:
  ns['arrow']([(290,219),(307,219),(307,166),(780,166),(780,189)])
  ns['arrow']([(570,253),(650,253)])
 if step>=5:
  ns['arrow']([(290,477),(360,477)]);ns['arrow']([(170,315),(170,365),(480,365),(480,414)])
 if step>=6:
  ns['arrow']([(600,477),(680,477)]);ns['arrow']([(800,414),(800,315)])
 if step>=7:ns['arrow']([(910,253),(990,253)])
 ns['talk'](['The track gives the road shape and boundaries.','Physics defines how controls move the car.','The simulator joins the road and physics into a timed driving session.','Telemetry supplies real recorded examples.','Alignment matches those examples to the road.','Learning turns matched examples into control predictions.','Evaluation compares drivers and reports completion, departures and time.'][step-1])
 ns['end']()
ns['c'].save();print(tmp/'pipeline_build.pdf')
