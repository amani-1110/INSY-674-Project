from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor
from reportlab.platypus import Paragraph
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from PIL import Image
from pypdf import PdfReader,PdfWriter
r=Path.cwd();o=r/'output/pdf/presentation';tmp=r/'tmp/pdfs/canva_appendix';tmp.mkdir(parents=True,exist_ok=True)
c=canvas.Canvas(str(tmp/'optional_slides.pdf'),pagesize=(1280,720))
def text(s,x,y,w,size=22,bold=False):
 p=Paragraph(s,ParagraphStyle('p',fontName='Helvetica-Bold' if bold else 'Helvetica',fontSize=size,leading=size*1.25,textColor=HexColor('#1d2d3e')));_,h=p.wrap(w,800);p.drawOn(c,x,720-y-h)
def image(path,x,y,w,h):
 im=Image.open(path);iw,ih=im.size;q=min(w/iw,h/ih);ww,hh=iw*q,ih*q;c.drawImage(ImageReader(im),x+(w-ww)/2,720-y-h+(h-hh)/2,width=ww,height=hh)
def page(n,title,assets,caption):
 c.setFillColor(HexColor('#ffffff'));c.rect(0,0,1280,720,fill=1,stroke=0)
 text('OPTIONAL VISUALS',50,28,1000,13,True);text(title,50,66,1160,36,True);text(str(n),1190,45,45,15)
 for x,path,label in assets:
  image(path,x,156,570,423);text(label,x,603,570,18,True)
 text(caption,50,658,1180,13);c.showPage()
m=r/'output/analysis_pack/media'
page(14,'Monza: additional turn screenshots',[(50,m/'monza_turn_2_screenshot.png','Assisted driver: second Monza turn'),(660,m/'monza_full_action_screenshot.png','Learned steering and pedals at Monza')],'Actual simulator replays. Choose one as an alternative photo; these independent runs are not synchronized.')
page(15,'Silverstone: additional screenshots',[(50,m/'silverstone_turn_1_screenshot.png','Assisted driver at Silverstone'),(660,m/'silverstone_full_action_screenshot.png','Learned full-control driver at Silverstone')],'Silverstone is the UK transfer circuit. These images are not Spa footage.')
page(16,'Track geometry: detailed reference plots',[(50,r/'output/monza_track.png','Monza: centreline, boundaries and curvature'),(660,r/'output/silverstone_track.png','Silverstone: centreline, boundaries and curvature')],'Detailed project plots for an optional technical slide; use the simple maps in the main story for the audience introduction.')
page(17,'Alignment: additional recorded laps',[(50,r/'output/pipeline_500m/lap11_overlay.png','Lap 11: transformed telemetry on the map'),(660,r/'output/pipeline_500m/lap14_overlay.png','Lap 14: transformed telemetry on the map')],'The transform was fitted using training lap 5. Lap 14 was excluded from model fitting, then inspected during development; its scores are exploratory.')
page(18,'Full-lap animation references',[(50,m/'monza_full_action_screenshot.png','Optional_Stage3_Monza_FullLap_10x.gif'),(660,m/'silverstone_full_action_screenshot.png','Optional_Stage3_Silverstone_FullLap_10x.gif')],'PDF uses stills. Insert the real replay GIF files in Canva for motion; full-lap playback is nominal 10x speed.')
c.save();w=PdfWriter()
for path in [o/'F1_Project_Presentation.pdf',tmp/'optional_slides.pdf']:
 for p in PdfReader(path).pages:w.add_page(p)
final=o/'F1_Presentation_With_Optional_Visuals.pdf'
with final.open('wb') as f:w.write(f)
assert len(PdfReader(final).pages)==18
print(final)
