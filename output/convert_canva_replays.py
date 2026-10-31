from pathlib import Path
import subprocess,json
from PIL import Image
import imageio_ffmpeg
r=Path.cwd();src=r/'output/F1_Team_Presentation_Package/02_GIFs';out=r/'output/canva_video';out.mkdir(exist_ok=True)
clips=[('Slide_01_Monza_LearnedDriver_FullLap_10x.gif','Monza_FullLap_LearnedDriver_10x'),('Optional_Stage3_Silverstone_FullLap_10x.gif','Silverstone_FullLap_LearnedDriver_10x'),('Slides_04_07_Stage1_Monza_NoSpeedPlanner_1x.gif','Stage1_Monza_NoSpeedPlanner_1x'),('Slides_04_08_Stage2_Monza_WithSpeedPlanner_1x.gif','Stage2_Monza_WithSpeedPlanner_1x'),('Slides_04_09_Stage3_Monza_LearnedSteeringAndPedals_1x.gif','Stage3_Monza_LearnedFullControl_1x'),('Slides_11_13_Stage3_Silverstone_TransferTurn_1x.gif','Stage3_Silverstone_TransferTurn_1x')]
items=[]
for gif,name in clips:
 im=Image.open(src/gif);duration=0
 for i in range(im.n_frames):im.seek(i);duration+=im.info.get('duration',100)
 target=out/(name+'.mp4')
 subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-i',str(src/gif),'-movflags','+faststart','-pix_fmt','yuv420p','-vf','pad=ceil(iw/2)*2:ceil(ih/2)*2','-c:v','libx264','-crf','17',str(target)],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
 items.append(dict(name=name,path=str(target),source=gif,width=im.width,height=im.height,frames=im.n_frames,duration_s=duration/1000))
(out/'manifest.json').write_text(json.dumps(items,indent=2),encoding='utf-8');print(json.dumps(items))
