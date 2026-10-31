from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from track import Track
from physics.physics import update_car
out=Path('output/pdf/presentation/media');out.mkdir(exist_ok=True)
for name in ['monza','silverstone','spa']:
 t=Track(name);points=np.array([t.position(s) for s in np.linspace(0,t.length,600)])
 fig,ax=plt.subplots(figsize=(5,3.7),facecolor='white');ax.plot(*points.T,color='#235f88',lw=4,solid_capstyle='round');ax.scatter(*points[0],color='#d96942',s=65,zorder=3);ax.set_aspect('equal');ax.axis('off');fig.tight_layout(pad=.3);fig.savefig(out/f'{name}_map.png',dpi=180,bbox_inches='tight');plt.close(fig)
state=dict(x=0.,y=0.,heading=0.,speed=0.);rows=[]
for i in range(200):
 time=i*.1;act=dict(steering=0.,throttle=1. if time<10 else 0.,brake=0. if time<10 else .6);rows.append((time,state['speed']*3.6));state=update_car(state,act,.1)
a=np.array(rows);fig,ax=plt.subplots(figsize=(9,3.2),facecolor='white');ax.plot(a[:,0],a[:,1],color='#235f88',lw=3);ax.axvline(10,color='#c56545',ls='--');ax.text(2,a[:,1].max()*.85,'Accelerate',fontsize=13);ax.text(11,a[:,1].max()*.85,'Brake',fontsize=13);ax.set(xlabel='Simulated time (s)',ylabel='Speed (km/h)');ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.15);fig.tight_layout();fig.savefig(out/'physics_acceleration_braking.png',dpi=180);plt.close(fig)
print('Created circuit maps from actual project geometry and a controlled run of the shared physics code.')
