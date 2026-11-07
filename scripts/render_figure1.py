"""Render an illustrative MuJoCo scene for Figure 1 (not an experiment snapshot).

Usage: python scripts/render_figure1.py --source /path/to/LATTE-MV/MujocoRecon \
           --output Figures/figure1.png --workdir /path/to/scratch
Dependencies: mujoco, numpy, scipy, matplotlib, Pillow.
Robot meshes/MJCF: sastry-group/LATTE-MV, MujocoRecon (BSD-3-Clause).
The gantries, coordinates, poses, ball state, and annotations are illustrative.
"""
from pathlib import Path
import argparse, copy, json, xml.etree.ElementTree as ET
import numpy as np
import mujoco
from scipy.optimize import least_squares
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle, FancyArrowPatch

P=argparse.ArgumentParser()
P.add_argument('--source',type=Path,required=True)
P.add_argument('--output',type=Path,required=True)
P.add_argument('--workdir',type=Path,required=True)
a=P.parse_args();a.workdir.mkdir(parents=True,exist_ok=True);a.output.parent.mkdir(parents=True,exist_ok=True)
W,H=2400,1450
src=ET.parse(a.source/'iiwa14_gantry.xml').getroot()
r=ET.Element('mujoco',model='illustrative_two_player_table_tennis')
ET.SubElement(r,'compiler',angle='radian',meshdir=str((a.source/'assets').resolve()))
ET.SubElement(r,'option',integrator='implicitfast')
v=ET.SubElement(r,'visual')
ET.SubElement(v,'global',offwidth=str(W),offheight=str(H))
ET.SubElement(v,'quality',shadowsize='4096',offsamples='8')
ET.SubElement(v,'headlight',diffuse='.5 .5 .5',ambient='.25 .25 .25',specular='.1 .1 .1')
ET.SubElement(v,'rgba',haze='1 1 1 1')
r.append(copy.deepcopy(src.find('default')));r.append(copy.deepcopy(src.find('asset')))
w=ET.SubElement(r,'worldbody')
ET.SubElement(w,'light',pos='0 -3 7',dir='0 .2 -1',directional='true',diffuse='.55 .55 .55',ambient='.15 .15 .15')
ET.SubElement(w,'light',pos='0 4 5',dir='0 -.4 -1',directional='true',diffuse='.30 .30 .30',castshadow='false')
def geom(parent=w,**kwargs):
    args={'contype':'0','conaffinity':'0'};args.update({k:str(x) for k,x in kwargs.items()});return ET.SubElement(parent,'geom',**args)
def box(name,pos,size,rgba):return geom(name=name,type='box',pos=' '.join(map(str,pos)),size=' '.join(map(str,size)),rgba=rgba)
geom(name='floor',type='plane',size='200 200 .05',rgba='.985 .986 .985 1')
# Standard table footprint. Net at x=0.
box('table',(0,0,.735),(1.37,.7625,.025),'.075 .29 .36 1')
box('table_edge',(0,0,.704),(1.39,.7825,.014),'.12 .17 .20 1')
for y in [-.75,.75]:box('sideline'+str(y),(0,y,.762),(1.37,.008,.002),'1 1 1 1')
for x in [-1.36,1.36]:box('endline'+str(x),(x,0,.762),(.008,.7625,.002),'1 1 1 1')
box('centerline',(0,0,.762),(1.37,.004,.002),'1 1 1 1')
for x in [-.95,.95]:
    for y in [-.55,.55]:box(f'leg{x}{y}',(x,y,.36),(.035,.035,.35),'.22 .26 .28 1')
box('net',(0,0,.84),(.006,.80,.075),'.30 .35 .38 .43')
box('net_top',(0,0,.917),(.009,.82,.007),'1 1 1 1')
for y in [-.81,.81]:box('netpost'+str(y),(0,y,.82),(.014,.014,.12),'.18 .22 .25 1')
# A schematic, visible XY gantry under each real KUKA model.
for side,base in [('left',-2.15),('right',2.15)]:
    for xx in [base-.37,base+.37]:
        box(f'{side}_rail_{xx}',(xx,0,.31),(.032,.98,.042),'.42 .49 .54 1')
        for yy in [-.88,.88]:box(f'{side}_foot{xx}{yy}',(xx,yy,.14),(.055,.055,.14),'.29 .35 .40 1')
    for yy in [-.94,.94]:box(f'{side}_end{yy}',(base,yy,.31),(.4,.035,.042),'.42 .49 .54 1')
    box(side+'_moving_crossrail',(base,0,.36),(.4,.06,.025),'.63 .69 .72 1')
    box(side+'_carriage',(base,0,.395),(.16,.12,.025),'.23 .30 .35 1')
    body=copy.deepcopy(src.find('worldbody/body'))
    # Prefix every robot identifier, leaving shared class/mesh/material names intact.
    for e in body.iter():
        if 'name' in e.attrib:e.set('name',side+'_'+e.get('name'))
        if e.tag=='geom' and e.get('class')=='collision':e.set('rgba','0 0 0 0')
    body.set('pos',f'{base} 0 .42');body.set('quat','1 0 0 0' if side=='left' else '0 0 0 1')
    # Fix gantry joints for this static render; the rails show the motion envelope.
    for e in list(body):
        if e.tag=='joint':body.remove(e)
    flange=body.find(".//body[@name='"+side+"_link7']")
    racket=flange.find('body');flange.remove(racket)
    racket=ET.SubElement(flange,'body',name=side+'_paddle',pos='0 0 .045')
    geom(racket,name=side+'_handle',type='cylinder',pos='0 0 .07',size='.015 .07',rgba='.30 .19 .12 1')
    geom(racket,name=side+'_racket_edge',type='cylinder',pos='0 0 .22',size='.10 .008',quat='.707107 .707107 0 0',rgba='.10 .12 .14 1')
    geom(racket,name=side+'_racket_face',type='cylinder',pos='0 -.009  .22',size='.096 .002',quat='.707107 .707107 0 0',rgba='.77 .10 .12 1')
    ET.SubElement(racket,'site',name=side+'_racket_center',pos='0 0 .22',size='.001',rgba='0 0 0 0')
    w.append(body)
ball=np.array([-.35,-.12,1.24]);velocity=np.array([3.8,.35,-.65])
geom(name='ball',type='sphere',pos=' '.join(map(str,ball)),size='.026',rgba='1 .60 .08 1')
xml=a.workdir/'figure1_scene.xml';ET.ElementTree(r).write(xml,encoding='unicode')
m=mujoco.MjModel.from_xml_path(str(xml));d=mujoco.MjData(m)
# Choose a symmetric, reachable pose by inverse kinematics of the paddle center.
poses={}
for side,sign in [('left',1),('right',-1)]:
    ids=[m.joint(side+'_joint'+str(i)).qposadr[0] for i in range(1,8)]
    site=m.site(side+'_racket_center').id;bid=m.body(side+'_paddle').id
    target=np.array([-1.42*sign,-.05*sign,1.18])
    initial=np.array([0,.75,0,-1.40,0,.4,0])
    def objective(q):
        d.qpos[ids]=q;mujoco.mj_forward(m,d)
        R=d.xmat[bid].reshape(3,3)
        return np.r_[4*(d.site_xpos[site]-target), .3*(R[:,1]-np.array([-.8*sign,.6,0])), .12*(R[:,2]-np.array([0,0,1])),.015*(q-initial)]
    q=least_squares(objective,initial,bounds=(-2.0*np.ones(7),2.0*np.ones(7)),max_nfev=500).x
    d.qpos[ids]=q;poses[side]=q.tolist()
mujoco.mj_forward(m,d)
centers={side:d.site_xpos[m.site(side+'_racket_center').id].copy() for side in ['left','right']}
cam=mujoco.MjvCamera();cam.lookat[:]=[0,0,.55];cam.distance=6.4;cam.azimuth=90;cam.elevation=-30
renderer=mujoco.Renderer(m,height=H,width=W);opt=mujoco.MjvOption();opt.geomgroup[3]=0
renderer.update_scene(d,camera=cam,scene_option=opt)
raw=renderer.render()
# Use MuJoCo's actual render camera rather than reconstructing its convention.
c0,c1=renderer.scene.camera
camera_eye=(np.array(c0.pos)+np.array(c1.pos))/2
camera_forward=np.array(c0.forward).copy()
camera_up=np.array(c0.up).copy()
print('Render camera:',camera_eye,camera_forward,camera_up)
renderer.close()
from PIL import Image
Image.fromarray(raw).save(a.workdir/'figure1_raw.png')
# Project vector annotations through exactly the same centered camera.
# MuJoCo azimuth=90 puts the render eye on -y; +x appears on the right.
eye=camera_eye
f=camera_forward/np.linalg.norm(camera_forward)
up=camera_up/np.linalg.norm(camera_up)
right=np.cross(f,up);right/=np.linalg.norm(right)
focal=H/(2*np.tan(np.deg2rad(m.vis.global_.fovy)/2))
def proj(p):
    t=np.array(p)-eye;z=t@f
    return np.array([W/2+focal*(t@right)/z,H/2-focal*(t@up)/z])
fig=plt.figure(figsize=(10,4.375),dpi=240);ax=fig.add_axes([0,0,1,1]);ax.imshow(raw);ax.set(xlim=(0,W),ylim=(1300,250));ax.axis('off')
blue='#20709a';green='#237a61';gold='#bd7b0c';ink='#263440';red='#b43c42'
# Workspace outlines and translucent fills lie on the gantry plane.
for base in [-2.15,2.15]:
    pts=np.array([proj((base+dx,y,.43)) for dx,y in [(-.40,-.98),(.40,-.98),(.40,.98),(-.40,.98)]])
    ax.add_patch(Polygon(pts,closed=True,fc=blue,alpha=.08,ec='none'))
    ax.add_patch(Polygon(pts,closed=True,fill=False,ec=blue,lw=1.5,ls=(0,(5,3))))
# Five target landing choices, on the right half only.
targets=[(.43,-.48,.775),(1.08,-.48,.775),(.76,0,.775),(.43,.48,.775),(1.08,.48,.775)]
for i,p in enumerate(targets,1):
    xy=proj(p);ax.add_patch(Circle(xy,24,fc='#f3c45e',ec='#8c5900',lw=1.0,zorder=7))
    ax.text(*xy,str(i),ha='center',va='center',fontsize=11,weight='bold',color='#473000',zorder=8)
# Preparation markers occupy right robot's workspace, separated laterally.
preps=[(2.40,-.68,.44),(2.40,0,.44),(2.40,.68,.44)]
for i,p in enumerate(preps,1):
    xy=proj(p);ax.add_patch(Circle(xy,27,fc='#eef7f1',ec=green,lw=1.4,zorder=8))
    ax.text(*xy,rf'$m_{i}$',ha='center',va='center',fontsize=11,color=green,zorder=9)
# Ghost paddle target poses, plus short normal vectors.
for side,center in centers.items():
    c=center+np.array([.22 if side=='left' else -.22,-.15,.30])
    angles=np.linspace(0,2*np.pi,80)
    sgn=1 if side=='left' else -1
    pts=np.array([proj(c+np.array([.6,.8*sgn,0])*.10*np.cos(t)+np.array([0,0,.10*np.sin(t)])) for t in angles])
    ax.add_patch(Polygon(pts,closed=True,fc=red,alpha=.12,ec='none',zorder=9))
    ax.plot(pts[:,0],pts[:,1],color=red,lw=1.4,ls=(0,(4,2)),zorder=10)
    h=np.array([proj(c+[0,0,-.10]),proj(c+[0,0,-.20])]);ax.plot(h[:,0],h[:,1],color=red,lw=2,ls='--')
    tip=c+[.16 if side=='left' else -.16,-.12,0]
    ax.add_patch(FancyArrowPatch(proj(c),proj(tip),arrowstyle='-|>',mutation_scale=11,color=red,lw=1.3,zorder=11))
# Ball marker and velocity share a physical 3-D origin; velocity length is illustrative.
bp=proj(ball);end=proj(ball+.19*velocity)
ax.add_patch(FancyArrowPatch(bp,end,arrowstyle='-|>',mutation_scale=19,color='#dd7d0b',lw=2.2,zorder=15))
ax.add_patch(Circle(bp,8,fc='#ffad24',ec='#775000',lw=1,zorder=16))
def label(text,xy,p,color=ink,align='center'):
    ax.annotate(text,xy=proj(p),xytext=xy,ha=align,va='center',fontsize=12,color=color,
        bbox=dict(boxstyle='round,pad=.3',fc='white',ec='none',alpha=.94),
        arrowprops=dict(arrowstyle='-',color=color,lw=1.1,connectionstyle='arc3,rad=0'),zorder=20)
label('Gantry movable region',(370,1170),(-2.48,-.67,.43),blue)
label('Gantry movable region',(2030,1170),(2.48,-.67,.43),blue)
label('KUKA iiwa + racket',(320,410),d.xpos[m.body('left_link4').id])
label('KUKA iiwa + racket',(2080,410),d.xpos[m.body('right_link4').id])
label('Target racket pose',(700,350),centers['left']+[.22,-.15,.30],red)
label('Target racket pose',(1700,350),centers['right']+[-.22,-.15,.30],red)
label('Ball',(1000,430),ball,gold)
label(r'Ball velocity $\mathbf{v}^{b}$',(1320,350),ball+.12*velocity,gold)
label(r'Return choices $k_1,\ldots,k_5$',(1480,1150),(.76,-.48,.775),gold)
label('Preparation positions\n'+r'$m_1,m_2,m_3$',(2020,1055),(2.40,-.68,.44),green)
# Symmetric camera means the table center and net remain on the image centerline.
fig.savefig(a.output,dpi=240,facecolor='white',bbox_inches='tight',pad_inches=.04)
fig.savefig(a.output.with_suffix('.pdf'),facecolor='white',bbox_inches='tight',pad_inches=.04)
plt.close(fig)
(a.workdir/'figure1_parameters.json').write_text(json.dumps({'illustrative':True,'ball_position':ball.tolist(),'ball_velocity':velocity.tolist(),'joint_poses':poses,'racket_centers':{k:v.tolist() for k,v in centers.items()},'return_targets':targets,'preparation_positions':preps},indent=2))
print('Saved',a.output,'and vector-annotation PDF; MuJoCo',mujoco.__version__)
print('Racket centers:',centers)
