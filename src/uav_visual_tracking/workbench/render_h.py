"""1920x1080 presentation compositor. It never writes experiment data."""
from bisect import bisect_right
import subprocess
import cv2
import numpy as np
from .common import SETTINGS,read,save
from .timeline import Timeline
from .control import angles,geom
from .u3_g3 import theta
from .u3_gazebo import quat,camera_transform
from .u3_surface import Surface,look_at

STYLE=read(SETTINGS/'style.json')
W,HGT=STYLE['width'],STYLE['height']


def color(key):
    c=STYLE[key].lstrip('#')
    return tuple(int(c[i:i+2],16) for i in (4,2,0))


def txt(im,s,x,y,size=.61,key='white',weight=1):
    cv2.putText(im,str(s),(int(x),int(y)),cv2.FONT_HERSHEY_SIMPLEX,size,color(key),weight,cv2.LINE_AA)


def panel(im,x,y,w,h,title):
    a=im[y:y+h,x:x+w]
    a[:]=color('panel');cv2.rectangle(a,(0,0),(w-1,h-1),color('border'),1)
    txt(a,title,15,29,.64,'white',1)
    return a


def segment(im,points,key,width=2):
    points=np.asarray(points,dtype=np.int32).reshape(-1,2)
    if len(points)>1:cv2.polylines(im,[points],False,color(key),width,cv2.LINE_AA)


def dashed(im,points,key,width=2):
    for a,b in zip(points[:-1],points[1:]):
        a,b=np.asarray(a),np.asarray(b);length=np.linalg.norm(b-a)
        steps=max(1,int(length/10))
        for i in range(steps):
            segment(im,[a+(b-a)*i/steps,a+(b-a)*(i+.56)/steps],key,width)


def marker(im,p,key,square,outer=False):
    x,y=map(int,p)
    r=10 if outer else 6
    if square:cv2.rectangle(im,(x-r,y-r),(x+r,y+r),color(key),2,cv2.LINE_AA)
    else:cv2.circle(im,(x,y),r+2,color(key),2,cv2.LINE_AA)


def mark_plan(im,r,proj,targets=True):
    p=r['plan'] or r['historical']
    if not p:return
    history=r['plan'] is None
    v=[p['current_uav_pose']['position_m'],*[q['position_m'] for q in p['planned_trajectory']]]
    dashed(im,proj(v),'expired' if history else 'cyan',2)
    for key,k in [('0p2','green'),('0p5','orange')]:
        style='expired' if history else k
        marker(im,proj([p[f'uav_reference_{key}']])[0],style,True,key=='0p5')
        target=p[f'target_prediction_{key}']
        if target is not None and targets:marker(im,proj([target])[0],style,False,key=='0p5')


def visibility(uv,in_front):
    if not in_front:return 'Behind camera'
    lo=uv.min(0);hi=uv.max(0)
    if hi[0]<0 or hi[1]<0 or lo[0]>=640 or lo[1]>=480:return 'Outside image'
    if np.any(lo<0) or np.any(hi>=[640,480]):return 'Target cropped'
    return 'None'


class View:
    def __init__(self,path,derived=None):
        self.tl=Timeline(path,derived)
        self.positions=np.array([q['camera_pose'][:3] for q in self.tl.states])
        self.times=np.array(self.tl.times)
        self.omega=np.rad2deg([q['joint_v_rad_s'] for q in self.tl.states])
        self.K=np.asarray(self.tl.spec['K']);self.g=self.tl.spec['geometry'].copy();self.g['tilt']=np.asarray(self.g['tilt'])
        self.goal=np.asarray(self.tl.spec['composition']['goal_px']);self.banks=self.tl.spec['banks']
        self.surface=Surface();self.mesh=self.surface.vertices.reshape(57,96,3).mean(1)
        self.cam=look_at(np.array([128.,-185.,155.]),np.array([0.,-5.,101.]))
        self.photos=sorted(self.tl.quality,key=lambda q:q['pts']);self.tr=self.tl.transitions

    def pose(self,r):
        if r is None or r['state'] is None:return None,None,None
        p,T=camera_transform(r['state']['camera_pose'])
        return p,T,angles(p,p+T[2,:3])

    def offset(self,r):
        if r is None or r['plan'] is None:return None
        xyz,T,_=self.pose(r)
        if T is None:return None
        p=r['plan'];phase=p['phase_at_capture_deg']+(p['cv_rate_deg_s'] or 0)*(r['t']-p['source_capture_pts'])
        uv,q=geom(self.banks[r['station']],phase,T,self.K,self.g,(640,480))
        return np.asarray(q['center_px'])-self.goal,uv,q

    def global_map(self,im,r):
        if r is None:return
        h,w=im.shape[:2];area=im[40:h-36,12:w-12]
        ah,aw=area.shape[:2]
        def proj(p):
            cam=np.asarray(p)@self.cam[:3,:3].T+self.cam[:3,3]
            return np.rint(cam[:,:2]/cam[:,2:]*(ah*1.3)+[aw*.49,ah*.46]).astype(int)
        p,_,_=self.pose(r)
        if p is not None:
            s=self.surface;phase=theta(s,r['state'])
            segment(area,proj([[0,0,0],s.hub]),'muted',2)
            for j in range(3):segment(area,proj(self.mesh@s.rotation(phase,j).T+s.hub),'muted',3)
            points=np.asarray(self.banks[r['station']]['points'])@s.rotation(phase,0).T+s.hub
            for v in proj(points[::max(1,len(points)//60)]):cv2.circle(area,tuple(v),2,color('purple'),-1)
            segment(area,proj(self.positions[:r['state_ix']+1:4]),'history',2)
            self.los(area,r,proj,15)
            mark_plan(area,r,proj)
        txt(im,'White actual LOS  |  gray executed  |  cyan planned  |  purple patch',15,h-11,.49,'muted')

    def los(self,im,r,proj,length):
        p,_,_=self.pose(r)
        if p is None:return
        rot=quat(r['state']['camera_pose'][3:]);q=p+length*rot[:,0]
        rays=[q+length*(a*320/self.K[0,0]*rot[:,1]+b*240/self.K[1,1]*rot[:,2])
              for a,b in [(-1,-1),(1,-1),(1,1),(-1,1)]]
        segment(im,proj([p,q]),'white',2)
        for x in rays:segment(im,proj([p,x]),'white',1)
        segment(im,proj([*rays,rays[0]]),'white',1)
        cv2.circle(im,tuple(proj([p])[0]),5,color('white'),-1)

    def local(self,im,r,other=None,rr=None,station_anchor=False):
        if r is None and rr is None:return
        h,w=im.shape[:2]
        a=im[42:h-77,58:w-24];ah,aw=a.shape[:2]
        rows=[(self,r,'baseline' if other else 'history')]
        if other:rows.append((other,rr,'joint'))
        if station_anchor:
            z=self.banks[(r or rr)['station']]['position_ref_m'][2]
            xmin,xmax=-STYLE['local_x_half_m'],STYLE['local_x_half_m']
            zmin,zmax=z-STYLE['local_z_half_m'],z+STYLE['local_z_half_m']
            scale=min((aw-12)/(xmax-xmin),(ah-12)/(zmax-zmin))
        else:
            base=np.array([b['position_ref_m'] for b in self.banks])
            zmin,zmax=float(base[:,2].min()-6),float(base[:,2].max()+6)
            xmin,xmax=-STYLE['local_x_half_m'],STYLE['local_x_half_m']
            scale=min((aw-12)/(xmax-xmin),(ah-12)/(zmax-zmin))
        cx=(xmin+xmax)/2;cz=(zmin+zmax)/2
        def proj(p):
            q=np.asarray(p)
            return np.rint(np.stack([(q[:,0]-cx)*scale+aw*.5,(cz-q[:,2])*scale+ah*.5],axis=1)).astype(int)
        xlo,xhi=cx-aw*.5/scale,cx+aw*.5/scale
        zlo,zhi=cz-ah*.5/scale,cz+ah*.5/scale
        step=5 if station_anchor else 10
        for u in np.arange(np.ceil(xlo/step)*step,xhi,step):
            x=proj([[u,0,cz]])[0,0];segment(a,[(x,0),(x,ah)],'grid',1)
            if x>4 and x<aw-30:txt(im,f'{u:.0f}',x+59,h-57,.43,'muted')
        for z in np.arange(np.ceil(zlo/step)*step,zhi,step):
            y=proj([[cx,0,z]])[0,1];segment(a,[(0,y),(aw,y)],'grid',1)
            if 8<y<ah-3:txt(im,f'{z:.0f}',16,y+47,.44,'muted')
        for view,row,key in rows:
            visible=row is not None
            past=row if visible else view.tl.rows[max(view.tl.rows)]
            p,_,_=view.pose(past)
            if p is None:continue
            end=past['state_ix'];segment(a,proj(view.positions[:end+1:5]),key,3)
            if visible:view.los(a,row,proj,7)
            if other:
                xy=proj([p])[0];cv2.circle(a,tuple(xy),7,color(key),2)
            if visible and (view is other or other is None):mark_plan(a,row,proj,targets=station_anchor)
        txt(im,'X (m) / Z (m), equal scale',15,h-29,.49,'muted')
        if other:
            p=self.pose(r)[0];q=other.pose(rr)[0]
            diff='N/A' if p is None or q is None else f'{q[2]-p[2]:+.2f} m'
            txt(im,f'Height difference D - B   {diff}',15,h-9,.57,'white')
            txt(im,'B baseline',w-266,h-29,.50,'baseline');txt(im,'D joint',w-141,h-29,.50,'joint')
        else:
            if r['plan']:
                p=r['plan'];z0=p['current_uav_pose']['position_m'][2]
                txt(im,f'dZ +0.2s {p["uav_reference_0p2"][2]-z0:+.2f} m',15,h-8,.53,'green')
                txt(im,f'dZ +0.5s {p["uav_reference_0p5"][2]-z0:+.2f} m',240,h-8,.53,'orange')
            else:txt(im,'dZ +0.2s  N/A      dZ +0.5s  N/A',15,h-8,.53,'muted')

    def image(self,im,r):
        h,w=im.shape[:2]
        if r is None:
            txt(im,'Recording ended',15,58,.70,'muted');return
        rgb=cv2.imread(str(self.tl.path/r['frame']['file']))
        if rgb is None:raise IOError(r['frame']['file'])
        goal=tuple(map(round,self.goal));cv2.circle(rgb,goal,54,color('white'),1)
        cv2.drawMarker(rgb,goal,color('white'),cv2.MARKER_CROSS,15,1)
        q=self.offset(r)
        def rect(bb,key):
            x,y,X,Y=np.rint(np.clip(bb,[-640,-480,-640,-480],[1280,960,1280,960])).astype(int)
            cv2.rectangle(rgb,(x,y),(X,Y),color(key),2,cv2.LINE_AA)
        if q:
            _,uv,quality=q
            if visibility(uv,quality['all_in_front']) not in ('Behind camera','Outside image'):
                rect(np.r_[uv.min(0),uv.max(0)],'white')
        if r['plan']:
            for k,c in [('0p2','green'),('0p5','orange')]:
                box=r['plan'][f'image_prediction_{k}']
                if box:rect(box['bbox_px'],c)
        iw=min(w-22,int((h-46)*4/3));ih=int(iw*.75);x=(w-iw)//2;y=38
        im[y:y+ih,x:x+iw]=cv2.resize(rgb,(iw,ih),interpolation=cv2.INTER_LINEAR if iw>640 else cv2.INTER_AREA)

    def metadata(self,im,r,large=False):
        h,w=im.shape[:2]
        if r is None:r=dict(status='unavailable',mode='ENDED',obs=None,plan=None,reason='Recording ended',station=None,
                            t=self.times[-1],state=None)
        offset=self.offset(r);p=r['plan'];status=r['status'];o=r['obs'];_,_,yp=self.pose(r)
        sz=.58 if large else .51;gap=26 if large else 24
        def entry(label,value,x,y,key='white'):
            dx=244 if large else 212
            assert cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,sz,1)[0][0]<dx-5,label
            assert cv2.getTextSize(value,cv2.FONT_HERSHEY_SIMPLEX,sz,1)[0][0]<(w//2 if large else w)-16-dx-5,value
            txt(im,label,x,y,sz,'muted');txt(im,value,x+dx,y,sz,key)
        status_key='green' if status=='live' else 'orange' if status=='expired' else 'reject' if status=='rejected' else 'muted'
        p_status=('Live / CV' if p['target_prediction_0p2'] else 'Image only / world N/A' if p['image_prediction_0p2']
                  else 'Reference only / target N/A') if p else (
            'Expired / history only' if r.get('historical') and status=='expired' else 'N/A')
        age=f'{1000*(r["t"]-o["capture_pts"]):.0f} ms' if o else '--'
        comp=f'{offset[0][0]:+.1f} / {offset[0][1]:+.1f}' if offset else '-- / --'
        region='--' if r['station'] is None else f'A / {r["station"]+1:02d}'
        mode=r['mode']
        cmd=r.get('command') or {};cause=cmd.get('reason','')
        trigger=(p['body_triggers'][0].replace('_',' ') if p and p['body_triggers'] else
                 'Joint correction' if cause=='joint_visual_body' else 'Static transit' if cause=='static_radial_transit'
                 else 'Station reference' if cause else '--')
        sight='--' if yp is None else f'{yp[0]:+.1f} / {yp[1]:+.1f} deg'
        progress=sum(q['region_complete'] and q['pts']<=r['t'] for q in self.photos)
        cut=[q['sim_s'] for q in self.tr if q['sim_s']<=r['t']]
        reason=visibility(offset[1],offset[2]['all_in_front']) if offset else r['reason']
        fields=[('Mode',mode,'white'),('Target region',region,'white'),('Observation status',status.upper(),'status'),
                ('Prediction status',p_status,'status'),('State age',age,'white'),('Center offset X/Y (px)',comp,'white'),
                ('Rejection reason',reason,'status'),('Yaw / pitch',sight,'white'),('Trigger',trigger,'white'),
                ('Qualified / last cut',f'{progress}/12  |  '+(f'{cut[-1]:.2f} s' if cut else '--'),'white')]
        n=5 if large else 10
        for j,(label,value,key) in enumerate(fields):
            x=16+(w//2)*(j//n);y=20+gap*(j%n)
            entry(label,value,x,y,status_key if key=='status' else key)


def omega_chart(im,view,t,end,other=None):
    h,w=im.shape[:2];x0,x1=82,w-33;y0,y1=47,h-31;mid=(y0+y1)/2
    radius=STYLE['omega_abs_deg_s']
    def point(x,v):return (int(x0+x/end*(x1-x0)),int(mid-v/radius*(y1-y0)/2))
    for v in [-1.5,0,1.5]:
        y=point(0,v)[1];segment(im,[(x0,y),(x1,y)],'muted' if v==0 else 'grid',1)
        txt(im,f'{v:+.1f}',20,y+5,.47,'muted')
    for x in np.arange(0,end+.01,2):
        px=point(x,0)[0];segment(im,[(px,y0),(px,y1)],'grid',1)
        txt(im,f'{x:.0f}',px-6,y1+20,.47,'muted')
    count=bisect_right(view.times,t);keys=np.where(np.abs(view.omega[:count])<.025,0,np.sign(view.omega[:count]))
    boundaries=np.r_[0,np.flatnonzero(np.diff(keys))+1,len(keys)]
    for a,b in zip(boundaries[:-1],boundaries[1:]):
        if b<=a:continue
        key={1:'green',0:'muted',-1:'orange'}[keys[a]]
        span=[point(x,v) for x,v in zip(view.times[max(0,a-1):b:2],view.omega[max(0,a-1):b:2])]
        span.append(point(view.times[b-1],view.omega[b-1]));segment(im,span,key,2)
    if other:
        for v,key,yy in [(view,'baseline',y1-3),(other,'joint',y1-11)]:
            for cut in v.tr:
                if cut['sim_s']>t:break
                xx=point(cut['sim_s'],0)[0];segment(im,[(xx,yy-7),(xx,yy)],key,2)
    xx=point(t,0)[0];segment(im,[(xx,y0-5),(xx,y1)],'white',1)
    txt(im,'+ Forward',w-452,28,.54,'green');txt(im,'Near zero',w-327,28,.54,'muted')
    txt(im,'- Reverse',w-204,28,.54,'orange');txt(im,'s',w-22,y1+20,.47,'muted')


def proof(view,r):
    if r is None:return None
    p=r['plan'];h=r['historical'];state=r['state']
    return dict(display_sim_pts=r['t'],frame_index=r['frame']['index'],source_file=r['frame']['file'],actual_pose_pts=state['sim_s'] if state else None,
        actual_mode=r['mode'],status=r['status'],origin_pts=p['display_pts'] if p else None,source_capture_pts=p['source_capture_pts'] if p else None,
        source_wall_pts=r['w'],valid_until=p['valid_until'] if p else None,valid_until_wall=p['valid_until_wall_s'] if p else None,
        live_uav_reference=bool(p),live_target_prediction=bool(p and p['target_prediction_0p2'] is not None),
        historical_origin_pts=h['display_pts'] if h else None,historical_only=bool(h),held=r['hold'],
        omega_last_pts=view.times[bisect_right(view.times,r['t'])-1] if bisect_right(view.times,r['t']) else None)


def render(run,out,ffmpeg,derived=None,comparison=None):
    scene=run.parent.name
    v=View(run,derived);d=View(comparison) if comparison else None
    n=max(max(v.tl.frames),max(d.tl.frames) if d else 0)
    end=float(np.ceil(n/20/2)*2)
    cmd=[str(ffmpeg),'-v','error','-n','-f','rawvideo','-pix_fmt','bgr24','-s',f'{W}x{HGT}','-r','20','-i','-',
         '-an','-c:v','libx264','-preset','fast','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(out)]
    cv2.setNumThreads(1)
    data=[]
    with (out.with_suffix('.encode.log')).open('wb') as log:
        proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=log)
        try:
            for tick in range(n+1):
                r=v.tl.rows.get(tick);rr=d.tl.rows.get(tick) if d else None
                t=(r or rr)['t'];im=np.full((HGT,W,3),color('background'),np.uint8)
                txt(im,'U3-H   /   '+scene.upper(),24,44,.94,'white',2)
                label='B   STATION REFERENCE   vs   D   JOINT PLANNING' if d else f'{run.name}   /   {r["mode"]}   /   REGION {r["station"]+1:02d}'
                txt(im,label,825,42,.65,'muted')
                txt(im,f'{t:05.2f} s',1744,44,.86,'cyan',2)
                if d:
                    view=panel(im,24,72,686,752,'01   SHARED WORLD / X-Z')
                    v.local(view,r,d,rr)
                    for x,a,row,title in [(724,v,r,'02   B  /  STATION RGB'),(1316,d,rr,'03   D  /  JOINT RGB')]:
                        a.image(panel(im,x,72,580,461,title),row)
                        a.metadata(panel(im,x,545,580,279,'') ,row)
                else:
                    v.global_map(panel(im,24,72,714,375,'01   WORLD / ACTUAL CAMERA VIEW'),r)
                    v.local(panel(im,24,459,714,365,'02   LOCAL X-Z / STATION-ANCHORED'),r,station_anchor=True)
                    v.image(panel(im,752,72,1144,602,'03   ACTUAL GAZEBO RGB'),r)
                    v.metadata(panel(im,752,686,1144,138,''),r,large=True)
                chart=panel(im,24,838,1872,218,'04   ROTOR SPEED  /  deg/s  /  independent telemetry, history only')
                omega_chart(chart,v,t,end,d)
                if d:txt(chart,'Cut ticks: B gray / D purple',420,201,.47,'muted')
                proc.stdin.write(im.tobytes())
                data.append(dict(tick=tick,display_sim_pts=t,B=proof(v,r) if d else None,D=proof(d,rr) if d else proof(v,r)))
        finally:
            proc.stdin.close()
            code=proc.wait()
            if code:raise RuntimeError(f'FFmpeg failed ({code}), see {out.with_suffix(".encode.log")}')
    save(out.with_suffix('.frames.json'),data)
    print(out.name,len(data),flush=True)
