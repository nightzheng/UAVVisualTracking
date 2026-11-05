"""Conditional local binding consumer and causal photo/actuation policy."""
from collections import deque
from pathlib import Path
import os
import sys
import time
import cv2
import numpy as np
from .common import STAGE
from .binding import Binder,VisualBank,project_bank
from .u3_local_rgb import LocalContour
from .u3_surface import ry


def angles(position,target):
    x,y,z=np.asarray(target)-position
    return np.array([(np.rad2deg(np.arctan2(y,x))+90)%360-180,np.rad2deg(np.arctan2(-z,np.hypot(x,y)))])


def geom(bank,phase,T,K,g,wh):
    uv,z=project_bank(bank['points'],[phase],K,T,g['hub'],g['tilt']);uv=uv[0];z=z[0]
    eta=np.asarray(bank['framing_eta']);limits=np.linspace(*bank['region_eta'],4)
    slices=[uv[(eta>=a)&(eta<=b)] for a,b in zip(limits[:-1],limits[1:])]
    if all(len(p) for p in slices):
        delta=slices[-1].mean(0)-slices[0].mean(0);normal=np.array([-delta[1],delta[0]])/max(np.linalg.norm(delta),1e-9)
        chord=min(float(np.ptp(p@normal)) for p in slices)
    else:chord=0.
    center=project_bank([bank['center']],[phase],K,T,g['hub'],g['tilt'])[0][0,0]
    world=np.asarray(bank['center'])@(np.asarray(g['tilt'])@ry(np.deg2rad(phase))).T+g['hub']
    pos=-T[:3,:3].T@T[:3,3];d=pos-world;distance=float(np.linalg.norm(d))
    cosine=float((-np.asarray(g['tilt'])[:,1])@d/max(distance,1e-9))
    return uv,dict(min_margin_px=float(np.minimum(uv,np.asarray(wh)-1-uv).min()),center_error_px=float(np.linalg.norm(center-K[:2,2])),
        min_chord_proxy_px=chord,distance_m=distance,axis_incidence_proxy_deg=float(np.rad2deg(np.arccos(np.clip(cosine,-1,1)))),
        all_in_front=bool((z>.1).all()),center_px=center.tolist())


def worker(conn,spec,run_directory):
    cv2.setNumThreads(spec['source']['opencv_threads']);repo=STAGE.parents[2]
    def guard(event,args):
        if event=='open' and isinstance(args[0],(str,bytes)) and any(Path(os.fsdecode(args[0])).resolve().is_relative_to(root) for root in (repo, run_directory)):
            raise PermissionError('Visual worker denied workspace IO')
    sys.addaudithook(guard)
    assert spec['G6_C1_admitted']
    K=np.array(spec['K']);g=spec['geometry'];g['tilt']=np.array(g['tilt']);cfg=spec['binding_config'];q=spec['quality']
    contour=LocalContour(spec['contour']);index=-1;last_pts=-np.inf;prior_geom=None;last_error=False
    conn.send('ready')
    try:
        while True:
            p=conn.recv()
            if p is None:break
            if set(p)!={'index','pts','arrived_wall','camera_T','raw','wh','step','station','phase_epoch'} or p['pts']<=last_pts:
                raise ValueError('Current visual packet violation')
            last_pts=p['pts'];started=time.monotonic();T=np.asarray(p['camera_T']);w,h=p['wh']
            raw=np.frombuffer(p['raw'],np.uint8).reshape(h,p['step'])[:,:3*w].reshape(h,w,3)
            bgr=cv2.cvtColor(raw,cv2.COLOR_RGB2BGR);gray=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)
            roi,mask=contour.step(bgr,p['pts'])
            if p['station']!=index:
                index=p['station'];bank=spec['banks'][index];epoch=p['phase_epoch'];prior_geom=None
                init=dict(pts=epoch[0],phase_deg=epoch[1],blade='A',region_eta=bank['region_eta'],anchors=bank['anchors']) if epoch else None
                binder=Binder('C',cfg,init,K,g);binder.lost=True
                vision=VisualBank(cfg,bank['anchors'] if init else [])
            matches,ambiguous=vision.step(gray,p['pts'],False);row=binder.step(p['pts'],T,roi,matches,ambiguous)
            qual=None
            if row['coverage_update_eligible']:
                uv,qual=geom(bank,row['phase_deg'],T,K,g,p['wh']);inside=(uv>=0).all(1)&(uv<=np.array(p['wh'])-1).all(1)
                pix=np.rint(uv[inside]).astype(int);support=float(mask[pix[:,1],pix[:,0]].sum()/len(uv)) if len(pix) else 0.
                qual['observed_contour_support_fraction']=support
                qual['exposure_motion_px']=None
                if prior_geom is not None:
                    delta=p['pts']-prior_geom[0]
                    if 0<delta<=.12:qual['exposure_motion_px']=float(np.max(np.linalg.norm(uv-prior_geom[1],axis=1))*q['exposure_s']/delta)
                prior_geom=(p['pts'],uv)
            else:prior_geom=None
            done=time.monotonic()
            row.update(index=p['index'],station=index,arrived_wall=p['arrived_wall'],visual_started_wall=started,available_at=done,
                fresh_until_sim_pts=p['pts']+spec['control']['max_state_age_s'],fresh_until_arrival_wall_s=p['arrived_wall']+spec['control']['max_state_age_s'],
                estimated_quality=qual,identity_candidates_on_loss=list('ABC') if not row['coverage_update_eligible'] else [],
                registration_kind='independent per-station material bank, descriptor and geometric reacquisition')
            conn.send(row)
    except Exception:
        import traceback
        conn.send(dict(error=traceback.format_exc()))
    finally:conn.close()


class Controller:
    def __init__(self,spec):
        self.spec=spec;self.g=spec['geometry'];self.K=np.asarray(spec['K']);self.history=deque();self.index=0
        self.epoch=(0.,spec['initial_phase_deg']);self.last_fresh=None;self.visited=set();self.started_sim=0.;self.navigation_pending=True

    @property
    def bank(self):return self.spec['banks'][self.index]

    def advance(self,sim):
        self.index+=1;self.started_sim=sim;self.history.clear();self.navigation_pending=True
        fresh=self.last_fresh
        self.epoch=fresh if fresh is not None and sim-fresh[0]<=self.spec['control']['max_transition_phase_age_s'] else None

    def navigation(self):
        self.navigation_pending=False;p=np.asarray(self.bank['position_ref_m']);target=self.bank['target_ref_m']
        return p,angles(p,target),'predeclared_safe_station_transition',1e9,1e12

    def step(self,row,sim,wall,position,T):
        q=self.spec['quality'];c=self.spec['control'];bank=self.bank
        current=bool(self.spec['G6_C1_admitted'] and row['station']==self.index and row['coverage_update_eligible'] and
            row['blade_identity']['value']=='A' and row['surface_side']['value']=='suction_geometric' and row['radial_region']['value']==bank['region_eta'] and
            row['eta']['value'] is not None and row['xi']['value'] is not None and
            np.ptp(row['eta']['value'])<=.0200000001 and np.ptp(row['xi']['value'])<=.2200000001 and
            row['capture_pts']<=sim<=row['fresh_until_sim_pts'] and row['available_at']<=wall<=row['fresh_until_arrival_wall_s'])
        if not current:
            self.history.clear();return dict(legal=False,reason='unavailable_ambiguous_expired_or_wrong_station',command=None,capture=False)
        t=row['capture_pts'];phase=row['phase_deg'];self.last_fresh=(t,phase)
        if self.history and t-self.history[-1][0]>.12:self.history.clear()
        self.history.append((t,phase))
        while self.history and t-self.history[0][0]>.35:self.history.popleft()
        forecast=None;future=phase
        if self.spec['arm']=='C' and len(self.history)>=3:
            times,values=np.asarray(self.history).T
            omega=float(np.clip(np.polyfit(times-times[-1],values,1)[0],-c['phase_speed_limit_deg_s'],c['phase_speed_limit_deg_s']))
            future=phase+omega*(sim-t+c['cv_horizon_s'])
            _,forecast=geom(bank,future,T,self.K,self.g,(640,480))
            forecast.update(phase_deg=future,horizon_from_decision_s=c['cv_horizon_s'],source_last_pts=t)
        arrived=bool(np.linalg.norm(position-np.asarray(bank['position_ref_m']))<.95)
        command=None
        if arrived and self.spec['arm'] in ('B','C'):
            center=np.asarray(bank['center'])@(np.asarray(self.g['tilt'])@ry(np.deg2rad(future))).T+self.g['hub']
            ref=np.asarray(bank['position_ref_m']).copy();yp=angles(position,center)
            if abs(yp[0])>8:ref[0]+=np.clip(center[0]-position[0],-.4,.4)
            command=(ref,yp,'visual_cv' if forecast else 'visual_current',row['fresh_until_sim_pts'],row['fresh_until_arrival_wall_s'])
        g=row['estimated_quality'];novel=sum(w for i,w in zip(bank['sample_ids'],bank['sample_area_m2']) if i not in self.visited)
        checks=dict(arrived=arrived,dwell=sim-self.started_sim>=c['minimum_dwell_s'],front=g['all_in_front'],edge=g['min_margin_px']>=q['min_margin_px'],
            center=g['center_error_px']<=q['max_center_error_px'],chord=g['min_chord_proxy_px']>=q['min_chord_px'],
            support=g['observed_contour_support_fraction']>=.70,incidence=g['axis_incidence_proxy_deg']<=q['max_incidence_deg'],
            distance=q['min_distance_m']<=g['distance_m']<=q['max_distance_m'],exposure=g['exposure_motion_px'] is not None and g['exposure_motion_px']<=q['max_exposure_motion_px'],
            novel=novel>=q['min_novel_area_m2'],forecast_window=forecast is None or forecast['min_margin_px']>=q['min_margin_px'])
        return dict(legal=True,capture=all(checks.values()),checks=checks,forecast=forecast,command=command,provisional_novel_m2=novel,
                    reason='candidate' if all(checks.values()) else 'quality_or_transition_gate')
