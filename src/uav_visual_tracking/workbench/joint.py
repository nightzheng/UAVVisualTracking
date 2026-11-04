"""Causal two-layer joint reference and immutable two-horizon snapshots."""
import numpy as np
from .imaging import ImagingControl
from .control import angles,geom
from .u3_surface import ry


def limit(x,b):
    return x*min(1.,b/max(np.linalg.norm(x),1e-12))


def rollout(p,v,aim,angular,goal,pointing,control,dt):
    pos=np.array(p);v=np.array(v);yp=np.array(aim);rate=np.array(angular);out=[]
    for i in range(1,round(.5/dt)+1):
        e=goal-pos;cap=min(control['max_body_speed_m_s'],np.sqrt(2*control['max_body_accel_m_s2']*np.linalg.norm(e)))
        v+=limit(limit(2*e,cap)-v,control['max_body_accel_m_s2']*dt);pos+=v*dt
        e=pointing-yp;a=control['max_gimbal_accel_deg_s2']
        cap=np.minimum(control['max_gimbal_rate_deg_s'],np.sqrt(2*a*np.abs(e)))
        rate+=np.clip(np.clip(4*e,-cap,cap)-rate,-a*dt,a*dt);yp+=rate*dt
        if i%round(.1/dt)==0:
            out.append(dict(horizon_s=round(i*dt,3),position_m=pos.tolist(),gimbal_yaw_pitch_deg=yp.tolist()))
    return out


class Planner(ImagingControl):
    def __init__(self,spec):
        super().__init__(spec)
        self.cfg=spec['joint_planning'];self.plan=None
        self.motion=(np.zeros(3),np.zeros(2));self.offset=np.zeros(3);self.error=np.zeros(2)
        self.last_update=None;self.last_legal_pts=None

    def advance(self,sim):
        super().advance(sim)
        self.clear()
        self.plan=None

    def clear(self):
        self.offset=np.zeros(3);self.error=np.zeros(2);self.last_update=None;self.last_legal_pts=None

    def target(self,phase):
        return np.asarray(self.bank['center'])@(np.asarray(self.g['tilt'])@ry(np.deg2rad(phase))).T+self.g['hub']

    def step(self,row,sim,wall,position,T):
        self.plan=None
        r=super().step(row,sim,wall,position,T)
        if not r['legal']:
            self.clear();r['joint_plan']=None
            return r
        phase=row['phase_deg'];obs=row['capture_pts'];v,angular=self.motion
        valid_cv=self.spec['arm']=='C' and len(self.history)>=3
        speed=None
        if valid_cv:
            ts,values=np.array(self.history).T
            speed=float(np.clip(np.polyfit(ts-ts[-1],values,1)[0],-self.spec['control']['phase_speed_limit_deg_s'],self.spec['control']['phase_speed_limit_deg_s']))
        base,aim,_,expiry,expiry_wall=r['command'];goal=base.copy();reasons=[]
        nominal=self.target(phase)
        projected=r['composition']['center_px'];err=np.asarray(projected)-self.config['goal_px']
        if self.last_legal_pts is None or obs-self.last_legal_pts>.13:self.error=err
        else:self.error=.65*self.error+.35*err
        self.last_legal_pts=obs
        if self.cfg['enabled']:
            q=row['estimated_quality'];c=self.cfg;depth=q['distance_m']
            if self.last_update is None or sim-self.last_update>=c['update_period_s']-.001:
                dt=.1 if self.last_update is None else min(.25,sim-self.last_update)
                self.last_update=sim
                proposed=self.offset.copy()
                # Scale and centering give independent inputs to one bounded position reference.
                d=self.target(phase+(speed or 0.)*(sim-obs+.2))-nominal
                right=T[0,:3]
                if abs(self.error[0])>c['horizontal_error_px']:
                    proposed[0]=np.clip(.7*(nominal[0]-self.bank['target_ref_m'][0])+.25*right[0]*self.error[0]*depth/self.K[0,0]+d[0],-2.5,2.5)
                    reasons.append('persistent_horizontal')
                if abs(self.error[1])>c['vertical_error_px']:
                    proposed[2]=np.clip(-.85*self.error[1]*depth/self.K[1,1]+d[2],-1.8,1.8)
                    reasons.append('persistent_vertical')
                if q['min_chord_proxy_px']<c['scale_target_px']:
                    proposed[1]=.4;reasons.append('insufficient_scale')
                if speed is not None and (('persistent_horizontal' in reasons and abs(d[0])>1e-4)
                    or ('persistent_vertical' in reasons and abs(d[2])>1e-4)):
                    reasons.append('cv_lead')
                self.offset+=limit(proposed-self.offset,c['offset_rate_m_s']*dt)
            goal=base+np.clip(self.offset,-np.array(c['offset_limit_m']),c['offset_limit_m'])
            axis=np.asarray(self.g['tilt'])[:,1];margin=(np.asarray(self.g['hub'])-goal)@axis
            goal-=max(0.,c['stand_off_floor_m']-margin)*axis
            # Feed forward the pointing change caused by the coordinated body movement.
            trial=rollout(position,v,self.actual,angular,goal,aim,self.spec['control'],c['trajectory_dt_s'])[1]
            prev=rollout(position,v,self.actual,angular,base,aim,self.spec['control'],c['trajectory_dt_s'])[1]
            pp=np.array(trial['position_m']);pb=np.array(prev['position_m'])
            aim=aim+np.clip(angles(pp,nominal)-angles(pb,nominal),-.8,.8)
        aim=np.clip(aim,[-75,-65],[75,65]);offset=goal-base
        reason='joint_visual_body' if np.linalg.norm(offset)>1e-6 else 'station_body_visual_gimbal'
        checks=r['checks'];checks['arrived']=bool(np.linalg.norm(position-goal)<.95)
        r['capture']=all(value for key,value in checks.items() if key!='novel')
        mode='capture' if r['capture'] else 'track' if checks['arrived'] else 'transit'
        r['mode']=mode.upper()
        r['command']=(goal,aim,reason,expiry,expiry_wall)
        self.plan=dict(source_capture_pts=obs,plan_created_pts=sim,plan_created_wall_s=wall,valid_until=expiry,
            valid_until_wall_s=expiry_wall,station=self.index,phase_at_capture_deg=phase,cv_rate_deg_s=speed,
            body_goal_m=goal.tolist(),station_goal_m=base.tolist(),gimbal_goal_deg=aim.tolist(),
            reason=reason,body_triggers=reasons or (['bounded_offset_retained'] if np.linalg.norm(offset)>1e-6 else []),
            control_mode=mode,state_source='C1_current_RGB_static_mesh_independent_camera_pose')
        r['joint_plan']=self.snapshot(sim,wall,position,T,v,angular)
        return r

    def snapshot(self,pts,wall,position,T,v,angular):
        p=self.plan
        if p is None or p['station']!=self.index or not(p['plan_created_pts']<=pts<=p['valid_until'] and p['plan_created_wall_s']<=wall<=p['valid_until_wall_s']):
            return None
        aim=angles(position,position+T[2,:3]);c=self.spec['control'];goal=np.array(p['body_goal_m'])
        path=rollout(position,v,aim,angular,goal,np.array(p['gimbal_goal_deg']),c,self.cfg['trajectory_dt_s'])
        result=dict(**p,display_pts=pts,state_age_s=pts-p['source_capture_pts'],current_uav_pose=dict(position_m=position.tolist(),velocity_m_s=v.tolist()),
            current_camera_pose=dict(world_to_camera=T.tolist(),yaw_pitch_deg=aim.tolist()),planned_trajectory=path)
        for k,item in [('0p2',path[1]),('0p5',path[4])]:
            horizon=item['horizon_s'];prediction=None;image=None
            if p['cv_rate_deg_s'] is not None:
                future=p['phase_at_capture_deg']+p['cv_rate_deg_s']*(pts-p['source_capture_pts']+horizon)
                prediction=self.target(future).tolist()
                uv,q=geom(self.bank,future,T,self.K,self.g,(640,480))
                if q['all_in_front']:
                    image=dict(center_px=q['center_px'],bbox_px=np.r_[uv.min(0),uv.max(0)].tolist(),coordinate_frame='current_camera')
            result[f'target_prediction_{k}']=prediction;result[f'image_prediction_{k}']=image
            result[f'uav_reference_{k}']=item['position_m'];result[f'gimbal_reference_{k}']=item['gimbal_yaw_pitch_deg']
        return result
