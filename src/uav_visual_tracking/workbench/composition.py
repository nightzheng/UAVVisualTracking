"""One gimbal reference for station transition, visual feedback and CV feedforward."""
import numpy as np
from .control import Controller, angles, geom


class Composition(Controller):
    def __init__(self,spec):
        super().__init__(spec)
        self.config=spec['composition']
        self.actual=None
        self.reference=None
        self.reference_pts=None
        self.inv_K=np.linalg.inv(self.K)

    def advance(self,sim):
        super().advance(sim)
        self.reference=None
        self.reference_pts=None

    def navigation(self):
        ref,initial,_,sim,wall=super().navigation()
        return ref,initial if self.actual is None else self.actual.copy(),'transit_preserve_aim',sim,wall

    def step(self,row,sim,wall,position,T):
        self.actual=angles(position,position+T[2,:3])
        decision=super().step(row,sim,wall,position,T)
        decision['command']=None
        if not decision['legal']:
            self.reference=None
            self.reference_pts=None
            decision.update(mode='RECOVER',composition=None)
            return decision
        cfg=self.config
        _,quality=geom(self.bank,row['phase_deg'],T,self.K,self.g,(640,480))
        uv=np.asarray(quality['center_px']);goal=np.asarray(cfg['goal_px'])
        forecast=decision['forecast'];ff=np.zeros(2)
        if forecast is not None:
            ff=np.clip(np.asarray(forecast['center_px'])-uv,-cfg['max_feedforward_px'],cfg['max_feedforward_px'])
        proposed=goal+cfg['feedback_gain']*(uv-goal)+cfg['feedforward_gain']*ff
        # Camera axes from the independent actual pose. Correct the ray offset from the desired pixel.
        desired_ray=self.inv_K@np.r_[goal,1.]
        target_ray=self.inv_K@np.r_[proposed,1.]
        correction=angles(position,position+T[:3,:3].T@target_ray)-angles(position,position+T[:3,:3].T@desired_ray)
        absolute=np.clip(self.actual+correction,[-75,-65],[75,65])
        if self.reference_pts is None:
            previous=self.actual
            dt=.1
        else:
            previous=self.reference
            dt=min(.15,sim-self.reference_pts)
        step=cfg['reference_rate_deg_s']*dt
        self.reference=previous+np.clip(absolute-previous,-step,step)
        self.reference_pts=sim
        arrived=decision['checks']['arrived']
        mode='CAPTURE' if decision['capture'] else 'TRACK' if arrived else 'TRANSIT'
        # Hold original safe station reference for both arms. The common actuator limits yaw/pitch acceleration.
        body=np.asarray(self.bank['position_ref_m'])
        decision['command']=(body,self.reference,'composition_cv' if forecast is not None else 'composition_current',
                             row['fresh_until_sim_pts'],row['fresh_until_arrival_wall_s'])
        decision.update(mode=mode,composition=dict(center_px=uv.tolist(),goal_px=goal.tolist(),feedback_px=(uv-goal).tolist(),
            feedforward_px=ff.tolist(),actual_aim=self.actual.tolist(),reference=self.reference.tolist(),
            station_reference=angles(body,self.bank['target_ref_m']).tolist()))
        return decision
