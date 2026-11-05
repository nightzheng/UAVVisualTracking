"""Bounded reference management with static geometric lead at station transition."""
import numpy as np
from .composition import Composition
from .control import angles


class ImagingControl(Composition):
    def __init__(self,spec):
        super().__init__(spec)
        self.imaging=spec['imaging']
        self.filtered=None
        self.filter_pts=None
        self.position=None
        self.transition_delta=np.zeros(2)

    def advance(self,sim):
        prior=self.bank
        super().advance(sim)
        self.transition_delta=np.zeros(2)
        if self.index<len(self.spec['banks']) and self.position is not None:
            # Relative elevation comes solely from the two registered static station centers.
            new=angles(self.position,self.bank['target_ref_m'])
            old=angles(self.position,prior['target_ref_m'])
            self.transition_delta[1]=np.clip((new-old)[1]*self.imaging['transition_radial_lead'],-10,10)
        self.filtered=None
        self.filter_pts=None

    def navigation(self):
        ref,aim,_,sim,wall=super().navigation()
        return ref,aim+self.transition_delta,'static_radial_transit',sim,wall

    def step(self,row,sim,wall,position,T):
        self.position=position.copy()
        result=super().step(row,sim,wall,position,T)
        if not result['legal']:
            self.filtered=None
            self.filter_pts=None
            return result
        ref,requested,reason,expiry,wall_expiry=result['command']
        cfg=self.imaging;detail=result['composition']
        err=np.asarray(detail['feedback_px']);ff=np.asarray(detail['feedforward_px'])
        # In a small deadband use actual pointing as the reference instead of chasing pixel noise.
        if np.linalg.norm(err)+np.linalg.norm(ff)<cfg['deadband_px']:
            requested=self.actual
        prior=self.actual if self.filtered is None else self.filtered
        dt=.1 if self.filter_pts is None else min(.15,sim-self.filter_pts)
        alpha=-np.expm1(-dt/cfg['filter_tau_s'])
        bound=cfg['ref_rate_deg_s']*dt
        self.filtered=prior+np.clip(alpha*(requested-prior),-bound,bound)
        self.filter_pts=sim
        result['command']=(ref,self.filtered,reason.replace('composition','imaging'),expiry,wall_expiry)
        # Preserve every photo gate except the original fixed novelty requirement.
        checks=result['checks'];result['capture']=all(v for k,v in checks.items() if k!='novel')
        detail.update(raw_reference=requested.tolist(),reference=self.filtered.tolist(),
                      planned_standoff_m=self.bank['standoff_m'],measured_distance_m=row['estimated_quality']['distance_m'])
        result.update(mode='CAPTURE' if result['capture'] else 'TRACK' if checks['arrived'] else 'TRANSIT',
                      reason='candidate' if result['capture'] else 'quality_or_transition_gate')
        return result
