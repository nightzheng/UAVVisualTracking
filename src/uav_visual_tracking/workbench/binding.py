"""Causal image association and conditional mesh binding. No evaluator dependency."""
import cv2
import numpy as np


def observation_field(status, value=None, candidates=None):
    return dict(status=status,value=value,candidates=[] if candidates is None else candidates)


def project_bank(local, phases, K, T, hub, tilt):
    """Return pixels and camera z for each phase hypothesis and material point."""
    angles=np.deg2rad(phases)
    c,s=np.cos(angles)[:,None],np.sin(angles)[:,None]
    x,y,z=np.asarray(local).T
    rotated=np.stack([c*x+s*z,np.broadcast_to(y,(len(angles),len(y))),-s*x+c*z],axis=-1)
    camera=(rotated@np.asarray(tilt).T+hub)@T[:3,:3].T+T[:3,3]
    return (camera@K.T)[...,:2]/np.maximum(camera[...,2:],1e-8),camera[...,2]


class VisualBank:
    def __init__(self,cfg,anchors):
        self.cfg=cfg;self.anchors=anchors
        self.sift=cv2.SIFT_create(nfeatures=cfg['feature_count'],contrastThreshold=cfg['sift_contrast'])
        self.match=cv2.BFMatcher(cv2.NORM_L2)
        self.reference=np.asarray([a['descriptor'] for a in anchors],np.float32).reshape(-1,128)
        self.previous_gray=None;self.previous_pts=None;self.tracks={}

    def step(self,gray,pts,initial):
        c=self.cfg;matches={};ambiguous=0
        if initial:
            matches={i:(np.asarray(a['xy'],np.float32),'anchor') for i,a in enumerate(self.anchors)}
        elif len(self.reference)>=2:
            if self.tracks and pts-self.previous_pts<=c['max_continuous_gap_s']:
                ids=list(self.tracks);xy=np.asarray([self.tracks[i][0] for i in ids],np.float32).reshape(-1,1,2)
                curr,ok,_=cv2.calcOpticalFlowPyrLK(self.previous_gray,gray,xy,None,winSize=(21,21),maxLevel=3)
                if curr is not None:
                    back,back_ok,_=cv2.calcOpticalFlowPyrLK(gray,self.previous_gray,curr,None,winSize=(21,21),maxLevel=3)
                    if back is not None:
                        for i,p,d,a,b in zip(ids,curr[:,0],np.linalg.norm(xy[:,0]-back[:,0],axis=1),ok.ravel(),back_ok.ravel()):
                            if a and b and d<=c['lk_fb_px'] and np.all(p>=4) and np.all(p<np.array(gray.shape[::-1])-4):
                                matches[i]=(p,'lk')
            keys,desc=self.sift.detectAndCompute(gray,None)
            if desc is not None and len(desc)>=2:
                reverse={m.queryIdx:m.trainIdx for m in self.match.match(desc,self.reference)}
                for pair in self.match.knnMatch(self.reference,desc,k=2):
                    if len(pair)<2:continue
                    m,n=pair
                    if m.distance<=c['max_descriptor_distance'] and m.distance<c['descriptor_ratio']*n.distance and reverse[m.trainIdx]==m.queryIdx:
                        matches[m.queryIdx]=(np.asarray(keys[m.trainIdx].pt,np.float32),'descriptor')
                    elif m.distance<=c['max_descriptor_distance']:
                        ambiguous+=1
        self.previous_gray=gray;self.previous_pts=pts;self.tracks=matches
        return matches,ambiguous


class Binder:
    """Immutable initialization, explicit phase hypothesis rejection and re-entry gate."""
    def __init__(self,mode,cfg,init,K,geometry=None):
        self.mode,self.cfg,self.init,self.K,self.geometry=mode,cfg,init,np.asarray(K),geometry
        self.last_pts=-np.inf;self.last_legal=None;self.last_phase=None;self.lost=False

    def step(self,pts,T,roi,matches,ambiguous,history_fault=False):
        if pts<=self.last_pts:raise ValueError('Nonmonotonic capture PTS')
        self.last_pts=pts;c=self.cfg
        row=dict(capture_pts=pts,available_at=None,last_legal_pts=self.last_legal,tracklet_id=roi.get('tracklet_id'),
            roi_bbox=roi.get('roi_bbox'),observation_status=roi['observation_status'],
            information_source={'A':'RGB only','B':'RGB + once-only Oracle material anchors','C':'RGB + once-only Oracle anchors and independent camera/static mesh'}[self.mode],
            coverage_update_eligible=False,blade_identity=observation_field('unknown'),surface_side=observation_field('unknown'),
            radial_region=observation_field('unknown'),eta=observation_field('unknown'),xi=observation_field('unknown'),material_patch=observation_field('unknown'),
            phase_deg=None,world_omega_deg_s=None,reassociation='not_attempted',previous_loss=self.lost,
            evidence=dict(lk=sum(v[1]=='lk' for v in matches.values()),descriptor=sum(v[1]=='descriptor' for v in matches.values()),
                          competing_descriptor=ambiguous),reason='no_observation')
        for key in ('roi_scale_ratio','blade_pixel_ratio','image_center_error_px','local_direction'):
            row[key]=roi.get(key)
        if self.mode=='A':
            if roi['observation_status']=='relative_roi':
                row['blade_identity']=observation_field('ambiguous',candidates=list('ABC'))
                row['reason']='C3_label_symmetry_or_missing_name_mapping'
            return row
        if not self.init or pts<self.init['pts']-1e-9:
            row['reason']='no_legal_initial_anchor';return row
        last=self.init['pts'] if self.last_legal is None else self.last_legal
        if pts-last>c['max_continuous_gap_s']:self.lost=True
        initial=abs(pts-self.init['pts'])<1e-9
        row['reassociation']='attempt' if self.lost else ('initialization' if initial else 'continuous')
        def reject(reason,ambiguity=False):
            row['reason']=reason
            row['blade_identity']=observation_field('ambiguous' if ambiguity else 'rejected',candidates=list('ABC') if ambiguity else [])
            if self.lost:row['reassociation']='rejected'
            return row
        if self.mode=='B' and self.lost:
            return reject('B_no_independent_constraint_after_loss',True)
        selected=[i for i,v in matches.items() if not self.lost or v[1]=='descriptor']
        minimum=c['min_reentry_descriptor_matches'] if self.lost else c['min_geometry_matches']
        if len(selected)<minimum:return reject('insufficient_current_material_support',ambiguous>0)
        anchors=self.init['anchors'];chosen=[];theta=None
        if self.mode=='C':
            if T is None:return reject('independent_camera_pose_missing')
            g=self.geometry;theta0=self.init['phase_deg'];bound=c['max_speed_deg_s']*(pts-self.init['pts'])+.35
            prior=self.last_phase+120 if history_fault and self.last_phase is not None else self.last_phase
            if prior is not None and abs(prior-theta0)>bound+.35:
                self.last_phase=None;self.lost=True
                return reject('history_inconsistent_with_immutable_anchor',True)
            if bound>=60:return reject('cyclic_reachability_overlap',True)
            step=c['phase_grid_deg'];d=np.arange(-int(bound/step),int(bound/step)+1)*step
            phases=theta0+d
            points=np.array([anchors[i]['point'] for i in selected])
            xy=np.array([matches[i][0] for i in selected]);T=np.asarray(T)
            projected,z=project_bank(points,phases,self.K,T,g['hub'],g['tilt'])
            error=np.linalg.norm(projected-xy[None],axis=2)
            support=(error<=c['pixel_tolerance'])&(z>.1)
            scores=support.sum(axis=1)
            cost=np.minimum(error,c['pixel_tolerance']*2).sum(axis=1)
            best=int(np.lexsort((cost,-scores))[0]);theta=float(phases[best]);n=int(scores[best])
            contender=(scores>=n)&(cost<=cost[best]+1)&(np.abs(phases-theta)>c['phase_ambiguity_deg'])
            row['evidence'].update(tested_phases=len(phases),inliers=n,viable_alternative_phases=phases[contender].tolist(),
                phase_bounds_deg=[float(phases[0]),float(phases[-1])],cyclic_offsets_excluded_by_anchor=[-120,120])
            if n<minimum or n/len(selected)<.6:return reject('geometry_support_or_consensus_failed')
            if contender.any():return reject('multiple_phase_basins',True)
            if prior is not None and abs(theta-prior)>c['max_speed_deg_s']*(pts-self.last_legal)+.75:
                return reject('motion_inconsistent_with_last_legal_phase')
            chosen=[(i,float(error[best,j])) for j,i in enumerate(selected) if support[best,j]]
        else:
            chosen=[(i,None) for i in selected]
        # Representative anchor chosen deterministically from supported matches.
        options=[]
        for i,residual in chosen:
            a=anchors[i];xy=matches[i][0]
            if self.mode=='C':
                tri,z=project_bank(np.asarray(a['triangle']),[theta],self.K,T,g['hub'],g['tilt'])
                uv=tri[0];edge=np.stack([uv[1]-uv[0],uv[2]-uv[0]],axis=1)
                if np.min(z)<=.1 or abs(np.linalg.det(edge))<1e-4:continue
                params=np.asarray(a['triangle_uv']);jac=(params[1:]-params[0]).T@np.linalg.inv(edge)
                half=np.linalg.norm(jac,axis=1)*(c['pixel_tolerance']+(residual or 0))
            else:
                half=np.array([c['max_eta_width'],c['max_xi_width']])*.5
            point=np.array([a['eta'],a['xi']]);lower=np.maximum(point-half,0);upper=np.minimum(point+half,1)
            if self.mode=='C' and (upper-lower>np.array([c['max_eta_width'],c['max_xi_width']])+1e-10).any():continue
            if lower[0]<self.init['region_eta'][0] or upper[0]>self.init['region_eta'][1]:continue
            options.append((float(np.sum(upper-lower)),i,residual,lower,upper))
        if not options:return reject('surface_interval_too_wide_or_crosses_target')
        _,i,residual,lower,upper=min(options,key=lambda o:(o[0],o[1]));a=anchors[i]
        status='anchored' if initial or self.mode=='B' else 'supported'
        row.update(blade_identity=observation_field(status,self.init['blade']),surface_side=observation_field(status,a['side']),
            eta=observation_field(status,[float(lower[0]),float(upper[0])]),xi=observation_field(status,[float(lower[1]),float(upper[1])]),
            radial_region=observation_field(status,self.init['region_eta']),material_patch=observation_field(status,a['face_id']),phase_deg=theta,
            coverage_update_eligible=True,reason='conditional_material_binding',matched_xy=matches[i][0].tolist(),
            reassociation='accepted' if self.lost else row['reassociation'])
        row['evidence'].update(anchor_index=i,representative_match=matches[i][1],reprojection_residual_px=residual)
        self.last_legal=pts;self.last_phase=theta;self.lost=False
        return row
