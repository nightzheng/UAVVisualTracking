"""Join plans, real actuator acknowledgements and independent imaging evaluation."""
from collections import Counter
import numpy as np
from .common import read,save
from .score_f import split_evaluate
from .control import angles
from .u3_gazebo import camera_transform


def audit(run, derived=None):
    derived = derived or run
    spec=read(run/'consumer_spec.json');states=read(run/'states.json')
    commands={x['sequence']:x for x in read(run/'commands.json')}
    obs={x['index']:x for x in read(run/'observations.json')}
    frames={x['index']:x for x in read(run/'frames.json')}
    first={};triggers=Counter();departures={};paths=Counter();motion=Counter()
    max_deviation=0.;stale=[]
    for x in commands.values():
        p=np.array(x['ref_m']);base=np.array(spec['banks'][x['station']]['position_ref_m']);delta=p-base
        if np.linalg.norm(delta)>1e-6:
            departures[x['sequence']]=delta
            plan=obs[x['source_index']]['decision']['joint_plan']
            assert np.allclose(p,plan['body_goal_m'])
            assert np.allclose(x['yaw_pitch_deg'],plan['gimbal_goal_deg'])
            triggers.update(plan['body_triggers'])
    for i,s in enumerate(states):
        seq=s['executed_sequence'];c=commands.get(seq)
        if c and seq not in first:
            first[seq]=s
            if s['command_applied_sim_s']>c['valid_until_sim'] or s['command_applied_wall_s']>c['valid_until_wall']:stale.append(seq)
        if i==0:continue
        p=np.asarray(s['camera_pose'][:3]);q=np.asarray(states[i-1]['camera_pose'][:3]);dx=p-q
        source='joint' if seq in departures else 'station' if c else 'hold'
        paths[source]+=float(np.linalg.norm(dx))
        if seq in departures:
            d=departures[seq]
            if s['executed_sequence']==states[i-1]['executed_sequence']:
                for k,axis in [(0,'x'),(1,'y'),(2,'z')]:
                    if abs(d[k])>1e-6 and dx[k]*d[k]>1e-9:motion[axis]+=1
            max_deviation=max(max_deviation,float(np.linalg.norm(d)))
    preview=[x for x in read(run/'display_plans.json') if x['plan'] is not None]
    n_cv=0;min_sampled_standoff=float('inf');errors=[]
    tilt=np.asarray(spec['geometry']['tilt'])[:,1];hub=np.asarray(spec['geometry']['hub'])
    for x in preview:
        p=x['plan'];t=x['pts'];f=frames[x['index']]
        assert p['source_capture_pts']<=p['plan_created_pts']<=t<=p['valid_until']
        assert p['plan_created_wall_s']<=f['arrived_wall']<=p['valid_until_wall_s']
        assert abs(p['state_age_s']-(t-p['source_capture_pts']))<1e-8
        path=p['planned_trajectory']
        assert [r['horizon_s'] for r in path]==[.1,.2,.3,.4,.5]
        assert p['uav_reference_0p2']==path[1]['position_m'] and p['uav_reference_0p5']==path[4]['position_m']
        min_sampled_standoff=min(min_sampled_standoff,*[float((hub-r['position_m'])@tilt) for r in path])
        if p['cv_rate_deg_s'] is not None:
            n_cv+=1
            assert p['target_prediction_0p2'] is not None and p['target_prediction_0p5'] is not None
    pose=np.array([s['camera_pose'][:3] for s in states]);t=np.array([s['sim_s'] for s in states])
    aim=np.array([angles(p,p+T[2,:3]) for p,T in (camera_transform(s['camera_pose']) for s in states)])
    dt=np.diff(t);velocity=np.diff(aim,axis=0)/dt[:,None]
    accel=np.diff(velocity,axis=0)/((dt[1:]+dt[:-1])/2)[:,None]
    visual=[x for x in commands.values() if x['source_index'] is not None]
    result=dict(body_offset_commands=len(departures),body_offset_applied=sum(k in first for k in departures),
        max_body_offset_m=max_deviation,body_triggers=dict(triggers),axis_feedback_steps=dict(motion),path_by_active_command_m=dict(paths),
        visual_application_fraction=sum(x['sequence'] in first for x in visual)/max(1,len(visual)),
        visuals_published=len(visual),causal_display_frames=len(preview),cv_display_frames=n_cv,
        min_rollout_standoff_m=None if not preview else min_sampled_standoff,
        combined_gimbal_rate_p95=float(np.percentile(np.linalg.norm(velocity,axis=1),95)),
        combined_gimbal_accel_p95=float(np.percentile(np.linalg.norm(accel,axis=1),95)),
        first_applied_stale=stale,frame_count=len(frames),
        applied_state_age_sim_p95_ms=None if not visual else float(np.percentile([
            (first[x['sequence']]['command_applied_sim_s']-x['source_pts'])*1000 for x in visual if x['sequence'] in first],95)))
    save(derived/'JOINT_AUDIT.json',result)
    return result


def compare(run,baseline):
    a=read(run/'states.json');b=read(baseline/'states.json')
    at=np.array([r['sim_s'] for r in a]);bt=np.array([r['sim_s'] for r in b])
    ap=np.array([r['camera_pose'][:3] for r in a]);bp=np.array([r['camera_pose'][:3] for r in b])
    common=(at>=8)&(at<=min(16,bt[-1]));at=at[common];ap=ap[common]
    diff=ap-np.stack([np.interp(at,bt,bp[:,k]) for k in range(3)],axis=1)
    return dict(pair=f'{run.name}-{baseline.name}',sample_count=len(diff),
        rms_xyz_m=np.sqrt(np.mean(diff**2,axis=0)).tolist(),max_distance_m=float(np.linalg.norm(diff,axis=1).max()))


def evaluate(root):
    results=[]
    for p in sorted(root.glob('*/*/runtime.json')):
        run=p.parent
        joint=audit(run);result=split_evaluate(run);result['joint']=joint
        paired={'C':'A','D':'B'}.get(run.name)
        if paired and (run.parent/paired/'runtime.json').exists():result['paired_actual_path']=compare(run,run.parent/paired)
        results.append(result)
    if not results:raise RuntimeError('No preserved U3-G recordings')
    save(root/'COMPARISON.json',results)
