from collections import Counter


from pathlib import Path


import cv2


import numpy as np


import yaml


from .common import STAGE,read,save,sha


from .evaluate import IndependentScene,align


from .prepare import mission_for


from .u3_gazebo import camera_transform,quat


from .u3_g3 import theta


from .u3_surface import Surface,fraction,windows


from .control import angles


class DistanceMesh:
    """Exact point-to-triangle distance with an AABB lower-bound filter."""
    def __init__(self,triangles):
        self.tri=triangles;self.lo=triangles.min(axis=1);self.hi=triangles.max(axis=1)

    def distance(self,p):
        p=np.asarray(p);delta=p-self.tri[:,0];upper=float(np.min(np.sum(delta*delta,axis=1)))
        box=np.maximum(np.maximum(self.lo-p,p-self.hi),0);tri=self.tri[np.sum(box*box,axis=1)<=upper+1e-10]
        a,b,c=tri.transpose(1,0,2);ab=b-a;ac=c-a;d=p-a
        dot=lambda x,y:np.sum(x*y,axis=1)
        aa,cc,abac=dot(ab,ab),dot(ac,ac),dot(ab,ac);den=aa*cc-abac**2
        good=den>1e-20;u=np.zeros(len(tri));v=u.copy()
        u[good]=(cc[good]*dot(ab,d)[good]-abac[good]*dot(ac,d)[good])/den[good]
        v[good]=(aa[good]*dot(ac,d)[good]-abac[good]*dot(ab,d)[good])/den[good]
        inside=good&(u>=0)&(v>=0)&(u+v<=1)
        best=upper
        if inside.any():
            q=d-u[:,None]*ab-v[:,None]*ac;best=min(best,float(dot(q,q)[inside].min()))
        for start,end in ((a,b),(a,c),(b,c)):
            edge=end-start;x=p-start
            t=np.clip(dot(x,edge)/np.maximum(dot(edge,edge),1e-20),0,1);q=x-edge*t[:,None]
            best=min(best,float(dot(q,q).min()))
        return float(np.sqrt(best))


def pct(x):return dict(p50=float(np.percentile(x,50)),p95=float(np.percentile(x,95))) if len(x) else dict(p50=None,p95=None)


def quality_config(c):
    q=c['quality']
    return dict(mission=mission_for(c),evaluation=dict(depth_tolerance_m=.04),camera=dict(width=640,height=480,
        min_edge_margin_px=q['min_margin_px'],min_projected_chord_px=q['min_chord_px'],max_center_error_px=q['max_center_error_px'],
        min_in_frame_ratio=.98,max_incidence_deg=q['max_incidence_deg'],min_patch_area_fraction=q['min_patch_coverage'],
        max_exposure_motion_px=q['max_exposure_motion_px']))


def motion(s,states,aligned,cmds,obs,c):
    lookup={r['sequence']:r for r in cmds};obs={r['index']:r for r in obs};acks={};movement={};violations=[]
    poses=np.array([r['camera_pose'][:3] for r in states]);pts=np.array([r['sim_s'] for r in states]);dt=np.diff(pts)
    axis=np.array([angles(p,p+quat(r['camera_pose'][3:])[:,0]) for r,p in zip(states,poses)])
    speed=np.diff(poses,axis=0)/dt[:,None];rate=np.diff(axis,axis=0)/dt[:,None];span=(dt[:-1]+dt[1:])/2
    for i,r in enumerate(states):
        seq=r.get('executed_sequence',-1)
        if seq not in lookup:continue
        cmd=lookup[seq]
        if r['sim_s']>cmd['valid_until_sim'] and seq not in violations:violations.append(seq)
        if seq not in acks:
            v=obs.get(cmd['source_index']);a=r['command_applied_sim_s'];w=r['command_applied_wall_s']
            if (a>cmd['valid_until_sim'] or w>cmd['valid_until_wall'] or (v and a<v['capture_pts'])) and seq not in violations:violations.append(seq)
            acks[seq]=dict(**cmd,applied_sim_s=a,applied_wall_s=w,ack_wall_s=r['telemetry_wall'],
                state_age_sim_s=None if v is None else a-v['capture_pts'],arrival_to_action_wall_s=None if v is None else w-v['arrived_wall'])
        if i and states[i-1].get('executed_sequence')==seq:
            ref=np.array(cmd['ref_m']);yp=np.array(cmd['yaw_pitch_deg']);prior=movement.setdefault(seq,[False,False])
            prior[0]|=bool(np.linalg.norm(ref-poses[i])-np.linalg.norm(ref-poses[i-1])<-1e-7)
            prior[1]|=bool(np.linalg.norm(yp-axis[i])-np.linalg.norm(yp-axis[i-1])<-1e-7)
    mesh=DistanceMesh(s.triangles);fixed=DistanceMesh(s.fixed);dist=[]
    for r in aligned:
        if not r:continue
        p=np.array(r['camera_pose'][:3]);hub=np.array(r['rotor_pose'][:3]);R=quat(r['rotor_pose'][3:]);d=fixed.distance(p)
        from .u3_surface import ry
        for j in range(3):d=min(d,mesh.distance((p-hub)@R@ry(j*2*np.pi/3)))
        dist.append(d)
    ctrl=c['control'];sweep=(s.hub-poses)@s.tilt[:,1]-ctrl['sweep_half_thickness_m'];used=[r for r in acks.values() if r['source_index'] is not None]
    m=dict(path_m=float(np.linalg.norm(np.diff(poses,axis=0),axis=1).sum()),gimbal_travel_deg=float(np.abs(np.diff(axis,axis=0)).sum()),
        min_swept_envelope_clearance_m=float(sweep.min()),min_mesh_point_distance_m=min(dist) if dist else None,
        mesh_distance_samples=len(dist),max_body_speed_m_s=float(np.linalg.norm(speed,axis=1).max()),
        max_body_accel_m_s2=float(np.linalg.norm(np.diff(speed,axis=0)/span[:,None],axis=1).max()),
        max_gimbal_rate_deg_s=float(np.abs(rate).max()),max_gimbal_accel_deg_s2=float(np.abs(np.diff(rate,axis=0)/span[:,None]).max()),
        stale_applied=violations,visual_commands_applied=len(used),commands_applied=len(acks),
        visual_translation_response=sum(v[0] for seq,v in movement.items() if lookup[seq]['source_index'] is not None),
        visual_rotation_response=sum(v[1] for seq,v in movement.items() if lookup[seq]['source_index'] is not None),
        applied_state_age_sim_s=pct([r['state_age_sim_s'] for r in used]),arrival_to_action_wall_s=pct([r['arrival_to_action_wall_s'] for r in used]),
        expired_sequences=len({r.get('expired_sequence') for r in states if r.get('expired_sequence',-1)>=0}))
    m['safety_pass']=bool(not violations and not any(r.get('safety_fault') for r in states) and m['min_swept_envelope_clearance_m']>=ctrl['min_clearance_m']
        and m['max_body_speed_m_s']<=ctrl['max_body_speed_m_s']+1e-5 and m['max_body_accel_m_s2']<=ctrl['max_body_accel_m_s2']+1e-5
        and m['max_gimbal_rate_deg_s']<=ctrl['max_gimbal_rate_deg_s']+1e-5 and m['max_gimbal_accel_deg_s2']<=ctrl['max_gimbal_accel_deg_s2']+1e-5)
    return m,list(acks.values())


def one(out, derived=None):
    derived = derived or out
    r=lambda name:read(out/f'{name}.json')
    c=yaml.safe_load((out/'config_snapshot.yaml').read_text(encoding='utf8'));q=c['quality'];mission=mission_for(c);patches=windows(mission)
    states,frames,obs,receipts=(r(n) for n in ('states','frames','observations','capture_receipts'));runtime=r('runtime')
    if len(states)<3 or not r('camera_info') or len(frames)<2:
        result=dict(arm=c['executed_arm'],saved=len(receipts),accepted=0,passed=False,runtime=runtime,reason='missing_sensor_or_telemetry')
        save(derived/'SUMMARY.json',result);return result
    info=r('camera_info')[0];K=np.array(info['K']).reshape(3,3);invK=np.linalg.inv(K);size=info['width'],info['height']
    aligned=align(frames,states,.004);s=Surface();sample=s.sample(4);area=sample['area'];domain=s.target(sample,mission,mission['radial_range'])
    ledger=np.zeros(len(area),bool);photos=[];masks=[];observations={a['index']:a for a in obs};identity=[]
    for a in obs:
        if not a['coverage_update_eligible']:continue
        st=aligned[a['index']];hit=IndependentScene(s,st).hit(a['matched_xy'],invK) if st else None
        match=bool(hit and hit['blade']=='A' and hit['side']=='suction_geometric' and all(a[k]['value'][0]<=hit[k]<=a[k]['value'][1] for k in ('eta','xi')))
        identity.append(dict(index=a['index'],station=a['station'],joint_correct=match,wrong_blade=bool(hit and hit['blade']!='A'),truth=hit))
    byindex={r['index']:r for r in identity}
    for receipt in receipts:
        i=receipt['frame_index'];st=aligned[i];j=receipt['station'];patch=patches[j];phase=theta(s,st);p,T=camera_transform(st['camera_pose'])
        raster=s.raster(phase,T,K,size);region=s.target(sample,mission,patch)
        def uv(state):
            _,matrix=camera_transform(state['camera_pose']);world=sample['points'][region]@quat(state['rotor_pose'][3:]).T+state['rotor_pose'][:3]
            camera=world@matrix[:3,:3].T+matrix[:3,3];return (camera@K.T)[:,:2]/camera[:,2:]
        delta=frames[i]['pts']-frames[i-1]['pts'] if i else 0
        exposure=float(np.linalg.norm(uv(st)-uv(aligned[i-1]),axis=1).max()*q['exposure_s']/delta) if i and delta>0 and aligned[i-1] else None
        metrics,mask=s.footprint(sample,quality_config(c),patch,phase,0,p,T,K,raster,exposure)
        original=out/receipt['actual_rgb_file'];photo=out/receipt['file'];exact=sha(photo)==sha(original)==receipt['sha256']
        rgb=cv2.imread(str(photo));pixels=rgb[raster['instance_id']==1];clipped=float(np.mean((pixels.min(1)>248)|(pixels.max(1)<12))) if len(pixels) else 1.
        novel=float(area[mask&~ledger].sum());a=observations[i]
        checks=dict(exact=exact,joint_binding=byindex.get(i,{}).get('joint_correct',False),geometry=metrics['geometric_pass'],
            distance=q['min_distance_m']<=metrics['distance_m']<=q['max_distance_m'],exposure=exposure is not None and exposure<=q['max_exposure_motion_px'],
            exposure_clipped_rgb=clipped<=.10,novel=novel>=q['min_novel_area_m2'],
            fresh=a['decision_sim_s']<=a['fresh_until_sim_pts'] and receipt['saved_wall']<=a['fresh_until_arrival_wall_s'])
        accepted=all(checks.values());masks.append(mask if accepted else np.zeros(len(area),bool))
        if accepted:ledger|=mask
        photos.append(dict(**receipt,accepted=accepted,checks=checks,geometry=metrics,new_independent_area_m2=novel,
            cumulative_fraction=fraction(ledger,domain,area),camera_pose=st['camera_pose'],region_eta=patch,clipped_rgb_fraction=clipped))
    mot,acks=motion(s,states,aligned,r('commands'),obs,c)
    times=[a['available_at']-a['arrived_wall'] for a in obs];span=frames[-1]['arrived_wall']-frames[0]['arrived_wall']
    good_count=sum(a['accepted'] for a in photos);union=fraction(ledger,domain,area);per=[fraction(ledger,s.target(sample,mission,p),area) for p in patches]
    overlaps=[fraction(a&b,a|b,area) for a,b in zip(masks[:-1],masks[1:])]
    multiplicity=np.sum(np.array(masks),axis=0) if masks else np.zeros(len(area),int)
    modes=Counter(a.get('decision',{}).get('reason','not_consumed') for a in obs)
    accepted_ids={x['station'] for x in photos if x['accepted']}
    tests=dict(sensor_wall_rate=(len(frames)-1)/span>=18,latency=bool(times and pct(times)['p95']<=c['control']['max_e2e_p95_wall_s']),
        safety=mot['safety_pass'],identity=not any(x['wrong_blade'] for x in identity),count=good_count==len(patches),
        area=union>=q['min_total_area_fraction'] and min(per)>=q['min_patch_coverage'],
        neighbor_overlap=len(overlaps)==len(patches)-1 and min(overlaps,default=0)>=q['min_neighbor_iou'],runtime=runtime['failure'] is None)
    save(derived/'capture_evaluation.json',photos);save(derived/'visual_independent.json',identity);save(derived/'action_alignment.json',acks)
    np.savez_compressed(derived/'surface_ledger.npz',**sample,required=domain,union=ledger,multiplicity=multiplicity,
                        photo_footprints=np.array(masks,dtype=bool).reshape(-1,len(area)))
    result=dict(arm=c['executed_arm'],scenario=c['executed_scenario'],passed=all(tests.values()),checks=tests,frames=len(frames),
        saved=len(photos),accepted=good_count,coverage=union,per_patch_union_coverage=per,neighbor_iou=overlaps,
        surface_duplicate_fraction=fraction(multiplicity>1,domain,area),missing_station_ids=[i+1 for i in range(len(patches)) if i not in accepted_ids],
        task_completion_pts=max(p['pts'] for p in photos) if good_count==len(patches) else None,
        minimum_accepted_chord_px=min((x['geometry']['projected_min_chord_px'] for x in photos if x['accepted']),default=None),
        wrong_blade_frames=sum(x['wrong_blade'] for x in identity),joint_bad_frames=sum(not x['joint_correct'] for x in identity),
        independent_material_registrations=len(patches),additional_online_oracle_initializations=0,
        received_sensor_hz=(len(frames)-1)/span,processed_wall_hz=len(obs)/span,arrival_to_state_wall_s=pct(times),
        decision_reasons=dict(modes),drop_reasons=dict(Counter(d['reason'] for d in r('drops'))),
        rejection_checks=dict(Counter(k for a in obs for k,v in a.get('decision',{}).get('checks',{}).items() if not v)),
        motion=mot,runtime=runtime,scope='C1 only, textured sensor RGB, independent material banks and camera telemetry')
    save(derived/'SUMMARY.json',result);print(out.name,'saved',len(photos),'accepted',good_count,'coverage',round(union,5),tests,flush=True);return result


def evaluate(root):
    results=[one(p.parent) for p in sorted(Path(root).glob('*/*/runtime.json'))]
    if not results:raise RuntimeError('No preserved actual runs')
    save(Path(root)/'COMPARISON.json',results)
    return results
