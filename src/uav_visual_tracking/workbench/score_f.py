"""Independent U3-F coverage metrics and same-input legacy evaluation."""
import numpy as np
import yaml
from .common import SETTINGS,read,save
from .evaluate import align
from .prepare import mission_for
from .review import frame_data,statistics
from .score import one,quality_config,fraction
from .u3_surface import Surface,windows
from .u3_gazebo import camera_transform
from .u3_g3 import theta


def gsd(points,normals,R,T,K):
    xyz=points@T[:3,:3].T+T[:3,3]
    n=normals@R.T@T[:3,:3].T
    z=xyz[:,2];J=np.zeros((len(z),2,3))
    J[:,0,0]=K[0,0]/z;J[:,1,1]=K[1,1]/z
    J[:,:,2]=-xyz[:,:2]*[K[0,0],K[1,1]]/z[:,None]**2
    tangential=J-(J@n[:,:,None])*n[:,None,:]
    singular=np.linalg.svd(tangential,compute_uv=False)[:,1]
    values=1000/np.maximum(singular,1e-9)
    return [float(np.median(values)),float(np.percentile(values,95))]


def split_evaluate(run, derived=None):
    derived = derived or run
    legacy=one(run, derived);cfg=read(SETTINGS/'imaging_settings.json')
    c=yaml.safe_load((run/'config_snapshot.yaml').read_text());m=mission_for(c)
    receipts=read(derived/'capture_evaluation.json');frames=read(run/'frames.json');states=read(run/'states.json')
    rows=align(frames,states,.004);s=Surface();sample=s.sample(4);area=sample['area'];patches=windows(m)
    domain=s.target(sample,m,m['radial_range']);K=np.array(read(run/'camera_info.json')[0]['K']).reshape(3,3)
    union=np.zeros(len(area),bool);seen={};details=[]
    for r in receipts:
        i=r['station'];patch=patches[i];st=rows[r['frame_index']];p,T=camera_transform(st['camera_pose']);phase=theta(s,st)
        R=s.rotation(phase,0)
        _,foot=s.footprint(sample,quality_config(c),patch,phase,0,p,T,K,s.raster(phase,T,K,(640,480)),r['geometry']['exposure_motion_px'])
        valid=all(v for k,v in r['checks'].items() if k!='novel')
        region=s.target(sample,m,patch);covered=region&foot
        area_coverage=fraction(foot,region,area)
        strat=[]
        for axis,bounds in [('eta',patch),('xi',m['chord_range'])]:
            slices=np.linspace(*bounds,4)
            for a,b in zip(slices[:-1],slices[1:]):
                sub=region&(sample[axis]>=a)&(sample[axis]<=b)
                strat.append(fraction(foot,sub,area))
        complete=valid and min([area_coverage,*strat])>=cfg['min_patch']
        novel=float(area[foot&~union].sum())
        if valid:union|=foot;seen[i]=foot
        points=sample['points'][covered]@R.T+s.hub
        details.append(dict(station=i+1,pts=r['pts'],file=r['file'],quality_pass=valid,region_complete=complete,
            eta_xi_coverage=area_coverage,radial_thirds=strat[:3],chord_thirds=strat[3:],independent_new_m2=novel,
            footprint_m2=float(area[foot].sum()),gsd_worst_axis_p50_p95_mm_px=gsd(points,sample['normals'][covered],R,T,K) if len(points) else None,
            min_chord_px=r['geometry']['projected_min_chord_px'],old_qualified=r['accepted']))
    overlaps=[fraction(seen[i]&seen[i+1],seen[i]|seen[i+1],area) if i in seen and i+1 in seen else None for i in range(len(patches)-1)]
    coverage=fraction(union,domain,area);complete=sum(r['region_complete'] for r in details);good=sum(r['quality_pass'] for r in details)
    observed=frame_data(run);mid=[r for r in observed if 8<=r['pts']<=16]
    motion=legacy['motion'];obs=read(run/'observations.json');commands=read(run/'commands.json')
    deltas=np.linalg.norm(np.diff([c['yaw_pitch_deg'] for c in commands],axis=0),axis=1)
    checks=dict(photos=good==cfg['required_photos'],regions=complete==len(patches),area=coverage>=cfg['min_area'],
        overlap=all(v is not None and v>=cfg['min_neighbor_iou'] for v in overlaps),safety=legacy['checks']['safety'],
        latency=legacy['checks']['latency'] and legacy['checks']['sensor_wall_rate'],identity=legacy['joint_bad_frames']==0,
        runtime=legacy['checks']['runtime'])
    if run.name!='A':
        checks['new_actuator_acceleration']=motion['max_gimbal_accel_deg_s2']<=cfg['angular_acceleration_deg_s2']+1e-5
    result=dict(scenario=run.parent.name,arm=run.name,passed=all(checks.values()),checks=checks,
        quality_count=good,region_count=complete,saved=len(details),coverage=coverage,neighbor_iou=overlaps,
        completion_s=details[-1]['pts'] if all(checks.values()) else None,legacy_count=legacy['accepted'],legacy_coverage=legacy['coverage'],
        legal_fraction=sum(r.get('decision',{}).get('legal') is True for r in obs)/len(obs),arrival_p95_ms=legacy['arrival_to_state_wall_s']['p95']*1000,
        joint_bad=legacy['joint_bad_frames'],actual_wall_vision_hz=legacy['processed_wall_hz'],
        motion=motion,ref_jump_p95_deg=float(np.percentile(deltas,95)),ref_jump_max_deg=float(deltas.max()),
        composition=statistics(observed),comparison_8_16=statistics(mid) if len(mid)>1 else None,photos=details)
    save(derived/'IMAGING_EVALUATION.json',result);print(run.parent.name,run.name,good,complete,coverage,checks,flush=True)
    return result
