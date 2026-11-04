from collections import Counter


import numpy as np


from .common import read,save


from .evaluate import align


from .control import geom,angles


from .binding import project_bank


from .u3_gazebo import camera_transform


from .u3_g3 import theta


from .u3_surface import Surface


def frame_data(run):
    spec=read(run/'consumer_spec.json');frames=read(run/'frames.json');states=read(run/'states.json')
    aligned=align(frames,states,.004);surface=Surface();K=np.asarray(spec['K']);g=spec['geometry']
    g['tilt']=np.asarray(g['tilt']);goal=spec['composition']['goal_px']
    obs={r['index']:r for r in read(run/'observations.json')}; transitions=read(run/'transitions.json')
    cmds={r['sequence']:r for r in read(run/'commands.json')};shots={r['frame_index']:r for r in read(run/'capture_receipts.json')}
    output=[]
    for f,st in zip(frames,aligned):
        if st is None:continue
        t=f['pts'];j=min(sum(x['sim_s']<=t for x in transitions),len(spec['banks'])-1);b=spec['banks'][j]
        p,T=camera_transform(st['camera_pose']);phase=theta(surface,st)
        uv,q=geom(b,phase,T,K,g,(640,480));err=np.array(q['center_px'])-goal
        r=obs.get(f['index'],{});dec=r.get('decision',{});src=cmds.get(st['executed_sequence'],{})
        projected,z=project_bank(b['points'],[phase],K,T,g['hub'],g['tilt'])
        inside=(z[0]>0)&(projected[0]>=0).all(1)&(projected[0]<[640,480]).all(1)
        mode=('CAPTURE' if f['index'] in shots else 'RECOVER' if not src else
              'TRANSIT' if np.linalg.norm(p-np.asarray(b['position_ref_m']))>=.95 else
              'TRACK' if src.get('source_index') is not None else 'RECOVER')
        output.append(dict(**f,station=j+1,independent_center=q['center_px'],error_xy=err.tolist(),error=float(np.linalg.norm(err)),
            raw_roi=r.get('roi_bbox'),material=r.get('matched_xy'),legal_center=(r.get('estimated_quality') or {}).get('center_px'),
            scale_pass=q['min_chord_proxy_px']>=spec['quality']['min_chord_px'],out_of_frame=not bool(inside.any()),cropped=not bool(inside.all()),
            margin_px=q['min_margin_px'],mode=mode,decision_mode=dec.get('mode'),source=src.get('reason','hold'),
            captured=f['index'] in shots))
    return output


def statistics(rows):
    errors=np.array([r['error_xy'] for r in rows]);norm=np.linalg.norm(errors,axis=1);bad=norm>54;maxrun=run=0
    for flag in bad:
        run=run+1 if flag else 0;maxrun=max(maxrun,run)
    return dict(frames=len(rows),p50=float(np.median(norm)),p95=float(np.percentile(norm,95)),
        axis_mean=errors.mean(0).tolist(),axis_rms=np.sqrt(np.mean(errors**2,axis=0)).tolist(),rms=float(np.sqrt(np.mean(norm**2))),
        above_54px_s=float(bad.sum()/20),longest_above_54px_s=maxrun/20,
        out_of_frame_s=sum(r['out_of_frame'] for r in rows)/20,scale_pass_fraction=sum(r['scale_pass'] for r in rows)/len(rows),
        crop_fraction=sum(r['cropped'] for r in rows)/len(rows),mode_counts=dict(Counter(r['mode'] for r in rows)),
        mode_changes=sum(a['mode']!=b['mode'] for a,b in zip(rows,rows[1:])))
