"""Generic Offline Tracking demo. Explicit user video, queries, source and weights."""
import argparse
import json
from pathlib import Path
import numpy as np
from .common.provenance import sha256,write_json,config_hash
from .data.video import read_clip,resize_coordinates
from .data.queries import Queries,make_queries
from .data.results import save_result,load_result
from .tracking.interface import validate_rgb_queries


def load_queries(path,clip):
    config=json.loads(Path(path).read_text('utf8'))
    if 'queries_txy' in config:
        q=np.asarray(config['queries_txy'],dtype=np.float32)
        if q.ndim!=2 or q.shape[1]!=3 or len(q)==0:raise ValueError('queries_txy must be nonempty N,3')
        ids=config.get('point_ids',[f'p{i:03}' for i in range(len(q))])
        if config.get('coordinate_space','processing')=='original':
            video=clip.metadata
            source=video.get('source_processing_resolution',video['processing_resolution'])
            q[:,1:]=resize_coordinates(q[:,1:],video['original_resolution'],source)
            if 'crop_roi_xyxy' in video:q[:,1:]-=video['crop_roi_xyxy'][:2]
        elif config.get('coordinate_space','processing')!='processing':raise ValueError('Unknown query coordinate_space')
        queries=Queries(q,ids,['manual']*len(q))
    else:
        queries=make_queries(config,clip)
    if not all(isinstance(p,str) and p for p in queries.point_ids):raise ValueError('Nonempty string point IDs required')
    validate_rgb_queries(clip,queries)
    return queries


def export_overlay(path,clip,queries,prediction):
    """CFR only: encoded frame intervals must agree with saved PTS."""
    import cv2
    fps=clip.metadata['fps']
    if not clip.metadata.get('cfr_observed') or not np.allclose(np.diff(clip.timestamps),1/fps,rtol=1e-4,atol=1e-6):
        raise ValueError('MP4 export requires verified CFR PTS; VFR/assumed timing is preserved in NPZ but cannot be exported as CFR')
    path=Path(path)
    if path.exists():raise FileExistsError('Overlay output exists')
    t,h,w,_=clip.frames.shape
    # Keep native pixels; pad odd dimensions and add a label area.
    width=max(640,w+(w%2));height=h+(h%2)+84
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),fps,(width,height))
    if not writer.isOpened():raise RuntimeError('MP4 encoder unavailable')
    try:
        for i,rgb in enumerate(clip.frames):
            frame=np.full((height,width,3),245,np.uint8)
            frame[:h,:w]=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
            for p,pid in enumerate(queries.point_ids):
                if i<queries.array[p,0]:continue
                xy=prediction.tracks[i,p]
                if not np.isfinite(xy).all() or (xy<0).any() or (xy>[w-1,h-1]).any():continue
                pt=tuple(np.rint(xy).astype(int));color=(180,90,30)
                if prediction.visibility[i,p]:
                    cv2.circle(frame,pt,4,color,-1,cv2.LINE_AA)
                    start=max(int(queries.array[p,0]),i-12)
                    for j in range(start,i):
                        line=prediction.tracks[j:j+2,p]
                        if prediction.visibility[j:j+2,p].all() and np.isfinite(line).all() and (line>=0).all() and (line<=[w-1,h-1]).all():
                            cv2.line(frame,tuple(np.rint(line[0]).astype(int)),tuple(np.rint(line[1]).astype(int)),color,1,cv2.LINE_AA)
                else:cv2.drawMarker(frame,pt,color,cv2.MARKER_TILTED_CROSS,9,1,cv2.LINE_AA)
                cv2.putText(frame,str(pid), (pt[0]+6,pt[1]-6),cv2.FONT_HERSHEY_SIMPLEX,.4,color,1,cv2.LINE_AA)
            for j,text in enumerate(['Offline Tracking | native acceptance, no identity verification',
                f'source frame {clip.frame_indices[i]} | PTS {clip.timestamps[i]:.6f} s',
                'filled point: native accept | x: native reject | last 12 accepted frames']):
                cv2.putText(frame,text,(8,h+20+j*23),cv2.FONT_HERSHEY_SIMPLEX,.46,(25,25,25),1,cv2.LINE_AA)
            writer.write(frame)
    finally:writer.release()
    cap=cv2.VideoCapture(str(path));count=0
    try:
        while cap.read()[0]:count+=1
    finally:cap.release()
    if count!=t:raise RuntimeError('Overlay encoded frame count mismatch')
    return dict(file=path.name,frames=t,fps=fps,first_source_PTS_seconds=float(clip.timestamps[0]),
        output_time='relative to first selected source PTS; original frame/PTS burned in',
        temporal_policy='verified CFR only; no resampling',sha256=sha256(path))


def run(config,tracker_factory=None):
    """tracker_factory is an engineering test seam, never a model-run claim."""
    output=Path(config['output'])
    if output.exists():raise FileExistsError('Use a new output directory')
    for key in ('video','queries','checkpoint'):
        if not Path(config[key]).is_file():raise FileNotFoundError('Provide an existing external '+key+' file; no bundled assets or private defaults')
    if not Path(config['source']).is_dir():raise FileNotFoundError('Provide the pinned official source directory')
    backend=config['backend']
    if backend not in {'B0','B1'}:raise ValueError('Choose B0 (CoTracker3) or B1 (BootsTAPIR)')
    clip=read_clip(config['video'],**config.get('clip',{}))
    if config.get('roi'):
        roi=config['roi'];x0,y0,x1,y1=roi;w,h=clip.metadata['processing_resolution']
        if not all(type(x) is int for x in roi) or not (0<=x0<x1<=w and 0<=y0<y1<=h) or min(x1-x0,y1-y0)<2:
            raise ValueError('ROI must be integer xyxy inside decoded processing video, size>=2')
        clip.frames=clip.frames[:,y0:y1,x0:x1].copy()
        clip.metadata.update(source_processing_resolution=[w,h],crop_roi_xyxy=roi,processing_resolution=[x1-x0,y1-y0])
    queries=load_queries(config['queries'],clip)
    if config.get('overlay') and not clip.metadata.get('cfr_observed'):
        raise ValueError('Overlay requires verified CFR PTS; omit --overlay for VFR or assumed timing')
    model=dict(source=config['source'],checkpoint=config['checkpoint'],device=config.get('device','cuda'),
        mode='offline',seed=0,export_raw_scores=True,support_grid=config.get('support_grid',True),
        model_resolution_hw=config.get('b1_resolution',[256,256]),query_chunk_size=64)
    if tracker_factory is None:
        if backend=='B0':
            from .tracking.cotracker3 import CoTracker3
            tracker_factory=CoTracker3
        else:
            from .tracking.bootstapir import BootsTAPIR
            tracker_factory=BootsTAPIR
    output.mkdir(parents=True,exist_ok=False)
    try:
        tracker=tracker_factory(model);prediction=tracker.predict(clip,queries)
        required={'visibility_probability','confidence_probability'} if backend=='B0' else {'visibility_score','occlusion_logit','expected_dist_logit'}
        if not required<=prediction.raw_scores.keys():raise ValueError('Backend failed the continuous raw-score contract')
        prediction.validate(len(clip.frames),len(queries.array))
        video_hash=sha256(config['video'])
        semantic=dict(backend=backend,model=tracker.identity,video_sha256=video_hash,clip=clip.metadata,
            queries=queries.array.tolist(),point_ids=queries.point_ids)
        manifest=dict(schema_version=2,video_id='sha256:'+video_hash,video_sha256=video_hash,
            video=clip.metadata,prediction=prediction.metadata,config=semantic,config_sha256=config_hash(semantic),
            queries=queries.array.tolist(),point_ids=queries.point_ids,initialization=queries.methods,
            status='success',purpose='user Offline Tracking demo; not a formal heldout experiment',
            demo_differences='User clip/query/resize/ROI; no frozen experimental bindings or resource controller. B0 native support on unless explicitly disabled; B1 256x256 by default.')
        save_result(output,clip,queries,prediction,manifest);load_result(output)
        if config.get('overlay'):manifest['overlay']=export_overlay(output/'overlay.mp4',clip,queries,prediction)
        manifest['artifacts_sha256']={p.name:sha256(p) for p in output.iterdir() if p.name!='manifest.json'}
        write_json(output/'manifest.json',manifest)
        return manifest
    except Exception as error:
        write_json(output/'failure.json',dict(status='failed',error_type=type(error).__name__,message=str(error),no_success_claim=True))
        raise


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend',choices=['B0','B1'],required=True)
    for name in ('source','checkpoint','video','queries','output'):p.add_argument('--'+name,required=True,type=Path)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--start-sec',type=float,default=0.);p.add_argument('--duration',type=float,default=8.)
    p.add_argument('--max-frames',type=int,default=120);p.add_argument('--max-width',type=int,default=640)
    p.add_argument('--roi',nargs=4,type=int,metavar=('X0','Y0','X1','Y1'))
    p.add_argument('--b1-resolution',nargs=2,type=int,default=[256,256],metavar=('H','W'))
    p.add_argument('--no-support-grid',action='store_true');p.add_argument('--overlay',action='store_true')
    a=p.parse_args(argv)
    config={k:str(getattr(a,k)) for k in ('source','checkpoint','video','queries','output')}
    config.update(backend=a.backend,device=a.device,support_grid=not a.no_support_grid,roi=a.roi,
        b1_resolution=a.b1_resolution,overlay=a.overlay,
        clip=dict(start_sec=a.start_sec,duration=a.duration,max_frames=a.max_frames,max_width=a.max_width))
    try:m=run(config)
    except (ValueError,FileNotFoundError,FileExistsError,RuntimeError,ImportError) as error:p.exit(1,str(error)+'\n')
    print(json.dumps(dict(status=m['status'],backend=a.backend,mode='Offline Tracking',output=a.output.name)))
    return 0


if __name__=='__main__':raise SystemExit(main())
