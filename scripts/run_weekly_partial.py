"""Small fixed crop pilot with matched one-point full-view controls, not new GT."""
import argparse
import csv
import json
from pathlib import Path
import time
import cv2
import numpy as np
from uav_visual_tracking.common.provenance import sha256,write_json,config_hash
from uav_visual_tracking.data.video import read_clip,resize_coordinates
from uav_visual_tracking.data.crops import crop_clip,source_to_crop,crop_to_source
from uav_visual_tracking.data.queries import make_queries
from uav_visual_tracking.data.results import save_result
from uav_visual_tracking.tracking.cotracker3 import CoTracker3
from uav_visual_tracking.evaluation.metrics import diagnostics
from uav_visual_tracking.visualization.video import export_video


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['video','source','checkpoint','output']:parser.add_argument('--'+key,required=True,type=Path)
    args=parser.parse_args();repo=Path(__file__).resolve().parents[1]
    protocol_path=repo/'configs/evaluation/weekly_partial_20261013.json'
    protocol=json.loads(protocol_path.read_text('utf-8'))
    source_hash=sha256(args.video)
    if source_hash!=protocol['video_sha256']:raise ValueError('Unexpected source video')
    args.output.mkdir(parents=True,exist_ok=False)
    base=read_clip(args.video,start_sec=0,duration=8,max_frames=120,max_width=960)
    inputs={};identities=[]
    for view in protocol['views']:
        clip=crop_clip(base,view['roi']);xy=source_to_crop([protocol['source_query_xy']],view['roi'])
        query_config=dict(strategy='manual',frame=0,coordinate_space='processing',point_ids=[protocol['point_id']],points=xy.tolist())
        queries=make_queries(query_config,clip);inputs[view['id']]=(clip,queries,query_config)
        raw=args.output/(view['id']+'_raw');raw.mkdir()
        for frame in [0,20,28,54,61,62,68,90,119]:
            cv2.imwrite(str(raw/f'{frame:06d}.png'),cv2.cvtColor(clip.frames[frame],cv2.COLOR_RGB2BGR))
        identities.append(dict(view=view['id'],roi=view['roi'],processing_size=clip.metadata['processing_resolution'],
            array_sha256=__import__('hashlib').sha256(clip.frames.tobytes()).hexdigest(),query=queries.array.tolist()))
    tracker=CoTracker3(dict(source=str(args.source),checkpoint=str(args.checkpoint),support_grid=True,device='cuda',mode='offline',seed=0))
    measured=[];predictions={}
    # Warm the shared model in both modes. No speed ranking across crop shapes.
    for enabled in [True,False]:
        tracker.support_grid=enabled;tracker.identity=dict(tracker.identity,support_grid=enabled)
        print('Excluded warmup '+str(enabled),flush=True)
        tracker.predict(inputs['full_matched'][0],inputs['full_matched'][1])
    for view in protocol['views']:
        clip,queries,qconfig=inputs[view['id']]
        for enabled in [True,False]:
            name=view['id']+('_on' if enabled else '_off');out=args.output/name;out.mkdir()
            tracker.support_grid=enabled;tracker.identity=dict(tracker.identity,support_grid=enabled)
            print('Real offline crop pilot '+name,flush=True)
            prediction=tracker.predict(clip,queries)
            semantic=dict(source_processing_clip=base.metadata,queries=qconfig,model=tracker.identity,
                video_sha256=source_hash,roi=view['roi'],device='cuda',pilot_protocol_sha256=sha256(protocol_path))
            manifest=dict(schema_version=1,status='in_progress',video_id='sha256:'+source_hash,video_sha256=source_hash,
                config=semantic,config_sha256=config_hash(semantic),video=clip.metadata,prediction=prediction.metadata,
                queries=queries.array.tolist(),point_ids=queries.point_ids,initialization=queries.methods,
                initialization_source='unchanged source query translated to crop; no material identity confirmation',
                source_sequence_count=1,derived_crop_proxy=True,independent_sequence=False)
            save_result(out,clip,queries,prediction,manifest)
            # Generic result writer assumes whole-frame scaling. Replace its CSV
            # original coordinates with the explicit crop-to-source transform.
            with (out/'tracks.csv').open(newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
            for row in rows:
                if row['x_processing']:
                    source_xy=crop_to_source([float(row['x_processing']),float(row['y_processing'])],view['roi'])
                    original_xy=resize_coordinates(source_xy,base.metadata['processing_resolution'],base.metadata['original_resolution'])
                    row.update(x_source_processing=float(source_xy[0]),y_source_processing=float(source_xy[1]),
                        x_original=float(original_xy[0]),y_original=float(original_xy[1]))
                else:row.update(x_source_processing='',y_source_processing='')
            with (out/'tracks.csv').open('w',newline='',encoding='utf-8') as f:
                writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            metric=diagnostics(prediction.tracks,prediction.visibility,queries.array[:,0],clip.metadata['processing_resolution'])
            source_tracks=crop_to_source(prediction.tracks,view['roi']);predictions[(view['id'],enabled)]=(source_tracks,prediction.visibility.copy())
            steps=np.linalg.norm(np.diff(source_tracks[:,0],axis=0),axis=1)
            metric.update(largest_step_source_processing_pixels=float(steps.max()),largest_step_destination_frame=int(steps.argmax()+1),
                material_identity_confirmed=False,gt_accuracy=None)
            write_json(out/'diagnostics.json',metric)
            manifest['visualization']=export_video(out/'tracked.mp4',clip,queries,prediction)
            manifest.update(status='success',artifacts_sha256={p.name:sha256(p) for p in out.iterdir() if p.name!='manifest.json'})
            write_json(out/'manifest.json',manifest)
            measured.append(dict(run=name,view=view['id'],roi=view['roi'],support_grid=enabled,
                input_point_count=1,config_sha256=manifest['config_sha256'],manifest_sha256=sha256(out/'manifest.json'),
                inference_seconds=prediction.metadata['inference_seconds'],inference_fps=prediction.metadata['inference_fps'],
                peak_allocated_bytes=prediction.metadata['gpu_peak_allocated_bytes'],diagnostics=metric))
    comparisons=[]
    for view in protocol['views'][1:]:
        for enabled in [True,False]:
            full,fullvis=predictions[('full_matched',enabled)];local,localvis=predictions[(view['id'],enabled)]
            finite=np.isfinite(full).all(2)&np.isfinite(local).all(2)
            disagreement=np.linalg.norm(local-full,axis=2)[finite]
            comparisons.append(dict(view=view['id'],support_grid=enabled,
                prediction_disagreement_mean_pixels=float(disagreement.mean()),
                prediction_disagreement_p90_pixels=float(np.percentile(disagreement,90)),
                denominator_finite_paired_point_frames=int(finite.sum()),
                meaning='two predictions mapped to common source pixels; NOT GT EPE',
                predicted_visibility_disagreements=int((localvis!=fullvis).sum())))
    if sha256(args.video)!=source_hash:raise RuntimeError('Source changed')
    write_json(args.output/'protocol.json',dict(status='success',date='2026-10-13',protocol=protocol,
        protocol_sha256=sha256(protocol_path),input_identities=identities,records=measured,
        prediction_comparisons=comparisons,source_sequence_count=1,explicit_prompt_count=1,
        reinitializations=0,warmups_excluded=2,measured_model_runs=8,gt_available=False,
        material_identity_confirmed=False,model_commit=tracker.identity['commit'],checkpoint_sha256=tracker.weight_hash,
        timing_policy='descriptive per-run throughput only; crop order/shape not balanced for speed ranking',
        model_architecture_changed=False,online=False,training=False))
    print(json.dumps(dict(status='success',model_runs=8,independent_sequences=1,gt_available=False)),flush=True)


if __name__=='__main__':main()
