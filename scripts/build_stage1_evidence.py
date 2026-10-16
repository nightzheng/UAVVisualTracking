"""Prediction-only visual evidence and resource plots; no invented GT overlays."""
import argparse
import json
from pathlib import Path
import shutil
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from uav_visual_tracking.common.provenance import write_json,sha256
from uav_visual_tracking.data.results import load_result
from uav_visual_tracking.data.video import resize_coordinates


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol',required=True,type=Path);p.add_argument('--raw-frames',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    protocol=json.loads(args.protocol.read_text('utf-8'));root=args.protocol.parent
    runnames=['measured_02_on','measured_03_off'];results=[load_result(root/name)[0] for name in runnames]
    colors=[(255,255,255),(0,255,255),(0,0,255),(255,255,0)]
    cases=[dict(id='C01',frames=[20,28,36],focus_point='p002',hypothesis='visibility_drop_near_tower_overlap_and_hidden_coordinates'),
           dict(id='C02',frames=[61,62,68],focus_point='p001',hypothesis='large_jump_and_periodic_blade_reassociation_candidate'),
           dict(id='C03',frames=[68,90,119],focus_point='p003',hypothesis='predicted_return_does_not_prove_identity_recovery'),
           dict(id='C04',frames=[56,57,62],focus_point='p003',hypothesis='large_jump_with_support_disabled')]
    strips=[]
    for case in cases:
        directory=args.output/case['id'];directory.mkdir();rows=[]
        for run,data in zip(runnames,results):
            panels=[]
            for f in case['frames']:
                raw=args.raw_frames/f'{f:06d}.png';shutil.copyfile(raw,directory/f'raw_{f:06d}.png')
                image=cv2.imread(str(raw));t=int(np.flatnonzero(data['frame_indices']==f)[0])
                for point,pid in enumerate(data['point_ids']):
                    xy=data['tracks'][t,point];color=colors[point]
                    history=data['tracks'][max(0,t-12):t+1,point]
                    if np.isfinite(history).all():cv2.polylines(image,[np.rint(history).astype(np.int32)],False,color,1)
                    if np.isfinite(xy).all():
                        center=tuple(np.rint(xy).astype(int));vis=bool(data['visibility'][t,point])
                        cv2.circle(image,center,6,color,-1 if vis else 2)
                        cv2.putText(image,str(pid)+(' V' if vis else ' H'),tuple(np.rint(xy+[8,-8]).astype(int)),0,.5,color,2)
                banner=np.zeros((52,960,3),np.uint8)
                cv2.putText(banner,f'{run} | frame {f} | t={data["timestamps"][t]:.4f}s', (8,20),0,.55,(255,255,255),1)
                cv2.putText(banner,'PREDICTION ONLY; NO HUMAN GT. Filled=pred visible; ring=pred hidden.',(8,43),0,.5,(255,255,255),1)
                panel=np.vstack([banner,image]);cv2.imwrite(str(directory/f'{run}_{f:06d}.png'),panel)
                panels.append(cv2.resize(panel,(480,296)))
            rows.append(np.hstack(panels))
        contact=np.vstack(rows);cv2.imwrite(str(directory/'before_during_after.jpg'),contact);strips.append(contact)
        case.update(gt_available=False,review_status='qualitative_visual_inspection_pending_human_GT',
            exact_failure_onset=None,physical_identity_switch_verified=False,
            gt_prediction_overlay_status='awaiting_reviewed_human_GT',
            raw_frames_sha256={str(f):sha256(directory/f'raw_{f:06d}.png') for f in case['frames']})
    cv2.imwrite(str(args.output/'cases_contact.jpg'),np.vstack(strips))
    diagnostics=[]
    for name,data in zip(runnames,results):
        for p,pid in enumerate(data['point_ids']):
            v=data['visibility'][:,p];finite=np.isfinite(data['tracks'][:,p]).all(1)
            diagnostics.append(dict(run=name,point_id=str(pid),pred_visible_frames=int(v.sum()),total_frames=len(v),
                hidden_frames=int((~v).sum()),finite_coordinates_while_hidden=int((~v & finite).sum()),
                visibility_returns=data['frame_indices'][1:][~v[:-1] & v[1:]].tolist(),
                first_pred_hidden=int(data['frame_indices'][np.flatnonzero(~v)[0]]) if (~v).any() else None,
                largest_coordinate_step_processing_pixels=float(np.linalg.norm(np.diff(data['tracks'][:,p],axis=0),axis=1).max()),
                largest_step_destination_frame=int(data['frame_indices'][1:][np.linalg.norm(np.diff(data['tracks'][:,p],axis=0),axis=1).argmax()])))
    # OpenCV half-pixel and official align-corners are separate transforms.
    xy=np.array([[0.,0.],[959.,539.],[591.,75.],[626.,470.]])
    orig=resize_coordinates(xy,[960,540],[1920,1080]);roundtrip=resize_coordinates(orig,[1920,1080],[960,540])
    internal=xy*np.array([511/959,383/539]);returned=internal*np.array([959/511,539/383])
    mapping=dict(processing_resolution=[960,540],original_resolution=[1920,1080],model_resolution_wh=[512,384],
        opencv_roundtrip_max_error=float(np.max(np.abs(roundtrip-xy))),official_align_corners_roundtrip_max_error=float(np.max(np.abs(returned-xy))),
        query_frame_max_deviation={name:float(np.max(np.abs(data['tracks'][0]-data['queries'][:,1:]))) for name,data in zip(runnames,results)},
        warning='Official predictor overwrites query frame with input queries and visibility=True; query-frame match is not accuracy evidence',
        gt_alignment='frame_index + point_id + decoded timestamp + video SHA + processing size; no interpolated annotation')
    measured=[r for r in protocol['records'] if not r['warmup_excluded']]
    fig,ax=plt.subplots(figsize=(7,4));ax.scatter(range(1,len(measured)+1),[r['inference_seconds'] for r in measured],
        c=['#1775c5' if r['support_grid'] else '#dc6a25' for r in measured]);ax.set(xlabel='Measured trial (ABBA BAAB)',ylabel='Inference seconds / 120 frames',title='Warm controlled offline trials; blue=on, orange=off');ax.grid(alpha=.25);fig.tight_layout();fig.savefig(args.output/'controlled_latency.png',dpi=150);plt.close(fig)
    samples=[json.loads(l) for l in (root/'gpu_samples.jsonl').read_text('utf-8-sig').splitlines() if l.strip()]
    good=[s for s in samples if s.get('status')=='ok' and s['process_memory']]
    start=samples[0]['sample_start_unix_ms'];x=[(s['sample_end_unix_ms']-start)/1000 for s in good]
    fig,axes=plt.subplots(2,1,figsize=(10,6),sharex=True)
    for field,label in [('DedicatedUsage','Process WDDM dedicated usage'),('SharedUsage','Process WDDM shared usage')]:
        axes[0].plot(x,[sum(int(p[field]) for p in s['process_memory'])/2**30 for s in good],label=label)
    axes[0].axhline(16303/1024,color='gray',ls='--',label='Device physical capacity (15.92 GiB)')
    axes[0].set(ylabel='GiB',title='Independent sampled OS usage proxies (not certified residency)');axes[0].legend(fontsize=8)
    axes[1].plot(x,[sum(int(p['TotalCommitted']) for p in s['process_memory'])/2**30 for s in good],label='WDDM process total committed')
    axes[1].plot(x,[float(s['nvidia_smi'].split(',')[-2])/1024 for s in good],label='nvidia-smi device FB used (includes other processes)')
    axes[1].set(xlabel='Seconds since sampler start (includes load and warmups)',ylabel='GiB');axes[1].legend();fig.tight_layout();fig.savefig(args.output/'gpu_memory_samples.png',dpi=150);plt.close(fig)
    write_json(args.output/'evidence.json',dict(cases=cases,prediction_only_diagnostics=diagnostics,coordinate_mapping=mapping,
        gt_available=False,protocol_sha256=sha256(args.protocol),run_manifest_sha256={name:sha256(root/name/'manifest.json') for name in runnames},
        artifacts_sha256={p.name:sha256(p) for p in args.output.glob('*.png')}))
    print(json.dumps(dict(status='prediction_only_evidence_ready',cases=len(cases),mapping=mapping),indent=2))


if __name__=='__main__':main()
