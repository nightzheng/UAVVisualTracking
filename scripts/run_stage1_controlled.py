"""Ten bounded real runs: two excluded warmups and eight ABBA/BAAB trials.

Single model, fixed input/query/weight, unchanged official model configuration.
Independent sampler runs in a hidden PowerShell process. Its OS counters are
usage proxies, not a definitive per-allocation residency measurement.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np
from uav_visual_tracking.common.provenance import sha256,write_json,config_hash
from uav_visual_tracking.data.video import read_clip
from uav_visual_tracking.data.queries import make_queries
from uav_visual_tracking.data.results import save_result
from uav_visual_tracking.tracking.cotracker3 import CoTracker3
from uav_visual_tracking.evaluation.metrics import diagnostics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['video','source','checkpoint','output']:p.add_argument('--'+name,required=True,type=Path)
    args=p.parse_args();repo=Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True,exist_ok=False)
    config=json.loads((repo/'configs/tracking/wind_offline_support.json').read_text('utf-8'))
    config['model'].update(source=str(args.source),checkpoint=str(args.checkpoint))
    video_hash=sha256(args.video)
    clip=read_clip(args.video,**config['clip']);queries=make_queries(config['queries'],clip)
    phase=args.output/'phase.json';stop=args.output/'sampler.stop';ready=args.output/'sampler.ready'
    def set_phase(name):
        temporary=phase.with_suffix('.tmp')
        write_json(temporary,dict(name=name,unix_seconds=time.time()))
        os.replace(temporary,phase)
    set_phase('load')
    log=(args.output/'sampler_console.txt').open('w',encoding='utf-8')
    sampler=subprocess.Popen(['powershell.exe','-NoProfile','-ExecutionPolicy','Bypass','-File',str(repo/'scripts/sample_gpu_memory.ps1'),
        '-TargetProcessId',str(os.getpid()),'-OutputFile',str(args.output/'gpu_samples.jsonl'),'-StopFile',str(stop),
        '-ReadyFile',str(ready),'-PhaseFile',str(phase)],stdout=log,stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NO_WINDOW)
    records=[]
    try:
        deadline=time.monotonic()+30
        while not ready.exists():
            if sampler.poll() is not None:raise RuntimeError('GPU sampler exited; inspect sampler_console.txt')
            if time.monotonic()>deadline:raise TimeoutError('GPU sampler did not initialize')
            time.sleep(.1)
        tracker=CoTracker3(config['model'])
        driver=subprocess.run(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.strip()
        order=[True,False,True,False,False,True,False,True,True,False]
        predictions={}
        for index,support in enumerate(order):
            warmup=index<2;name=('warmup' if warmup else 'measured')+f'_{index:02d}_'+('on' if support else 'off')
            set_phase(name)
            tracker.support_grid=support
            tracker.identity=dict(tracker.identity,support_grid=support)
            print('Real GPU '+name,flush=True)
            prediction=tracker.predict(clip,queries)
            set_phase(name+'_export')
            out=args.output/name;out.mkdir()
            semantic=dict(clip=config['clip'],queries=config['queries'],model=tracker.identity,video_sha256=video_hash,device=tracker.device)
            manifest=dict(schema_version=1,video_id='sha256:'+video_hash,video_sha256=video_hash,
                video=clip.metadata,config=semantic,config_sha256=config_hash(semantic),prediction=prediction.metadata,
                queries=queries.array.tolist(),point_ids=queries.point_ids,initialization=queries.methods,
                protocol='stage1_controlled_abba_baab',warmup_excluded=warmup,status='success')
            save_result(out,clip,queries,prediction,manifest)
            write_json(out/'diagnostics.json',diagnostics(prediction.tracks,prediction.visibility,queries.array[:,0],clip.metadata['processing_resolution']))
            manifest['artifacts_sha256']={x.name:sha256(x) for x in out.iterdir() if x.name!='manifest.json'}
            write_json(out/'manifest.json',manifest)
            records.append(dict(run=name,support_grid=support,warmup_excluded=warmup,
                inference_seconds=prediction.metadata['inference_seconds'],inference_fps=prediction.metadata['inference_fps'],
                gpu_peak_allocated_bytes=prediction.metadata['gpu_peak_allocated_bytes'],
                gpu_peak_reserved_bytes=prediction.metadata['gpu_peak_reserved_bytes'],config_sha256=manifest['config_sha256'],
                manifest_sha256=sha256(out/'manifest.json'),tracks_sha256=sha256(out/'tracks.npz')))
            if support in predictions:
                old=predictions[support]
                records[-1]['max_coordinate_difference_from_same_config_first']=float(np.max(np.abs(old[0]-prediction.tracks)))
                records[-1]['visibility_equal_same_config_first']=bool(np.array_equal(old[1],prediction.visibility))
            else:predictions[support]=(prediction.tracks.copy(),prediction.visibility.copy())
            write_json(args.output/'protocol.in_progress.json',dict(records=records))
        if sha256(args.video)!=video_hash or sha256(args.checkpoint)!=tracker.weight_hash:raise RuntimeError('Input changed')
        set_phase('complete')
    finally:
        stop.write_text('stop',encoding='utf-8')
        try:sampler.wait(timeout=15)
        except subprocess.TimeoutExpired:sampler.terminate();sampler.wait(timeout=5)
        log.close()
    samples=[json.loads(line) for line in (args.output/'gpu_samples.jsonl').read_text('utf-8-sig').splitlines() if line.strip()]
    summary={}
    for enabled in [True,False]:
        selected=[r for r in records if r['support_grid']==enabled and not r['warmup_excluded']]
        times=np.array([r['inference_seconds'] for r in selected])
        summary['on' if enabled else 'off']=dict(n=len(times),seconds_median=float(np.median(times)),
            seconds_min=float(times.min()),seconds_max=float(times.max()),seconds_mean=float(times.mean()),
            fps_from_median=len(clip.frames)/float(np.median(times)))
        for record in selected:
            matches=[s for s in samples if s.get('status')=='ok' and (s.get('phase') or {}).get('name')==record['run']]
            # Aggregate matching process instances across adapters; raw names are retained.
            def peak(field):
                values=[sum(int(x[field]) for x in s['process_memory']) for s in matches if s['process_memory']]
                return max(values) if values else None
            record['os_sampling']=dict(samples=len(matches),process_dedicated_usage_peak_bytes=peak('DedicatedUsage'),
                process_shared_usage_peak_bytes=peak('SharedUsage'),process_total_committed_peak_bytes=peak('TotalCommitted'),
                meaning='WDDM reported counters across adapter instances; sampled, asynchronous, may miss peak; not certified physical residency')
    write_json(args.output/'protocol.json',dict(status='success',records=records,summary=summary,
        source_commit=tracker.identity['commit'],checkpoint_sha256=tracker.weight_hash,video_sha256=video_hash,
        gpu_driver=driver,query_array=queries.array.tolist(),point_ids=queries.point_ids,frames=clip.frame_indices.tolist(),
        warmups=2,measured_order='ABBA BAAB (A=support on)',single_model_loaded=True,
        cache_policy='same warmed process; no empty_cache; same sampler active for both; no baseline resolution/iterations changes',
        gt_available=False,gt_accuracy_comparison_status='awaiting_reviewed_human_gt',
        sampler=dict(total_samples=len(samples),error_samples=sum(s.get('status')!='ok' for s in samples),
            sample_file_sha256=sha256(args.output/'gpu_samples.jsonl'),interval_floor_seconds=.2,
            interpretation='OS dedicated/shared/commit usage and device nvidia-smi sampled independently; per-process true residency not directly measured'),
        training=False,online=False))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
