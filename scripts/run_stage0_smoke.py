"""Small fixed real-model smoke protocol; never reads legacy prediction caches."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

from uav_visual_tracking.cli import run
from uav_visual_tracking.common.provenance import write_json, sha256

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source",required=True,type=Path)
parser.add_argument("--checkpoint",required=True,type=Path)
parser.add_argument("--data-root",required=True,type=Path)
parser.add_argument("--output",required=True,type=Path)
args=parser.parse_args()
if args.output.exists():raise FileExistsError("Use a new protocol output directory")
args.output.mkdir(parents=True)
repo=Path(__file__).resolve().parents[1]
protected={p:sha256(p) for p in (args.source/'cotracker').rglob('*.py')}
protected.update({p:sha256(p) for p in (args.source/'experiments/camera_motion_compensation').glob('*.py')})
protected.update({p:sha256(p) for p in args.data_root.glob('*.mp4')})
records=[]
protocol=[('wind_support','wind_offline_support','wind_turbine.mp4'),
          ('wind_no_support','wind_offline_no_support','wind_turbine.mp4'),
          ('ordinary','ordinary_offline','woman_walking.mp4'),
          ('official','official_offline_grid',None),
          ('wind_support_repeat','wind_offline_support','wind_turbine.mp4')]
for name,template,video in protocol:
    config=json.loads((repo/'configs/tracking'/f'{template}.json').read_text('utf-8'))
    config.update(video=str(args.data_root/video if video else args.source/'assets/apple.mp4'),output=str(args.output/name))
    config['model'].update(source=str(args.source),checkpoint=str(args.checkpoint))
    print(f'Running real GPU inference: {name}',flush=True)
    manifest=run(config)
    records.append(dict(run=name,frames=manifest['video']['frame_count'],points=len(manifest['point_ids']),
                        config_sha256=manifest['config_sha256'],video_sha256=manifest['video_sha256'],
                        support_grid=manifest['prediction']['model']['support_grid'],
                        inference_seconds=manifest['prediction']['inference_seconds'],
                        inference_fps=manifest['prediction']['inference_fps'],
                        gpu_peak_allocated_bytes=manifest['prediction']['gpu_peak_allocated_bytes'],
                        end_to_end_seconds=manifest['end_to_end_seconds'],status=manifest['status']))
    write_json(args.output/'protocol.json',dict(records=records,status='in_progress'))
    print(records[-1],flush=True)
with np.load(args.output/'wind_support/tracks.npz',allow_pickle=False) as a,np.load(args.output/'wind_support_repeat/tracks.npz',allow_pickle=False) as b:
    repeat=dict(max_absolute_coordinate_difference=float(np.max(np.abs(a['tracks']-b['tracks']))),
                visibility_equal=bool(np.array_equal(a['visibility'],b['visibility'])),
                same_config=records[0]['config_sha256']==records[-1]['config_sha256'])
unchanged=all(sha256(p)==value for p,value in protected.items())
driver=subprocess.run(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],capture_output=True,text=True)
write_json(args.output/'protocol.json',dict(records=records,repeatability=repeat,legacy_protected_count=len(protected),
                                          legacy_unchanged=unchanged,gpu_driver=driver.stdout.strip(),status='success',
                                          model_training=False,ground_truth_used=False))
if not unchanged:raise RuntimeError('Legacy protected inputs changed')
print(json.dumps(dict(status='success',repeatability=repeat,legacy_unchanged=unchanged),indent=2),flush=True)
