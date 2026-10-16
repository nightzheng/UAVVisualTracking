"""Freeze current baseline identity without modifying prior run artifacts."""
import json
from pathlib import Path
from uav_visual_tracking.common.provenance import sha256,write_json
from uav_visual_tracking.data.results import load_result

repo=Path(__file__).resolve().parents[1]
output=repo/'configs/evaluation/v0_frozen_20261013.json'
if output.exists():raise FileExistsError('A frozen baseline cannot be overwritten; use a new version')
root=repo/'experiments/runs/stage1_20261013_controlled'
references={}
for label,name in [('support_on','measured_02_on'),('support_off','measured_03_off')]:
    data,manifest=load_result(root/name)
    for artifact,value in manifest['artifacts_sha256'].items():
        if sha256(root/name/artifact)!=value:raise ValueError('Historical artifact changed')
    references[label]=dict(config_sha256=manifest['config_sha256'],model=manifest['prediction']['model'],
        semantic_config=manifest['config'],manifest_sha256=sha256(root/name/'manifest.json'),
        tracks_sha256=sha256(root/name/'tracks.npz'),reference_run=f'experiments/runs/stage1_20261013_controlled/{name}')
implementation=[repo/'src/uav_visual_tracking'/name for name in ['tracking/cotracker3.py','data/video.py',
    'data/queries.py','data/results.py','evaluation/metrics.py','evaluation/annotations.py','cli.py']]
write_json(output,dict(schema_version=1,baseline_id='V0-offline-wind120-20261013-r1',frozen_date='2026-10-13',
    frozen_scope='engineering model/input/query/evaluation contract; not verified material identities or a frozen test dataset',
    references=references,initial_queries=data['queries'].tolist(),point_ids=data['point_ids'].tolist(),
    source_frame_range_inclusive=[0,119],processing_resolution=[960,540],
    video_sha256=manifest['video_sha256'],future_frame_access=True,causal=False,online_weight_status='not_found',
    explicit_queries=4,query_frame=0,reinitializations=0,query_semantics_human_confirmed=False,
    support_grid_policy='retain both pinned strategies; do not choose by predicted visibility ratio',
    metric_protocol=json.loads((repo/'configs/evaluation/stage1_protocol.json').read_text('utf-8')),
    metric_protocol_sha256=sha256(repo/'configs/evaluation/stage1_protocol.json'),
    implementation_sha256={p.relative_to(repo).as_posix():sha256(p) for p in implementation},
    tracking_template_sha256={name:sha256(repo/'configs/tracking'/name) for name in ['wind_offline_support.json','wind_offline_no_support.json']},
    GT_revision=None,GT_reviewed_point_frames=0,frozen_test=False,scientific_quality_validated=False,
    crop_pilots_are_separate_query_subset=True,change_policy='New ID and new output directory for baseline changes; old evidence remains intact'))
print(json.dumps(dict(frozen_baseline=str(output.relative_to(repo)),sha256=sha256(output))))
