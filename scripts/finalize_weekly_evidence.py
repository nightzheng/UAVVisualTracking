"""Verify weekly outputs and preserved baseline; export a human todo, not GT."""
import csv
import json
from pathlib import Path
from uav_visual_tracking.common.provenance import sha256,write_json

repo=Path(__file__).resolve().parents[1]
output=repo/'experiments/weekly/20261013_evidence_record.json'
if output.exists():raise FileExistsError('Do not overwrite weekly evidence')
freeze_path=repo/'configs/evaluation/v0_frozen_20261013.json'
freeze=json.loads(freeze_path.read_text('utf-8'))
for path,value in freeze['implementation_sha256'].items():
    if sha256(repo/path)!=value:raise ValueError('Frozen baseline implementation changed: '+path)
for name,value in freeze['tracking_template_sha256'].items():
    if sha256(repo/'configs/tracking'/name)!=value:raise ValueError('Tracking template changed')
previous=json.loads((repo/'docs/v0/STAGE1_EVIDENCE_RECORD.json').read_text('utf-8'))
historical=[]
for relative,value in previous['artifacts_sha256'].items():
    if sha256(repo/relative)!=value:raise ValueError('Stage 1 evidence changed: '+relative)
    historical.append(relative)
for item in previous['controlled_protocol']['records']:
    directory=repo/'experiments/runs/stage1_20261013_controlled'/item['run']
    if sha256(directory/'manifest.json')!=item['manifest_sha256']:raise ValueError('Historical model run changed')
    manifest=json.loads((directory/'manifest.json').read_text('utf-8'))
    for name,value in manifest['artifacts_sha256'].items():
        if sha256(directory/name)!=value:raise ValueError('Historical artifact changed')
doc=json.loads((repo/'experiments/annotations/stage1_wind_120/annotations.blank.json').read_text('utf-8'))
if any(r['review_status']!='pending' or r['x'] is not None or r['y'] is not None for r in doc['annotations']):
    raise ValueError('Blank annotation changed')
queue=[]
groups=[('query_definition',[0],['p000','p001','p002','p003'],'确认点定义及旧查询对应，不凭模型补标签'),
    ('representative',list(range(0,120,10))+[119],['p001','p002','p003'],'原视频保持 A/B/C 身份，不能辨认标不确定'),
    ('C01',[20,28,36],['p002'],'确认可见/遮挡/出画与塔身附近材料对应'),
    ('C02',[54,61,62,68],['p001'],'复核跨相似叶片关联候选'),
    ('C04',[56,57,62],['p003'],'复核关闭网格跳变的物理身份'),
    ('returns',[83,90,105,119],['p001','p002','p003'],'确认可见返回是否恢复原材料身份'),
    ('partial_pilot',[18,19,20,21,22,50,51,52,53,54,55,56,61,62,63,64,83,84,85,86,87,88],['p001'],'先确认源材料身份再判裁剪视域；预测候选不是 GT')]
for kind,frames,points,note in groups:
    for frame in frames:
        for pid in points:queue.append(dict(case_id=kind,source_frame_index=frame,timestamp_seconds=doc['frames'][frame]['timestamp_seconds'],
            point_id=pid,question=note,review_status='pending_human',annotator='',reviewer='',gt_file='',note=''))
queue_path=repo/'experiments/weekly/20261013_manual_review_queue.csv'
if queue_path.exists():raise FileExistsError('Human review queue exists')
with queue_path.open('w',newline='',encoding='utf-8-sig') as f:
    writer=csv.DictWriter(f,fieldnames=list(queue[0]));writer.writeheader();writer.writerows(queue)
tests=(repo/'.local/weekly_contract_tests.txt').read_text('utf-8-sig')
if 'Ran 37 tests' not in tests or 'FAILED' in tests or 'OK' not in tests:raise ValueError('Tests not passed')
review=json.loads((repo/'.local/review_reference.json').read_text('utf-8'))
pilot_root=repo/'experiments/runs/weekly_20261013_partial'
pilot=json.loads((pilot_root/'protocol.json').read_text('utf-8'))
verification=json.loads((pilot_root/'pilot_review/verification.json').read_text('utf-8'))
if len(verification['checks'])!=8 or any(x['csv_original_coordinate_max_error']!=0 for x in verification['checks']):
    raise ValueError('Crop verification failed')
v1=json.loads((repo/'experiments/weekly/20261013_v1_known_transforms/protocol.json').read_text('utf-8'))
if v1['status']!='mathematical_contract_passed':raise ValueError('Known transforms failed')
legacy_path=repo/'experiments/weekly/20261013_legacy_integrity.json'
legacy=json.loads(legacy_path.read_text('utf-8'))
if not legacy['unchanged']:raise ValueError('Legacy changed')
files=[freeze_path,repo/'configs/evaluation/weekly_partial_20261013.json',repo/'.local/weekly_contract_tests.txt',queue_path,legacy_path,
    pilot_root/'protocol.json',pilot_root/'pilot_review/verification.json',repo/'experiments/weekly/20261013_v1_known_transforms/protocol.json']
files += [repo/'docs'/name for name in ['v0/V0_BASELINE_STATUS.md','v0/V0_GT_PROGRESS.md',
    'v0/V0_PARTIAL_VISIBILITY_PLAN.md','v1/V1_EGOMOTION_PLAN.md','WEEKLY_PROGRESS_20261012_1018.md',
    'v1/history/V1_EGOMOTION_PLAN_stage0_20261013.md']]
files += [repo/'scripts'/name for name in ['run_weekly_partial.py','prepare_v1_known_transforms.py',
    'summarize_weekly_partial.py','freeze_v0_weekly.py','finalize_weekly_evidence.py']]
files += [repo/'src/uav_visual_tracking/data/crops.py',repo/'src/uav_visual_tracking/geometry/transforms.py',repo/'tests/test_weekly_contracts.py']
write_json(output,dict(week='2026-10-12/2026-10-18',as_of='2026-10-13',status='scope_delivered_awaiting_review',
    review_reference=review,review_sections_checked=['3.1','3.2','3.3','3.4','4.3','6.4','7.3'],
    baseline_id=freeze['baseline_id'],frozen_baseline_sha256=sha256(freeze_path),
    frozen_baseline_implementation_unchanged=True,historical_stage1_artifacts_verified=historical,historical_model_runs_verified=10,
    GT_reviewed_point_frames=0,possible_GT_point_frames=480,no_codex_generated_human_GT=True,
    review_queue=dict(rows=len(queue),kind='human_todo_not_GT',file=queue_path.relative_to(repo).as_posix()),
    partial_pilot=pilot,partial_verification=verification,v1_mathematical_fixtures=v1,legacy_integrity=legacy,
    tests=dict(passed=37,total=37,new_crop_and_transform_tests=10),
    real_camera_estimation_validated=False,real_partial_nearfield_validated=False,
    artifacts_sha256={p.relative_to(repo).as_posix():sha256(p) for p in files},
    stop_state='awaiting_review; no training/fusion/full_V1/V2/remote_actions',
    original_v1_plan_archive_sha256=sha256(repo/'docs/v1/history/V1_EGOMOTION_PLAN_stage0_20261013.md')))
print(json.dumps(dict(saved=output.relative_to(repo).as_posix(),status='awaiting_review',GT_reviewed=0,tests_passed=37)))
