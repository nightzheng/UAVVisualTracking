"""Verify retained artifacts and write the compact Stage 1 evidence record."""
import json
from pathlib import Path
import numpy as np
from uav_visual_tracking.common.provenance import sha256,write_json
from uav_visual_tracking.data.results import load_result

repo=Path(__file__).resolve().parents[1]
root=repo/'experiments/runs/stage1_20261013_controlled'
protocol=json.loads((root/'protocol.json').read_text('utf-8'))
verified=[]
for record in protocol['records']:
    directory=root/record['run'];data,manifest=load_result(directory)
    if sha256(directory/'manifest.json') != record['manifest_sha256']:raise ValueError('Manifest changed')
    for name,expected in manifest['artifacts_sha256'].items():
        if sha256(directory/name)!=expected:raise ValueError('Artifact changed: '+name)
    if len(data['frame_indices'])!=120 or data['tracks'].shape!=(120,4,2):raise ValueError('Unexpected shape')
    verified.append(record['run'])
legacy=json.loads((repo/'docs/migration/STAGE1_INTEGRITY.json').read_text('utf-8'))
if not legacy['unchanged']:raise ValueError('Legacy integrity failed')
tests=(repo/'.local/stage1_tests.txt').read_text('utf-8-sig')
if 'Ran 27 tests' not in tests or 'OK' not in tests or 'FAILED' in tests:raise ValueError('Test log does not confirm success')
annotation=repo/'experiments/annotations/stage1_wind_120/annotations.blank.json'
doc=json.loads(annotation.read_text('utf-8'))
if any(r['x'] is not None or r['y'] is not None or r['review_status']!='pending' for r in doc['annotations']):
    raise ValueError('Blank annotation artifact was populated')
source_followup=dict(date='2026-10-13',exact_sources_verified=False,downloaded_remote_media=False,
    user_confirmed_platform='Pexels',formal_local_author_and_source_url_remain_null=True,
    candidate_pages=[dict(local_video='wind_turbine.mp4',url='https://www.pexels.com/video/close-up-of-a-wind-turbine-9403173/',
        page_author='Steve B',page_resolution=[3840,2160],page_fps=25,page_duration_seconds=79,
        local_resolution=[1920,1080],local_fps=29.97002997002997,local_duration_seconds=794/29.97002997002997,
        verification='unconfirmed; metadata mismatch and no download/edit provenance'),
        dict(local_video='woman_walking.mp4',url='https://www.pexels.com/video/a-woman-walking-on-the-streets-6000644/',
        page_author='Sam Lion',page_resolution=[3840,2160],page_fps=23.98,page_duration_seconds=10,
        local_resolution=[3840,2160],local_fps=23.976023976023978,local_duration_seconds=186/23.976023976023978,
        verification='unconfirmed; no visual identity/download/edit provenance')],
    next_evidence='original material URL/download history and frame sequence plus editing record; no automatic private browser history access',
    license_url='https://www.pexels.com/license/',terms_url='https://www.pexels.com/terms-of-service/')
write_json(repo/'docs/v0/SOURCE_FOLLOWUP_STAGE1.json',source_followup)
artifact_paths=[repo/'docs/v0'/f for f in ['V0_GT_ANNOTATION_SPEC.md','V0_ERROR_ANALYSIS.md','V0_CONTROLLED_COMPARISON.md','V0_DATA_REQUIREMENTS.md','SOURCE_FOLLOWUP_STAGE1.json']]
artifact_paths += [repo/'docs/STAGE1_REPORT.md',repo/'configs/evaluation/stage1_protocol.json',annotation,
    repo/'experiments/annotations/stage1_wind_120/index.html',repo/'experiments/stage1/evidence_v2/evidence.json',
    repo/'experiments/stage1/gt_comparison_pending.json',repo/'.local/stage1_tests.txt']
artifact_paths += [repo/'scripts'/name for name in ['run_stage1_controlled.py','sample_gpu_memory.ps1',
    'prepare_stage1_annotations.py','annotation_tool.html','evaluate_stage1.py','compare_stage1_gt.py','build_stage1_evidence.py']]
artifact_paths += list((repo/'src/uav_visual_tracking').rglob('*.py'))
artifact_paths += [repo/'tests/test_stage1.py']
write_json(repo/'docs/v0/STAGE1_EVIDENCE_RECORD.json',dict(stage='Stage 1',date='2026-10-13',
    status='prepared_and_diagnosed_awaiting_human_GT_and_review',stage1_full_acceptance_passed=False,
    reviewed_gt_point_frames=0,possible_active_point_frames=480,real_accuracy_metrics_available=False,
    synthetic_test_fixtures_are_not_wind_GT=True,tests=dict(passed=27,total=27,log_sha256=sha256(repo/'.local/stage1_tests.txt')),
    browser_qa=dict(raw_frames_loaded=True,representative_frame_navigation_verified=True,
        numeric_frame_navigation_verified=[0,62],unannotated_review_rejected=True,
        no_GT_created_during_automated_UI_checks=True,screenshot='.local/annotation_tool_preview.png'),
    controlled_protocol=protocol,verified_real_run_artifacts=verified,
    qualitative_evidence=json.loads((repo/'experiments/stage1/evidence_v2/evidence.json').read_text('utf-8')),
    annotation_preparation=json.loads((repo/'experiments/annotations/stage1_wind_120/preparation.json').read_text('utf-8')),
    legacy_integrity=legacy,artifacts_sha256={p.relative_to(repo).as_posix():sha256(p) for p in artifact_paths},
    stop_state='awaiting_user_review; no automatic optimization/V1/training/remote actions',
    gpu_residency_limit='independent WDDM dedicated/shared counters and device FB samples; no per-allocation certified residency trace'))
print(json.dumps(dict(verified_runs=len(verified),gt_reviewed=0,legacy_files_checked=legacy['checked_existing_files'],status='awaiting_review')))
