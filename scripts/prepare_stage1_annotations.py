"""Export raw frames and blank human annotation records, never predictions."""
import argparse
import csv
import json
from pathlib import Path
import cv2
import numpy as np
from uav_visual_tracking.common.provenance import sha256, write_json
from uav_visual_tracking.data.video import read_clip
from uav_visual_tracking.data.results import load_result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', required=True, type=Path)
    parser.add_argument('--reference-run',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    data,manifest=load_result(args.reference_run)
    if sha256(args.video) != manifest['video_sha256']: raise ValueError('Video identity mismatch')
    clip=read_clip(args.video, **manifest['config']['clip'])
    np.testing.assert_array_equal(clip.frame_indices,data['frame_indices'])
    np.testing.assert_allclose(clip.timestamps,data['timestamps'],rtol=0,atol=1e-9)
    args.output.mkdir(parents=True,exist_ok=False)
    raw=args.output/'raw';raw.mkdir()
    hashes={}
    for i,frame in enumerate(clip.frames):
        path=raw/f'{int(clip.frame_indices[i]):06d}.png'
        if not cv2.imwrite(str(path),cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)): raise RuntimeError('Frame export failed')
        hashes[path.name]=sha256(path)
    # These descriptions are hypotheses about query intent, not validated GT.
    names=['p000: hub/axis reference (geometric center, not a material point)',
           'p001: blade A, initially upper blade; original query must be verified',
           'p002: blade B, initially lower-right blade; original query must be verified',
           'p003: blade C, initially left blade; original query must be verified']
    points=[dict(point_id=str(pid), label=names[p], definition='', blade_id=[None,'A','B','C'][p],
        point_kind='geometric_reference' if p==0 else 'material_point_pending_confirmation',
        initial_query_processing=data['queries'][p,1:].tolist(),
        definition_reviewed=False,definition_reviewer='',query_correspondence_confirmed=False)
        for p,pid in enumerate(data['point_ids'])]
    representative=list(range(0,120,10))+[119]
    dense=sorted({t for center in [28,36,52,54,55,68] for t in range(center-3,center+4) if 0<=t<120})
    doc=dict(schema_version=1,annotation_kind='human_visual',video_sha256=manifest['video_sha256'],
        video_id=manifest['video_id'],resolution=clip.metadata['processing_resolution'],
        original_resolution=clip.metadata['original_resolution'],coordinate_space='processing',
        time_basis=clip.metadata['time_basis'],timestamp_meaning=clip.metadata['timestamp_meaning'],
        point_definition_policy='Human must confirm original query material identity; do not silently redefine a query as a different tip/feature.',
        points=points,case_reviews=[],representative_frames=representative,dense_frames=dense,
        frame_hashes=hashes,frames=[dict(frame_index=int(f),timestamp_seconds=float(t),
            image=f'raw/{int(f):06d}.png') for f,t in zip(clip.frame_indices,clip.timestamps)],
        annotations=[dict(frame_index=int(f),timestamp_seconds=float(t),point_id=str(pid),x=None,y=None,
            state='pending',occluded=None,out_of_frame=None,uncertainty='pending',source='unannotated',
            annotator='',reviewer='',review_status='pending',reviewed_at_utc='',note='')
            for f,t in zip(clip.frame_indices,clip.timestamps) for pid in data['point_ids']])
    write_json(args.output/'annotations.blank.json',doc)
    with (args.output/'annotations.blank.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        rows=[dict(video_sha256=doc['video_sha256'],coordinate_space='processing',**r) for r in doc['annotations']]
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    template=(Path(__file__).parent/'annotation_tool.html').read_text('utf-8')
    (args.output/'index.html').write_text(template.replace('__ANNOTATION_DOCUMENT__',json.dumps(doc,ensure_ascii=False).replace('</','<\\/')),encoding='utf-8')
    write_json(args.output/'preparation.json',dict(status='awaiting_human_annotation_and_review',frames=len(clip.frames),
        blank_point_frames=len(doc['annotations']),reviewed_gt_point_frames=0,raw_frames_only=True,
        reference_run=str(args.reference_run.resolve()),reference_manifest_sha256=sha256(args.reference_run/'manifest.json'),
        annotation_blank_sha256=sha256(args.output/'annotations.blank.json')))
    print(json.dumps(dict(status='prepared_blank_no_gt',frames=len(clip.frames),output=str(args.output)),ensure_ascii=False))


if __name__=='__main__': main()
