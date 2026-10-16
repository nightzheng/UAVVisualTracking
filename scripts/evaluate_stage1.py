"""Evaluate reviewed human JSON and export aligned GT overlays. No interpolation."""
import argparse
import csv
import json
from pathlib import Path
import cv2
import numpy as np
from uav_visual_tracking.common.provenance import sha256,write_json
from uav_visual_tracking.data.results import load_result
from uav_visual_tracking.evaluation.annotations import evaluate_annotations


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True,type=Path)
    parser.add_argument('--gt',required=True,type=Path)
    parser.add_argument('--raw-frames',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    data,manifest=load_result(args.run)
    if sha256(args.run/'tracks.npz') != manifest['artifacts_sha256']['tracks.npz']: raise ValueError('Prediction artifact changed')
    doc=json.loads(args.gt.read_text('utf-8'))
    metrics,(gt,gv,valid)=evaluate_annotations(doc,data,manifest)
    args.output.mkdir(parents=True,exist_ok=False)
    metrics.update(annotation_sha256=sha256(args.gt),prediction_sha256=sha256(args.run/'tracks.npz'),
        result_config_sha256=manifest['config_sha256'],video_sha256=manifest['video_sha256'])
    write_json(args.output/'evaluation.json',metrics)
    # Reuse Stage 0's aligned NPZ interface, preserving sparse valid masks.
    # This export is review-gated, and has zero valid entries for a blank form.
    np.savez_compressed(args.output/'gt_aligned_review_gated.npz',tracks=gt,visibility=gv,valid=valid,
        frame_indices=data['frame_indices'],point_ids=data['point_ids'],video_sha256=doc['video_sha256'],
        coordinate_space='processing',resolution=doc['resolution'],annotation_sha256=sha256(args.gt),
        gt_available=metrics['gt_available'],provenance='human_visual_review_gated_no_interpolation')
    rows=[]
    for t,f in enumerate(data['frame_indices']):
        for p,pid in enumerate(data['point_ids']):
            if not valid[t,p]:continue
            error=np.linalg.norm(data['tracks'][t,p]-gt[t,p]) if gv[t,p] else None
            rows.append(dict(frame_index=int(f),timestamp_seconds=float(data['timestamps'][t]),point_id=str(pid),
                gt_visible=bool(gv[t,p]),predicted_visible=bool(data['visibility'][t,p]),
                epe_processing_pixels=float(error) if error is not None and np.isfinite(error) else None))
        if not valid[t].any():continue
        path=args.raw_frames/f'{int(f):06d}.png'
        if sha256(path) != doc['frame_hashes'][path.name]:raise ValueError('Raw annotation frame changed')
        image=cv2.imread(str(path))
        for p,pid in enumerate(data['point_ids']):
            if not valid[t,p]:continue
            xy=data['tracks'][t,p]
            if np.isfinite(xy).all():
                cv2.drawMarker(image,tuple(np.rint(xy).astype(int)),(0,0,255),cv2.MARKER_TILTED_CROSS,12,2)
                cv2.putText(image,str(pid)+' P'+str(int(data['visibility'][t,p])),tuple(np.rint(xy+[8,0]).astype(int)),0,.45,(0,0,255),1)
            if gv[t,p]:
                cv2.circle(image,tuple(np.rint(gt[t,p]).astype(int)),6,(0,255,0),2)
                cv2.putText(image,str(pid)+' GT',tuple(np.rint(gt[t,p]+[8,15]).astype(int)),0,.45,(0,255,0),1)
        cv2.putText(image,f'frame {f}: green=reviewed human GT; red=prediction', (12,24),0,.55,(255,255,255),2)
        cv2.imwrite(str(args.output/f'overlay_{int(f):06d}.png'),image)
    with (args.output/'point_frame_errors.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=['frame_index','timestamp_seconds','point_id','gt_visible','predicted_visible','epe_processing_pixels']);writer.writeheader();writer.writerows(rows)
    print(json.dumps(dict(status=metrics['status'],annotated_point_frames=metrics['annotated_point_frames'])))


if __name__=='__main__':main()
