"""Require identical input/query/model identity and one reviewed GT document."""
import argparse
import json
from pathlib import Path
import numpy as np
from uav_visual_tracking.common.provenance import sha256,write_json
from uav_visual_tracking.data.results import load_result
from uav_visual_tracking.evaluation.annotations import evaluate_annotations


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['on','off','gt','output']:p.add_argument('--'+name,required=True,type=Path)
    args=p.parse_args();a,ma=load_result(args.on);b,mb=load_result(args.off)
    for key in ['frame_indices','timestamps','point_ids','queries']:
        if not np.array_equal(a[key],b[key]):raise ValueError('Different '+key)
    ca,cb=dict(ma['config']),dict(mb['config'])
    ca['model']=dict(ca['model']);cb['model']=dict(cb['model'])
    if ca['model'].pop('support_grid') is not True or cb['model'].pop('support_grid') is not False:
        raise ValueError('Expected support on/off')
    if ca!=cb or ma['video']!=mb['video']:raise ValueError('Configuration differs beyond support grid')
    doc=json.loads(args.gt.read_text('utf-8'))
    metrics={}
    for label,data,manifest,directory in [('on',a,ma,args.on),('off',b,mb,args.off)]:
        if sha256(directory/'tracks.npz') != manifest['artifacts_sha256']['tracks.npz']:raise ValueError('Result artifact changed')
        metrics[label]=evaluate_annotations(doc,data,manifest)[0]
    if metrics['on']['coverage'] != metrics['off']['coverage']:raise ValueError('GT denominators differ')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.output.exists():raise FileExistsError('Use a new comparison result path')
    write_json(args.output,dict(status='evaluated' if metrics['on']['gt_available'] else 'awaiting_reviewed_human_gt',
        same_input_query_weight_and_gt=True,annotation_sha256=sha256(args.gt),
        on_manifest_sha256=sha256(args.on/'manifest.json'),off_manifest_sha256=sha256(args.off/'manifest.json'),
        metrics=metrics,best_configuration=None,selection_policy='Do not select using predicted visibility alone'))
    print(json.dumps(dict(status='evaluated' if metrics['on']['gt_available'] else 'awaiting_reviewed_human_gt')))


if __name__=='__main__':main()
