"""One B0/B1 entry to pinned official TAP-Vid metrics; unknown GT is rejected."""
import ast
import hashlib
from importlib.resources import files
from pathlib import Path
from typing import Mapping
import warnings
import numpy as np
from ..tracking.interface import processing_to_original

OFFICIAL_SHA256 = '90cd01e53e23f6d489d3a6cd840cfd93fed4c1a6f4a0dfd373933cc164f0e570'
OFFICIAL_COMMIT = '730cda1c730877cfedbe01bf87fb1cadb78a565d'


def official_compute(source=None):
    source = files('uav_visual_tracking.evaluation._official').joinpath('evaluation_datasets.py') if source is None else Path(source)
    content = source.read_bytes()
    if hashlib.sha256(content).hexdigest() != OFFICIAL_SHA256:
        raise ValueError('Official metric source hash mismatch')
    tree = ast.parse(content)
    tree.body = [node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='compute_tapvid_metrics']
    if len(tree.body) != 1:
        raise ValueError('Missing official metric function')
    namespace={'np':np,'Mapping':Mapping}
    exec(compile(tree,str(source),'exec'),namespace)
    return namespace['compute_tapvid_metrics']


def aligned_inputs(data, manifest, gt):
    """GT is explicitly in original pixels, with independent identity and source alignment."""
    if str(np.asarray(gt['video_id']).item()) != manifest['video_id']:
        raise ValueError('GT video identity mismatch')
    for key in ('frame_indices','timestamps','point_ids'):
        if not np.array_equal(np.asarray(gt[key]),data[key]):
            raise ValueError('GT alignment mismatch: '+key)
    tracks=np.asarray(data['tracks'])
    t,n,_=tracks.shape
    if t < 2 or n < 1 or data['frame_indices'].shape != (t,) or data['timestamps'].shape != (t,):
        raise ValueError('Require T>=2, N>=1 and aligned time arrays')
    if (np.diff(data['frame_indices'])<=0).any() or not np.isfinite(data['timestamps']).all() or (np.diff(data['timestamps'])<=0).any():
        raise ValueError('Frame indices and PTS must increase strictly')
    if len(set(data['point_ids'].tolist())) != n or data['queries'].shape != (n,3):
        raise ValueError('Unique point IDs and N,3 queries required')
    queries=data['queries'].astype(float).copy()
    queries[:,1:]=processing_to_original(queries[:,1:],manifest['video'])
    if not np.allclose(queries,np.asarray(gt['queries_original_txy']),rtol=0,atol=1e-5):
        raise ValueError('GT evaluation queries differ from model task queries')
    if not np.isfinite(queries).all() or (queries[:,0] != np.floor(queries[:,0])).any() or (
            (queries[:,0]<0) | (queries[:,0]>=t)).any():
        raise ValueError('Invalid query time')
    truth=np.asarray(gt['tracks_original'],dtype=float)
    if truth.shape != (t,n,2) or data['visibility'].shape != (t,n) or data['visibility'].dtype!=np.bool_:
        raise ValueError('Invalid prediction/GT dimensions')
    for k in ('visibility','visibility_known','position_known'):
        if np.asarray(gt[k]).shape != (t,n) or np.asarray(gt[k]).dtype != np.bool_:
            raise ValueError('GT flags must be boolean T,N: '+k)
    pred=processing_to_original(tracks,manifest['video'])
    if 'tracks_original' in data and not np.allclose(data['tracks_original'],pred,equal_nan=True):
        raise ValueError('Saved original coordinates disagree with manifest transform')
    return pred,truth,queries


PROTOCOL = 'actual-video-256-v2'


def original_to_processing(xy, video):
    """Inverse crop then pixel-center resize, from the frozen evaluator."""
    original = np.asarray(video['original_resolution'], float)
    source = np.asarray(video.get('source_processing_resolution', video['processing_resolution']), float)
    mapped = (np.asarray(xy, float) + .5) * source / original - .5
    if 'crop_roi_xyxy' in video:
        mapped -= np.asarray(video['crop_roi_xyxy'][:2])
    return mapped


def evaluate_actual_video(data, manifest, gt, query_mode='first'):
    """Unmodified official TAP-Vid math, normalized in the actual video raster.

    The inverse source transform is used only to map source-image GT/queries
    into the prediction raster. C3 uses W=180,H=260, never source W=480,H=360.
    Incomplete GT is rejected; the CLI reports NA without swallowing alignment errors.
    """
    pred, truth, queries = aligned_inputs(data, manifest, gt)
    if query_mode not in {'first', 'strided'}:
        raise ValueError('Use explicit official first or strided query_mode')
    video = manifest['video']
    w,h = video['processing_resolution']
    if not np.isfinite([w,h]).all() or min(w,h)<2:
        raise ValueError('Invalid actual video resolution')
    vis=gt['visibility'];known=gt['visibility_known'];pk=gt['position_known']
    if not known.all() or not (pk | ~vis).all():
        raise ValueError('Official TAP-Vid requires complete visibility and visible-position GT')
    if not np.isfinite(truth[vis]).all():
        raise ValueError('Visible GT positions must be finite')
    qi=queries[:,0].astype(int);ni=np.arange(len(queries))
    if not vis[qi,ni].all() or not np.allclose(truth[qi,ni],queries[:,1:],atol=1e-5,rtol=0):
        raise ValueError('Queries must be visible GT locations')
    np.testing.assert_allclose(original_to_processing(pred,video),data['tracks'],atol=1e-5,rtol=0,equal_nan=True)
    local_truth=original_to_processing(truth,video)
    q=queries[:,[0,2,1]][None].copy()
    q[...,1:]=original_to_processing(queries[:,1:],video)[:,::-1]*[256/h,256/w]
    invalid=~np.isfinite(pred).all(-1)
    normalized_pred=np.where(invalid[...,None],1e9,data['tracks'])*[256/w,256/h]
    normalized_truth=np.where(vis[...,None],local_truth,0)*[256/w,256/h]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        result=official_compute()(q,(~vis).T[None],normalized_truth.transpose(1,0,2)[None],
            (~data['visibility']).T[None],normalized_pred.transpose(1,0,2)[None],query_mode)
    scalar=lambda x:float(x[0]) if np.isfinite(x[0]) else None
    active=np.arange(len(pred))[:,None]>qi if query_mode=='first' else np.arange(len(pred))[:,None]!=qi
    return dict(name='official_TAPVid_actual_video_complete_GT_v2',protocol=PROTOCOL,
        AJ=scalar(result['average_jaccard']),delta_avg=scalar(result['average_pts_within_thresh']),
        OA=scalar(result['occlusion_accuracy']),all_metrics={k:scalar(v) for k,v in result.items()},
        query_mode=query_mode,evaluation_count=int(active.sum()),visible_count=int((active&vis).sum()),
        nonfinite_prediction_count=int((active&invalid).sum()),source_commit=OFFICIAL_COMMIT,
        source_sha256=OFFICIAL_SHA256,actual_evaluation_resolution_wh=[w,h],
        source_reference_resolution_wh=video['original_resolution'],
        coordinate_norm='actual video raster xy * [256/W_video,256/H_video]; no pixel-center offset',
        aggregation='per sequence; complete explicit plan required for parent macro',
        execution='exact verified official function body; AST omits unrelated dataset imports')

