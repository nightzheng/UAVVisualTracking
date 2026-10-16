"""Adapted visibility statistics from legacy owner target_tracking.py.

Added aligned GT metrics. Prediction events are never labelled true occlusions.
"""
import numpy as np


def ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def diagnostics(tracks, visibility, query_frames, size):
    tracks, visibility = np.asarray(tracks), np.asarray(visibility)
    if tracks.ndim != 3 or tracks.shape[-1] != 2 or visibility.shape != tracks.shape[:2] or visibility.dtype != np.bool_:
        raise ValueError("Invalid tracks or visibility")
    active = np.arange(len(tracks))[:, None] >= np.asarray(query_frames)[None]
    finite = np.isfinite(tracks).all(2)
    width, height = size
    outside = finite & ((tracks[..., 0] < 0) | (tracks[..., 0] > width-1) |
                        (tracks[..., 1] < 0) | (tracks[..., 1] > height-1))
    per_point = []
    for p in range(tracks.shape[1]):
        hidden = ~visibility[:, p]
        valid_transitions = active[1:, p] & active[:-1, p]
        indices = np.flatnonzero(hidden & active[:, p])
        per_point.append(dict(point_index=p, predicted_visible_ratio=ratio(int((visibility[:, p] & active[:, p]).sum()), int(active[:, p].sum())),
                              first_predicted_invisible_clip_frame=int(indices[0]) if len(indices) else None,
                              predicted_invisibility_entries=int((~hidden[:-1] & hidden[1:] & valid_transitions).sum()),
                              predicted_visibility_returns=int((hidden[:-1] & ~hidden[1:] & valid_transitions).sum()),
                              predicted_out_of_image_frames=int((outside[:, p] & active[:, p]).sum()),
                              predicted_exit_events=int((~outside[:-1, p] & outside[1:, p] & valid_transitions & finite[:-1, p] & finite[1:, p]).sum()),
                              predicted_reentry_events=int((outside[:-1, p] & ~outside[1:, p] & valid_transitions & finite[:-1, p] & finite[1:, p]).sum())))
    return dict(gt_available=False, metric_kind="prediction_only_diagnostics",
                event_meaning="predicted visibility/bounds transitions; not annotated occlusion or verified recovery",
                active_point_frames=int(active.sum()), nonfinite_active_predictions=int((active & ~finite).sum()),
                predicted_visible_ratio=ratio(int((active & visibility).sum()), int(active.sum())), per_point=per_point)


def evaluate_gt(predicted_tracks, predicted_visible, gt_tracks, gt_visible, valid, active, thresholds=(2, 5, 10, 20)):
    prediction = np.asarray(predicted_tracks)
    gt = np.asarray(gt_tracks)
    shape = prediction.shape[:2]
    if prediction.ndim != 3 or prediction.shape[-1] != 2 or gt.shape != prediction.shape:
        raise ValueError("GT and prediction coordinates must align")
    for value in [predicted_visible, gt_visible, valid, active]:
        if np.asarray(value).shape != shape or np.asarray(value).dtype != np.bool_:
            raise ValueError("Visibility/annotation/active masks must be aligned bool arrays")
    annotated = valid & active
    localization_required = annotated & gt_visible
    if not np.isfinite(gt[localization_required]).all():
        raise ValueError("Visible annotated GT coordinates must be finite")
    finite = np.isfinite(prediction).all(2)
    localization = localization_required & finite
    errors = np.linalg.norm(prediction[localization]-gt[localization], axis=1)
    tp = int((annotated & gt_visible & predicted_visible).sum())
    fp = int((annotated & ~gt_visible & predicted_visible).sum())
    fn = int((annotated & gt_visible & ~predicted_visible).sum())
    tn = int((annotated & ~gt_visible & ~predicted_visible).sum())
    return dict(gt_available=True, coordinate_units="processing pixels", annotated_point_frames=int(annotated.sum()),
                localization_required_count=int(localization_required.sum()), localization_evaluated_count=len(errors),
                nonfinite_visible_predictions=int((localization_required & ~finite).sum()),
                mean_endpoint_error=float(errors.mean()) if len(errors) else None,
                median_endpoint_error=float(np.median(errors)) if len(errors) else None,
                p90_endpoint_error=float(np.percentile(errors, 90)) if len(errors) else None,
                position_correctness={str(t):dict(correct=int((errors <= t).sum()),
                    denominator=int(localization_required.sum()),
                    rate=ratio(int((errors <= t).sum()), int(localization_required.sum()))) for t in thresholds},
                localization_coverage=ratio(len(errors), int(localization_required.sum())),
                localization_policy="all annotated GT-visible active point frames regardless of predicted visibility; nonfinite reported separately",
                visibility=dict(tp=tp, fp=fp, fn=fn, tn=tn, accuracy=ratio(tp+tn, tp+fp+fn+tn),
                                precision=ratio(tp, tp+fp), recall=ratio(tp, tp+fn), f1=ratio(2*tp, 2*tp+fp+fn)))


def evaluate_gt_file(data, manifest, path):
    with np.load(path, allow_pickle=False) as source:
        gt = {name: source[name] for name in source.files}
    if str(gt["video_sha256"].item()) != manifest["video_sha256"]:
        raise ValueError("GT belongs to a different source video")
    if str(gt["coordinate_space"].item()) != "processing" or not np.array_equal(gt["resolution"], manifest["video"]["processing_resolution"]):
        raise ValueError("GT coordinate space/resolution mismatch; convert explicitly first")
    gt_ids, gt_frames = gt["point_ids"].tolist(), gt["frame_indices"].tolist()
    if len(set(gt_ids)) != len(gt_ids) or len(set(gt_frames)) != len(gt_frames):
        raise ValueError("GT IDs must be unique")
    frame_map, point_map = {v:i for i,v in enumerate(gt_frames)}, {v:i for i,v in enumerate(gt_ids)}
    try:
        fi = np.array([frame_map[v] for v in data["frame_indices"].tolist()])
        pi = np.array([point_map[v] for v in data["point_ids"].tolist()])
    except KeyError as error:
        raise ValueError("GT missing frame/point; mark unannotated entries with valid=False") from error
    ix = np.ix_(fi, pi)
    active = np.arange(len(data["frame_indices"]))[:, None] >= data["queries"][:, 0][None]
    return evaluate_gt(data["tracks"], data["visibility"], gt["tracks"][ix], gt["visibility"][ix], gt["valid"][ix], active)
