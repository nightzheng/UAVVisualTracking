"""Explicit coordinate bridges; upstream TAPIR raster scaling differs from OpenCV."""
import numpy as np
from ..data.video import resize_coordinates


def raster_scale(xy, source_wh, target_wh):
    source, target = np.asarray(source_wh), np.asarray(target_wh)
    if source.shape != (2,) or target.shape != (2,) or (source < 2).any() or (target < 2).any():
        raise ValueError("Resolutions must be [width,height] >=2")
    return np.asarray(xy) * (target / source)


def processing_to_original(xy, video):
    if 'crop_roi_xyxy' in video:
        xy = np.asarray(xy) + np.asarray(video['crop_roi_xyxy'][:2])
        size = video['source_processing_resolution']
    else:
        size = video['processing_resolution']
    return resize_coordinates(xy, size, video['original_resolution'])


def validate_rgb_queries(clip, queries):
    frames, q = clip.frames, queries.array
    if frames.ndim != 4 or frames.shape[-1] != 3 or frames.dtype != np.uint8 or len(frames) < 2:
        raise ValueError("Require T,H,W,3 uint8 RGB frames, T>=2; BGR must be converted by caller")
    t, h, w, _ = frames.shape
    if clip.metadata.get('color_space', 'RGB') != 'RGB':
        raise ValueError("BGR is not a supported input color space")
    if clip.metadata['processing_resolution'] != [w, h]:
        raise ValueError("Processing resolution differs from frames")
    if q.ndim != 2 or q.shape[1] != 3 or not len(q) or not np.isfinite(q).all():
        raise ValueError("Require finite N,3 txy queries")
    if ((q[:, 0] != np.floor(q[:, 0])) | (q[:, 0] < 0) | (q[:, 0] >= t)).any():
        raise ValueError("Query time must be a clip-local integer in [0,T)")
    if (q[:, 1:] < 0).any() or (q[:, 1:] > [w-1, h-1]).any():
        raise ValueError("Query xy outside processing pixel centers")
    if len(queries.point_ids) != len(q) or len(set(queries.point_ids)) != len(q):
        raise ValueError("Point IDs must be unique and align with queries")
    if clip.frame_indices.shape != (t,) or clip.timestamps.shape != (t,) or not np.isfinite(clip.timestamps).all():
        raise ValueError("Source frames/PTS must align with frames")
    if (np.diff(clip.frame_indices) <= 0).any() or (np.diff(clip.timestamps) <= 0).any():
        raise ValueError("Source frame indices and timestamps must increase strictly")


def prepare_tapir_input(clip, queries, model_hw):
    """Return float32 RGB [-1,1] 1,T,H,W,3 and raster-scaled 1,N,3 tyx."""
    import cv2
    validate_rgb_queries(clip, queries)
    mh, mw = model_hw
    if any(type(v) is not int or v < 256 or v % 8 for v in (mh, mw)):
        raise ValueError("TAPIR inference dimensions must be integer multiples of 8 and >=256")
    frames = np.stack([cv2.resize(f, (mw, mh), interpolation=cv2.INTER_LINEAR) for f in clip.frames])
    frames = frames.astype(np.float32) / 255 * 2 - 1
    q = queries.array.astype(np.float32).copy()
    q[:, 1:] = raster_scale(q[:, 1:], clip.metadata['processing_resolution'], (mw, mh))
    return frames[None], q[:, [0, 2, 1]][None]


def decode_tapir_output(raw, processing_wh, model_hw, t, n):
    """No GT, query overwrite or clipping; preserve raw logits and continuous rule."""
    tracks, occ, dist = (np.asarray(raw[k]) for k in ('tracks', 'occlusion', 'expected_dist'))
    if tracks.shape != (1, n, t, 2) or occ.shape != (1, n, t) or dist.shape != occ.shape:
        raise ValueError("Official TAPIR must return B,N,T,2 and B,N,T with B=1")
    # sigmoid(-logit) equals 1-sigmoid(logit), without overflow at extreme logits.
    score = np.exp(-np.logaddexp(0, occ)) * np.exp(-np.logaddexp(0, dist))
    fields = dict(occlusion_logit=occ[0].T.copy(), expected_dist_logit=dist[0].T.copy(),
                  visibility_score=score[0].T.copy())
    mapped = raster_scale(tracks[0].transpose(1, 0, 2), model_hw[::-1], processing_wh)
    return mapped, (score[0].T > .5), fields
