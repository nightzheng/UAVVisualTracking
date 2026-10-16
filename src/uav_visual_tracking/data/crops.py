"""Fixed views of one source sequence. No new material labels or GT are created."""
import numpy as np
from .video import VideoClip


def validate_roi(roi, size):
    if len(roi)!=4 or any(type(v) is not int for v in roi):
        raise ValueError('ROI must be four integer pixel bounds [x0,y0,x1,y1)')
    x0,y0,x1,y1=roi;w,h=size
    if not (0<=x0<x1<=w and 0<=y0<y1<=h and x1-x0>=2 and y1-y0>=2):
        raise ValueError('ROI outside source processing image or too small')
    return np.array(roi,dtype=int)


def source_to_crop(points, roi):
    return np.asarray(points,dtype=float)-np.asarray(roi[:2],dtype=float)


def crop_to_source(points, roi):
    return np.asarray(points,dtype=float)+np.asarray(roi[:2],dtype=float)


def crop_clip(clip, roi):
    x0,y0,x1,y1=validate_roi(roi,clip.metadata['processing_resolution'])
    metadata=dict(clip.metadata)
    metadata.update(processing_resolution=[int(x1-x0),int(y1-y0)],
        source_processing_resolution=clip.metadata['processing_resolution'],
        crop_roi_xyxy=list(map(int,roi)),crop_bounds='left/top inclusive; right/bottom exclusive; no post-crop resize',
        crop_to_source_processing_matrix=[[1.,0.,float(x0)],[0.,1.,float(y0)],[0.,0.,1.]],
        original_coordinate_mapping='crop translation -> source processing -> source original pixel-center resize',
        independent_sequence=False,derived_crop_proxy=True)
    return VideoClip(np.ascontiguousarray(clip.frames[:,y0:y1,x0:x1]),clip.frame_indices.copy(),clip.timestamps.copy(),metadata)
