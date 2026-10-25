"""Adapted from repository owner's legacy run.py::read_clip (2026-10-13).

Reuse: OpenCV clip selection, width resize, RGB frames and metadata.
Changes: retain decoded PTS, explicit source frame indices and time provenance.
Attribution and source hashes: THIRD_PARTY_NOTICES.md and SOURCE_MANIFEST.json.
"""
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class VideoClip:
    frames: np.ndarray
    frame_indices: np.ndarray
    timestamps: np.ndarray
    metadata: dict


def resize_coordinates(points, source_size, destination_size):
    """OpenCV pixel-center resize; sizes are [width,height], points are x,y."""
    source = np.asarray(source_size, dtype=float)
    destination = np.asarray(destination_size, dtype=float)
    if source.shape != (2,) or destination.shape != (2,) or (source < 2).any() or (destination < 2).any():
        raise ValueError("Image sizes must be [width,height], each >=2")
    return (np.asarray(points, dtype=float) + .5) * (destination / source) - .5


def read_clip(path, start_sec=0., duration=8., max_frames=120, max_width=640):
    if not np.isfinite([start_sec, duration, max_width, max_frames]).all() or start_sec < 0 or duration <= 0 or max_frames < 2 or max_width < 2:
        raise ValueError("Invalid clip limits")
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot read video: {path.name}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        width, height = int(cap.get(3)), int(cap.get(4))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not np.isfinite(fps) or fps <= 0 or min(width, height) < 2:
            raise ValueError("Invalid video FPS or size")
        first = int(round(start_sec * fps))
        if first >= total:
            raise ValueError("start_sec is beyond end of video")
        count = min(total-first, int(max_frames), max(1, int(np.ceil(duration * fps))))
        if first and not cap.set(cv2.CAP_PROP_POS_FRAMES, first):
            raise ValueError("Decoder cannot seek to requested frame")
        scale = min(1., max_width / width)
        size = (max(2, int(round(width*scale))), max(2, int(round(height*scale))))
        frames, indices, pts = [], [], []
        for _ in range(count):
            ok, image = cap.read()
            if not ok:
                break
            index = int(round(cap.get(cv2.CAP_PROP_POS_FRAMES))) - 1
            if index != first + len(frames):
                raise ValueError("Decoder frame index differs from requested clip")
            indices.append(index)
            pts.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.)
            if size != (width, height):
                image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
            frames.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if len(frames) < 2:
            raise ValueError("At least two frames required")
        pts = np.asarray(pts, dtype=np.float64)
        valid_pts = np.isfinite(pts).all() and (np.diff(pts) > 0).all()
        time_basis = "opencv_decoded_pts_seconds" if valid_pts else "nominal_fps_assumption"
        timestamps = pts if valid_pts else np.asarray(indices) / fps
        cfr = valid_pts and bool(np.allclose(np.diff(pts), 1 / fps, rtol=1e-4, atol=1e-6))
        return VideoClip(np.stack(frames), np.asarray(indices, dtype=np.int64), timestamps,
                         dict(video_name=path.name, fps=fps, original_resolution=[width, height],
                              processing_resolution=list(size), frame_count=len(frames), start_frame=first,
                              time_basis=time_basis, cfr_observed=bool(cfr),
                              timestamp_meaning="video playback time; not sensor acquisition wall clock",
                              requested_start_sec=start_sec, actual_start_sec=float(timestamps[0]),
                              raw_decoder_pts_seconds=[float(t) if np.isfinite(t) else None for t in pts],
                              decode_truncated=len(frames) != count))
    finally:
        cap.release()
