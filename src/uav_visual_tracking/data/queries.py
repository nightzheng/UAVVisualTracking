"""Adapted validation and grid initialization from owner's legacy point selector.

Manual locations never imply semantic target localization.
"""
from dataclasses import dataclass
import numpy as np
from .video import resize_coordinates


@dataclass
class Queries:
    array: np.ndarray  # N,3: clip frame index,x,y in processing pixels
    point_ids: list[str]
    methods: list[str]


def make_queries(config, clip):
    frame_count, height, width = clip.frames.shape[:3]
    strategy = config.get("strategy", "manual")
    if strategy not in {"manual", "grid"}:
        raise ValueError("strategy must be manual or grid")
    q = config.get("frame", 0)
    if not isinstance(q, int) or not 0 <= q < frame_count:
        raise ValueError("Query frame must be an integer inside the clip")
    if strategy == "grid":
        side = config.get("grid_size", 5)
        if not isinstance(side, int) or not 2 <= side <= 30:
            raise ValueError("grid_size must be 2..30")
        # Endpoint-interior uniform grid; explicitly differs from upstream grid helper.
        # Passed as explicit queries so support-grid policy remains independently controlled.
        x, y = np.meshgrid(np.linspace(.05*(width-1), .95*(width-1), side),
                           np.linspace(.05*(height-1), .95*(height-1), side))
        xy = np.column_stack([x.ravel(), y.ravel()])
        methods = ["automatic_uniform_grid_no_semantics"] * len(xy)
    else:
        xy = np.asarray(config.get("points", []), dtype=float)
        if xy.ndim != 2 or xy.shape[1] != 2 or not len(xy):
            raise ValueError("Manual points must be nonempty [x,y] pairs")
        space = config.get("coordinate_space", "processing")
        if space == "original":
            xy = resize_coordinates(xy, clip.metadata["original_resolution"], [width, height])
        elif space != "processing":
            raise ValueError("coordinate_space must be original or processing")
        methods = ["manual"] * len(xy)
    if not np.isfinite(xy).all() or (xy < 0).any() or (xy > [width-1, height-1]).any():
        raise ValueError("Queries must be finite and within pixel-center image bounds")
    if len(np.unique(xy, axis=0)) != len(xy):
        raise ValueError("Duplicate query points are not allowed")
    ids = config.get("point_ids", [f"p{i:03d}" for i in range(len(xy))])
    if len(ids) != len(xy) or len(set(ids)) != len(ids) or not all(isinstance(i, str) and i for i in ids):
        raise ValueError("Point IDs must be unique nonempty strings matching queries")
    return Queries(np.column_stack([np.full(len(xy), q), xy]).astype(np.float32), ids, methods)
