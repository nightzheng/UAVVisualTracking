"""Repository owner original drawing code; MIT. Migrated 2026-10-13.
Source: legacy experiments/camera_motion_compensation/visualization.py.
See docs/migration/SOURCE_MAP.json for provenance.
"""
import cv2
import numpy as np

def draw_point(image, point, visible, color, radius=4):
    if np.isfinite(point).all():
        xy = tuple(np.rint(np.clip(point, -10000000, 10000000)).astype(int))
        cv2.circle(image, xy, radius, color, -1 if visible else 1, cv2.LINE_AA)

def draw_history(image, tracks, visibility, start, end, color, transform=None):
    for i in range(tracks.shape[1]):
        for t in range(max(start + 1, end - 90), end + 1):
            pair = tracks[t-1:t+1, i]
            if transform is not None:
                pair = transform(pair)
            if not np.isfinite(pair).all():
                continue
            xy = np.rint(np.clip(pair, -10000000, 10000000)).astype(int)
            if visibility[t - 1, i] and visibility[t, i]:
                cv2.line(image, tuple(xy[0]), tuple(xy[1]), color, 2)
            else:
                # Blend only the segment's image region, avoiding full-frame copies.
                ok, a, b = cv2.clipLine((0, 0, image.shape[1], image.shape[0]),
                                      tuple(xy[0]), tuple(xy[1]))
                if not ok:
                    continue
                x0, y0 = max(0, min(a[0], b[0])-2), max(0, min(a[1], b[1])-2)
                x1, y1 = min(image.shape[1], max(a[0], b[0])+3), min(image.shape[0], max(a[1], b[1])+3)
                region = image[y0:y1, x0:x1]
                layer = region.copy()
                cv2.line(layer, (a[0]-x0, a[1]-y0), (b[0]-x0, b[1]-y0), color, 2)
                region[:] = cv2.addWeighted(layer, .25, region, .75, 0)
