from pathlib import Path
import cv2
import numpy as np
from .drawing import draw_history, draw_point

COLORS = [(40, 80, 255), (255, 180, 40), (60, 220, 80), (240, 70, 230), (30, 240, 240)]


def export_video(path, clip, queries, prediction):
    if clip.metadata["time_basis"] == "opencv_decoded_pts_seconds" and not clip.metadata["cfr_observed"]:
        raise ValueError("CFR video writer cannot preserve VFR timing; NPZ/CSV remain usable")
    height, width = clip.frames.shape[1:3]
    size = (width + width % 2, height + height % 2)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), clip.metadata["fps"], size)
    try:
        if not writer.isOpened():
            raise RuntimeError("MP4 writer unavailable")
        for t, frame in enumerate(clip.frames):
            image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            for p, point in enumerate(queries.point_ids):
                q = int(queries.array[p, 0])
                if t < q:
                    continue
                color = COLORS[p % len(COLORS)]
                draw_history(image, prediction.tracks[:, p:p+1], prediction.visibility[:, p:p+1], q, t, color)
                draw_point(image, prediction.tracks[t, p], prediction.visibility[t, p], color)
            cv2.putText(image, f"OFFLINE  frame {int(clip.frame_indices[t])}  t={clip.timestamps[t]:.3f}s", (8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1)
            cv2.putText(image, "Solid/hollow = predicted visibility (no GT)", (8, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 1)
            if t == 0:
                cv2.imwrite(str(Path(path).with_name("preview.png")), image)
            writer.write(cv2.copyMakeBorder(image, 0, size[1]-height, 0, size[0]-width, cv2.BORDER_CONSTANT))
    finally:
        writer.release()
    cap = cv2.VideoCapture(str(path))
    try:
        decoded = 0
        while cap.read()[0]:
            decoded += 1
        if decoded != len(clip.frames):
            raise RuntimeError(f"Export decoded {decoded} frames, expected {len(clip.frames)}")
        if abs(cap.get(cv2.CAP_PROP_FPS) - clip.metadata["fps"]) > .02:
            raise RuntimeError("Export FPS differs from source")
    finally:
        cap.release()
    return dict(decoded_frame_count=decoded, timing="CFR output at source nominal FPS", padded_resolution=list(size))
