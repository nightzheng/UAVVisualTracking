"""NPZ arrays + JSON manifest + inspectable per-frame CSV. No pickle needed."""
import csv
import json
from pathlib import Path
import numpy as np

from ..tracking.base import Prediction
from ..common.provenance import write_json
from .video import resize_coordinates


def save_result(directory, clip, queries, prediction, manifest):
    directory = Path(directory)
    prediction.validate(len(clip.frames), len(queries.array))
    np.savez_compressed(directory / "tracks.npz", schema_version=np.array(1),
                        tracks=prediction.tracks, visibility=prediction.visibility,
                        frame_indices=clip.frame_indices, timestamps=clip.timestamps,
                        point_ids=np.asarray(queries.point_ids, dtype=str), queries=queries.array,
                        initialization=np.asarray(queries.methods, dtype=str))
    write_json(directory / "manifest.json", manifest)
    original = resize_coordinates(prediction.tracks, clip.metadata["processing_resolution"],
                                   clip.metadata["original_resolution"])
    columns = ["video_id", "frame_index", "clip_frame_index", "timestamp_seconds", "point_id",
               "initialization", "active", "x_processing", "y_processing", "x_original", "y_original",
               "predicted_visibility", "generated_at_utc", "available_after_frame_index",
               "video_context_gap_seconds", "mode", "config_sha256", "model_commit", "checkpoint_sha256"]
    with (directory / "tracks.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for t, index in enumerate(clip.frame_indices):
            for p, pid in enumerate(queries.point_ids):
                active = t >= queries.array[p, 0]
                xy, raw = prediction.tracks[t, p], original[t, p]
                usable = active and np.isfinite(xy).all()
                writer.writerow(dict(video_id=manifest["video_id"], frame_index=int(index), clip_frame_index=t,
                                     timestamp_seconds=float(clip.timestamps[t]), point_id=pid, initialization=queries.methods[p],
                                     active=int(active), x_processing=float(xy[0]) if usable else "",
                                     y_processing=float(xy[1]) if usable else "", x_original=float(raw[0]) if usable else "",
                                     y_original=float(raw[1]) if usable else "", predicted_visibility=int(prediction.visibility[t, p]) if active else "",
                                     generated_at_utc=prediction.metadata["generated_at_utc"],
                                     available_after_frame_index=prediction.metadata["result_available_after_source_frame"],
                                     video_context_gap_seconds=float(clip.timestamps[-1]-clip.timestamps[t]),
                                     mode=prediction.metadata["model"]["mode"], config_sha256=manifest["config_sha256"],
                                     model_commit=prediction.metadata["model"]["commit"],
                                     checkpoint_sha256=prediction.metadata["model"]["checkpoint_sha256"]))


def load_result(directory):
    directory = Path(directory)
    with np.load(directory / "tracks.npz", allow_pickle=False) as stored:
        data = {name: stored[name] for name in stored.files}
    if int(data["schema_version"]) != 1:
        raise ValueError("Unsupported result schema")
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    prediction = Prediction(data["tracks"], data["visibility"], manifest["prediction"])
    prediction.validate(len(data["frame_indices"]), len(data["point_ids"]))
    if data["timestamps"].shape != data["frame_indices"].shape or data["queries"].shape != (len(data["point_ids"]), 3):
        raise ValueError("Invalid result frame or query arrays")
    if not np.isfinite(data["timestamps"]).all() or (np.diff(data["timestamps"]) <= 0).any():
        raise ValueError("Result timestamps must be strictly increasing")
    if len(set(data["point_ids"].tolist())) != len(data["point_ids"]) or len(set(data["frame_indices"].tolist())) != len(data["frame_indices"]):
        raise ValueError("Duplicate IDs in result")
    return data, manifest
