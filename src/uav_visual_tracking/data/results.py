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
    if not prediction.raw_scores:
        raise ValueError('Public writer requires schema2 continuous raw scores')
    if any((directory/name).exists() for name in ('tracks.npz','tracks.csv','manifest.json')):
        raise FileExistsError('Result artifacts already exist')
    extra = {}
    if prediction.raw_scores:
        from ..tracking.interface import processing_to_original
        extra = {"raw_score__" + k: v for k, v in prediction.raw_scores.items()}
        extra['tracks_original'] = processing_to_original(prediction.tracks, clip.metadata)
        extra['numerical_valid'] = np.isfinite(prediction.tracks).all(axis=-1)
        # Keep manifest and NPZ version consistent, including opt-in use from the old CLI.
        manifest['schema_version'] = 2
        manifest['raw_score_fields'] = sorted(prediction.raw_scores)
    np.savez_compressed(directory / "tracks.npz", schema_version=np.array(2 if extra else 1),
                        tracks=prediction.tracks, visibility=prediction.visibility,
                        frame_indices=clip.frame_indices, timestamps=clip.timestamps,
                        point_ids=np.asarray(queries.point_ids, dtype=str), queries=queries.array,
                        initialization=np.asarray(queries.methods, dtype=str), **extra)
    write_json(directory / "manifest.json", manifest)
    original = resize_coordinates(prediction.tracks, clip.metadata["processing_resolution"],
                                   clip.metadata["original_resolution"])
    if extra:
        original = extra['tracks_original']
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
    if int(data["schema_version"]) not in {1, 2}:
        raise ValueError("Unsupported result schema")
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get('schema_version')!=int(data['schema_version']):
        raise ValueError('Manifest/NPZ schema mismatch')
    raw_scores = {k.removeprefix('raw_score__'):v for k,v in data.items() if k.startswith('raw_score__')}
    prediction = Prediction(data["tracks"], data["visibility"], manifest["prediction"], raw_scores)
    prediction.validate(len(data["frame_indices"]), len(data["point_ids"]))
    if int(data['schema_version']) == 2:
        from ..tracking.interface import processing_to_original
        expected = processing_to_original(data['tracks'], manifest['video'])
        if not raw_scores or data['tracks_original'].shape != data['tracks'].shape or not np.allclose(
                data['tracks_original'], expected, equal_nan=True):
            raise ValueError('Invalid schema-2 original coordinate mapping')
        if not np.array_equal(data['numerical_valid'], np.isfinite(data['tracks']).all(axis=-1)):
            raise ValueError('Invalid numerical validity mask')
        if sorted(raw_scores)!=manifest.get('raw_score_fields'):
            raise ValueError('Manifest/NPZ raw score fields differ')
    if data["timestamps"].shape != data["frame_indices"].shape or data["queries"].shape != (len(data["point_ids"]), 3):
        raise ValueError("Invalid result frame or query arrays")
    if not np.isfinite(data["timestamps"]).all() or (np.diff(data["timestamps"]) <= 0).any():
        raise ValueError("Result timestamps must be strictly increasing")
    if len(data['frame_indices'])<2 or data['frame_indices'].dtype.kind not in 'iu' or (np.diff(data['frame_indices'])<=0).any():
        raise ValueError('Require at least two strictly increasing integer source frames')
    q=data['queries']
    if not np.isfinite(q).all() or (q[:,0]!=np.floor(q[:,0])).any() or (q[:,0]<0).any() or (q[:,0]>=len(data['frame_indices'])).any():
        raise ValueError('Invalid saved query time')
    if 'queries' in manifest and not np.array_equal(q,np.asarray(manifest['queries'],dtype=q.dtype)):
        raise ValueError('Manifest/NPZ query mismatch')
    if 'point_ids' in manifest and data['point_ids'].tolist()!=manifest['point_ids']:
        raise ValueError('Manifest/NPZ point IDs differ')
    if len(set(data["point_ids"].tolist())) != len(data["point_ids"]) or len(set(data["frame_indices"].tolist())) != len(data["frame_indices"]):
        raise ValueError("Duplicate IDs in result")
    return data, manifest
