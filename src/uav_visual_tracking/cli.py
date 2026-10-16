import argparse
import json
from pathlib import Path
import time

from .common.provenance import config_hash, sha256, write_json
from .data.video import read_clip
from .data.queries import make_queries
from .data.results import save_result, load_result
from .evaluation.metrics import diagnostics, evaluate_gt_file
from .tracking.cotracker3 import CoTracker3
from .visualization.video import export_video


def run(config):
    output = Path(config["output"]).resolve()
    if output.exists():
        raise FileExistsError("Use a new output directory; saved experiments cannot be overwritten")
    if not Path(config["video"]).is_file():
        raise FileNotFoundError(config["video"])
    output.mkdir(parents=True)
    start = time.perf_counter()
    try:
        clip = read_clip(config["video"], **config.get("clip", {}))
        queries = make_queries(config["queries"], clip)
        decode_seconds = time.perf_counter()-start
        video_hash = sha256(config["video"])
        tracker = CoTracker3(config["model"])
        prediction = tracker.predict(clip, queries)
        semantic_config = dict(clip=config.get("clip", {}), queries=config["queries"], model=tracker.identity,
                               video_sha256=video_hash, device=tracker.device)
        manifest = dict(schema_version=1, video_id=f"sha256:{video_hash}", video_sha256=video_hash,
                        config=semantic_config, config_sha256=config_hash(semantic_config), video=clip.metadata,
                        prediction=prediction.metadata, decode_seconds=decode_seconds,
                        queries=queries.array.tolist(), point_ids=queries.point_ids, initialization=queries.methods)
        save_result(output, clip, queries, prediction, manifest)
        metrics = diagnostics(prediction.tracks, prediction.visibility, queries.array[:, 0], clip.metadata["processing_resolution"])
        if config.get("gt"):
            data, saved_manifest = load_result(output)
            metrics["ground_truth"] = evaluate_gt_file(data, saved_manifest, config["gt"])
            metrics["gt_available"] = True
        write_json(output / "metrics.json", metrics)
        video_info = export_video(output / "tracked.mp4", clip, queries, prediction)
        manifest["visualization"] = video_info
        manifest["end_to_end_seconds"] = time.perf_counter() - start
        manifest["end_to_end_fps"] = len(clip.frames) / manifest["end_to_end_seconds"]
        manifest["end_to_end_timing"] = "decode, input hash, source validation, model load, inference, CSV/NPZ/JSON, video export + full decode verification; excludes final manifest hash"
        manifest["artifacts_sha256"] = {p.name:sha256(p) for p in output.iterdir() if p.name != "manifest.json" and p.is_file()}
        manifest["status"] = "success"
        write_json(output / "manifest.json", manifest)
        return manifest
    except Exception as error:
        write_json(output / "error.json", {"status": "failed", "error_type": type(error).__name__, "message": str(error),
                                          "no_success_claim": True})
        raise


def main():
    parser = argparse.ArgumentParser(description="V0 CoTracker3 baseline: explicit queries, offline inference, traceable output")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--video")
    parser.add_argument("--checkpoint")
    parser.add_argument("--cotracker-source")
    parser.add_argument("--output")
    parser.add_argument("--device", choices=["cuda", "cpu"])
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    for name in ["video", "output"]:
        if getattr(args, name):
            config[name] = getattr(args, name)
    for argument, key in [("checkpoint", "checkpoint"), ("cotracker_source", "source"), ("device", "device")]:
        if getattr(args, argument):
            config["model"][key] = getattr(args, argument)
    manifest = run(config)
    print(json.dumps({"status": manifest["status"], "frames": manifest["video"]["frame_count"],
                      "points": len(manifest["point_ids"]), "inference_seconds": manifest["prediction"]["inference_seconds"],
                      "inference_fps": manifest["prediction"]["inference_fps"],
                      "gpu_peak_allocated_bytes": manifest["prediction"]["gpu_peak_allocated_bytes"]}, indent=2))


if __name__ == "__main__":
    main()
