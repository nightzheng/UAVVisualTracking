"""Read-only Stage A evidence collection; never runs model inference.

Run with -B. All generated files go to --output in the new repository.
The only executed legacy tests use arrays, mock predictors, and drawing in memory.
"""
import argparse
import contextlib
import hashlib
import importlib
import importlib.metadata
import io
import json
import subprocess
import sys
import unittest
from pathlib import Path

sys.dont_write_bytecode = True


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def snapshot(root):
    return {str(p.relative_to(root)).replace("\\", "/"): sha(p)
            for p in root.rglob("*") if p.is_file()
            and not any(part in {".git", "__pycache__"} for part in p.parts)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    root = args.legacy_root.resolve()
    output = args.output.resolve()
    if output == root or root in output.parents:
        raise ValueError("Audit output must be outside the legacy directory")
    if output.exists():
        raise FileExistsError("Use a fresh output path; existing evidence is protected")
    before = snapshot(root)
    original = root / "co-tracker"
    packages = [p / "code" for p in root.iterdir() if p.is_dir() and (p / "code/file_manifest.json").is_file()]
    report = {"scope": "Stage A only; no model inference, migration, or dataset redistribution",
              "python": sys.version, "packages": {}, "legacy_file_count": len(before),
              "source_snapshot": before, "package_checks": []}
    for name in ["torch", "torchvision", "numpy", "opencv-python", "matplotlib", "imageio", "imageio-ffmpeg", "tqdm"]:
        report["packages"][name] = importlib.metadata.version(name)
    import torch
    import cv2
    import numpy as np
    report["cuda"] = {"torch_cuda": torch.version.cuda, "available": torch.cuda.is_available(),
                      "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    status = subprocess.run(["git", "--no-optional-locks", "-C", str(original), "status", "--short"], capture_output=True, text=True)
    report["legacy_git_status"] = {"exit_code": status.returncode, "stdout": status.stdout, "stderr": status.stderr}
    diff = subprocess.run(["git", "--no-optional-locks", "-C", str(original), "diff", "--name-only", "HEAD"], capture_output=True, text=True)
    report["tracked_differences"] = {"exit_code": diff.returncode, "stdout": diff.stdout, "stderr": diff.stderr}
    report["commit"] = (original / ".git/refs/heads/main").read_text().strip()
    for code in packages:
        manifest = json.loads((code / "file_manifest.json").read_text(encoding="utf-8"))
        failures = []
        for item in manifest["files"]:
            path = code.parent / item["relative_path"]
            if not path.is_file():
                failures.append({"file": item["relative_path"], "reason": "missing"})
            elif path.stat().st_size != item["bytes"] or sha(path) != item["sha256"]:
                failures.append({"file": item["relative_path"], "reason": "hash or size mismatch"})
        common, different, missing = [], [], []
        for p in code.rglob("*.py"):
            relative = p.relative_to(code)
            other = original / relative
            if not other.is_file():
                continue
            (common if sha(other) == sha(p) else different).append(relative.as_posix())
        for p in original.rglob("*.py"):
            relative = p.relative_to(original)
            if relative.parts[0] in {"cotracker", "experiments"} and not (code / relative).is_file():
                missing.append(relative.as_posix())
        report["package_checks"].append({"directory": code.parent.name, "manifest_count": len(manifest["files"]),
                                          "failures": failures, "identical_python": common,
                                          "different_python": different, "original_only_python": missing})
    if args.checkpoint:
        report["checkpoint"] = {"file": args.checkpoint.name, "bytes": args.checkpoint.stat().st_size,
                                "sha256": sha(args.checkpoint), "authenticity": "not verified against official hash"}
    report["videos"] = []
    for path in root.rglob("*.mp4"):
        rel = path.relative_to(root)
        if "data" not in rel.parts:
            continue
        cap = cv2.VideoCapture(str(path))
        try:
            fps, count = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            checks = []
            for index in sorted({0, count // 2, max(0, count - 1)}):
                seek = cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, frame = cap.read()
                checks.append({"index": index, "seek_ok": bool(seek), "decoded": bool(ok),
                               "pts_ms_reported": cap.get(cv2.CAP_PROP_POS_MSEC),
                               "shape": list(frame.shape) if ok else None})
            report["videos"].append({"file": rel.as_posix(), "fps": fps, "frame_count": count,
                                      "width": int(cap.get(3)), "height": int(cap.get(4)),
                                      "sampled_decode": checks, "sha256": before[rel.as_posix()],
                                      "rights_status": "source and redistribution permission undocumented"})
        finally:
            cap.release()
    report["cached_tracks"] = []
    for path in root.rglob("tracks.npz"):
        with np.load(path, allow_pickle=False) as data:
            entry = {"file": path.relative_to(root).as_posix(),
                     "arrays": {name: {"shape": list(data[name].shape), "dtype": str(data[name].dtype)}
                                for name in data.files}, "new_inference": False}
            for name in ["pred_tracks", "target_tracks", "background_tracks"]:
                if name in data:
                    entry[name + "_finite_ratio"] = float(np.isfinite(data[name]).mean())
            for name in ["pred_visibility", "target_visibility"]:
                if name in data:
                    entry[name + "_predicted_visible_ratio"] = float(data[name].mean())
            report["cached_tracks"].append(entry)
    sys.path[:0] = [str(original), str(original / "experiments/camera_motion_compensation")]
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    for name in ["test_motion_compensation", "test_affine_compensation"]:
        suite.addTests(loader.loadTestsFromModule(importlib.import_module(name)))
    for name in ["test_metrics_include_invisible_predictions_and_exclude_prequery",
                 "test_online_schedule_short_exact_and_partial_tail",
                 "test_hollow_point_and_faint_invisible_history"]:
        suite.addTests(loader.loadTestsFromName("test_target_tracking.TargetTrackingTests." + name))
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    report["tests"] = {"count": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
                       "passed": result.wasSuccessful(), "log": stream.getvalue(),
                       "meaning": "Legacy arithmetic, mocked scheduling and in-memory drawing; not neural model reproduction"}
    after = snapshot(root)
    report["legacy_unchanged"] = before == after
    report["changed_files"] = [key for key in before.keys() | after.keys() if before.get(key) != after.get(key)]
    output.mkdir(parents=True)
    (output / "LEGACY_AUDIT_EVIDENCE.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"legacy_file_count": len(before), "package_checks": [
        {"directory": p["directory"], "manifest_count": p["manifest_count"], "failures": p["failures"],
         "identical_python": len(p["identical_python"]), "different_python": p["different_python"],
         "original_only_python": p["original_only_python"]} for p in report["package_checks"]],
        "videos": len(report["videos"]), "caches": len(report["cached_tracks"]),
        "tests": report["tests"], "legacy_unchanged": report["legacy_unchanged"],
        "checkpoint": report.get("checkpoint")}, ensure_ascii=False, indent=2))
    if not result.wasSuccessful() or not report["legacy_unchanged"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
