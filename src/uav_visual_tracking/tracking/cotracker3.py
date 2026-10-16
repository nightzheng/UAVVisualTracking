"""Adapted legacy owner inference flow; official model stays an external dependency.

Offline public forward adds support points. Disabled support uses the pinned
private sparse API, matching legacy target_tracking.py; no model modifications.
"""
from datetime import datetime, timezone
from pathlib import Path
import importlib
import importlib.metadata
import platform
import subprocess
import sys
import time

from .base import Prediction, Tracker
from ..common.provenance import sha256

UPSTREAM_COMMIT = "82e02e8029753ad4ef13cf06be7f4fc5facdda4d"
OFFLINE_SHA256 = "2670d4562ed69326dda775a26e54883925cd11b6fc9b24cb7aa9f8078bce7834"


def verify_upstream(source):
    source = Path(source).resolve()
    def git(*args):
        return subprocess.run(["git", "--no-optional-locks", "-C", str(source), *args],
                              capture_output=True, text=True, check=True).stdout.strip()
    if git("rev-parse", "HEAD") != UPSTREAM_COMMIT:
        raise ValueError("CoTracker source must match pinned commit " + UPSTREAM_COMMIT)
    if git("diff", "HEAD", "--name-only", "--", "cotracker", "hubconf.py", "setup.py"):
        raise ValueError("Upstream tracked model files differ from pinned commit")
    if not (source / "LICENSE.md").is_file():
        raise ValueError("Missing upstream license")
    return source


class CoTracker3(Tracker):
    def __init__(self, config):
        if config.get("mode", "offline") != "offline":
            raise NotImplementedError("Window-online is documented but not implemented or relabeled as offline")
        self.config = dict(config)
        self.device = config.get("device", "cuda")
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("device must be cpu or cuda")
        self.support_grid = config.get("support_grid", True)
        if not isinstance(self.support_grid, bool):
            raise ValueError("support_grid must be a boolean")
        checkpoint = Path(config["checkpoint"]).resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError("Provide an existing offline checkpoint; no automatic download")
        self.weight_hash = sha256(checkpoint)
        if self.weight_hash != OFFLINE_SHA256:
            raise ValueError("Checkpoint differs from verified official scaled_offline.pth")
        source = verify_upstream(config["source"])
        # Local pinned source avoids mutating the old installation or fetching latest code.
        sys.path.insert(0, str(source))
        module = importlib.import_module("cotracker.predictor")
        if not Path(module.__file__).resolve().is_relative_to(source):
            raise ValueError("Imported CoTracker does not belong to the verified source")
        import torch
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; explicitly select cpu if desired")
        torch.manual_seed(config.get("seed", 0))
        if self.device == "cuda":
            torch.cuda.manual_seed_all(config.get("seed", 0))
        torch.backends.cudnn.benchmark = False
        start = time.perf_counter()
        self.model = module.CoTrackerPredictor(checkpoint=str(checkpoint), offline=True, v2=False,
                                               window_len=60).to(self.device).eval()
        self._sync()
        self.load_seconds = time.perf_counter() - start
        self.identity = dict(backend="cotracker3", mode="offline_full_clip", commit=UPSTREAM_COMMIT,
                             checkpoint_name=checkpoint.name, checkpoint_sha256=self.weight_hash,
                             support_grid=self.support_grid, model_resolution_hw=list(self.model.interp_shape),
                             window_len=60, backward_tracking=False, iterations=6,
                             model_license="CC-BY-NC-4.0", seed=config.get("seed", 0))

    def _sync(self):
        if self.device == "cuda":
            import torch
            torch.cuda.synchronize()

    def predict(self, clip, queries):
        import torch
        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        self._sync()
        start = time.perf_counter()
        video = torch.from_numpy(clip.frames).permute(0, 3, 1, 2)[None].float().to(self.device)
        prompt = torch.from_numpy(queries.array.copy())[None].to(self.device)
        with torch.inference_mode():
            if self.support_grid:
                tracks, visibility = self.model(video, queries=prompt, backward_tracking=False)
            else:
                tracks, visibility = self.model._compute_sparse_tracks(
                    video, queries=prompt, add_support_grid=False, backward_tracking=False)
        self._sync()
        elapsed = time.perf_counter() - start
        output = Prediction(tracks[0].detach().cpu().numpy(), visibility[0].detach().cpu().numpy(),
                            dict(model=self.identity, device=self.device, model_load_seconds=self.load_seconds,
                                 inference_seconds=elapsed, inference_fps=len(clip.frames)/elapsed,
                                 inference_timing="input tensor/upload + model, excluding loading/decode/export/download",
                                 generated_at_utc=datetime.now(timezone.utc).isoformat(),
                                 result_available_after_source_frame=int(clip.frame_indices[-1]),
                                 future_frame_access=True, causal=False, output_policy="one final full-clip prediction",
                                 gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated() if self.device == "cuda" else None,
                                 gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved() if self.device == "cuda" else None,
                                 environment={name: importlib.metadata.version(name) for name in
                                              ["torch", "torchvision", "numpy", "opencv-python"]},
                                 python=platform.python_version(), platform=platform.platform(), torch_cuda=torch.version.cuda,
                                 gpu=torch.cuda.get_device_name() if self.device == "cuda" else None))
        output.validate(len(clip.frames), len(queries.array))
        return output
