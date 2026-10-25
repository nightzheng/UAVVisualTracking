"""BootsTAPIR PyTorch offline adapter for the pinned unmodified official inference code."""
from datetime import datetime, timezone
import importlib
import importlib.metadata
from pathlib import Path
import platform
import sys
import time

import numpy as np
from .base import Prediction, Tracker
from .interface import prepare_tapir_input, decode_tapir_output
from ..common.provenance import sha256

UPSTREAM_COMMIT = "730cda1c730877cfedbe01bf87fb1cadb78a565d"
OFFLINE_SHA256 = "8493c7a69e02c85b9382fbb3c7b8b539b36bc08ede744b9e99feb739a0129f4b"
SOURCE_SHA256 = {
    "tapnet/torch/__init__.py": "f7c8d8c66e2f93ac350ce82c03f02152ffc82524cd120ebc944c931b6ea879e1",
    "tapnet/torch/nets.py": "4961a8e4882e8c9ac219c4714f6f2c811cc1ab665a439d4d2e79942002a9517c",
    "tapnet/torch/tapir_model.py": "87c05fb17d48d0f4353a00b67c9d40d3a264e13e43d88e3f7b5829119845e3ba",
    "tapnet/torch/utils.py": "a4267b3a6bd8ecf2dfe5effacab9f7a43664dc5d08966c98f6a0cc68f5ccedd3",
    "LICENSE": "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30",
}


def verify_source(source):
    source = Path(source).resolve()
    for name, expected in SOURCE_SHA256.items():
        if not (source / name).is_file() or sha256(source / name) != expected:
            raise ValueError("BootsTAPIR source differs from pinned official Git blobs: " + name)
    return source


class BootsTAPIR(Tracker):
    def __init__(self, config):
        if config.get('mode', 'offline') != 'offline':
            raise ValueError("Only official BootsTAPIR offline v2 is supported")
        self.config = dict(config)
        self.device = config.get('device', 'cuda')
        if self.device not in {'cuda', 'cpu'}:
            raise ValueError("Explicit cpu or cuda device required")
        self.model_hw = tuple(config.get('model_resolution_hw', [256, 256]))
        if len(self.model_hw) != 2 or any(type(v) is not int or v < 256 or v % 8 for v in self.model_hw):
            raise ValueError("Model resolution must be H,W multiples of 8 and >=256")
        self.query_chunk_size = config.get('query_chunk_size', 64)
        if type(self.query_chunk_size) is not int or self.query_chunk_size < 1:
            raise ValueError("Positive query_chunk_size required")
        checkpoint = Path(config['checkpoint']).resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError("Existing official checkpoint required; never auto-download")
        self.weight_hash = sha256(checkpoint)
        if self.weight_hash != OFFLINE_SHA256:
            raise ValueError("Not the audited official bootstapir_checkpoint_v2.pt")
        source = verify_source(config['source'])
        sys.path.insert(0, str(source))
        module = importlib.import_module('tapnet.torch.tapir_model')
        for name in ('tapnet.torch.tapir_model', 'tapnet.torch.nets', 'tapnet.torch.utils'):
            if not Path(sys.modules[name].__file__).resolve().is_relative_to(source):
                raise ValueError("Imported TAPIR module outside verified source: " + name)
        import torch
        if self.device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; no silent CPU fallback")
        torch.manual_seed(config.get('seed', 0))
        if self.device == 'cuda':
            torch.cuda.manual_seed_all(config.get('seed', 0))
        torch.backends.cudnn.benchmark = False
        start = time.perf_counter()
        # Official torch notebook defaults, including temperature=20 (training config differs).
        self.model = module.TAPIR(pyramid_level=1)
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        self.model.load_state_dict(state, strict=True)
        self.model = self.model.to(self.device).eval()
        self._sync()
        self.load_seconds = time.perf_counter() - start
        self.identity = dict(backend='bootstapir', version='offline_v2_pytorch', mode='offline_full_clip',
                             commit=UPSTREAM_COMMIT, checkpoint_name=checkpoint.name,
                             checkpoint_sha256=self.weight_hash, model_license='Apache-2.0',
                             model_resolution_hw=list(self.model_hw), initial_resolution_hw=[256,256],
                             query_chunk_size=self.query_chunk_size, pyramid_level=1, num_pips_iter=4,
                             extra_convs=True, use_casual_conv=False, softmax_temperature=20.0,
                             feature_extractor_chunk_size=10, refinement_resolutions='official_default',
                             support_point_count=0, seed=config.get('seed',0), strict_load=True)
        self.last_raw = None

    def _sync(self):
        if self.device == 'cuda':
            import torch
            torch.cuda.synchronize()

    def predict(self, clip, queries):
        import torch
        if self.device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        self._sync()
        start = time.perf_counter()
        frames, q = prepare_tapir_input(clip, queries, self.model_hw)
        video = torch.from_numpy(frames).to(self.device)
        prompt = torch.from_numpy(q).to(self.device)
        with torch.inference_mode():
            raw = self.model(video, prompt, is_training=False, query_chunk_size=self.query_chunk_size)
        self._sync()
        elapsed = time.perf_counter() - start
        # Preserve every final and intermediate output returned by the official model.
        self.last_raw = {k: ([x.detach().cpu().numpy() for x in v] if isinstance(v,list)
                             else v.detach().cpu().numpy()) for k,v in raw.items()}
        tracks, visibility, scores = decode_tapir_output(
            self.last_raw, clip.metadata['processing_resolution'], self.model_hw,
            len(clip.frames), len(queries.array))
        metadata = dict(model=self.identity, device=self.device, model_load_seconds=self.load_seconds,
                        inference_seconds=elapsed, inference_fps=len(clip.frames)/elapsed,
                        inference_timing='resize + input normalization/upload + new GPU/CPU forward; excludes decode/load/export',
                        generated_at_utc=datetime.now(timezone.utc).isoformat(),
                        result_available_after_source_frame=int(clip.frame_indices[-1]),
                        future_frame_access=True, causal=False, output_policy='one final full-clip prediction',
                        input_color_space='RGB', input_dtype='float32', input_range=[-1,1],
                        input_layout='B,T,H,W,C', query_layout='B,N,(t,y,x)',
                        output_layout='T,N,(x,y) in processing pixels',
                        coordinate_mapping=dict(processing_to_model='xy * [model_W/W,model_H/H]; official raster',
                                                model_to_processing='inverse raster scale',
                                                processing_to_original='crop translation then project pixel-center resize',
                                                crop_roi_xyxy=clip.metadata.get('crop_roi_xyxy')),
                        visibility_rule='(1-sigmoid(occlusion_logit))*(1-sigmoid(expected_dist_logit)) > 0.5',
                        score_semantics='occlusion and localization-error logits; no calibrated material identity probability',
                        query_frame_override=False, prequery_predictions_preserved=True,
                        numerical_valid=bool(np.isfinite(tracks).all() and all(np.isfinite(v).all() for v in scores.values())),
                        failure_reason=None if np.isfinite(tracks).all() and all(np.isfinite(v).all() for v in scores.values())
                        else 'nonfinite_official_output',
                        gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated() if self.device=='cuda' else None,
                        gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved() if self.device=='cuda' else None,
                        environment={x:importlib.metadata.version(x) for x in ['torch','numpy','opencv-python','dm-tree','einshape']},
                        python=platform.python_version(), platform=platform.platform(), torch_cuda=torch.version.cuda,
                        gpu=torch.cuda.get_device_name() if self.device=='cuda' else None)
        prediction = Prediction(tracks, visibility, metadata, scores)
        prediction.validate(len(clip.frames),len(queries.array))
        return prediction
