# Third-party notices and asset provenance

Original authorized project contributions: Copyright (c) 2026 nightzheng, MIT as stated in [LICENSE](LICENSE). Video/query/inference utilities derive from the owner's authorized contributions. Adaptations and source hashes are recorded in [SOURCE_MANIFEST.json](SOURCE_MANIFEST.json). MIT does not relicense upstream models, dependencies or third-party material.

## CoTracker3 (external B0)

Meta Platforms, Inc. and affiliates; Karaev et al., *CoTracker3: Simpler and Better Point Tracking by Pseudo-Labelling Real Videos* (2024). [Official source](https://github.com/facebookresearch/co-tracker), pinned at `82e02e8029753ad4ef13cf06be7f4fc5facdda4d`, is subject to [CC BY-NC 4.0](https://github.com/facebookresearch/co-tracker/blob/82e02e8029753ad4ef13cf06be7f4fc5facdda4d/LICENSE.md). The [official model card](https://huggingface.co/facebook/cotracker3) also declares CC BY-NC 4.0. Noncommercial restrictions and attribution remain applicable; project MIT terms grant no commercial authorization for the combined system.

External weight: [scaled_offline.pth](https://huggingface.co/facebook/cotracker3/blob/main/scaled_offline.pth), SHA256 `2670d4562ed69326dda775a26e54883925cd11b6fc9b24cb7aa9f8078bce7834`. The historical download transaction was undocumented; local content identity was verified by hash. Upstream model source and weights are not redistributed here.

## BootsTAPIR and TAP-Vid

Copyright Google LLC / DeepMind contributors. Attribution: Doersch et al., *TAPIR: Tracking Any Point with per-frame Initialization and temporal Refinement* (2023), and *BootsTAP: Bootstrapped Training for Tracking-Any-Point* (2024). [Official source](https://github.com/google-deepmind/tapnet), pinned at `730cda1c730877cfedbe01bf87fb1cadb78a565d`. Its [LICENSE](https://github.com/google-deepmind/tapnet/blob/730cda1c730877cfedbe01bf87fb1cadb78a565d/LICENSE) and [README](https://github.com/google-deepmind/tapnet/blob/730cda1c730877cfedbe01bf87fb1cadb78a565d/README.md#license-and-disclaimer) cover the software and linked pretrained checkpoints under Apache-2.0.

The [official PyTorch notebook](https://github.com/google-deepmind/tapnet/blob/730cda1c730877cfedbe01bf87fb1cadb78a565d/colabs/torch_tapir_demo.ipynb) identifies [bootstapir_checkpoint_v2.pt](https://storage.googleapis.com/dm-tapnet/bootstap/bootstapir_checkpoint_v2.pt), SHA256 `8493c7a69e02c85b9382fbb3c7b8b539b36bc08ede744b9e99feb739a0129f4b`. Model source/weights remain external. Upstream papers/repository supply full author lists and citations.

This snapshot redistributes the unchanged [official evaluation source](https://github.com/google-deepmind/tapnet/blob/730cda1c730877cfedbe01bf87fb1cadb78a565d/tapnet/tapvid/evaluation_datasets.py) as `src/uav_visual_tracking/evaluation/_official/evaluation_datasets.py`, SHA256 `90cd01e53e23f6d489d3a6cd840cfd93fed4c1a6f4a0dfd373933cc164f0e570`. Its original `Copyright 2026 Google LLC` / Apache-2.0 header remains intact. Full license: `evaluation/_official/LICENSE`. Formulas are unmodified; coordinate validation and NA handling occur outside the official function. No TAP-Vid videos or annotations are bundled.

## Included visual materials

Research PNGs/MP4s in `docs/assets` derive exclusively from project-owned frozen procedural synthetic rasters and numeric outputs. Background texture is procedural; no photograph, external texture, person or real turbine footage is used. These authorized original derivatives are covered by project MIT terms to the extent of the owner's rights. Model-derived tracks retain attribution/applicable model conditions; no unrestricted commercial-use claim is made for combined derivatives.

Source hashes, frame/PTS ranges, display transformations and selected cases are in [MEDIA_MANIFEST](docs/assets/MEDIA_MANIFEST.json). Original arrays/GT, full predictions, renderer and weights are not redistributed as a dataset. No Pexels or other real source footage is included: rights records for local legacy material remain incomplete.

## Workbench assets and dependencies

The workbench screenshots show the application and, in the results view, a render of a simulated turbine. The blade geometry derives from the [IEA-3.4-130-RWT reference model](https://github.com/IEAWindSystems/IEA-3.4-130-RWT/tree/aacff396f46b35f7104d83be095f859684013405) under its upstream Apache-2.0 terms. Reference attribution is Bortolotti et al. (2019), *IEA Wind TCP Task 37: Systems Engineering in Wind Energy – WP2.1 Reference Wind Turbines*, NREL/TP-5000-73492. The project derives the blade mesh and UV mapping from the reference geometry and applies procedural textures for the experiments. The screenshots do not include a reusable copy of the mesh, texture, original RGB recording or reference model. Full asset provenance, modifications and notices accompany the external research archive.

Vue and Vite are installed through npm, PyYAML through Python packaging, and ROS 2, Gazebo and FFmpeg through the selected runtime environment. Their installed distributions carry their respective license and dependency notices.

## External dependencies

Python (PSF), NumPy (BSD), OpenCV (Apache-2.0 plus bundled notices), PyTorch/torchvision (BSD plus bundled/CUDA terms), einops (MIT), dm-tree/einshape (Apache-2.0), tqdm (MPL/MIT), and optional Matplotlib (PSF-based Matplotlib license) retain their own distribution notices. Short labels do not replace installed license texts. No runtime or font binary is embedded. Figures rasterize installed DejaVu Sans without copying font files.

Upstream license/source links and weight provenance were checked on 2026-10-25. Revision and SHA verification establish content identity, not future URL availability or rights beyond upstream terms.
