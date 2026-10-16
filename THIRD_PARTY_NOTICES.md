# Third-party notices — UAVVisualTracking v0.1.0

Prepared on 2026-10-14; copyright decision confirmed on 2026-10-16.
The user explicitly confirmed `nightzheng` as the MIT copyright holder.
MIT applies only to original contributions that nightzheng has authority
to license; this confirmation is not inferred from Git author metadata.

CoTracker3 is an external dependency used by the implemented research baseline. Its official source and weights are not copied into this repository; original input videos are read from local files and excluded from version control. Project-owned experiment functions were migrated following the owner’s authorization on 2026-10-13.

## CoTracker3

- Author/publisher: Meta Platforms, Inc. and affiliates; facebookresearch/co-tracker.
- Official repository: https://github.com/facebookresearch/co-tracker
- Audited and pinned commit: `82e02e8029753ad4ef13cf06be7f4fc5facdda4d`.
- Code license: Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0); retain original attribution and applicable notices. See the [official license at the pinned commit](https://github.com/facebookresearch/co-tracker/blob/82e02e8029753ad4ef13cf06be7f4fc5facdda4d/LICENSE.md).
- Model publisher: [facebook/cotracker3](https://huggingface.co/facebook/cotracker3), whose model card declares CC BY-NC 4.0. Keep model weights outside version control.
- Audited weight: `scaled_offline.pth`, 101890938 bytes, SHA256 `2670d4562ed69326dda775a26e54883925cd11b6fc9b24cb7aa9f8078bce7834`. Local content matches the SHA256 published on the [official weight page](https://huggingface.co/facebook/cotracker3/blob/main/scaled_offline.pth), checked on 2026-10-12. The historical download transaction is undocumented; the content identity was verified by hash.
- `scaled_online.pth` was not found in the audited legacy tree or the current Torch Hub checkpoint directory. An existing cached online trajectory is not evidence that the online weight is currently installed.
- Research attribution: Karaev et al., *CoTracker3: Simpler and Better Point Tracking by Pseudo-Labelling Real Videos* (2024); the model card includes the full author/citation details.

The project MIT license applies only to original project contributions that nightzheng has authority to license. It does not relicense CoTracker3, other dependencies, model weights, or third-party videos including Pexels assets. It does not establish commercial authorization for the combined system. This phase is limited to research baseline reproduction within the original third-party terms.

## Independent legacy contributions and data

The repository owner confirmed on 2026-10-13 that the provided experiment contributions are their independent original work and do not contain other group members’ code. The authorized functions and adapted flows are identified with hashes in `docs/migration/SOURCE_MAP.json`; they are covered by this project’s MIT license. Contributions belonging to other members require internal research reuse permission and retained attribution; internal cooperation alone does not authorize public MIT distribution. Files of uncertain authorship are held individually.

The user confirms the legacy video platform as Pexels. Individual source URLs, authors and download records remain to be completed in `docs/v0/DATA_REGISTER.json`. Candidate URLs from a source search are explicitly unverified and do not establish authorship of the local files. Current use is local noncommercial inference/visualization of already-downloaded files, without endorsement claims for depicted people. Original assets are excluded from Git and are not distributed as a standalone asset pack. No bulk collection, training or public dataset was performed.

See the [Pexels license](https://www.pexels.com/license/) and [Pexels terms](https://www.pexels.com/terms-of-service/), reviewed on 2026-10-13. The license permits free use/modification with conditions including no implied endorsement. The terms restrict unauthorized automated collection and bulk/systematic copying. Public dataset construction and larger-scale machine-learning use require a separate terms/authorization review; no such rights are inferred here.

## Other components in the verified environment

These packages are external dependencies, not copied or relicensed project
source. Versions/licenses below were read from the installed distribution
metadata and license files on 2026-10-14. Upstream and bundled component
notices remain applicable; this table is not a license for their binaries.

| Component | Verified version | License / scope |
|---|---|---|
| Python | 3.10.21 | Python Software Foundation license; runtime external |
| PyTorch | 2.11.0+cu128 | BSD-3-Clause; bundled third-party/CUDA terms separate |
| torchvision | 0.26.0+cu128 | BSD-3-Clause; external model/runtime dependency |
| NumPy | 2.2.6 | BSD-3-Clause and bundled numerical-library notices |
| opencv-python / OpenCV | 5.0.0.93 | Apache-2.0; wheel contains LICENSE-3RD-PARTY.txt, including separate bundled components |
| Matplotlib | 3.10.9 | Matplotlib license (PSF-based); plotting scripts and separate font notices |
| imageio | 2.37.4 | BSD-2-Clause; existing upstream environment utility |
| imageio-ffmpeg | 0.6.0 | BSD-2-Clause Python wrapper; FFmpeg binary/build licensing is separate |
| Pillow | 12.3.0 | MIT-CMU; existing image/plotting dependency |
| tqdm | 4.70.1 | MPL-2.0 AND MIT, according to installed metadata |
| setuptools | 78.1.0 | MIT; build tooling |
| pip | 26.2.1 | MIT; installation tooling |

Only NumPy and OpenCV are declared base project dependencies. The
`cotracker` optional extra pins the upstream VCS version; its upstream
setup.py has no runtime install_requires, so it does not install the
verified PyTorch/torchvision environment for the user. Matplotlib is
needed by evidence plotting scripts; imageio/FFmpeg utilities and tqdm
are listed because they exist in the verified upstream environment,
not because every project command requires them.

## Final migration boundary check

The supplied legacy functions were rechecked against SOURCE_MAP.json and
the read-only source files. The migrated draw_point/draw_history functions
retain the owner's source bodies; the video/query/output/diagnostic flows
retain the recorded origin. The src tree contains the project adapter,
not an embedded cotracker model implementation or weight file. No
exact upstream match was found among function bodies with at least eight
top-level statements in the AST comparison performed for this preparation;
this bounded scan is not independent proof of authorship. Calling the pinned upstream API does not relicense
its implementation. Any subsequently identified upstream-derived fragment
must retain its original attribution/license and be excluded from blanket
MIT claims until resolved.

Public Git preparation includes source/configuration/protocols and
sanitized numeric summaries only. Original videos, derived video frames,
human annotation outputs, full local machine records and model weights
remain local. Raw records are preserved, rather than edited to remove
paths; public summaries link to their content hashes and clearly state
that full reproduction needs separately obtained inputs.
