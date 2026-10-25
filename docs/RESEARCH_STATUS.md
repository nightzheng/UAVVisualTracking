# Research status and reproducibility boundary

Snapshot v0.1, 2026-10-25. The question is whether offline point tracking provides reliable geometric observations when repetitive structures leave and reenter an observation region. This is baseline evidence and a fixed acceptance operating point, with no new tracking network.

A material point is a fixed physical location on a surface, with an identity that persists across time. A rotor hub/axis reference or an image-space tip extremum is not interchangeable with a blade material point. Repeated appearance and geometric proximity cannot by themselves confirm that identity has been recovered. If a query cannot be bound to an independently reviewable material definition, its identity must remain unknown; a newly chosen point requires a new ID/query rather than replacement GT for the old query.

## Method and formal conditions

B0 is pinned CoTracker3 offline (`82e02e8…`), with native 384×512 internal resolution, a 60-frame window, six iterations and native support grid (36 support points in formal runs). Native acceptance is visibility probability >0.9, with the official query-frame override. Confidence probability is exported but does not enter native offline visibility. B1 is pinned BootsTAPIR v2 PyTorch (`730cda1…`), using 256×256, no support points, four refinement iterations, pyramid level 1, temperature 20 and official offline defaults. B1 score is `(1-sigmoid(occlusion_logit))*(1-sigmoid(expected_dist_logit))`; native acceptance is strictly >0.5. Native system conditions and computational budgets differ.

Both consume full selected clips and can use future frames. Predictions become available after the final selected frame. No result establishes causal operation, onboard real-time execution or control-loop usefulness.

Stage5B used six procedural parents, one rendering family, 48 frames at 24 Hz, a 480×360 source reference and two material-point queries (`A_outer`, `A_inner`) per C0/C2/C3. C0 is full view with distinct appearance; C2 is full view with repeated appearance; C3 is 180×260 cropped at source `[295,65,475,325]`. The full study had 108 outputs including B0 no-grid; this public comparison selects the 72 native B0/B1 outputs without changing predictions. The original 18 outputs had a documented resource admission deviation; p01 information had previously been exposed during recovery. These limitations remain part of interpreting the frozen comparison. No-grid effects were condition- and metric-dependent.

Stage6A selected score operating points on development data; some candidate macro risks were NA because acceptance was empty. No threshold is selected again here. Stage6B used six new procedural parents of the same rendering family, three conditions and two queries per condition (72 outputs). Same-parent queries and crops are repeated opportunities, not independent scenes. This synthetic replication is not evidence of real-domain generalization.

M0 accepts native visible, finite positions. M1 accepts finite positions with the model's continuous visibility score **strictly >0.7**. ALL accepts all finite positions as a diagnostic. M1 neither updates coordinates nor calibrates identity. After the query, B0 M0 is a subset of M1; B1 M1 is a subset of M0. Count directions partly follow mechanically from these nested sets.

## Results and negative findings

These are frozen machine summaries, not new experiments. Rates are fractions; parent means weight planned query opportunities equally, then weight parents equally.

| Stage5B condition | Model | AJ | δavg | OA |
|---|---|---:|---:|---:|
| C0 | B0 | 0.829839 | 0.938652 | 0.843972 |
| C0 | B1 | 0.878564 | 0.928014 | 0.962766 |
| C2 | B0 | 0.232350 | 0.377305 | 0.952128 |
| C2 | B1 | 0.573949 | 0.685461 | 0.783688 |
| C3 | B0 / B1 | NA | NA | NA |

C0 has no stable advantage across metrics or parents: B1 has higher AJ in four parents, B0 in two, while mean δavg favors B0. B1 AJ is higher in all six C2 parents within this rendering family. C3 has only 5/12 complete-GT query tasks, so no full six-parent standard macro is reported. Known-only geometric evidence is not official AJ.

| Stage6B, 10 source px | R_sel M0 → M1 | C_good M0 → M1 | Wrong accepts M0 → M1 | Correct accepts M0 → M1 |
|---|---:|---:|---:|---:|
| B0 | 0.207191 → 0.236139 | 0.455038 → 0.515870 | 344 → 369 | 769 → 871 |
| B1 | 0.055840 → 0.040107 | 0.682995 → 0.644500 | 66 → 50 | 1153 → 1088 |

Rates are strict parent macros; counts are pooled supplements and must not replace macro rates. B0 pooled selective risk can move in a different direction from macro risk. The 5 px sensitivity analysis has the same tradeoff directions; exact 5/10 px macros and every parent are in [summary.json](../results/summary.json) and [reliability_parent.csv](../results/reliability_parent.csv).

At 10 px, B0 M0/M1 accept 1/2 unknown opportunities; B1 M0/M1 accept 3/2. There are 1,692 postquery opportunities per model, with 1,683 known. Unknown outcomes are excluded from known denominators and separately reported with best/worst risk bounds. Empty acceptance gives R_sel=NA. Undefined query values propagate to parent and full macros.

The selected C2 video shows a high-scoring B0 prediction far from synthetic material GT amid repeated blade appearances. It is geometric failure/candidate association, not an independently confirmed identity switch. C3 includes hidden/unknown periods and reentry; a correct position can still be rejected. Events are not additional independent samples. Public media uses only formal Stage5B frozen synthetic results; no Pilot image is relabeled. The separate development GPU clip validates engineering only and appears in no research statistic.

## Metric contract

Official AJ, δavg and OA execute the exact verified `compute_tapvid_metrics` body from TAPNet commit `730cda1c730877cfedbe01bf87fb1cadb78a565d`. The full source, copyright header and Apache-2.0 license are bundled as package resources. The loader verifies SHA256 and extracts only that function to avoid unrelated dataset/JAX dependencies. Formulas, threshold comparisons and query-mode semantics are unchanged. Official TAP-Vid uses strict distance < threshold in the 256-normalized domain; G_geo separately uses source-image distance ≤epsilon.

Source-image GT is mapped into the actual processing raster by the inverse project pixel-center resize and ROI translation. Predictions are already in that raster. Normalize **x×256/W_video, y×256/H_video**, without another half-pixel offset. C3 uses W=180, H=260; 480×360 never supplies its standard normalization. `first` scores strictly after each query; optional official `strided` scores all nonquery frames. Reliability always scores strictly after each query.

Complete visibility and visible-position GT are required for official full metrics. Incomplete GT returns `standard=null` and a reason. Identity, point-ID, query, source-frame, PTS and transform mismatches raise errors even with incomplete GT. Nonfinite predictions are never silently dropped: official position errors use an out-of-threshold sentinel while native visibility semantics remain intact; reliability rejects them and counts known failures.

Source-image EPE is Euclidean distance in original pixels over postquery known-visible/known-position observations. Mean, median and P95 are conditional on finite predictions; required/finite counts and large-error counts disclose nonfinite predictions. EPE does not disappear when a model rejects an observation.

For G_geo, good means known visible GT, finite prediction and source-image error ≤epsilon (5 or 10 px). Known hidden is bad. Unknown visibility, or visible with unknown position, is excluded from known denominator N. U counts all postquery opportunities. Cross good/bad with accept/reject to obtain AC, AB, RC, RB. `R_sel=AB/(AC+AB)`, `C_acc=(AC+AB)/N`, `C_good=AC/N`; zero denominators give null. Accepted unknown AU is separate, with risk bounds `[AB/(AC+AB+AU), (AB+AU)/(AC+AB+AU)]`. This is geometric correctness, not an identity oracle. Unknown breaks known continuous runs; bracketed returns receive no interpolated exact boundaries.

## Evaluation input and complete plan

Use demo `tracks.npz` and `manifest.json` schema2. The reader also supports aligned legacy schema1 for frozen regression; public writing always requires continuous raw scores and emits schema2. CSV and NPZ use the same source/crop transform. The manifest describes time basis, original/processing resolution, optional ROI/source-processing resolution, model/weight identity and temporal permissions. PTS means playback time, not sensor acquisition wall-clock time; invalid decoder PTS are marked as nominal-FPS assumptions. CFR export refuses assumed timing and VFR.

GT NPZ is loaded with `allow_pickle=False` and contains:

| Field | Shape / meaning |
|---|---|
| `video_id` | scalar Unicode string equal to prediction manifest identity |
| `frame_indices`, `timestamps` | T source frame integers and exact saved PTS, matching predictions |
| `point_ids` | N unique strings in prediction order |
| `queries_original_txy` | N×3 clip-local query frame and source x,y; query must be known visible GT |
| `tracks_original` | T×N×2 source-image material GT coordinates |
| `visibility` | boolean T×N; meaningful where visibility_known is true |
| `visibility_known` | boolean T×N; false means unknown |
| `position_known` | boolean T×N; known visible positions must be finite |

Hidden positions need not be finite or known. Unknown must not be interpolated or converted to known hidden. GT creation and material identity adjudication remain independent responsibilities. Human material GT must be determined and reviewed from source frames independently of model predictions; query coordinates, predicted tracks and automatic interpolation do not constitute human GT. Synthetic oracle GT is a separate source type and does not establish real-video annotation validity.

A plan JSON declares nonempty unique `parents`, `conditions`, `queries`, `models` (B0/B1) and `slots`. Every Cartesian combination must appear exactly once. Each slot has `parent`, `condition`, `query`, `model`, `result` (directory) and `gt` (NPZ file); paths resolve relative to the plan. Each result binds exactly its declared query. Parent/condition metadata are checked when present. The plan is the caller's explicit declaration of the full intended population, not a list inferred from successful outputs.

Two parents × C0/C2/C3 × A_outer/A_inner × B0/B1 require 24 slots; six parents require 72. Missing slots/files give `status=refused_incomplete_plan`, exact missing identities, `macro=null` and exit code 2. Duplicate/unexpected or mismatched slots fail. Complete plans emit `rows`, `parent_metrics` and `macro` for M0/M1/ALL. Query rates average within parent, then parents receive equal weight. An undefined required rate makes its parent and macro null; undefined contributors are listed. Standard columns always use native visibility, even on M1/ALL rows.

The public API evaluates image-space observations. Camera-motion compensation and reference-recovery extensions are outside this snapshot. A compensated 2D reference remains a pixel coordinate system; a low background fitting residual alone does not prove target-motion accuracy, metric 3D motion or observer/target motion separation. Those claims require independent references, calibration and any necessary depth/scale evidence.

## Reproduction and acceptance boundary

Small numeric summaries and selected derivatives allow inspection of claims. Full renderer, frozen arrays/GT, model code/weights, recovery/resource controller and stage history are not distributed. This snapshot cannot regenerate the full research dataset or replay every historical execution from scratch. No model, GT or threshold was regenerated during research artifact export.

Read-only regression covered all 72 native Stage5B and 72 Stage6B outputs, C0/C2/C3, both models, fixed 5/10 px Stage6B gates, complete parent aggregation and saved official scores: 9,396 scalar checks, maximum difference 2.22×10⁻¹⁶. Complete C3 tasks used actual 180×260; incomplete tasks stayed NA. The 25 CPU tests cover scientific and I/O boundaries including deliberate missing plans.

Normal-wheel and editable installations were checked separately, importing the intended snapshot package and bundled official source/license with original PYTHONPATH ignored. Existing dependencies were reused; fully fresh internet dependency resolution and CPU model forwards were not tested. Both B0/B1 actually ran on a separate 6-frame development MP4, one query, processing width 320, native defaults and GPU. Schema2 reload, raw-score fields and CFR overlay decoding passed. This narrow engineering check is excluded from all research figures, videos and statistics.

There is no real material-point GT, demonstrated real turbine performance, validated material identity posterior, new training result, proven network contribution or onboard real-time capability. Evidence belongs to one synthetic rendering family; broader claims require independent data and experiments outside this snapshot.
