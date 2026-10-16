# Blank annotation template

[wind120.annotations.blank.json](wind120.annotations.blank.json) is a metadata
copy of the existing 120-frame / 4-query form. All 480 records are unannotated:
coordinates are null, physical point definitions unconfirmed, review pending.
It contains no source images or human labels and is not GT.

The frame hashes/point IDs refer to the documented development input. Generate
raw frames and the interactive page locally with the existing
`scripts/prepare_stage1_annotations.py` after obtaining an authorized matching
video and a traceable matching run. A different video requires a new form and
point definitions; do not change a video hash to force alignment.

Use the existing [annotation specification](../../docs/v0/V0_GT_ANNOTATION_SPEC.md)
and [GT progress report](../../docs/v0/V0_GT_PROGRESS.md). No model prediction or
interpolation may fill this template as human GT.
