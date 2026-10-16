# Local experiments

`runs/` is excluded from Git. Every run uses a fresh directory with input/model/config hashes, NPZ and CSV trajectories, predicted visibility, runtime metadata, weak diagnostics and verified MP4 output. Original data and weights remain external.

Stage 0 real-model protocol output: `runs/stage0_20261013/`. Public numeric records are in [summaries/](summaries/README.md); complete local records such as `docs/v0/STAGE0_RUN_RECORD.json` remain preserved but excluded from Git because they contain machine metadata. Interpretation is in `docs/v0/V0_PRELIMINARY_RESULTS.md`. Never replace fresh model runs with legacy cached predictions.

`templates/` contains unfilled metadata forms. `annotations/`, `stage1/` and
`weekly/` hold local frames, labels, plots and full evidence; those subpaths are
ignored. Configurations and protocols remain under `configs/`, and reports
under `docs/`. JSON/CSV/NPZ/PNG are not globally ignored, so small synthetic
fixtures and public summaries can be reviewed for Git individually.
