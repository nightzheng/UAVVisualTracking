# Public experiment summaries

These JSON files preserve selected numeric results and content hashes from
existing local experiments. They exclude absolute machine paths, full local
file inventories, original videos/frames, weights, trajectories and human
annotation outputs. No new GPU inference was run for release preparation.

- [Stage 0](STAGE0_SUMMARY.json): five real offline smoke runs and historical
  engineering checks; cold/warm times must not be used for configuration ranking.
- [Stage 1](STAGE1_SUMMARY.json): warm controlled on/off runs, independent
  reported GPU usage sampling and qualitative failure candidates. GT is 0/480.
- [Weekly pilot](WEEKLY_20261012_1018_SUMMARY.json): eight saved crop/full-view
  runs from one source sequence, and known 2D transform checks. Prediction
  disagreement is not GT EPE; transform arithmetic is not real camera estimation.
- [Legacy audit](LEGACY_AUDIT_SUMMARY.json): source inventory count and
  integrity checkpoint; full private inventory remains local.

`source_records_sha256` identifies preserved raw records, which are not
published here. Hashes identify evidence; they cannot substitute for rerunning
the model on separately obtained matching input and checkpoint. Current
research conclusions remain limited by absent reviewed human GT and independent
near-field/camera-motion validation. Source code/weight/data licenses remain
separate; see [THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md).
