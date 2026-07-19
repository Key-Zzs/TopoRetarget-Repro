# Reproduction log

## 2026-07-19 — Stage 0

- Confirmed repository root, branch, remote, initial commit, and existing GPLv3 license.
- Created package/CLI, path and dataset registry, CI, tests, documentation, and local-only policy.
- Imported the full local Arti-MANO URDF/mesh tree from the configured ManipTrans checkout and
  generated the ignored asset manifest.
- Ran dataset discovery against the configured NAS path without reading raw dataset files.

## 2026-07-19 — Stage 1

- Audited the 16-page arXiv 2606.16272 v2 PDF, including Appendices A.1–A.5.
- Recorded PDF SHA-256, equations 1–12, tables 1–6, figures 1–5, configurations, assumptions,
  author questions, and strict/extended boundaries.
- Ran the paper fidelity checker and unit tests.

## 2026-07-19 — Stage 2A

- Implemented `toporetarget.hoi.v1`, explicit scene/wrist/object transforms, lazy adapter contract,
  opt-in Zarr storage, deterministic synthetic data, comparison metrics, and Matplotlib rendering.
- Preserved timestamps and native FPS as metadata; no temporal resampling, spatial sampling, robot
  model, source-data modification, or full-dataset conversion was introduced.
- Stage 2A is complete and is staged as a separate Git snapshot. Stage 2B is not started in this
  snapshot.

## Reproducibility boundary

This log does not claim that numerical results, private Pen-Spin data, Wuji hardware transfer, or
later algorithm stages have been reproduced.
