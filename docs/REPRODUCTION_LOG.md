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
- Stage 2A is complete and is staged as a separate Git snapshot. Stage 2B work remains unstaged
  so the two delivery boundaries stay separate.

## 2026-07-19 — Stage 2B

- Implemented the explicit-path GRAB NPZ reader, object/PLY loader, replaceable MANO backend,
  canonical adapter, fake-backend tests, and opt-in licensed-data test.
- Inspected one local sequence, `cubemedium_inspect_1`, at 120 FPS. `describe` passed and confirmed
  the selected right-hand fields and object/personalized-vtemp resources.
- With the user-provided MANO root, converted the explicit `cubemedium_inspect_1` right-hand clip
  `[0, 60)` at native 120 FPS through the SMPL-X package's MANO backend. Canonical Zarr, raw-to-
  canonical side-by-side/overlay reports, and first/middle/last frame renders passed. Source hash
  and mtime are unchanged; evidence is in ignored `.local/reports/stage2b/`.

## Reproducibility boundary

This log does not claim that numerical results, private Pen-Spin data, Wuji hardware transfer, or
later algorithm stages have been reproduced.

## 2026-07-29 -- W2.3 Wuji sequential finalization

Added the separately named `wuji_continuous_sequential_v1` candidate and the
bounded finalization harness. W1/W2/W3 formal artifacts are audited as
immutable inputs; selected-frame replay, multi-threshold signed-distance
audits, the W3 window oracle, nonblocking window shadow, versioned exports,
HTML smoke, and final integrity evidence are isolated under
`.local/experiments/wuji_hand2_continuous_v1/w2_3_finalization/`. The scope is
offline reference generation only; author-exact, RL, real-time, and
cross-subject claims remain unresolved.

## 2026-07-19 — Stage 3

- Audited the existing real `cubemedium_inspect_1` right-hand cache: `mano16`, `[60,16,3]`,
  778 vertices, 120 FPS, and the Stage 2B MANO translation-based wrist pose.
- Compared installed `smplx` fingertip candidates with the local ManipTrans candidates on MANO
  neutral geometry and the real clip. Selected installed-smplx anchors in a versioned profile and
  retained the discrepancy in `.local/reports/stage3/mapping_sources.json`.
- Implemented explicit layout/profile registries, semantic joint conversion, dense/sparse regressor
  conversion, scene/wrist APIs, CLI validation, reports, and scene/wrist visualizations.
- Converted the bounded real clip to `.local/cache/hoi/grab/cubemedium_inspect_1_rh_f000000_f000060_mp21.zarr`.
  Direct copy metrics are zero; round-trip is floating-point precision; source/object/timing
  integrity is preserved. MANO joint 0 versus the stored wrist-pose origin is `0.08606943 m` and
  remains documented under `A_HAND_FRAME_001`.
- Repeated the adapter on the real left-hand `[0,10)` clip and wrote
  `.local/cache/hoi/grab/cubemedium_inspect_1_lh_f000000_f000010_mp21.zarr`; direct/tip checks are
  zero and scene↔wrist round-trip is floating-point precision. Added and smoke-tested the local
  interactive viewer with all display toggles and no coordinate mutation.

## 2026-07-20 — Stage 5

- Built the filename-first GRAB index at `.local/index/grab`: 1,335 active NPZ sequences across
  subjects `s1`–`s10`; index construction did not load MANO models or frame arrays and did not hash
  source files by default.
- Implemented `GrabDatasetAdapter` with explicit right/left/both hand selection, contiguous
  half-open frame ranges, native 120 Hz timestamps, personalized `vtemp`, native hand/object/table
  geometry, source/binary/official semantic contact modes, optional MediaPipe21, atomic Zarr output,
  and provenance.
- Converted the real `s7/cubemedium_inspect_1` right-hand and bimanual clips `[0, 60)` using the
  local GRAB root and MANO root. The canonical tracks contain native MANO16/SMPL-X joints, 778
  vertices, scene wrist poses, object/table poses, contacts, and optional MediaPipe21.
- Validation passed: timestamps/contact arrays/hand vertices/translation/world vertices match;
  maximum wrist/object rotation difference is approximately `1.71e-6` degrees and round-trip
  reconstruction is floating-point precision. The legacy Stage 2B native-keypoint comparison is
  explicitly unavailable because that cache lacks the formal native-keypoint field.
- Generated first/middle/last canonical and compare PNGs, a JSON/CSV validation report, index and
  provenance reports, and an interactive viewer smoke report. No raw dataset or external model was
  modified; all generated data remains under ignored `.local/`.
- Verified the official GRAB `contact_ids` mapping and added strict/non-strict semantic conversion,
  raw/binary/semantic Zarr round-trip checks, semantic viewer colors, and deprecated/canonical
  `--reference-frame` CLI alias validation. Fresh MANO-backed semantic conversion and validation
  pass for the bounded s1 contact and s7 bimanual geometry clips when the external MANO root is
  supplied explicitly; source/object/table/model integrity remains unchanged.
