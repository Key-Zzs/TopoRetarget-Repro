# Penetration signal stratification

This document defines the source-only half of S1.1. It is deliberately
independent of the old S1/E0 result tree.

Each raw `grab/*/*.npz` file is audited for native FPS, right-hand MANO
parameter completeness, finite object pose, vtemp resolution, and strict mesh
topology. The object mesh is not repaired, patched, convexified, or replaced.
Contact labels remain source semantic labels; `DATASET_PROXY` is provenance,
not ground truth.

For eligible sequences, one deterministic native 60-frame half-open window is
chosen by a fixed source score combining contact-frame ratio, hand-object
proximity, and continuity. The score is used for stratification only. The
shortlist caps subject/object repetition and records source hashes, mesh audit
results, frame relation, and exclusion reasons in:

- `reports/source_candidate_scan.json` and `.csv`;
- `selection/source_window_candidates.json` and `.csv`;
- `selection/shortlist_round1.json`, then round 2 or `shortlist_all.json` when
  the active gate requires expansion.

Only after those files are persisted does the workflow build canonical artifacts
and run the 12-frame E0 probe. A clip is active only when at least two of the
configured frame-level, excess-depth, SDF-energy, and link-coverage conditions
pass while all 12 frames satisfy the strict E0 acceptance contract. Classes A-F
separate clean-E0, source-and-E0 active, inactive, invalid-source,
backend-inconsistent, and solver-failure outcomes.

