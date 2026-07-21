# Stage 10 — GRAB → Arti-MANO end-to-end retargeting

Status: implemented orchestration, bounded real runs currently blocked at the
existing Stage 9 solver on the tested contact-rich `s1/airplane_lift` windows;
human acceptance remains explicit and separate.

## Entry gate

Stage 9 must be closed, the worktree/index must be clean at the gate, and
`.local/reports/stage9/manual_acceptance.json` must be a human-authored pass that
reviews frames `0,29,59`. `pre_contact` is admissible for the Stage 9 gate but is
not contact-rich evidence for Stage 10 acceptance.

## Definition of done

- one explicit native-time GRAB sequence/window is selected by official semantic
  contact labels and source geometry sanity;
- all Stage 5–9 artifacts are generated or safely reused through a content-signed
  DAG;
- frame count, timestamps, hand/robot/object identity, profile hashes, and source
  integrity are checked across stage boundaries;
- full-surface penetration and semantic sanity reports are retained;
- first/middle/last and metric-worst review frames are rendered without solver calls;
- a human can accept or reject the review bundle;
- `robot_reference.v1` can be exported without modifying source data;
- resume, invalidation, and deterministic plan/signature tests pass.

## Known bounded evidence

The selector passed on `s1/airplane_lift` for `[844,904)` with a 1.0 contact-frame
ratio, approximately `2.865 mm` source contact median distance, and a strict
watertight airplane mesh. The specified `[238,298)` window also passed with the
same strict source geometry; `[240,300)` had approximately `3.046 mm` source
median distance. The existing Stage 9 object `s7/cubemedium_inspect_1` passed
selection at `[363,423)`, and finite explicit queries additionally passed
`s1/airplane_fly_1 [729,789)` and `s1/cubemedium_inspect_1 [343,403)`. The
small-cube candidate passed selector/mesh but failed unchanged strict graph
validation at frame 13 due to two simplex volumes below tolerance. The completed
contact-rich runs reached frozen interaction evaluation, then failed at
final-refinement frame 0 or 1 with the existing `scipy_slsqp_active_set_v1`
`Iteration limit reached` result. A ratio-0.5 transition run exceeded 40 minutes
at roughly 100% CPU and was terminated with SIGTERM; it is not treated as a
trajectory. No solver profile or Stage 9 implementation was changed to conceal
any failure.

Additional finite explicit queries found strict candidates for
`s1/cylinderlarge_inspect_1 [327,387)` and `s1/apple_lift [1717,1777)`.
The cylinderlarge run again passed through Stage 8 and failed at Stage 9 frame 0;
the sphere candidate failed unchanged strict graph validation, and mug/phone/bunny
were rejected for non-watertight meshes. A single-frame diagnostic shows that the
frozen SLSQP profile reaches positive full-surface and residual margins but still
returns status 9 at `maxiter=30`; Stage 10 preserves that fail-fast result.
The corresponding finite left-hand query `s7/cubemedium_inspect_1 [513,573)` passed
selection but failed unchanged strict graph validation at frame 1 before refinement.

Therefore the workflow is not claimed as a passed end-to-end trajectory until a
bounded contact-rich run solves and its review is human-accepted.

## Boundary

This stage does not add Eq. 10–12 metrics, ContactPose evaluation, new adapters,
baselines, PPO/RL, physics, full-dataset conversion, or changes to Stage 7–9
coordinate/weight/solver contracts.
