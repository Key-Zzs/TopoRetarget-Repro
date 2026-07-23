# Stage 7.1 — warm-start fidelity and reachability audit

Stage 7.1 is a read-only audit boundary between the accepted Stage 10 reference
runtime and any later Stage 9.3.3 work. It does not regenerate Stage 7, Stage 8,
Stage 9.2, Stage 9.3.2, or Stage 10 artifacts. The formal Stage 7 objective stays
the relative bone-direction objective:

```text
E_bone = sum over the 15 adjacent pairs of ||f_robot - f_source||^2
E_2    = lambda_warm * E_bone + lambda_smooth * ||q_t - q_(t-1)||^2
```

Contact, surface distance, object-relative geometry, and Stage 8 Laplacian
fidelity are reported separately. A contact-retention proxy is not ground truth.
The base seed remains the explicit engineering convention
`T^S_B = T^S_Hs (T^B_Hr(q))^-1`; it is not silently promoted to a paper fact.

## Reproduction command

Use the isolated Python 3.12 environment and the manifest-driven accepted run:

```bash
env PYTHONNOUSERSITE=1 PYTHONPATH=src \
  /home/deepcybo/miniconda3/envs/topo-retarget/bin/python \
  -m toporetarget workflow audit-warm-start \
  --run .local/runs/stage10_reference_runtime/s1__airplane_lift__right__artimano_rh__f000240_f000300/manifest.json \
  --canonical-contact-audit .local/runs/stage9_3_2_canonical_reaudit/s1__airplane_lift__right__artimano_rh__f000240_f000300 \
  --output-root .local/runs/stage7_1_warmstart_audit/s1__airplane_lift__right__artimano_rh__f000240_f000300 \
  --html --run-reachability-diagnostics --diagnostic-frames auto
```

The read-only pass replays persisted qpos and recomputes frames, FK anchors,
bone pairs, Eq. (1)/(2), base alignment, per-finger attribution, Stage 8
evaluation metrics, and source/warm/final contact proxies. It also audits the
source MediaPipe semantic chain, Arti-MANO anchor links, thumb URDF ancestry and
axes, joint limits, local Jacobian observability, and diagnostic-only base
alignment alternatives.

## Reachability interpretation

Only five bounded representative frames are used for diagnostic solves. The
diagnostic root contains official Stage 7 replay records, thumb-only and
all-joint canonical-keypoint fits, thumb formal-feature fits, fixed-base and
base-adjusted fits, no/reduced temporal-weight comparisons, and a deterministic
4096-point Sobol thumb workspace sample. Diagnostic IK is not the paper method,
is not an accepted reference, and cannot write an official artifact.

The workspace report records raw source target, robot-length reconstructed target,
thumb tip/pad samples, nearest distances, direction error, sampled convex-hull
membership, and the explicit caveat that sampled workspace is not a strict global
reachability proof. Robot-length targets are a diagnostic morphology comparison,
not a replacement source trajectory.

## Current accepted-run result

For `s1/airplane_lift`, right Arti-MANO RH, local frames `[0,60)` / global
`[240,300)`, the current audit reports:

- `WARM_START_FORMALLY_VALID_CONTINUE_STAGE9_3_3`;
- `CONTINUE_STAGE9_3_3=YES`;
- persisted Stage 7 replay gates pass with maximum recomputation differences at
  approximately `4.4e-16` and official solver invocation count `0`;
- source mapping, robot anchor mapping, frame, and base-seed gates pass;
- raw thumb target nearest sampled-workspace distance averages about `12.51 mm`,
  while robot-length reconstructed targets average about `3.81 mm` and are within
  the 5 mm diagnostic proximity threshold on all selected frames;
- whole-hand final canonical-keypoint RMSE and reported `E_IM` are higher than
  warm, so final refinement degradation is retained as a separate ranked cause;
- all 45 solver calls are diagnostic-only and are confined to
  `.local/runs/stage7_1_reachability_diagnostics/`.

The morphology result is an embodiment-gap explanation, not evidence to modify
the formal Stage 7 math. The final-retargeting trade-off remains a Stage 9
question. Stage 9.3.3 may continue; Stage 9.4 remains a later readiness question,
and physics/RL readiness is not implied.

## Output contract

The audit root contains `stage7_1_summary.json/.md`,
`stage7_1_readiness.json`, `stage7_artifact_replay.json`, mapping/frame/base
audits, per-finger and warm-vs-final attribution CSV/JSON, joint-limit and
Jacobian reports, `root_cause_analysis.json`,
`warmstart_fidelity_and_reachability.html`, `html_headless_smoke.json`, and
`official_artifact_immutability.json`. The diagnostic root contains a separate
`diagnostic_manifest.json`, `reachability_results_per_frame.csv`, profile JSON,
`thumb_workspace_audit.json`, and `thumb_workspace_points.npz`.

All official input hashes and mtimes must remain unchanged. Any nonzero
`official_artifacts_changed`, any official solver invocation, a mapping/replay
failure, or an unexpected Git worktree change invalidates the readiness result.

## Handoff to Stage 9.3.3

The Stage 7.1 pass is a prerequisite, not evidence that the Stage 9.2 final
trajectory is numerically replayable. Stage 9.3.3 binds the current warm qpos
and base seed together with the official Stage 9.2 final `t-1` state, then
fail-closes if the official baseline is not `EXACT` or
`NUMERICALLY_EQUIVALENT`. Its long-finger attribution is diagnostic and does
not replace the persisted warm-start artifact.

Stage 9.3.4 consumes this audit as a provenance input only. Its base-seed
Kabsch diagnostics do not replace the official warm-start artifact or prove a
final optimizer basin.
