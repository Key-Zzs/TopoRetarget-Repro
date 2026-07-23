# Stage 9.3.3 Shadow Equivalence and Long-Finger Ablation

Stage 9.3.3 is a diagnostic gate around the accepted Stage 9.2 reference runtime.
It does not define a paper method, does not change Eq. (1)-(9), and does not
write the Stage 9.2 final/repeat/checkpoint artifacts or Stage 10 exports.

## Gate order

The workflow first binds each isolated frame to the manifest, canonical source,
Stage 7 warm state, Stage 8 graph, object pose, 512 collision samples, QuerySet
profile, formal solver/execution profiles, paper weights, canonical reference SDF,
and the official Stage 9.2 final state at frame `t-1`. It then runs three fresh
official-profile repeats. The numerical contract is
`toporetarget.shadow_equivalence.v1`:

```text
epsilon_f = max(float64_floor_f, 20 * max_pairwise_repeat_difference_f)
```

Hard caps are fixed before comparing the official replay. In particular, qpos,
keypoint, collision-point, and canonical-SDF tolerances cannot exceed `1e-7`
or `1e-6` for qpos, and objective relative tolerance cannot exceed `1e-6`.
Millimetre-scale replay differences therefore cannot be reclassified as float
noise. Status `9`, failed strict acceptance, identity mismatch, or a context
mismatch is fail-closed. `FEASIBILITY_EQUIVALENT_ONLY` is not an accepted
baseline level.

Only when every selected frame is `EXACT` or `NUMERICALLY_EQUIVALENT` may the
six bounded diagnostic profiles run:

1. official baseline reproduction;
2. half active margin;
3. zero active margin;
4. full 512-point QuerySet;
5. minimal soft-safe projection from warm;
6. official-slack projection from warm.

All profile outputs use the same canonical reference-winding evaluation backend.
Projection states are not formal trajectories, and cross-profile total-objective
ranking is not valid because the projection objectives are intentionally different.
Per-finger keypoint RMSE, MCP/PIP/DIP/tip error, `E_IM`, `E_bone`, contact proxy,
QuerySet counts, slack, and full-512 SDF are reported in shared mm/rad/m scales.

## Checkpoints and immutability

Each `frame/profile/repeat` has an atomic diagnostic checkpoint under
`.local/runs/stage9_3_3_shadow_ablation/<run>/shadow_checkpoints/`. A checkpoint
must match its frame, profile, repeat, and trajectory schema; a missing or
corrupt checkpoint is recomputed. `--resume` never treats the checkpoint as a
formal Stage 9 artifact. `--max-wall-time` pauses before the next isolated
profile and preserves completed diagnostic rows.

The equivalence root records before/after path, hash, size, and mtime identity
for the Stage 9.2 final/repeat/checkpoint chain, Stage 7.1 audit, Stage 9.3.2
audit, Stage 10 manifest, acceptance files, and robot exports. Any change is a
separate blocker, regardless of numerical results.

## Current bounded run

The current selected local frames are `[49, 10, 14]`, corresponding to global
frames `[289, 250, 254]`. The official replay reaches the same final QuerySet
IDs/order and strict status/feasibility flags, but is not numerically equivalent:
the largest qpos differences are approximately `1.526e-3`, `1.348e-4`, and
`9.306e-5` rad across the three frames; canonical full-512 SDF differences are
approximately `6.291e-5`, `2.768e-5`, and `1.137e-5` m. The manifest also exposes
an internal provenance mismatch (`git_commit=23e6465` versus recorded runtime
environment commit `58fa77c`). The result is therefore:

```text
SHADOW_BASELINE_NOT_NUMERICALLY_EQUIVALENT
RETURN_TO_STAGE9_3_2_SHADOW_HARNESS_FIX
ENTER_STAGE9_4=NO
mandatory shadow profiles run=0
```

The required evidence is under
`.local/runs/stage9_3_3_shadow_equivalence/<run>/`. Because the baseline gate
failed, the ablation command writes only a blocked `shadow_manifest.json` and
does not execute any mandatory shadow profile.

## Commands

```bash
export PYTHONNOUSERSITE=1 PYTHONPATH=src
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

/home/deepcybo/miniconda3/envs/topo-retarget/bin/python -m toporetarget workflow \
  calibrate-shadow-equivalence \
  --run .local/runs/stage10_reference_runtime/<run>/manifest.json \
  --stage7-audit .local/runs/stage7_1_warmstart_audit/<run> \
  --canonical-audit .local/runs/stage9_3_2_canonical_reaudit/<run> \
  --frames 49,10,14 --baseline-repeats 3 --resume \
  --output-root .local/runs/stage9_3_3_shadow_equivalence/<run>

/home/deepcybo/miniconda3/envs/topo-retarget/bin/python -m toporetarget workflow \
  run-stage9-shadow-ablation \
  --run .local/runs/stage10_reference_runtime/<run>/manifest.json \
  --equivalence-root .local/runs/stage9_3_3_shadow_equivalence/<run> \
  --canonical-audit .local/runs/stage9_3_2_canonical_reaudit/<run> \
  --profiles official_baseline_reproduction,half_active_margin,zero_active_margin,full_512_query_reference,minimal_soft_safe_projection_from_warm,official_slack_projection_from_warm \
  --resume --max-wall-time 1800 \
  --output-root .local/runs/stage9_3_3_shadow_ablation/<run>
```

The second command is intentionally fail-closed until the first root reports
`baseline_pass=true`. Stage 9.4 remains a separate future decision and is not
implemented by this workflow.

Stage 9.3.4 is the subsequent provenance-rebased current-lane experiment. It
keeps this shadow-equivalence gate and all accepted artifacts immutable; see
[`STAGE9_PROVENANCE_MULTISTART_AND_CAUSAL_ABLATION.md`](STAGE9_PROVENANCE_MULTISTART_AND_CAUSAL_ABLATION.md).
