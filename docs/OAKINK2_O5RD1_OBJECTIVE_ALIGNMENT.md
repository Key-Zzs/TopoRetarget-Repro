# OakInk2 O5R-D1 objective-alignment audit

O5R-D1 is a diagnostic audit, not another optimization method. It explains why
the low-interaction-error states found by O5R-B did not become production
solutions, using only the 25 B1 development frames and the 30 failed, now
consumed, O5R-C SparseValidationV1 frames.

## Result

`StructuredSolverV1` is scientifically rejected and must not be reused as a
production solver. Its 30/30 technical completion did not satisfy the frozen
Semantic V1 sparse gate.

The production objective contains exactly the Semantic V1 per-frame
graph-Laplacian `E_IM`, with weight 500. This does not make the full objective
semantically aligned: the per-frame scalar also contains bone fidelity,
continuous temporal regularization, base correction, and collision slack, and
it does not optimize the trajectory-level Semantic V1 p95 gate.

On all 25 B1 frames, Q1 has lower `E_IM` and a worse exact production objective.
The median weighted changes from Q0 to Q1 are:

- continuous temporal: `+9.961803871`
- bone direction: `+0.075531161`
- interaction mesh: `-0.040541888`

All 25 Q1 states pass the production hard-collision bound. The rejection of Q1
is therefore an objective trade-off, not a hard-feasibility failure. Ten
deterministic interpolation paths show monotonic semantic improvement together
with monotonic objective degradation. In a separate frozen 10-frame Q1-start
micro-polish control, full polish worsens `E_IM` on 10/10 frames while improving
the exact production objective on 10/10.

Contributor selection is not the cause: dominant-contributor/selected-block
agreement is 55/55. The structured block stage is weak and inconsistent (32
effective, 22 regressed, one numerically weak). The ordinary Q2-to-Q3 polish is
not a rollback mechanism on this population; it generally improves `E_IM`.

The primary diagnosis is therefore
`OBJECTIVE_SEMANTIC_MISALIGNMENT` with `HIGH` confidence. O5R-C also omitted the
production continuous-prediction inputs when constructing its structured-stage
contexts, causing a different temporal branch to run. That is a secondary V1
binding defect and does not rehabilitate the rejected solver.

## Reproduction

The real CLI is `scripts/data/run_oakink2_o5rd1.py`. Inspect it first with:

```bash
conda run -n toporetarget-rl \
  python scripts/data/run_oakink2_o5rd1.py --help
```

Its actions separately audit the objective, replay Q2/Q3, decompose Q0-Q3,
audit contributor/block selection and polish rollback, run bounded directional,
interpolation, and development-only micro-polish diagnostics, and summarize the
decision. Evidence is written under
`.local/reports/oakink2_o5rd1_objective_alignment_v1/`.

## Validation hygiene

The 25 B1 frames and 30 SparseValidationV1 frames are permanently consumed for
method development/diagnosis. A future method must freeze a new untouched
SparseValidationV2 only after its V2 contract is frozen, excluding all 55 frame
IDs. O5R-D1 created no V2 objective, solver, semantic metric, or validation set,
and it ran no full DEV1 trajectory or DEV2 solve.

