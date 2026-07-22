# Stage 7 — relative bone directions and sequential warm start

## Scope

Stage 7 closes the source MediaPipe-21 → wrist-centered features → FK anchors →
Eq. (1) → Eq. (2) qpos warm-start path. The output is a bounded initialization
trajectory. Object surface samples, SDF, Delaunay, interaction graphs,
Laplacian coordinates, slack variables, Eq. (8), final collision queries,
PPO, and full GRAB conversion remain outside the stage.

## Implementation and audit

- Frame definitions and strict degeneracy policy are in `retarget/frames.py`.
- Semantic directed bone profiles and differentiable features are in
  `retarget/bones.py`.
- Eq. (1) diagnostics and Eq. (2) residual scaling are in `retarget/objectives.py`.
- Torch-autograd Jacobians, SciPy TRF solving, first-frame policy, and failure
  handling are in `retarget/solver.py`.
- Base observability and canonical alignment are in `retarget/alignment.py`.
- `retarget/pipeline.py` builds the independent `toporetarget.warm_start.v1`
  artifact; storage is in `retarget/artifacts.py`.
- CLI and Matplotlib static/interactive diagnostics are in
  `cli/retarget.py` and `retarget/visualization.py`.

Paper-provided values are limited to the displayed relative direction and
temporal objectives and the two locked weights. Frame axes, pair topology,
initialization, base treatment, bounds, solver, tolerances, and time scaling are
explicit assumptions registered in `docs/ASSUMPTIONS.md` and retained in every
artifact.

## Acceptance

The bounded real acceptance uses `s7/cubemedium_inspect_1`, frames `[0,60)`,
native 120 FPS, and local Arti-MANO RH/LH assets. Definition of Done requires
20 bones/15 pairs, strict frame invariance, Jacobian agreement, bounded
sequential solves, exact base-frame alignment, artifact round-trip, source
integrity, bilingual documentation, and paper-fidelity status
`implemented_with_assumptions`. Stage 8 now consumes the artifact read-only for bounded Eq. (7)
evaluation; it does not modify the warm-start trajectory.

### Stage 7.1 audit boundary

[`../WARM_START_FIDELITY_AND_REACHABILITY_AUDIT.md`](../WARM_START_FIDELITY_AND_REACHABILITY_AUDIT.md)
replays the accepted artifact without regenerating it. It audits mapping, thumb
URDF ancestry/axes, frame and base conventions, joint limits, Jacobian
observability, morphology-normalized targets, and warm-vs-final attribution.
Bounded IK and workspace sampling are diagnostic-only; sampled workspace is not a
global reachability proof. Formal Stage 7 remains the relative bone-direction
objective and contact fidelity is evaluated separately downstream.
