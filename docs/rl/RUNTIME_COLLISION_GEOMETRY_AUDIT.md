# Runtime Collision Geometry Audit

Stage 16-D formally measures collision against the geometry actually authored
for the Isaac Lab runtime: 21 Wuji hand convex proxies and one object convex
proxy per clip. Visual OBJ meshes, the reference ghost, inactive objects,
self-collision, ground, and support geometry are excluded from the formal pair
set. The visual meshes are non-watertight and are used only for unsigned
surface-distance diagnostics.

`RuntimeCollisionProxyPenetrationV1` uses
`python-fcl==0.7.0.11` with libccd GJK signed distance and contact EPA/MTD. The
sign convention is positive for separation, zero for touching, and negative
for overlap. Penetration is `max(0, -signed separation)`. Any exception,
non-finite value, missing overlap MTD, or sign inconsistency fails closed. The
fixed numeric tolerance is 10 nanometres and the source-relative numeric
epsilon is 0.5 micrometres. `python-fcl` does not expose a maximum-iteration
field on `DistanceRequest` or `CollisionRequest`; the audit records that as
`null` rather than claiming control over the compiled FCL default, and relies
on fail-closed finite/MTD checks plus the analytic suite.

For each frame and replica, the metric takes the maximum penetration across all
allowed hand-object pairs. Formal p95 is computed only over positive
contact-active per-frame-worst values; the all-frame p95 is diagnostic and may
not dilute the gate. A corrected trajectory must satisfy all four conditions:

- maximum penetration strictly below 10 mm;
- contact-active p95 no greater than 3 mm;
- corrected maximum no greater than source maximum ×1.10 plus epsilon;
- corrected contact-active p95 no greater than source p95 ×1.10 plus epsilon.

The backend passed 13 analytic cases covering separation, touching, overlap,
boxes, rotation, capsule/box, convex translation, rigid-transform invariance,
symmetry, q/-q, scale, near-touch, deterministic repetition, and
depenetration. Runtime FK transform and PhysX contact/sign crosschecks passed;
all formal pair queries converged.

The corrected `170105` terminal candidate measures 1.088 mm maximum and 0.972
mm active p95 against a 0.014/0.014 mm source. `170650` measures 0.838 mm and
0.427 mm against a 0.188/0.187 mm source. Both pass absolute limits and fail
both source-relative limits. This does not imply visual-mesh penetration or
physical calibration, and it does not authorize PPO.

Reproduce the frozen baseline audit with:

```bash
conda run -n toporetarget-isaaclab python \
  scripts/rl/isaaclab/audit_stage16d_runtime_collision_geometry.py \
  --phase inventory --accept-eula
conda run -n toporetarget-isaaclab python \
  scripts/rl/isaaclab/audit_stage16d_runtime_collision_geometry.py \
  --phase backend
conda run -n toporetarget-isaaclab python \
  scripts/rl/isaaclab/audit_stage16d_runtime_collision_geometry.py \
  --phase audit
```

## V1 attainability versus comparability

The D.4R2 audit does not revoke the same-geometry comparability result above.
It tests the separate claim that a kinematic source-relative threshold is
attainable in dynamic contact. A 1,000-repeat query floor and both no-contact
runs pass. `170105` zero-residual dynamic source following produces only
transient required contact and about 0.837/0.797 mm max/active-p95; `170650`
produces no PhysX contact. Bounded source-only stable-contact trials fail to
preserve required topology across 20 replicas. This is insufficient both to
retain V1 as demonstrated-attainable and to freeze a shared stable floor for
V2, so the decision is `STAGE16D_GEOMETRY_GATE_REVISION_BLOCKED`. No corrected
candidate metric was used as a calibration floor.
