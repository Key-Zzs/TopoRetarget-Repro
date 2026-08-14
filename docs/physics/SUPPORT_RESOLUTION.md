# Stage 16 support resolution and reconstruction

This contract resolves the physical support of a recorded sequence before a
runtime scene is built. It is deliberately separate from PPO, reward, contact
semantics, and the existing absolute hand-object geometry gate.

## Resolution order

`auto` follows this order:

1. audit the sequence directory and source metadata for explicit support;
2. validate recovered scene geometry only when a source adapter supplies an
   asset, transform, and validation receipt;
3. if no source support is validated, detect the earliest stable pre-contact
   interval from object pose/twist, mesh height, gravity, and optional contact
   evidence;
4. fit a gravity-aligned planar support from the visual and runtime collision
   mesh trajectories and create a finite, static/kinematic box proxy;
5. qualify object/table and hand/table geometry, then run a matched full-
   gravity PhysX A/B: object-only with the proxy versus object-only without it.

`source_only` never falls back to an inferred plane. `inferred_planar` is an
explicit opt-in to steps 3–5. A missing source asset is not itself evidence
that a table existed.

## Frozen contracts

The tracked algorithm contract is
[`configs/physics/support_resolution_v1.yaml`](../../configs/physics/support_resolution_v1.yaml).
The reusable implementation is under
`src/toporetarget/physics/support/`:

- `source_evidence.py`: source-first adapter and validation boundary;
- `planar_inference.py`: stable interval, gravity normal, plane, patch, and
  finite extent inference;
- `geometry_validation.py`: visual/collision object-table and optional full
  hand-table checks;
- `runtime_support.py`: local-frame finite collision actor asset;
- `physics_validation.py`: backend-neutral force, stability, and A/B reduction;
- `resolver.py`: fail-closed resolution and final status.

The support box is authored in a local frame. Isaac Lab applies the audited
table center and quaternion through `RigidObjectCfg.init_state` at spawn. The
box has finite extent and thickness, nominal uncalibrated material, and no
force injection, attachment, object teleport, guidance, or rollout state
write.

## Current HOCap qualification

The source audit for `hocap_170105` and `hocap_170650` found no recoverable
source support asset in the mounted sequence directories. Both clips therefore
resolve as `INFERRED_PLANAR_SUPPORT`, with stable pre-contact intervals `0:18`
and `0:21`, respectively. Object/table geometry passes and full hand/table
geometry remains `DEFERRED` because the available reference contains tracked
link points, not the full hand collision mesh.

The real Isaac Lab 5.1 / GPU PhysX receipts show the expected causal split:
without support, both objects fall under full gravity; with support, contact is
continuous, the normal force is approximately `mg`, and position plus
quaternion pose drift remain within the static qualification limits. Both
inferred supports therefore pass geometry and object-only physics. Runtime
reference-following transfer remains deferred because the full hand collision
mesh and the existing P3 hand-object geometry gate are not available; this is
not a reason to hide the motion by moving the table or injecting guidance.

Runtime reference-following transfer remains
`DEFERRED_BY_HAND_OBJECT_GEOMETRY`, and P3/G3/P4 remain blocked. The exact
receipts are in ignored local storage under
`.local/reports/stage16_support_reconstruction/`.

## Reproduction commands

Geometry inference and overlays:

```bash
PYTHONPATH=src python scripts/physics/visualize_support_reconstruction.py \
  --support auto --static --replay
```

The source-only audit is fail-closed:

```bash
PYTHONPATH=src python scripts/physics/visualize_support_reconstruction.py \
  --sequence hocap_170105 --support source_only
```

Full-gravity object-only PhysX receipts use the generated local proxy. Run the
`with_support` and `without_support` commands for each clip, then reduce them:

```bash
conda run --no-capture-output -n toporetarget-isaaclab \
  env OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=src \
  python scripts/physics/validate_support_physx.py \
  --clip hocap_170105 --case with_support --steps 360 --accept-eula \
  --support-asset .local/support_assets/hocap/hocap_170105/support_proxy.usda \
  --proxy-json .local/reports/stage16_support_reconstruction/inference/hocap_170105/table_proxy.json

conda run --no-capture-output -n toporetarget-isaaclab \
  env OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=src \
  python scripts/physics/validate_support_physx.py \
  --clip hocap_170105 --case without_support --steps 360 --accept-eula

PYTHONPATH=src python scripts/physics/finalize_support_reconstruction.py
```

The same commands apply to `hocap_170650` after changing the clip and asset
paths. A non-zero finalizer exit is intentional when the physical qualification
is blocked.

## Non-goals

This stage does not retrain PPO, change rewards, alter C0/C1/C2/C3/C4 or G3
gates, repair hand-object reference penetration, add a floor/table to the main
RL environment, or promote a failed support qualification. The support actor
is a diagnostic/reconstruction artifact until both geometry and physics pass.
