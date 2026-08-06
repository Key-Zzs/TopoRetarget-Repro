# Stage 16-D — Physics-Consistent Retargeting

## Closeout status

Stage 16-D changes the goal from strict reproduction of the source object path
to causal contact-driven reproduction of the motion intent. The source
wrist/finger motion, contact evidence, object path, Stage 12 outputs, and Stage
16-C reports remain immutable inputs. The runtime object is a free PhysX rigid
body; its corrected trajectory is an output, never an optimizer variable.

The closeout status is `STAGE16D_BLOCKED_WITH_BOUNDED_EVIDENCE`.

| Stage | Status | Evidence |
| --- | --- | --- |
| D.0 freeze | `VALIDATED` | Source hashes and Stage 16-C failure ledger frozen. |
| D.1 semantics | `VALIDATED_WITH_GENERIC_FALLBACK` | Complete 321-step signed-proxy telemetry was regenerated; frozen source traces have no force/impulse fields, and confidence remains below 0.60. |
| D.2 environment | `STAGE16D_PHYSICS_CORRECTION_ENV_VALIDATED` | Real 1/128/4096-env CUDA PhysX smokes; 26-D action and 764-D observation. |
| D.3 optimization | `PARTIAL_BLOCKED` | 170105 terminal repair stayed 15/20; its only global fallback regressed to 12/20. |
| D.4 qualification | `PARTIAL_BLOCKED` | 20-replica empirical success 0.75/1.00; both fail source-relative runtime-proxy geometry. |
| D.5 single PPO | `NOT_RUN_GATE_BLOCKED` | No demonstration dataset, zero samples, no checkpoints. |
| D.6 two-clip PPO | `NOT_RUN_GATE_BLOCKED` | Both single-clip PPO gates are unmet. |
| D.7 export/audit | `PARTIAL_BLOCKED` | Partial V1 packages exist; V2 and sensitivity audit are not authorized. |

The two physics packages are ignored local artifacts under
`.local/physics_consistent_retargeting/hocap_170105/` and
`.local/physics_consistent_retargeting/hocap_170650/`. Each contains
`trajectory.npz`, `rollout.zarr`, `manifest.json`, `quality.json`,
`contacts.parquet`, `action_trace.npy`, and `comparison.csv`.

## Semantic result

Both clips use the shared low-confidence
`generic_contact_preserving_motion` fallback; there are no clip-specific
controller branches or reward profiles.

| Clip | Required groups | Contact onset | Minimum persistence | Source motion |
| --- | --- | ---: | ---: | --- |
| `hocap_170105` | index; thumb optional | 163–179 | 2 control steps | 0.246828 m, 17.378 deg |
| `hocap_170650` | index and pinky | 98–114 | 1 control step | 0.245653 m, 13.106 deg |

The task label is deliberately partial: the validated C3 traces contain only
six and two contact steps. They do not support a high-confidence task-specific
classification.

## Qualification result

| Clip | Success | Semantic | Contact | Causality | Stability | Formal runtime-proxy max / active-p95 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `170105` | 0.75 | 1.00 | 1.00 | 1.00 | 0.75 | 1.088 / 0.972 mm; source 0.014 / 0.014 mm; relative fail |
| `170650` | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.838 / 0.427 mm; source 0.188 / 0.187 mm; relative fail |

`RuntimeCollisionProxyPenetrationV1` queries every allowed 21-by-1 hand/object
convex pair with `python-fcl==0.7.0.11`. Positive is separation and negative is
overlap; per-frame aggregation selects the worst pair, and formal p95 uses only
contact-active per-frame-worst samples. Source and corrected trajectories use
the same proxies, transforms, backend, tolerance, and aggregation. Both clips
pass max <10 mm and active-p95 <=3 mm, but fail corrected <= source×1.10 plus
the frozen 0.5 micrometre numeric epsilon. Non-watertight visual meshes remain
unsigned diagnostics only. An empirical optimized seed is not a validated
trajectory and is not PPO data.

## Reproducible commands

The EULA variable is process-scoped and appears only on commands that launch
Isaac Sim. These commands use concrete repository paths and reproduce the
selected Stage 16-D flow.

```bash
conda run -n toporetarget-rl python scripts/rl/isaaclab/freeze_stage16d_inputs.py
conda run -n toporetarget-rl python scripts/rl/isaaclab/extract_stage16d_task_semantics.py \
  --output-root .local/reports/stage16d_physics_consistent_retargeting \
  --reference-time-scale 8 --overwrite

conda run -n toporetarget-isaaclab env OMNI_KIT_ACCEPT_EULA=YES \
  python scripts/rl/isaaclab/optimize_stage16d_physics_trajectory.py \
  --accept-eula --stage env-smoke --clip hocap_170105 --num-envs 128 \
  --output .local/reports/stage16d_physics_consistent_retargeting/env_smoke_128.json

conda run -n toporetarget-isaaclab env OMNI_KIT_ACCEPT_EULA=YES \
  python scripts/rl/isaaclab/optimize_stage16d_physics_trajectory.py \
  --accept-eula --stage d3-s3 --clip hocap_170105 --knots 16 \
  --population 64 --replicas 4 --iterations 5 --elites 12 \
  --output .local/reports/stage16d_physics_consistent_retargeting/optimizer_170105_s3.json

conda run -n toporetarget-isaaclab env OMNI_KIT_ACCEPT_EULA=YES \
  python scripts/rl/isaaclab/qualify_stage16d_trajectory.py --accept-eula \
  --clip hocap_170105 \
  --actions .local/reports/stage16d_physics_consistent_retargeting/optimizer_170105_s3.actions.npy \
  --replicas 20 \
  --trace .local/reports/stage16d_physics_consistent_retargeting/trajectory_trace_170105_v3.npz \
  --output .local/reports/stage16d_physics_consistent_retargeting/trajectory_qualification_170105_v3.json

conda run -n toporetarget-isaaclab python \
  scripts/rl/isaaclab/audit_stage16d_runtime_collision_geometry.py \
  --phase audit

conda run -n toporetarget-isaaclab python \
  scripts/rl/isaaclab/audit_stage16d_runtime_collision_geometry.py \
  --phase candidate --clip hocap_170105 \
  --corrected-trace .local/reports/stage16d_metric_qualification_and_ppo/trajectory_trace_170105_terminal_refined.npz \
  --candidate-label terminal_refined

conda run -n dp3 python scripts/rl/isaaclab/materialize_stage16d_trajectory.py \
  --clip hocap_170105 \
  --trace .local/reports/stage16d_physics_consistent_retargeting/trajectory_trace_170105_v3.npz \
  --actions .local/reports/stage16d_physics_consistent_retargeting/optimizer_170105_s3.actions.npy \
  --qualification .local/reports/stage16d_physics_consistent_retargeting/trajectory_qualification_170105_v3.json \
  --geometry .local/reports/stage16d_physics_consistent_retargeting/geometry_audit_170105_v3.json \
  --output .local/physics_consistent_retargeting/hocap_170105
```

The same commands with `170650` reproduce the second clip. BC, PPO, two-clip
training, V2 export, and sensitivity commands are implemented but exit
fail-closed while the geometry gate is blocked.

## Scope boundary

Factor-8 retiming changes time semantics. The six wrist joints are a virtual
3P+3R articulation, not a physical arm. Object mass, inertia, and friction are
engineering-nominal rather than physically calibrated. Qualified simulation
data are not robot data, and this stage makes no sim-to-real claim.

## D.4R2 attainability stop

The D.4R2 audit preserves every frozen source, factor-8, 26D/764D,
controller, physics, asset, and absolute 10 mm/3 mm contract. Numerical and
no-contact floors pass, but source-only dynamic/stable calibration does not
establish V1 under required contact topology or a 20-replica stable shared
floor for V2. The result is `STAGE16D_GEOMETRY_GATE_REVISION_BLOCKED`; no new
trajectory optimization or PPO is authorized. Exact python-fcl remains the
formal authority. The online signal and exact top-K design code is not a
qualification result and was not used for reward or gate acceptance.
