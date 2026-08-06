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
| D.1 semantics | `STAGE16D_TASK_SEMANTICS_PARTIAL` | Both C3 contact traces are sparse and below the 0.60 confidence gate. |
| D.2 environment | `STAGE16D_PHYSICS_CORRECTION_ENV_VALIDATED` | Real 1/128/4096-env CUDA PhysX smokes; 26-D action and 764-D observation. |
| D.3 optimization | `PARTIAL_BLOCKED` | Shared 16-knot, population-64, four-replica, five-iteration spline CEM produced two candidates. |
| D.4 qualification | `PARTIAL_BLOCKED` | 20-replica empirical success 0.75/1.00, but formal penetration comparability is blocked. |
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

| Clip | Success | Semantic | Contact | Causality | Stability | Penetration diagnostic |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `170105` | 0.75 | 1.00 | 1.00 | 1.00 | 0.75 | max lower bound 1.553 mm; p95 0 mm |
| `170650` | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | max lower bound 1.117 mm; p95 0.096 mm |

These penetration numbers are lower bounds from the runtime convex collision
proxy, not signed upper bounds. The original OBJ files are non-watertight, and
the Stage 12 SDF uses a different geometry, scale, and metric. Therefore the
formal gate is `BLOCKED_METRIC_COMPARABILITY_AND_VISUAL_SIGN` for both clips.
An empirical optimized seed is not a validated trajectory and is not PPO data.

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

conda run -n toporetarget-rl python scripts/rl/audit_stage16d_geometry.py \
  --clip hocap_170105 \
  --trace .local/reports/stage16d_physics_consistent_retargeting/trajectory_trace_170105_v3.npz \
  --source-stage12 .local/experiments/stage12_dataset_validation_v4/stage12_v4_20260730T155200Z_b31d179_de6ba696_13d502c3_r7_active_jacobian/hocap/hocap_subject_1_20231025_170105/final/final_refinement_fast_exact_v2_r1/final_retarget.zarr \
  --output .local/reports/stage16d_physics_consistent_retargeting/geometry_audit_170105_v3.json

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
