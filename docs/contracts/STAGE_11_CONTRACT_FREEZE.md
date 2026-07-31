# Stage 11 contract freeze

Stage 11 freezes the interfaces used by future dataset adapters, robot-hand
plugins, playback, and reference-tracking consumers. It does not add a
dataset, invoke retargeting, change a solver, or change a paper formula.

## Contract map

| Contract | Python entry point | Serialized version | Current instances |
| --- | --- | --- | --- |
| Canonical HOI | `toporetarget.contracts.canonical` | `toporetarget.hoi.v2` | GRAB migrated from v1 |
| Dataset adapter | `toporetarget.contracts.dataset` | `toporetarget.dataset_adapter.v1` | GRAB |
| Robot hand plugin | `toporetarget.contracts.robot` | `toporetarget.robot_hand_plugin.v1` | Arti-MANO, Wuji Hand2 Beta1 |
| Robot reference | `toporetarget.contracts.reference` | `toporetarget.robot_reference.v2` | NPZ, Zarr |
| Metric registry | `toporetarget.contracts.metrics` | `toporetarget.metric_registry.v1` | paper exact, proxies, geometry, diagnostics |

Future code must register a dataset or robot through the corresponding registry.
It must not add dataset- or robot-name conditionals to the canonical schema,
retarget solver, or reference exporter. Dataset proxy evidence must remain
declared as `DATASET_PROXY`; it is never ground truth.

## Compatibility

`migrate_v1_to_v2()` reads an existing `toporetarget.hoi.v1` cache and returns a
v2 facade without modifying the source. `load_canonical_hoi()` accepts both
unmarked historical v1 caches and v2-marked copies. Existing
`toporetarget.data.*`, `toporetarget.robots.*`, and metric imports remain valid.

`RobotReferenceV2` stores `qpos_reference`, scene-frame base pose, object pose in
robot-base coordinates, tracked link positions in robot-base coordinates,
timestamps/FPS, explicit joint order, robot hash, and dataset provenance. Both
NPZ and repository-compatible Zarr output are supported.

## Readiness boundary

The current Wuji status is offline reference generation ready. It is not RL
ready, not real-time ready, and not cross-dataset validated. Those claims belong
to later stages and require their own evidence gates.
