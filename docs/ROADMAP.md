# Roadmap

| Stage | Objective | Major deliverables | Definition of done | Status |
| --- | --- | --- | --- | --- |
| 0 | Repository creation and architecture | Package, CLI, path policy, dataset discovery, Arti-MANO importer | CI-safe scaffold and local doctors pass | complete |
| 1 | Paper fidelity audit | Full PDF manifest, equations/tables/figures, configs, assumptions | Fidelity checker and audit report pass | complete |
| 2 | Canonical HOI schema and coordinate conventions | Schema, lazy storage, comparison visualization, bounded GRAB inspection | Stage 2A and bounded Stage 2B real-data acceptance pass | complete (bounded scope) |
| 3 | MANO to MediaPipe-style 21-keypoint conversion | Explicit layout/profile registry, converter, reports, visualizations, synthetic and bounded real-GRAB validation | Scene-frame converter validated with source/object/timestamp preservation; assumptions remain explicit | complete (bounded, with assumptions) |
| 4 | Generic robot-hand kinematics and Arti-MANO target adapter | RobotHandSpec/Model, strict URDF parser, differentiable/reference FK, MediaPipe21 anchors, geometry inspection, RH/LH CLI and reports | Synthetic and both real Arti-MANO sides validate; assets remain unchanged; docs and fidelity record pass | complete (with explicit assumptions) |
| 5 | GRAB dataset adapter | Lazy index, native single-sequence/bimanual adapter, contacts, validation, provenance, comparison, interactive viewer | Real right/both-hand clips convert at native time, raw/canonical checks pass, and no raw source is modified | complete (bounded); fresh semantic closeout passed |
| 6 | Object surface sampling, collision geometry and SDF | Mesh audit, deterministic 50-point anchors, collision-only robot samples, closest-point/SDF backends, probes, reports, visualizations | Synthetic/default checks, bounded GRAB object, RH/LH Arti-MANO, source integrity, and fidelity pass; later graph/optimization remains out of scope | complete (bounded, with assumptions) |
| 7 | Relative bone-direction initialization | Eq. 1-2 implementation, frame audit, sequential warm-start artifact | Public/default tests, bounded RH/LH real acceptance, validation, visualization, and fidelity pass | complete (with explicit assumptions) |
| 8 | Shared interaction graph and Laplacian coordinates | Eq. 3-7 implementation, source-only artifacts, RH/LH reports and views | Graph/loss tests, source integrity, identity/Jacobian checks, and bounded 60-frame acceptance pass; Eq. 8-9 remains out of scope | complete (bounded, with explicit assumptions) |
| 9 | Constrained final optimization with slack variables | Eq. 8-9 implementation | Constraint tests pass | not started |
| 10 | GRAB → Arti-MANO end-to-end retargeting | Pipeline | Reproducible trajectory | not started |
| 11 | Metrics and ContactPose evaluation | Eq. 10-12 | Metric fixtures and report | not started |
| 12 | OakInk, DexYCB and HO-Cap adapters | Dataset adapters | Adapters validated | not started |
| 13 | ARCTIC, OakInk2 and TACO extensions | Dataset adapters | Adapters validated | not started |
| 14 | Arbitrary dexterous-hand plugin interface | URDF/MJCF interface | Plugin contract tested | not started |
| 15 | Baselines and ablations | OmniRetarget, Mink, DexPilot, GeoRT | Fair baseline runs | not started |
| 16 | Reference-tracking PPO | RL controller | Training/eval pipeline | not started |
| 17 | Paper experiment reproduction | Tables/Figures | Result report | not started |
| 18 | Performance optimization and v1.0 release | Packaging and benchmarks | Release criteria pass | not started |
| 19 | Non-paper extensions | MANO cleanup, SPIDER, other extensions | Separately labeled extensions | not started |
