# Roadmap

## Current foundation and forward sequence

M0 is complete. F0 on `main` is complete: tracked Arti-MANO assets and the generic target-hand
contract are implemented and validated without changing Stage 7–9 mathematics, solver profiles,
or existing Stage 10 artifacts. The planned sequence is:

| Milestone | Scope | Status |
| --- | --- | --- |
| F0 main | Tracked robot assets and generic target-hand contract; Arti-MANO first registry instance | complete |
| P0 | Create `develop/pene-loss` and its worktree | next, not created by F0 |
| W0/W1 main | Wuji Hand2 Beta1 tracked assets, generic registration, bounded GRAB→Wuji validation | complete (bounded) |
| S1 `develop/pene-loss` | Generic SDF penetration loss | planned |
| I1 | Rebase pene-loss onto latest main; Arti/Wuji × baseline/SDF integration | planned |
| W2/W3 | At least three watertight-clip Wuji retargeting and Stage 10 export | planned |
| R0/R1 | MJCF playback and PPO tracking | later |
| CP | ContactPose formal evaluation | later |

F0 deliberately does not add Wuji Hand2, SDF penetration loss, `develop/pene-loss`, RL, or new
artifacts. See [`stages/F0_TARGET_HAND_FOUNDATION.md`](stages/F0_TARGET_HAND_FOUNDATION.md).

W0/W1 is now complete on `main`: both Wuji Hand2 Beta1 sides are tracked, registered through the
generic contract, and passed bounded Stage 7/8/9 construction checks. W2 is intentionally still
open and requires at least three watertight clips with full Stage 7–9 execution and contact/collision
audits. S1 remains isolated to `develop/pene-loss`, initially on Arti-MANO; I1 updates that branch
to current `main` and validates Arti-MANO plus Wuji under baseline/SDF conditions. W3 is export,
R0 is MJCF playback/PD, R1 is PPO tracking, and CP remains later.

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
| 9 | Constrained final optimization with slack variables | Eq. 8-9 implementation, QuerySets/slack, RH/LH bounded artifacts, independent audit | Constraint, determinism, source-integrity, and bounded 60-frame acceptance pass; assumptions remain explicit | complete (bounded, with explicit assumptions) |
| 10 | GRAB → Arti-MANO end-to-end retargeting | Resumable bounded DAG, contact-window selector, provenance, review, reference export | Implementation and focused tests pass; accepted bounded contact-rich reference-runtime run with human acceptance | implemented; bounded reference-runtime accepted; preferred performance, production, and real-time scopes open |
| 11 | Core Contract Freeze | Canonical HOI v2, DatasetAdapter v1, RobotHandPlugin v1, RobotReference v2, MetricRegistry v1, v1→v2 migration | Contracts, registries, compatibility tests, bilingual docs, and immutable Stage 10 migration report pass | complete |
| Q1–Q3 | Multi-dataset interaction benchmark and unified automatic evaluation | Frozen selection, Eq. 10-12 registry, GRAB proxies, automatic gates, baseline execution, aggregation, reports, dashboard | Selection is frozen before runs; applicable profiles execute or preserve failures; reports and integrity checks are generated | implemented, bounded; current local ContactPose gate blocks before freeze |
| Q4 | Morphology-aware warm-start | Morphology gap analysis and separately versioned seed/prior candidates | Four-clip evidence without changing paper-core Eq. (1)–(9) | complete (bounded, diagnostic extension) |
| Q5 | Arti-MANO surface contact proxies | Generic robot surface-region interface and deterministic Arti-MANO profile | Proxy is distinct from source ground truth and collision samples | complete (bounded, paper-external) |
| Q6 | Contact-aware final extension | Fixed contact candidate grid, source/object-local targets, Huber/direction diagnostics | Separate method, tests, rejection lineage, and rollback evidence | complete (bounded; no accepted contact extension) |
| Q7 | Cross-trajectory automatic profile selection | Frozen A–E 2×2 matrix, hard/regression/improvement/Pareto gates | Selection uses only predeclared cross-trajectory metrics | complete (bounded; baseline fallback retained) |
| 12 | Dataset Adapter Expansion | OakInk, DexYCB and HO-Cap adapters plus paused final-job controls/performance repair | P2 v2 five-frame evidence is available; batch final remains paused pending user approval and long-clip qualification | in progress; final queue paused |
| 13 | Complex HOI Expansion | ARCTIC, OakInk2 and TACO adapters | Articulated/complex HOI remains canonical and validated | DEFERRED |
| 14 | Universal Robot Hand Plugin | Broader arbitrary-hand URDF/MJCF plugin validation | Plugins satisfy RobotHandPlugin v1 and reference export gates | DEFERRED |
| 15 | Baseline Comparison | OmniRetarget, Mink, DexPilot, GeoRT | Frozen fair baseline runs and reports | DEFERRED |
| 16-A | Paper/minimal reference tracking | Base-relative 20D residual controller | Appendix A.5 MDP/PPO implementation and frame-0 controllability qualification | Preserved: 16.0 functional complete; 16.1a confirms current fixed-base/20D contact timing is dynamically infeasible; 16.2/16.3 not authorized or started |
| 16-B.0 | `ENGINEERING_EXTENSION` world-reference export | World wrist/object/links at 20 Hz plus wrist-relative reconstructions | Direct Stage-12 export validates provenance, quaternion convention, ordering, and world-to-wrist reconstruction | complete, bounded |
| 16-B.1 | Finite-wrench wrist controller | Shared 6D finite-wrench wrist plus 20D fingers | W2 controller gates pass without teleport or direct object control | complete |
| 16-B.1b | Fixed-horizon per-trajectory controllability | Frozen H1/H5/H10 matrix | At least one bounded controller per trajectory | complete: H5 passes `170105`; H10 passes `170650` |
| 16-B.1c | Shared adaptive multi-horizon MuJoCo oracle | Shared state-only H1/H5/H10 selector, terminal contraction, 20-episode frame-0 gate | Both clips pass one shared bounded oracle | partial / closed: selected 32x3 has `170650` 20/20 pass and `170105` 0/20 at 80%; bounded 48x4 `170105` reaches 82.5% with 5.002 cm axis error |
| 16-B.2 | MuJoCo single-clip PPO | One gated 26D policy per clip | Requires B.1c validation | blocked / `NOT_STARTED_GATE_BLOCKED`; 0 samples, no checkpoints |
| 16-B.3 | MuJoCo two-clip PPO | Balanced two-clip 26D policy | Requires both single-clip gates | TODO / not started |
| 16-B.4 | Full domain randomization | World-wrist DR suite | Gated by a qualified two-clip nominal policy | TODO |
| 16-B.5 | Geometry/zero/PPO comparison | Comparable trajectory/contact audit | Gated by a qualified PPO rollout | TODO |
| 16-C.0 | Isaac Lab platform qualification | Host, isolated install, Isaac Sim/Lab, GPU PhysX, headless and 128-env vector smoke | Platform-only evidence; no custom task/PPO | COMPLETE WITH LIMITATIONS: all hard gates pass; no display |
| 16-C.1 | Wuji and object asset migration | Wuji Hand2 Beta1 plus HO-Cap 170105/170650 | Floating articulation, rigid dynamics, contact and 1/128-env CUDA evidence | COMPLETE / `STAGE16C1_ISAACLAB_ASSET_MIGRATION_VALIDATED` |
| 16-C.2 | Isaac Lab `DirectRLEnv` | Stage-16 task shell | Asset migration complete | COMPLETE / `STAGE16C2_DIRECT_RL_ENV_VALIDATED` |
| 16-C.3 | Single-environment semantic parity | Action/observation/reward/termination parity | DirectRLEnv exists | COMPLETE / `STAGE16C3_SEMANTIC_QUALIFICATION_VALIDATED`: one authorized global factor-8 retiming preserves both source hashes and all 41 source keys in a derived 321-sample view; C3-0 through C3-5 pass with the unchanged shared bounded wrist profile |
| 16-C.4 | GPU vectorization benchmark | Stage-16 task throughput | Semantic parity passes | COMPLETE / `STAGE16C4_GPU_VECTOR_BACKEND_VALIDATED`: all 128/512/1024/2048/4096 aggregate-contact counts exit clean/finite; selected 4096 envs x rollout 16 = 65536 samples/update |
| 16-C.5A | Distributional candidate replication | Natural 20-replica baseline, frozen distribution metrics, persistent GPU pool | C.3/C.4 qualified; R3 deterministic failure retained | PARTIAL / `REPLICATION_DISTRIBUTION_FAIL`: pre-contact passes both clips, all contact-bearing phases fail frozen natural envelopes; 384/576/768 pools pass and 384 is selected |
| 16-C.5B | Multi-horizon robust CEM | H1/H5/H10, 32x3x4 persistent evaluation, lexical robust selector | C.5A-R4 distribution gate passes | IMPLEMENTED / GATE-BLOCKED: B0/B1 and two 30-step B2 runs complete; both B2 outcomes have failure probability 1.0, so B3 is `NOT_STARTED_GATE_BLOCKED` |
| 16-C.5C | Two-clip formal robust qualification | New B3 trace plus 20 fresh frame-zero PhysX episodes per clip | Both B3 full rollouts complete | NOT STARTED / gate-blocked by C.5A-R4 and B2; `STAGE16C5_PHYSX_ROBUST_ORACLE_PARTIAL` |
| 16-C.6 | Single-clip GPU PPO | Per-clip PPO in Isaac Lab | PhysX oracle passes both clips | NOT AUTHORIZED; 0 samples, 0 checkpoints |
| 16-C.7 | Two-clip GPU PPO | Shared policy | Both single-clip policies pass | TODO |
| 16-C.8 | Dynamics randomization | Bounded PhysX DR | Nominal policy qualified | TODO |
| 16-C.9 | Geometry/MuJoCo/Isaac/PPO comparison | Cross-backend diagnostic comparison | Qualified evidence exists | TODO |
| 17 | Paper Experiment Reproduction | Tables/Figures | Full result report with provenance | not started |
| 18 | Performance Optimization | Packaging and benchmarks | Release criteria pass | not started |
| 19 | Non-paper Extensions | MANO cleanup, SPIDER, other extensions | Separately labeled extensions | not started |

MuJoCo is not removed: it remains the correctness, deterministic-regression,
contact-diagnostic, action-replay, and interactive-visualization backend.
Isaac Lab is the planned GPU-parallel training backend. MuJoCo and PhysX are
not required to match bitwise, but semantic parity and a new PhysX oracle gate
are required before any Isaac Lab PPO; MuJoCo evidence cannot authorize it.

Stage 16-C.0 freezes the independent platform stack to Python 3.11.15, Isaac
Sim 5.1.0, Isaac Lab v2.3.2 at exact commit
`37ddf626871758333d6ed89cf64ad702aef127d0`, and Torch 2.7.0 cu128. Its
qualification is limited to host/install/import, finite official GPU-PhysX,
headless, viewer, and 128-environment evidence. Its current result is
`STAGE16C0_ISAACLAB_PLATFORM_VALIDATED_WITH_LIMITATIONS`; all hard gates pass,
while viewer review is unavailable without a display. C.1 independently
validates the floating Wuji Hand2 Beta1 articulation, both frozen HO-Cap
objects, named PhysX contact pairs, CUDA tensors, and 128-env subset reset.
C.2 is `STAGE16C2_DIRECT_RL_ENV_VALIDATED` with real 1/128-env GPU evidence.
C3R4 remains immutable original-timing evidence: the explicit serial 3P+3R
architecture exhausted computed-torque and bounded-preview paths after the
false MPC reporter termination was repaired; independent 1/6-step holdout and
both 41-frame MPC gates still failed. The later user-authorized C3R5 structural
choice applies one factor-8 retiming globally to both clips, preserving the two
source hashes and every one of the 41 source keys in a derived 321-sample 20 Hz
view. With unchanged gains, effort bounds, 26-D action, 764-D observation and
acceptance gates, `high_authority_bounded` passes both clips, contact causality,
and C3-0 through C3-5 as `STAGE16C3_SEMANTIC_QUALIFICATION_VALIDATED`. This does
not turn the abstract fixed-anchor wrist into a real arm. C.4 passes all five
formal counts as `STAGE16C4_GPU_VECTOR_BACKEND_VALIDATED`; 4096 environments
sustain 700.35 samples/s under shared GPU load with 3731 MiB process-VRAM peak
and zero contact warnings. C.5A-R1 validates the frozen inputs, candidate-state
contract, repaired harness, exact single-environment replication, origin
normalization, cross-process controls, and read-only contact telemetry.
C.5A-R3 then finds T0/T1 passing but T2, every natural T4 shard, and every
natural T5 shard (including 8x12) failing raw and derived gates. The result is
`TRUE_CONTACT_SOLVER_NONDETERMINISM`; T3 staggered starts are diagnostic only.
R3 implements an independent frame-zero robust contract, but C5C's unchanged
20-replica physical gate fails for both selected traces. R4 freezes a
20-replica distributional baseline and all seven metrics before candidate
results. Both clips pass pre-contact but fail contact-onset, sustained-contact,
and post-contact envelopes. Persistent 384/576/768 pools pass their mapping and
resource gates; 384 is selected. The H1/H5/H10 robust CEM is implemented and
B0/B1 plus two 30-step B2 runs complete, but both B2 runs finish with formal
failure probability 1.0. B3 and the new C5C are therefore
`NOT_STARTED_GATE_BLOCKED`. The exact status is
`STAGE16C5_PHYSX_ROBUST_ORACLE_PARTIAL`: C.6/PPO is not authorized and remains
started=false, samples=0, checkpoints=0, without tolerance softening or
solver/reference/controller/reward/termination mutation.

The Wuji three-clip implementation is available through the generic
`workflow run-grab-suite` command; its completion status is determined only by
the runtime `final_status.json` under the experiment root.
## W2.1 Wuji continuity repair

The frozen Wuji Hand2 three-clip baseline is retained alongside the
engineering-only continuous profile. See
`docs/stages/W2_1_WUJI_CONTINUITY_REPAIR.md` for the state-chart, acceptance,
retry, and five-frame-window contract. This milestone does not establish
cross-subject generalization or RL readiness.

## W2.2 Wuji continuity closeout

W2.2 is complete as a diagnostic closeout, not as a recommendation. The
closeout includes W2 q-step attribution, isolated B0/B1/B2 transport-versus-
temporal evidence, seven fixed anomaly windows, synthetic routing, a real W3
five-frame shadow, deterministic replay, HTML review, and artifact-integrity
checks. The formal trajectories remain immutable. The recorded status is
`WUJI_CONTINUOUS_PROFILE_NOT_RECOMMENDED_WINDOW_FALLBACK_FAILED` because the
real window returns SLSQP status 4 and fails center continuity; a W3
penetration-rate regression independently fails the quality gate. See
`docs/stages/W2_2_WUJI_CONTINUITY_CLOSEOUT.md`. Further work must resolve
these gates before the continuous profile is used for offline reference
generation.

## W2.3 Wuji sequential finalization

W2.3 adds `wuji_continuous_sequential_v1` as a separately audited offline
candidate. It disables production window fallback, isolates the five-frame
repair/shadow, reruns the frozen W1/W2/W3 evidence with selected replay and
multi-threshold collision gates, and exports only versioned artifacts under
`.local/experiments/wuji_hand2_continuous_v1/w2_3_finalization/`. The result
does not establish RL, real-time, cross-subject, or author-exact validity.
# P3 compiled ambiguous spatial-FD

The portable compiled CPU probe kernel remains experimental: its measured
five-frame overall benefit is below the merge threshold.

# P4 certified compiled exact sign

P4 adds optional exact float64 compiled generalized winding and certified
FD-probe sign reuse. Near-threshold winding retains the qualified reference
fallback. It is experimental and non-default pending fixed-frame and 60-frame
qualification; Stage-12 remains untouched.

# Stage-12 ContactPose mug solver closeout

The recoverable status-8/9 path reconstructs representable active-set slack at
fixed q/base and reruns the unchanged strict solver/audit. This generic solver
engineering is not a paper-method extension and does not establish the
ContactPose Eq.10/Eq.11 benchmark.

## Stage 16-D physics-consistent retargeting closeout

Stage 16-C strict object-trajectory tracking is `PARTIAL / CLOSED WITH
EVIDENCE`. The Stage 16-D progression is:

| Stage | Result |
| --- | --- |
| D.0 contract/freeze | `VALIDATED` |
| D.1 task/contact semantics | `STAGE16D_TASK_SEMANTICS_PARTIAL` |
| D.2 physics-correction environment | `STAGE16D_PHYSICS_CORRECTION_ENV_VALIDATED` |
| D.3 corrected trajectories | both `PARTIAL_BLOCKED` |
| D.4 qualification/V1 export | both `PARTIAL_BLOCKED` |
| D.5 single-clip PPO | both `NOT_RUN_GATE_BLOCKED` |
| D.6 two-clip PPO | `NOT_RUN_GATE_BLOCKED` |
| D.7 V2/sensitivity | `PARTIAL_BLOCKED` |

The source object path is now a soft prior, not a hard target; source NPZs and
Stage 12 results remain frozen. The corrected object motion comes only from a
free PhysX rollout. The formal blocker is a non-comparable lower-bound
penetration audit on non-watertight visual meshes. Factor-8 changes timing, the
virtual wrist is not a physical arm, physical provenance is unresolved,
qualified simulation data are not robot data, and no sim-to-real claim is
made. The next gate is a signed, metric-compatible runtime-geometry audit.
