# Paper fidelity audit

## 1. Paper identity and PDF hash

The audited source is *TopoRetarget: Interaction-Preserving Retargeting for Dexterous
Manipulation*, arXiv `2606.16272` v2. The local PDF is `docs/TopoRetarget.pdf`, 16 pages,
SHA-256 `21c06a125430854dcff0d778283963b7fe107c8dfa79e3982639a80c21b206ab`.

## 2. Scope of the strict reproduction

Stage 1 audits the complete PDF, including Sections 3.1–3.4, Section 4, Sections 5–7, Equations
1–12, Figures 1–5, Tables 1–6, and Appendices A.1–A.5. The machine-readable index is
[`PAPER_FIDELITY.yaml`](PAPER_FIDELITY.yaml). The numerical method, simulator, datasets, and
baselines are not implemented in this stage.

## 3. Equation-to-code traceability

Equations 1–9 map to `src/toporetarget/retarget/` and are implemented with explicit assumptions;
Equations 3–7 provide the frozen interaction graph/Laplacian term, and Equations 8–9 provide the
bounded constrained final refinement. Equations 10–12 remain future ContactPose and penetration
metrics. Each entry records its PDF page,
known values, unknowns, assumptions, and future implementation/test targets in the YAML manifest.

## 4. Table-to-config traceability

Retargeting values from Table 3 are in `configs/paper/retarget.yaml`; MDP/reward and termination
values from Table 4 and RL/domain-randomization/PPO values from Tables 5–6 are in
`configs/paper/rl.yaml`. ContactPose metric definitions are in `configs/paper/metrics.yaml` and
baseline adaptations are in `configs/paper/baselines.yaml`. The reproduction notes transcribe all
rows of Tables 1–6 rather than referring to them only by name.

## 5. Figure reproduction plan

`docs/reproduction/figure1.md` through `figure5.md` record each figure's purpose, required data
and code, blocker, and expected output. No synthetic figure is presented as a paper reproduction.

## 6. Dataset dependencies

The paper uses ContactPose (25 of 28 grasps), Ho-cap (32 clips), and a self-collected MoCap
Pen-Spin set (32 clips, average 12.4 s). These are external and are never copied into this
repository. Stage 0 only discovers registered dataset paths under the configured storage root.

## 7. Baseline dependencies

OmniRetarget, Mink, DexPilot, and GeoRT are recorded in `configs/paper/baselines.yaml`. The
OmniRetarget dexterous adaptation follows Appendix A.2: MediaPipe keypoints replace humanoid
keypoints, the wrist/base is optimized rather than fixed, and collision uses full hand geometry.
No upstream repository is copied in this stage.

## 8. Stage 3 and Stage 4 repository boundaries

The paper accepts MediaPipe-style 21-point input but does not disclose a MANO-to-MediaPipe
conversion module. Stage 3 therefore implements a repository-local, explicit source-hand adapter
with assumptions: named MANO joints plus audited 778-vertex tip anchors become scene-frame
`mediapipe21`; the original source track and all object/timestamp data remain available. This is
tracked as `implemented_with_assumptions`, not as an implementation of Equations 1–9, interaction
graphs, robot mapping, or retargeting optimization. See [`MANO_TO_MEDIAPIPE21.md`](MANO_TO_MEDIAPIPE21.md)
and [`stages/STAGE_3_MANO_MEDIAPIPE21.md`](stages/STAGE_3_MANO_MEDIAPIPE21.md).

Stage 4 now implements the target-hand side required to evaluate `P^r(q)`: a generic strict URDF
parser, differentiable Torch FK, independent NumPy FK, explicit named qpos order, separate visual
and collision geometry instances, canonical MediaPipe21-compatible robot anchors, and real
Arti-MANO RH/LH validation. It is tracked as `stage4_robot_keypoint_forward_kinematics` with status
`implemented_with_assumptions`, not as implementation of the paper's retargeting equations. The
engineering base is URDF root `palm`; the paper's exact robot wrist-centered orientation remains
`A_ROBOT_HAND_FRAME_001`. See [`ROBOT_HAND_INTERFACE.md`](ROBOT_HAND_INTERFACE.md),
[`ARTIMANO_ADAPTER.md`](ARTIMANO_ADAPTER.md), and
[`stages/STAGE_4_ARTIMANO_TARGET_HAND.md`](stages/STAGE_4_ARTIMANO_TARGET_HAND.md).

## F0 tracked target-hand foundation

F0 is repository infrastructure, not a new paper-method implementation. It moves the audited
Arti-MANO RH/LH payload into `third_party/robot_hands/artimano/`, records upstream provenance and
license evidence, and adds a data-driven `RobotHandSpec`/asset-bundle contract with registry
resolution and compatibility checks. The tracked URDFs rebase mesh filenames only; the F0 reports
record exact payload, topology, FK, anchor, Jacobian, and mesh-transform equality against the
historical local tree. F0 does not change Equations 1–9, add Wuji, add a penetration loss, alter
solver profiles, or create new Stage 10 artifacts. See [`ROBOT_HAND_TARGET_CONTRACT.md`](ROBOT_HAND_TARGET_CONTRACT.md),
[`THIRD_PARTY_ASSET_POLICY.md`](THIRD_PARTY_ASSET_POLICY.md), and the ignored reports under
`.local/reports/f0/`.

## W0/W1 Wuji Hand2 Beta1 target boundary

W0/W1 adds a second generic `RobotHandSpec`/registry instance for the pinned Wuji Hand2 Beta1 RH/LH
body assets. It records source provenance, explicit 20-DoF orders, URDF/MJCF consistency, semantic
anchors, visual/collision separation, and bounded generic Stage 7/8/9 construction evidence. This
is an engineering target-hand integration; it does not add a Wuji-specific adapter or solver, does
not change Equations 1–9 or solver profiles, and does not reproduce Wuji hardware, calibration,
MJCF playback, PPO, or the paper's zero-shot claim. At least three watertight clips are required for
the future W2 full-retargeting milestone. See [`WUJI_HAND2_BETA1_TARGET.md`](WUJI_HAND2_BETA1_TARGET.md),
[`WUJI_HAND2_SEMANTIC_MAPPING.md`](WUJI_HAND2_SEMANTIC_MAPPING.md), and
[`stages/W0_W1_WUJI_HAND2_INTEGRATION.md`](stages/W0_W1_WUJI_HAND2_INTEGRATION.md).

## 9. Stage 4 target-hand boundary

Stage 4 implements repository infrastructure for `P^r(q)`: a generic YAML/URDF robot-hand
specification, differentiable Torch FK, an independent NumPy reference path, explicit visual and
collision geometry instances, and MediaPipe-21-compatible Arti-MANO RH/LH joint/link anchors.
It is validated with synthetic URDF fixtures and the tracked Arti-MANO vendor assets. The paper
does not publish the target-hand anchors, qpos ordering, or exact robot wrist frame, so the adapter
is `implemented_with_assumptions`; it does not implement source-to-robot qpos retargeting, Eq. 1-9,
or optimization. Stage 6 adds a separately scoped surface-sampling, collision-query, and SDF
foundation without adding an interaction graph. See
[`ROBOT_HAND_INTERFACE.md`](ROBOT_HAND_INTERFACE.md), [`ARTIMANO_ADAPTER.md`](ARTIMANO_ADAPTER.md),
and [`stages/STAGE_4_ARTIMANO_TARGET_HAND.md`](stages/STAGE_4_ARTIMANO_TARGET_HAND.md).

## Stage 7 warm-start boundary

Stage 7 implements the displayed Eq. (1) adjacent-direction residual and Eq. (2) sequential
objective. Its assumptions are the 20-bone/15-pair semantic profile,
`canonical_keypoint_wrist_v1`, raw 22-joint radians, neutral first frame, previous-frame temporal
reference, direct URDF bounds, float64 SciPy TRF with Torch-autograd Jacobian, and a post-solver
canonical-frame base seed. The local direction objective's base observability is measured rather
than hidden. These choices are not presented as paper-exact facts. The independent artifact is
`toporetarget.warm_start.v1`; it is an initialization output, not final retargeting. See
[`stages/STAGE_7_BONE_DIRECTION_WARM_START.md`](stages/STAGE_7_BONE_DIRECTION_WARM_START.md).

## Stage 7.1 warm-start audit boundary

Stage 7.1 is an engineering audit of the Arti-MANO warm-start, not a new paper
objective. It verifies source semantic mapping, robot anchors/thumb URDF axes,
canonical frames, the non-paper base seed, persisted Eq. (1)/(2) replay, and
per-finger warm/final attribution. The formal Stage 7 target remains relative
bone-direction only; contact fidelity is not silently added to it. Raw source
targets and robot-length reconstructed targets are separate diagnostic evidence,
and local Jacobian/workspace sampling is not a global reachability certificate.
Diagnostic IK is `paper_method=false`, `accepted_reference=false`, and all
diagnostic solver calls are isolated from official artifacts. See
[`WARM_START_FIDELITY_AND_REACHABILITY_AUDIT.md`](WARM_START_FIDELITY_AND_REACHABILITY_AUDIT.md)
and its Chinese counterpart.

## 11. Stage 6 geometry foundation

Stage 6 implements three traceable engineering foundations with explicit assumptions:

- `object_surface_sampling_foundation`: paper-locked `N_o=50`, deterministic area-weighted
  triangle selection, PCG64 seed `20260720`, face+barycentric anchors, and fixed object-local
  temporal identity. The sampler, seed, temporal schedule, and diagnostic face normals are not
  disclosed by the paper.
- `signed_distance_query_foundation`: exact reference point-to-triangle closest points, strict
  watertight sign, generalized winding sign confidence, and an explicit unsigned-only mode. The
  convention is always positive outside; unsigned values are never relabeled as signed.
- `robot_collision_surface_foundation`: collision-only samples from the existing URDF geometry
  API, with engineering profile `32` samples per geometry. Visual-only links and tip spheres are
  reported, not silently synthesized.

Stage 6 does not construct the final `Q_t`, Delaunay tetrahedra, Laplacian coordinates, slack
variables, constraints, or optimizer. See
[`stages/STAGE_6_OBJECT_GEOMETRY_SDF.md`](stages/STAGE_6_OBJECT_GEOMETRY_SDF.md).

## 11. Unpublished implementation details

The solver, Delaunay backend/flags, SDF backend, first-frame seed, paper coordinate-frame details,
ContactPose intensity threshold, robot surface sampling, tracked links, axis-point geometry,
simulator/physics settings, low-level gains, and unlisted PPO values are explicitly registered in
[`ASSUMPTIONS.md`](ASSUMPTIONS.md). Configuration leaves undisclosed values null.

## Stage 9 final-refinement boundary

Stage 9 implements the paper's Eq. (8)-(9) objective and signed-distance/slack constraint contract
for the bounded RH/LH `s7/cubemedium_inspect_1`, `[0,60)` acceptance window. It uses explicit local
seed-delta coordinates, sequential warm-started solves, full and adaptive collision QuerySets,
float64 SLSQP, and an independent Stage 6 reference SDF audit over all 512 robot collision samples.
The optimizer, QuerySet construction, collision sample density, derivative policy, and termination
values are not disclosed by the paper and remain tracked as `implemented_with_assumptions`; this is
not a claim of result-level reproduction or Stage 10 end-to-end completion. See
[`FINAL_REFINEMENT_OPTIMIZATION.md`](FINAL_REFINEMENT_OPTIMIZATION.md) and
[`stages/STAGE_9_FINAL_CONSTRAINED_REFINEMENT.md`](stages/STAGE_9_FINAL_CONSTRAINED_REFINEMENT.md).

## Stage 9.2 execution boundary

Stage 9.2 adds only engineering execution machinery around the frozen contract:
immutable per-frame context, exact-x evaluation reuse, persistent exact reference-SDF
AABB resources, batched collision Jacobians, scheduled independent full-surface audits, atomic
strict-accepted checkpoints, soft pause/resume, and final artifact assembly.
The paper does not specify these mechanisms; they are recorded as
`A_REFINEMENT_EVALUATION_CACHE_001`, `A_REFINEMENT_CHECKPOINT_RESUME_001`,
`A_REFINEMENT_EXECUTION_DEVICE_001`, `A_REFINEMENT_SOLVER_VARIABLE_SCALING_001`,
`A_REFINEMENT_REFERENCE_SDF_ACCELERATION_001`, `A_REFINEMENT_BATCHED_JACOBIAN_001`,
`A_REFINEMENT_FULL_AUDIT_SCHEDULING_001`,
`A_REFINEMENT_PERFORMANCE_GATE_001`, and
`A_REFINEMENT_WALL_TIME_PAUSE_001`. They do not change Eq. (8)-(9), the SLSQP
profile, or the paper's 4.70 ms/frame reference claim.

## Stage 10 workflow boundary

Stage 10 adds only the bounded orchestration/provenance layer that composes the implemented source,
interaction, and refinement stages into an inspectable robot reference trajectory. Its contact-window
thresholds, cache policy, final sanity warning, and human-review gate are engineering assumptions; they
do not change Equations 1–9. A real contact-rich solve and human review are still required before the
workflow can be called accepted. Equations 10–12, ContactPose evaluation, RL, physics, and baselines
remain explicitly unimplemented.

## Stage 9.3.1 metric-reconciliation boundary

Stage 9.3.1 is an engineering audit, not a new paper method. It preserves
Equations 1-9, the paper weights, the Stage 9.2 solver profile and artifacts,
and the Stage 10 manifest/export/manual-acceptance boundary. It defines a
single reference signed-distance comparison for the persisted 512 collision
points, keeps the old solver-only convex-hull report diagnostic-only, and
replays the existing hard/soft/slack acceptance contract without running a
solver. Its bounded shadow boundary is limited to at most three representative
frames and is fail-closed; no profile is paper-faithful evidence unless the
reconciliation gate passes. The current accepted-window audit blocks on the
legacy backend mismatch, so Stage 9.4 is not entered. See
[`CONTACT_METRIC_RECONCILIATION_AND_SHADOW_ABLATION.md`](CONTACT_METRIC_RECONCILIATION_AND_SHADOW_ABLATION.md).

## 13. Current blockers

The private Pen-Spin data, Wuji deployment assets, target hand identity for Table 1, and several
solver/geometry/RL details are unavailable from the paper. These blockers prevent result-level
reproduction and are not silently resolved.

## 14. Definition of method-complete

Method-complete means all publicly specified method equations, configurations, constraints,
metrics, and evaluation code are implemented and tested, with every remaining assumption
explicitly resolved or marked as a deliberate extension. This repository is not method-complete.

## Stage 9.3.2 canonical contact audit boundary

Stage 9.3.2 is an engineering audit, not a new paper method. Formal contact
evaluation is pinned to the versioned reference winding SDF; solver-side
acceleration remains a separate profile. The old convex-hull Stage 9.3
values are diagnostic-only and superseded for formal contact claims. Dense
visual samples and retention values are proxies, and open visual meshes do
not establish signed collision/visual offset direction. The bounded shadow
profiles are paper-external diagnostics and cannot modify Eq. (1)-(9), paper
weights, accepted Stage 9.2 artifacts, or Stage 10 acceptance/export.

## 15. Definition of result-complete

Result-complete additionally requires the same datasets, private trajectories, robot assets,
hardware, simulator, seeds, and experimental conditions needed to reproduce the reported numbers.
This repository is not result-complete.

## 16. Stage 5 dataset boundary

The bounded GRAB dataset adapter is an engineering implementation around the paper's source-hand
input boundary, not an additional claim about the paper's undisclosed preprocessing. Its native
time/mesh/contact preservation, lazy index, validation, and viewer are tracked as
`dataset_adapter_grab` in [`PAPER_FIDELITY.yaml`](PAPER_FIDELITY.yaml). The official GRAB contact
label table is now verified and versioned locally; the unresolved GRAB scene, wrist,
personalized-template, downstream contact aggregation, table, and sequence-ID choices are listed in
[`ASSUMPTIONS.md`](ASSUMPTIONS.md); Stage 6 object sampling/SDF is a bounded engineering
foundation, while no interaction graph, retargeting, or RL behavior is implied.

## Stage 9.3.3 shadow-equivalence boundary

The Stage 9.3.3 workflow is diagnostic-only and paper-external. It preserves
Eq. (1)-(9), paper weights, formal Stage 9.2 solver/execution profiles, accepted
artifacts, Stage 10 exports, and manual/runtime acceptance. Numerical baseline
equivalence is calibrated from independent repeats with fixed floors and caps;
feasibility-only equivalence is rejected. The six bounded profiles and
long-finger attribution cannot authorize Stage 9.4. The current run is
`SHADOW_BASELINE_NOT_NUMERICALLY_EQUIVALENT` because both the replay state
differences and the accepted manifest's internal code-provenance mismatch must
be resolved at the Stage 9.3.2 shadow harness boundary.

Stage 9.3.4 remains diagnostic-only: provenance, multistart, base-seed, and
causal-ablation outputs cannot authorize Stage 9.4 or alter paper fidelity.

Stage 9.3.5 remains diagnostic-only as well. Its `projection_state_metric`,
warm-to-final feasibility path, counterfactual states, objective/constraint
attribution, and optional branch rollout are not paper-specified methods.
They use the canonical full-512 reference-winding audit and cannot modify
Eq. (1)-(9), formal solver profiles, accepted artifacts, or Stage 10 exports.

Stage 9.4 closes the implementation audit with a versioned faithful repair.
The repair preserves Eq. (9) weights and the separate base priors; the only
engineering correction is the temporal-vector membership identified in the
code map. Projection remains diagnostic-only. The repair subsequently received
case-A human manual acceptance in the faithful finalization bundle.

## Q1–Q3 metric and Eq. 9 benchmark boundary

Q1–Q3 implements the Appendix A.3 formulas from the local paper copy. ContactPose attribution
inputs are required: sigmoid-normalized/thresholded native contact annotations, nearest assignment
to the 20 hand bones, and the ten-vertex link rule. If those fields or the source-to-robot link
mapping are unavailable, the output is `N/A` and the unit is not silently promoted to `PAPER_EXACT`.
GRAB has no equivalent official in-contact bone attribution in the canonical contract, so its
contact metrics are separate `DATASET_PROXY` IDs and are never called Eq. (10) or Eq. (11).

The frozen comparison keeps `scipy_slsqp_active_set_contact_rich_v2` as
`literal_full_state_temporal` and `scipy_slsqp_active_set_contact_rich_v3_fixed` as
`decomposed_finger_temporal_plus_base_priors`. Both are paper-consistent interpretations;
`author_exact` remains unresolved. Any empirical preference is an engineering preference only.

## GRAB Arti-MANO A–E quality-extension boundary

The frozen quality experiment retains both Eq. (9) interpretations above and
does not claim `author_exact`. Its morphology seed, visual surface contact
proxies, contact losses, and 2×2 selection are paper-external engineering
extensions; GRAB contact values remain `DATASET_PROXY`. The active-set
continuation buffer recorded in `ASSUMPTIONS.md` is initialization conditioning
only and does not change Eq. (1)–(9), solver tolerances, collision queries, or
strict acceptance.

The initial local execution was not result-complete: G2 had valid 60-frame v2
and v3 artifacts, while G3 was blocked before final solving by the open source
banana mesh. The resumed run uses the separately documented derived sign proxy,
but its strict active-QuerySet boundary gate currently routes to
`SIGN_PROXY_CONTACT_REGION_CONFLICT`; no C–E result or recommendation may be
inferred from that partial run.

## Faithful reproduction finalization

The canonical faithful profile is
`scipy_slsqp_active_set_contact_rich_v3_fixed`; it corrects the Eq. (9)
temporal-vector membership and is classified as `validated_quality_neutral`.
The old `scipy_slsqp_active_set_contact_rich_v2` profile is explicitly
non-faithful because it includes base correction in temporal regularization,
but remains `historical_accepted` as a legacy engineering comparison.
Projection is not a paper method. No significant quality improvement is
claimed. The new fixed Stage 10 reference received repository-valid case-A
human acceptance and is the canonical faithful baseline. See
[`FAITHFUL_REPRODUCTION_FINALIZATION.md`](FAITHFUL_REPRODUCTION_FINALIZATION.md).

## Open-object signed-distance engineering boundary

The GRAB quality lane records `hybrid_original_distance_proxy_sign_v1` as
paper-unspecified geometry engineering. The raw object mesh is unchanged; a
derived watertight proxy supplies sign only, while the original mesh supplies
closest point and unsigned distance. Convex hull is not an accepted proxy.
Raw/proxy provenance, hashes, boundary loops, patch IDs, fixed deviation gates,
and strict contact-region conflicts are retained in the geometry artifacts.
This engineering policy does not change paper equations, solver profiles, or
old Stage 10 artifacts.

## Wuji target-hand boundary

Wuji Hand2 Beta1 is a registered generic target-hand embodiment used for a
fixed engineering reproduction lane. Its 20D qpos, anchor profile, formal
collision profile, and asset provenance are explicit; using it is not a claim
that the paper's original hardware has been reproduced. The solver remains
paper-core Eq. (1)--(9), with no SDF penetration loss, contact attraction, or
robot-specific objective term.
## Wuji continuous engineering extension

`wuji_continuous_full_state_v1` is explicitly not the paper method. It keeps
the paper-core frame objective and collision constraints unchanged and adds a
separate continuation/continuity acceptance layer. The canonical faithful
profile `scipy_slsqp_active_set_contact_rich_v3_fixed` remains immutable.

## W2.3 sequential profile boundary

`wuji_continuous_sequential_v1` is an engineering extension for offline
reference generation, not a paper-method or author-exact claim. It preserves
the full-state objective, temporal terms, constraints, and formal artifacts;
only the production window-fallback flag differs. The five-frame repair,
multi-threshold penetration interpretation, exports, and HTML are diagnostic
evidence and do not change Eq. (1)--(9), baseline/source/warm/graph data, or
historical Stage 10 outputs. RL, realtime, and cross-subject claims remain
false, and `author_exact` remains unresolved.

## W2.2 closeout boundary

The W2.2 closeout is a diagnostic validation layer, not a paper result. Its
B0/B1/B2 attribution, five-frame routing, analytic solver callbacks, HTML
review, and gate reports do not alter Eq. (1)--(9), the formal QuerySet or
collision profile, or any persisted formal trajectory. The real W3 shadow
currently fails with SLSQP status 4 and a continuity-gate failure, while W3
penetration rate regresses; the closeout therefore records
`WUJI_CONTINUOUS_PROFILE_NOT_RECOMMENDED_WINDOW_FALLBACK_FAILED`.
