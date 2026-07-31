# Assumptions and unpublished details

## W2.2 Wuji continuity closeout

- The closeout is diagnostic-only and writes under `closeout_v1`; formal
  baseline/continuous/export artifacts and raw inputs are immutable.
- W2 q-step attribution decomposes `final = warm + correction` in the local
  seed-delta chart. `SOURCE_OR_WARM_DRIVEN`, `CORRECTION_DRIVEN`, `LIMIT_DRIVEN`,
  `REACHABILITY_DRIVEN`, `MIXED`, and `NUMERICALLY_INCONCLUSIVE` are evidence
  categories, not source-ground-truth labels.
- B0/B1/B2 use the same QuerySet, collision profile, paper weights, and solver
  budget. B1 changes only previous-final transport; B2 adds only the declared
  correction temporal term in isolated mode. Retry/window evidence is kept
  operationally separate from the isolated causal ablation.
- The five-frame fallback fixes one left anchor, jointly optimizes the center
  and three future states, persists only center diagnostics, and treats future
  states as hints. Each frame keeps its own QuerySet, slack, and full-surface
  audit. Failure is propagated to the recommendation gate.
- The real W3 shadow currently returns SLSQP status 4 and fails the center
  continuity gate; W3 penetration-rate regression also blocks recommendation.

## W2.1 Wuji continuous extension

- The base chart is `scene_local_seed_delta_exp_left` at root `r_wrist`.
- `author_exact` remains `unresolved`; the new profile is engineering-only.
- Continuity is measured against propagated previous-final state, not against
  source MANO or baseline final artifacts.
- The five-frame fallback is bounded and cannot be expanded automatically.

## W0/W1 Wuji Hand2 Beta1 boundary

Wuji Hand2 Beta1 is a tracked target-hand asset and generic registry instance, not a hardware or
deployment reproduction. The pinned upstream body subset, license, exclusions, and hashes are
recorded in [`WUJI_HAND2_ASSET_PROVENANCE.md`](WUJI_HAND2_ASSET_PROVENANCE.md). The MediaPipe-21
anchors are engineering semantics derived from URDF joint/link origins and official MJCF tip sites;
the paper does not publish an author-exact mapping.

URDF is the differentiable/reference kinematics source and MJCF is the simulation-facing
actuator/tip/collision source. Formal collision uses the declared MJCF convex-hull geoms and ten
contact excludes; soft-pad tip meshes are not auto-promoted. Visual/contact proxies, source contact
labels, and signed-distance ground truth remain separate. The `engineering_collision_32_per_geometry`
surface profile is not a paper-specified value.

The W0/W1 bounded pipeline smoke checks generic construction only. It does not establish Wuji
hardware calibration, MuJoCo playback, PD gains, PPO tracking, full multi-clip retargeting, or the
paper's Wuji transfer claim. W2 requires at least three watertight clips; S1 and I1 remain separate
branch milestones.

This register intentionally records missing information instead of silently selecting values.
Each identifier is referenced from `docs/PAPER_FIDELITY.yaml` where it affects traceability.

| ID | Category | Impact | Provisional handling | Evidence | Status | Resolution criteria |
| --- | --- | --- | --- | --- | --- | --- |
| A_F0_TRACKED_ASSET_PROVENANCE_001 | asset distribution | P0 | Vendor the Arti-MANO payload from pinned ManipTrans commit `a3d08cfe3c3a5868a7f057533bcaf759c5af4705`, retain the upstream GPL notice, and record per-file hashes; no separate asset license was found. | `third_party/robot_hands/artimano/SOURCE.yaml`, `.local/reports/f0/license_audit.json` | implemented_with_assumptions | Upstream clarification or a separate asset license that changes redistribution terms. |
| A_F0_URDF_PATH_REBASE_001 | asset compatibility | P1 | Rebase only relative mesh filenames from the upstream flat directory into `urdf/` and `meshes/`; require exact topology, geometry transforms, and numerical regression. | `.local/reports/f0/numerical_regression.json` | implemented_with_assumptions | Upstream structured asset layout or an author-provided packaged asset. |
| A_F0_LEGACY_ASSET_REBIND_001 | artifact provenance | P0 | Rebind historical absolute `.local/assets/artimano` paths by robot ID/source URDF hashes and content manifest, with no solver invocation or artifact rewrite. | `.local/reports/f0/historical_artifact_compatibility.json` | implemented_with_assumptions | Versioned artifact reader contract from the original pipeline. |
| A_F0_GENERIC_TARGET_CONTRACT_001 | target-hand adapter | P0 | Make asset, kinematic, semantic-anchor, surface, collision, and simulation metadata data-driven; derive anchor/Jacobian shapes from the selected profile. | `docs/ROBOT_HAND_TARGET_CONTRACT.md`, registry tests | implemented_with_assumptions | Author/source contracts for additional hands, including Wuji Hand2. |
| A_HAND_FRAME_001 | coordinate convention | P0 | Use the explicit `canonical_keypoint_wrist_v1` frame: wrist origin, Gram-Schmidt lateral/longitudinal axes, and a right-handed cross-product third axis. Stored GRAB wrist poses remain separate evidence. | Sec. 3.2, Eq. 1; Stage 7 frame audit | implemented_with_assumptions | Author-provided source wrist-centered frame definition. |
| A_MANO_MEDIAPIPE_SEMANTICS_001 | source-hand adapter | P1 | Map audited MANO named joints to MediaPipe-style names, including thumb CMC/MCP/IP as an explicit semantic approximation. | Stage 3 MANO layout/profile audit | implemented_with_assumptions | Author/source-data documentation of the intended MANO-to-MediaPipe joint semantics. |
| A_MANO_FINGERTIP_VERTICES_001 | source-hand adapter | P1 | Use installed-smplx MANO tip anchors 744/320/443/554/671 for the audited 778-vertex model; retain differing ManipTrans candidates in the audit. | `smplx.vertex_ids`, ManipTrans `grab_dataset_dexhand.py`, neutral/real local geometry checks | implemented_with_assumptions | Confirm the intended MANO topology and fingertip-anchor profile for every model release. |
| A_MANO_BACKEND_LAYOUT_001 | source-hand adapter | P1 | Treat current SMPL-X output as audited `mano16_smplx`; do not infer semantics from shape alone. | Stage 2B cache and installed SMPL-X/MANO source inspection | implemented_with_assumptions | Public named joint-order contract for the backend output. |
| A_ROBOT_HAND_FRAME_001 | coordinate convention | P0 | Derive the robot canonical hand frame from its FK MediaPipe-21 anchors using the same profile as the source; do not equate URDF palm axes with the derived frame. | Sec. 3.1-3.2; Stage 7 frame audit | implemented_with_assumptions | Author-provided robot wrist-centered frame convention. |
| A_ROBOT_KEYPOINT_ANCHORS_001 | target-hand adapter | P1 | Use a versioned, explicit URDF anchor profile with the canonical Stage 3 MediaPipe21 order. | `configs/robots/keypoints/artimano_mediapipe21.yaml`, Stage 4 report | implemented_with_assumptions | Author-provided target-hand 21-point anchor definition. |
| A_ARTIMANO_KEYPOINT_MAPPING_001 | target-hand adapter | P1 | Map MediaPipe21 semantics to Arti-MANO joint origins; use the first joint of coincident multi-axis groups and fixed fingertip joint origins. | Arti-MANO RH/LH URDFs, Stage 4 profile and validation | implemented_with_assumptions | Author/source target-hand semantic mapping. |
| A_ROBOT_BASE_FRAME_001 | coordinate convention | P1 | Use the URDF root link `palm` as the engineering robot base frame; pass scene base pose separately. | RH/LH URDF root audit, Stage 4 interface | implemented_with_assumptions | Paper's exact wrist-centered robot frame orientation. |
| A_ARTIMANO_COLLISION_COVERAGE_001 | target-hand geometry | P1 | Keep visual and collision instances separate; report fixed tip visual-only spheres and do not synthesize collision geometry. | RH/LH URDF geometry audit, `asset_integrity.json` | implemented_with_assumptions | Author-confirmed collision coverage and later collision-query policy. |
| A_ARTIMANO_DOF_ORDER_001 | target-hand kinematics | P1 | Use explicit 22-name qpos order audited against both URDFs and ManipTrans `artimano.py`; never use parser order. | `configs/robots/artimano_{rh,lh}.yaml`, Stage 4 report | implemented_with_assumptions | Released target-hand qpos contract. |
| A_BASE_PARAMETERIZATION_001 | optimization variable | P1 | Do not choose Euler, axis-angle, or quaternion. | Appendix A.1, Eq. 9 | pending_author_confirmation | Exact `q_base` rotation coordinates and constraints. |
| A_FIRST_FRAME_INITIALIZATION_001 | initialization | P1 | Use neutral 22-DoF q at t=0, omit the temporal residual, and do not perform automatic multistart. | Sec. 3.2, Eq. 2; Stage 7 solver profile | implemented_with_assumptions | First-frame warm-start rule or released code. |
| A_BONE_DIRECTION_FRAME_001 | initialization frame | P0 | Default to the local canonical keypoint frame; provide `translation_centered_scene_axes` only as a bounded diagnostic interpretation. | Sec. 3.2, Eq. 1; Stage 7 frame audit | implemented_with_assumptions | Author-provided wrist-frame axis and rotation-removal rule. |
| A_BONE_PAIR_SET_001 | bone topology | P1 | Use five semantic full finger chains, 20 directed bones, and 15 within-finger consecutive pairs. Retain 15-bone/10-pair phalange-only as diagnostic. | Sec. 3.2, Eq. 1; Stage 7 profiles | implemented_with_assumptions | Author-provided directed bone and adjacent-pair indices, including wrist-to-MCP scope. |
| A_ZERO_LENGTH_BONE_POLICY_001 | numerical validity | P1 | Use an explicit 1e-10 m threshold and fail strict extraction with frame/bone diagnostics; no identity or previous-frame fallback. | Stage 7 feature extractor | implemented_with_assumptions | Author-provided zero-length handling. |
| A_WARMSTART_BASE_OBSERVABILITY_001 | optimization variable | P0 | Audit base Jacobians and optimize only the 22 raw finger DoFs because the default local direction objective removes base translation and rotation observability. | Eq. 1-2; Stage 7 observability report | implemented_with_assumptions | Author-provided full-q parameterization or base observability treatment. |
| A_BASE_SEED_ALIGNMENT_001 | initialization output | P1 | Produce `T^S_B=T^S_Hs(T^B_Hr(q))^-1` after q solving, outside Eq. 1/2, and preserve source scene motion. | Stage 7 artifact contract | implemented_with_assumptions | Author-provided base-seed/calibration rule. |
| A_WARMSTART_COORDINATES_001 | optimization coordinates | P1 | Optimize raw 22 joint radians; do not normalize ranges, use PCA, or introduce a latent parameterization. | Eq. 2; Stage 7 solver | implemented_with_assumptions | Author-provided q coordinate convention. |
| A_WARMSTART_SOLVER_001 | optimization backend | P1 | Use float64 `scipy.optimize.least_squares(method="trf")` with Torch-autograd Jacobians and engineering tolerances stored in the solver profile. | Eq. 2; Stage 7 solver profile | implemented_with_assumptions | Author-provided optimizer/backend and termination settings. |
| A_WARMSTART_JOINT_LIMITS_001 | robot constraints | P1 | Enforce Stage 4 URDF lower/upper bounds directly in the bounded solver, not by post-solve clamping. | Eq. 2; Stage 4 URDF limits | implemented_with_assumptions | Author-provided Eq. 2 joint-limit scope. |
| A_WARMSTART_TIME_DISCRETIZATION_001 | temporal discretization | P1 | Use contiguous native frames, preserve timestamps/FPS, do not resample, and do not dt-normalize the temporal weight. | Eq. 2; GRAB native 120 FPS | implemented_with_assumptions | Author-provided time-weight scaling rule. |
| A_OBJECT_SAMPLING_001 | geometry | P1 | Use 50 fixed object-local anchors and reuse their identity across bounded frames; this is an engineering foundation, not a paper claim. | Sec. 3.3, Table 3 | implemented_with_assumptions | Per-frame/per-sequence/per-object sampling rule. |
| A_OBJECT_SAMPLING_METHOD_001 | geometry | P1 | Use deterministic area-weighted triangle sampling with barycentric anchors; the paper does not disclose the sampler. | Sec. 3.3 | implemented_with_assumptions | Sampling algorithm and determinism policy. |
| A_OBJECT_SAMPLING_SEED_001 | geometry | P1 | Use explicit NumPy PCG64 seed 20260720 and record it in the profile/artifact. | Sec. 3.3 | implemented_with_assumptions | Author-provided seed or released code. |
| A_OBJECT_SAMPLE_TEMPORAL_REUSE_001 | geometry | P1 | Reuse object-local face+barycentric anchors after pose transforms; do not resample each frame. | Sec. 3.3 | implemented_with_assumptions | Author-provided temporal sampling schedule. |
| A_SURFACE_NORMAL_MODE_001 | geometry | P2 | Store face normals only for diagnostics, SDF probes, and visualization; they are not interaction-graph inputs. | Sec. 3.3 | implemented_with_assumptions | Author-provided normal use and interpolation policy. |
| A_DELAUNAY_BACKEND_001 | geometry | P0 | Use `scipy.spatial.Delaunay` in three dimensions with the audited environment version recorded in each graph artifact. | Sec. 3.3, Eq. 4; Stage 8 profile | implemented_with_assumptions | Author-confirmed library, dimensionality, flags, and version. |
| A_DELAUNAY_OPTIONS_001 | geometry | P0 | Use explicit non-incremental Qhull options `Qbb Qc Qz Q12`; do not use `QJ` in the strict profile. Normalize only by centroid translation and uniform bounding-box-diagonal scale for Qhull numerical conditioning. | Sec. 3.3, Eq. 4; `configs/retarget/interaction/strict_scipy_qhull_v1.yaml` | implemented_with_assumptions | Author-provided Delaunay flags and numerical conditioning policy. |
| A_DELAUNAY_DEGENERACY_001 | geometry | P1 | Fail fast on duplicate, near-duplicate, coplanar, or zero-volume source inputs; a deterministic jitter profile exists only as a diagnostic and is not used for acceptance artifacts. | Sec. 3.3; Stage 8 validation | implemented_with_assumptions | Coplanar, cospherical, duplicate-point, and near-degenerate handling. |
| A_INTERACTION_EDGE_FILTERING_001 | graph topology | P1 | Extract every unique tetrahedron edge and apply no semantic or distance filter; reject isolated vertices and require at least one hand-object edge per accepted frame. | Eq. 4; Stage 8 graph validation | implemented_with_assumptions | Author-confirmed edge subset, if any. |
| A_INTERACTION_GRAPH_FRAME_001 | graph topology | P1 | Build the graph in the canonical scene frame `S` using 21 source hand points followed by 50 posed Stage 6 object samples. | Eq. 3-4; Stage 8 artifact schema | implemented_with_assumptions | Author-confirmed graph coordinate frame and vertex ordering. |
| A_INTERACTION_GRAPH_REBUILD_001 | graph topology | P1 | Rebuild source connectivity when source geometry or object scale changes; otherwise reuse the saved source topology, directed weights, and object point identity during robot evaluation. | Eq. 4-7; Stage 8 CLI and provenance | implemented_with_assumptions | Author-confirmed cache invalidation/rebuild schedule. |
| A_LAPLACIAN_WEIGHT_NUMERICS_001 | graph numerics | P1 | Compute source exponential weights with a row-wise log-sum-exp stabilization, retain all directed neighbors without cutoff, and fail rows with no neighbors. | Eq. 5; Stage 8 weight validation | implemented_with_assumptions | Author-confirmed precision, underflow, and zero-neighbor policy. |
| A_INTERACTION_BASE_DIFFERENTIABILITY_001 | diagnostics | P1 | Evaluate Eq. 7 on frozen Stage 7 qpos/base and expose a qpos Jacobian plus bounded base perturbation diagnostics; do not start Stage 9 optimization or select its base parameterization. | Eq. 7; Stage 8 evaluation artifact | implemented_with_assumptions | Author-confirmed full parameterization and differentiability contract. |
| A_REFINEMENT_BASE_PARAMETERIZATION_001 | optimization variable | P0 | Use x=[delta p, delta omega, q theta, s], with R=Exp(delta omega) R_seed and p=p_seed+delta p; rotation is a scene-frame rotation vector in radians. | Eq. 8-9; configs/retarget/refinement/local_seed_delta_v1.yaml | implemented_with_assumptions | Author's floating-base rotation coordinates. |
| A_REFINEMENT_BASE_PRIOR_REFERENCE_001 | regularization | P0 | Interpret q_base,pos and q_base,rot as the correction delta relative to the Stage 7 base seed. | Eq. 9; refinement coordinate profile | implemented_with_assumptions | Author's base-prior origin. |
| A_REFINEMENT_FIRST_FRAME_001 | sequential optimization | P1 | Initialize t=0 at the Stage 7 seed with zero base correction and omit only the temporal previous-state term; retain base priors. | Eq. 9; refinement coordinate profile | implemented_with_assumptions | Author's t=0 policy. |
| A_REFINEMENT_COORDINATE_SCALING_001 | numerical optimization | P1 | Persist and evaluate the Stage 9 vector in raw metres, radians, and joint radians; no FPS or dt scaling is applied. Any internal solver reparameterization must be explicit and invertible. | Eq. 9; final_refinement.py | implemented_with_assumptions | Author's coordinate scaling; solver conditioning is tracked separately. |
| A_REFINEMENT_TIME_DISCRETIZATION_001 | temporal optimization | P1 | Use contiguous native frames and remap the previous final pose into the current seed coordinates before the temporal term. | Eq. 9; final_refinement.py | implemented_with_assumptions | Author's temporal discretization. |
| A_REFINEMENT_QUERY_SET_001 | collision constraints | P0 | Represent one robot collision surface sample to primary object mesh pair per query; use deterministic full or adaptive profiles and never use visual samples/contact labels. | Eq. 8; collision query profile | implemented_with_assumptions | Author's Q_t construction. |
| A_COLLISION_ACTIVE_MARGIN_001 | collision constraints | P1 | Adaptive profile starts with penetration, 10 mm margin, and nearest sample per collision geometry; 10 mm is not paper-specified. | Eq. 8; adaptive_active_set_v1.yaml | implemented_with_assumptions | Author's active-set margin. |
| A_COLLISION_ACTIVE_SET_REFINEMENT_001 | collision constraints | P0 | Add missed full-surface violations monotonically between solves and independently audit all 512 samples; stop on convergence or five rounds. | Eq. 8; final_refinement.py | implemented_with_assumptions | Author's outer-loop policy. |
| A_SDF_CONSTRAINT_JACOBIAN_001 | collision derivatives | P0 | Use scene-frame outward SDF normal times Torch autograd collision-point Jacobian; invalid-normal rows use central finite differences and are counted. | Eq. 8; final_refinement.py | implemented_with_assumptions | Author's SDF derivative and nonsmooth policy. |
| A_REFINEMENT_SDF_BACKEND_001 | collision backend | P0 | Use strict positive-outside Stage 6 reference SDF for acceptance; the convex-hull solver backend is enabled only after probe cross-validation and is never the final audit backend. | Stage 6 SDF; final_refinement.py | implemented_with_assumptions | Author's acceleration and SDF backend. |
| A_REFINEMENT_SOLVER_001 | constrained solver | P0 | Use float64 SciPy SLSQP with analytic objective and hybrid constraint Jacobians, URDF q bounds, slack bounds, and sequential per-frame solves. | Eq. 8; scipy_slsqp_active_set_v1.yaml | implemented_with_assumptions | Author's constrained solver. |
| A_REFINEMENT_SOLVER_TOLERANCES_001 | solver termination | P1 | Record SLSQP maxiter=100, ftol=1e-7, constraint audit tolerance=1e-6 m, finite-difference epsilon=1e-6, and fail-fast policy. | Eq. 8; solver profile | implemented_with_assumptions | Author's termination and line-search details. |
| A_REFINEMENT_SOLVER_CONTINUATION_002 | solver continuation | P0 | Preserve v1 warm-seed reinitialization for regression, and use v2 result.x continuation with query-ID slack remapping and minimum bounded initialization for newly added queries. | Stage 9.1 benchmark; final_refinement.py | implemented_with_assumptions | Paper does not disclose active-set continuation state. |
| A_REFINEMENT_CONTINUATION_BUFFER_001 | solver continuation numerics | P1 | Add a fixed 1e-9 m interior buffer only to newly added soft-constraint slack during v2 result.x continuation, preventing reference/solver SDF round-off from making the SLSQP starting point infinitesimally infeasible; objective, solver tolerances, active margin, and acceptance gates are unchanged. | `final_refinement.py`; G2 frame-22 reproduction and 60-frame rerun | implemented_with_assumptions | Confirm the published active-set implementation and its cross-backend initialization precision policy. |
| A_REFINEMENT_SOLVER_TERMINATION_002 | solver termination | P0 | Keep strict optimizer convergence separate from feasibility and full-surface audits; status 9 is rejected, while the fixed-grid uniform maxiter and deterministic repeats are recorded as provenance. | Stage 9.1 benchmark; solver_benchmark.py | implemented_with_assumptions | Paper does not disclose optimizer termination, stationarity, or KKT policy. |
| A_REFINEMENT_PROFILE_HASH_001 | solver provenance | P0 | Preserve v1 hash `6affff2fdb425a0402f643c291c0b8904d4dbec6c5b69a5006cf9829dcc220aa`; register v2 hash `c42c21d894c54d07b1d30943b5a3338b13628bf0429ab203b5540cf934d09b7c` and bind it into Stage 10 signatures. | Stage 9.1 profile YAML and workflow planner | implemented_with_assumptions | Profile identity and termination behavior are engineering choices not disclosed by the paper. |
| A_REFINEMENT_EVALUATION_CACHE_001 | execution performance | P1 | Use an exact float64 `x` identity within one immutable frame context; invalidate all cached layers on QuerySet change and never reuse across frames. | `refinement_performance.py`; Stage 9.2 report | implemented_with_assumptions | The paper does not specify callback reuse or cache identity. |
| A_REFINEMENT_CHECKPOINT_RESUME_001 | execution provenance | P0 | Persist only strict-accepted frames in an atomic hash chain; resume only from the last contiguous frame with matching input, solver, query, and execution profile hashes. | `refinement_checkpoint.py`; Stage 9.2 report | implemented_with_assumptions | The paper does not specify interruption or artifact checkpoint semantics. |
| A_REFINEMENT_EXECUTION_DEVICE_001 | numerical execution | P1 | Keep the validated default on CPU float64; execution device/cache policy is separate from the SLSQP solver profile and is selected by measurement. | `cached_checkpoint_cpu_float64_v1.yaml` | implemented_with_assumptions | The paper does not disclose device or precision execution policy. |
| A_REFINEMENT_SOLVER_VARIABLE_SCALING_001 | numerical execution | P1 | Use the explicit `seed_delta_normalized_v1` diagonal map only for the internal SLSQP vector; map every callback and persisted result back to raw Stage 9 coordinates before objective, constraint, audit, or checkpoint semantics. | `final_refinement.py`; execution profile | implemented_with_assumptions | An invertible conditioning transform; not a paper objective or coordinate claim. |
| A_REFINEMENT_REFERENCE_SDF_ACCELERATION_001 | collision backend | P1 | Build one exact triangle AABB tree for the reference SDF; prune only by AABB lower bounds and evaluate the existing exact triangle closest-point formula at leaves, while retaining independent winding/sign computation. | `closest_point.py`; `reference.py` | implemented_with_assumptions | The paper does not disclose reference SDF acceleration. |

| A_STAGE9_3_2_CANONICAL_REFERENCE_SDF_001 | contact audit backend | P0 | Use the versioned strict `reference_winding_v1` / `reference_triangle_winding` backend for all formal Stage 9.3.2 metrics; keep solver SDF selection separate. | `configs/audit/contact_distance/reference_winding_v1.yaml`; `contact_canonical_reaudit.py` | implemented_with_assumptions | The paper does not publish the exact audit backend contract. |
| A_STAGE9_3_2_CONTACT_PROXY_001 | contact semantics | P0 | Name source and retention values as proxies; never promote dense visual samples or semantic anchors to ground-truth contact labels. | Stage 9.3.2 v2 reports | implemented_with_assumptions | Visual/semantic contact attribution is not fully specified by the paper. |
| A_STAGE9_3_2_OFFSET_DIRECTION_001 | geometry audit | P0 | Report unsigned collision/visual coverage gaps, but keep offset direction inconclusive when visual meshes are open or normals are unvalidated. | `canonical_collision_visual_audit.json` | implemented_with_assumptions | Normal orientation and faithful visual/collision correspondence are not disclosed. |
| A_STAGE9_3_2_SHADOW_METHOD_001 | diagnostic solver | P0 | Keep all six bounded shadow profiles diagnostic-only, paper-external, canonical-evaluated, and isolated from accepted artifacts. | `contact_canonical_reaudit.py` | implemented_with_assumptions | The paper does not specify this engineering ablation. |
| A_REFINEMENT_BATCHED_JACOBIAN_001 | collision derivatives | P1 | Transform the complete active QuerySet and obtain point Jacobians with one vectorized Torch autograd call; retain the reference loop as a validation path. | `final_refinement.py`; Stage 9.2 report | implemented_with_assumptions | The paper does not disclose collision Jacobian implementation. |
| A_REFINEMENT_FULL_AUDIT_SCHEDULING_001 | collision validation | P0 | Run independent 512-point audits only at query initialization, active-set round boundaries, frame final acceptance, and final artifact validation; never inside SLSQP callbacks. | `final_refinement.py`; Stage 9.2 report | implemented_with_assumptions | The paper does not specify audit scheduling. |
| A_REFINEMENT_PERFORMANCE_GATE_001 | runtime gate | P1 | Treat the median/p95 and 60-frame runtime thresholds as engineering gates; never present them as paper performance or silently relax strict acceptance. | Stage 9.2 performance report | implemented_with_assumptions | The paper's reported runtime is not a local reproduction gate. |
| A_REFINEMENT_WALL_TIME_PAUSE_001 | recoverability | P0 | Interpret `--max-wall-time` as a soft boundary checked before frames/outer rounds; finish the current solve, checkpoint it if accepted, and report `paused`. | `retarget.py`; checkpoint report | implemented_with_assumptions | The paper does not specify wall-time interruption semantics. |
| A_REFINEMENT_HAND_SURFACE_SAMPLES_001 | final collision coverage | P1 | Use the existing 512 engineering collision samples (32 per each of 16 Arti-MANO collision geometries) without visual fallback; this count is not paper-specified. | Stage 6 artifact; final artifact | implemented_with_assumptions | Author's collision-surface sampling rule. |
| A_SOLVER_001 | optimization | P0 | Do not pick an optimizer. | Sec. 3.4, Eq. 8 | pending_author_confirmation | Exact constrained solver and variable parameterization. |
| A_SOLVER_TERMINATION_001 | optimization | P0 | Keep iteration/tolerance/line search null. | Sec. 3.4 and Table 3 boundary | not_provided | Released implementation or author response. |
| A_JOINT_LIMIT_001 | robot constraints | P0 | Record Eq. 8 constraint scope without inventing joint-limit equations. | Sec. 3.4 | pending_author_confirmation | Exact joint-limit and self-collision implementation. |
| A_COLLISION_QUERY_SET_001 | collision | P0 | Keep `Q_t` construction blocked. | Sec. 3.4, Eq. 8 | pending_author_confirmation | Hand-object pair query construction. |
| A_SIGNED_DISTANCE_BACKEND_001 | collision | P0 | Provide a reference triangle closest-point backend with strict watertight, generalized-winding, and explicit unsigned-only modes; no mesh repair is performed. | Sec. 3.4 and Appendix A.3 | implemented_with_assumptions | Author backend, gradient, and non-watertight policy. |

| A_MESH_WATERTIGHT_POLICY_001 | collision | P0 | Strict sign rejects open/non-manifold meshes; generalized winding reports confidence; unsigned-only never fabricates a signed value. | Sec. 3.4 and Appendix A.3 | implemented_with_assumptions | Author mesh preprocessing and sign policy. |
| A_HAND_SURFACE_SAMPLES_001 | metric | P1 | Use explicit engineering profile `engineering_collision_32_per_geometry` for Stage 6 only; it is not a paper sample count. | Appendix A.3, Eq. 12 | implemented_with_assumptions | Released metric code or explicit paper rule. |
| A_CONTACTPOSE_THRESHOLD_001 | metric | P1 | Keep intensity threshold null; preserve all other attribution rules. | Appendix A.3 | not_provided | Contact intensity threshold from authors or ContactPose code. |
| A_TABLE1_TARGET_HAND_001 | experiment | P1 | Report Table 1 without naming an unverified target hand. | Table 1 | pending_author_confirmation | Target hand identity and asset/version. |
| A_RL_TRACKED_LINKS_001 | RL observation | P1 | Keep tracked link list null. | Sec. 4 and Appendix A.5.2 | not_provided | Exact list/order of tracked links. |
| A_RL_AXIS_POINTS_001 | RL observation | P1 | Keep axis-point spatial offsets null; record six points only. | Appendix A.5.2 and Table 4 | not_provided | Exact six-point construction and offsets. |
| A_RL_SIMULATOR_001 | RL infrastructure | P0 | Do not install or select a simulator in Stage 0/1. | Appendix A.5 | pending_author_confirmation | Simulator, version, and physics solver settings. |
| A_RL_PD_GAINS_001 | RL infrastructure | P1 | Keep low-level gains null. | Appendix A.5.5 | not_provided | Gains or actuator configuration used for results. |
| A_PPO_UNLISTED_PARAMS_001 | RL optimization | P1 | Keep clip/value/gradient values null. | Appendix A.5.6 | not_provided | Full PPO configuration or released training code. |
| A_PRIVATE_PENSPIN_DATA_001 | dataset | P0 | Treat self-collected Pen-Spin data as unavailable. | Appendix A.4 | blocked | Author release or access procedure for 32 clips. |
| A_WUJI_ASSET_001 | hardware | P0 | Record transfer claim; do not claim hardware reproduction. | Sec. 5.2, Fig. 4 | blocked_missing_asset | Wuji URDF, calibration, controller, and deployment assets. |
| A_GRAB_SCENE_FRAME_001 | GRAB adapter | P1 | Treat the GRAB source scene as the canonical scene frame and preserve source-to-scene identity. | GRAB source fields, Stage 2B/5 raw-canonical comparison | implemented_with_assumptions | Official GRAB scene-frame contract for every release. |
| A_GRAB_WRIST_FRAME_001 | GRAB adapter | P1 | Preserve the source MANO translation as the wrist-pose origin and keep the existing Stage 2B frame convention. | SMPL-X/MANO backend and Stage 5 validation | implemented_with_assumptions | Author/source documentation of the intended GRAB wrist frame. |
| A_GRAB_PERSONALIZED_VTEMP_001 | GRAB adapter | P1 | Resolve and use the sequence-referenced personalized `vtemp`; do not replace it with a neutral template. | GRAB NPZ metadata and local `vtemp` files | implemented_with_assumptions | Official per-sequence personalized-template contract. |
| A_GRAB_CONTACT_MAPPING_001 | GRAB contacts | P1 | Preserve source contact labels, expose binary labels as `labels != 0`, and map semantic IDs from the official GRAB `contact_ids` table. | `otaheri/GRAB/tools/utils.py` at commit `4dab3211fae4fc5b8eb6ab86246ccc3a42d8f611`; tracked in `grab_contact_parts.yaml` | implemented_with_assumptions | Confirm whether downstream paper code intends a different contact aggregation. |
| A_GRAB_TABLE_ROLE_001 | GRAB scene geometry | P1 | Store the GRAB table as a static `support_surface`, not an interactable object. | GRAB `table.ply` and Stage 5 schema adapter | implemented_with_assumptions | Dataset documentation for table semantics in downstream tasks. |
| A_GRAB_SEQUENCE_ID_001 | GRAB index | P2 | Use `subject/object_action_repetition` filename-derived stable IDs and retain the source path. | GRAB layout and Stage 5 index manifest | implemented_with_assumptions | Official sequence identifier field or release manifest. |
| A_WORKFLOW_CONTACT_WINDOW_THRESHOLD_001 | Stage 10 selection | P1 | Use a 60-frame native window, require at least 0.5 contact-frame ratio, and rank by semantic contact density, contact count, source distance, start frame, and sequence. | `configs/workflows/contact_window_selection.yaml`; bounded selector reports | implemented_with_assumptions | Author-provided downstream contact-window policy. |
| A_WORKFLOW_SOURCE_CONTACT_SANITY_001 | Stage 10 selection | P1 | Require a strict watertight object mesh and MANO-backed source contact median distance at or below 0.02 m; do not infer contact from labels alone. | Stage 10 selector and source geometry reports | implemented_with_assumptions | Author-provided source-contact gate and mesh policy. |
| A_WORKFLOW_FINAL_CONTACT_SANITY_001 | Stage 10 validation | P1 | Report final contact-frame collision-surface distances with a 0.05 m engineering warning threshold; this is not Eq. 10–12. | `configs/workflows/semantic_acceptance.yaml`; semantic sanity report | implemented_with_assumptions | Author-provided final contact sanity criterion. |
| A_WORKFLOW_ARTIFACT_REUSE_001 | Stage 10 provenance | P1 | Reuse only matching implementation/config/input signatures with existing, hash-matching, passing outputs. | Workflow cache records and resume tests | implemented_with_assumptions | Author-provided cache trust/invalidation policy. |
| A_WORKFLOW_INVALIDATION_001 | Stage 10 provenance | P1 | A changed source, request/window, profile, implementation version, dependency, or artifact hash invalidates the node and downstream outputs. | Workflow plan, executor, and invalidation tests | implemented_with_assumptions | Author-provided invalidation graph. |
| A_WORKFLOW_MANUAL_ACCEPTANCE_001 | Stage 10 review | P1 | Require human pass reviewing frames 0/29/59; permit pre-contact Stage 9 interpretation but never treat it as contact-rich evidence. | Stage 9 manual gate and review bundle contract | implemented_with_assumptions | Human acceptance protocol and contact-rich review rubric. |
| A_INTERACTION_HTML_DISPLAY_WEIGHT_001 | visualization | P2 | Keep the saved directed Stage 8 weights unchanged; use only the symmetric display value `w_vis(i,j)=(w_ij+w_ji)/2` for an undirected HTML line's width/opacity/color. | `interaction_html.py`; Stage 8 graph artifact | implemented_with_assumptions | Paper does not specify an undirected line-rendering convention. |
| A_INTERACTION_HTML_FINAL_RESIDUAL_001 | visualization diagnostic | P2 | Compute final HTML residuals read-only from final hand keypoints and the frozen Stage 8 directed graph; preserve the 50 object sample vertices from Stage 8. | `mesh_visualization.py`; Stage 8/9 artifacts | implemented_with_assumptions | Paper does not specify a browser diagnostic or final-state visualization formula. |
| A_STAGE9_3_DENSE_SURFACE_APPROXIMATION_001 | Stage 9.3 audit geometry | P1 | Use deterministic area-uniform dense surface samples, retain all source vertices, and label the result as a continuous-surface approximation. | `contact_audit.py`; `CONTACT_RETENTION_AUDIT.md` | implemented_with_assumptions | The paper does not disclose a continuous mesh audit sampler. |
| A_STAGE9_3_CONTACT_PROXY_001 | Stage 9.3 contact attribution | P1 | Treat source contact and retention as diagnostic proxies based on signed-distance thresholds and nearest MediaPipe21 anchor regions, never as ground-truth contact labels. | `contact_audit.py`; `CONTACT_RETENTION_AUDIT.md` | implemented_with_assumptions | Contact labels are not available in the accepted final artifact contract. |
| A_STAGE9_3_PAD_PROXY_001 | Stage 9.3 fingertip geometry | P1 | Report semantic anchors separately from visual/collision surfaces; fingertip links are a pad proxy only when the available geometry exposes no dedicated pad surface. | `contact_audit.py`; `CONTACT_RETENTION_AUDIT.md` | implemented_with_assumptions | The accepted robot geometry artifact does not provide a validated contact-pad surface contract. |
| A_STAGE9_3_INTERPOLATION_001 | Stage 9.3 counterfactual path | P1 | Interpolate raw qpos linearly and base pose with SO(3) Slerp for diagnostics only; do not call an optimizer or treat intermediate states as accepted trajectories. | `contact_audit.py`; `CONTACT_RETENTION_AUDIT.md` | implemented_with_assumptions | The paper does not specify an interpolation-based feasibility diagnostic. |
| A_STAGE9_3_SLSQP_MULTIPLIER_001 | Stage 9.3 active-set audit | P1 | Record SciPy SLSQP multipliers as unavailable and use persisted slack, active-set provenance, and independent full-surface checks instead. | `contact_audit.py`; `CONTACT_RETENTION_AUDIT.md` | implemented_with_assumptions | The available SciPy result/artifact contract does not expose reliable multipliers. |
| A_STAGE9_3_1_SDF_RECONCILIATION_001 | Stage 9.3.1 metric audit | P0 | Use the persisted Stage 9.2 reference triangle/winding SDF as the formal comparison definition; keep the legacy convex-hull solver-only backend diagnostic-only until definitions reconcile. | `contact_metric_reconciliation.py`; `CONTACT_METRIC_RECONCILIATION_AND_SHADOW_ABLATION.md` | implemented_with_assumptions | The paper does not publish the solver/reference SDF implementation boundary. |
| A_STAGE9_3_1_OFFSET_DIRECTION_001 | Stage 9.3.1 geometry audit | P1 | Treat unsigned visual/collision offsets as non-directional and report inconclusive when mesh normals are not reliable; never infer outward inflation from unsigned distance alone. | `contact_metric_reconciliation.py`; `CONTACT_METRIC_RECONCILIATION_AND_SHADOW_ABLATION.md` | implemented_with_assumptions | The paper does not specify the robot collision surface or validated normal orientation. |
| A_STAGE9_3_1_SHADOW_GATE_001 | Stage 9.3.1 ablation | P0 | Allow at most three isolated representative frames and fail closed with zero solver invocations when identity, SDF, transform, or acceptance replay gates fail. | `contact_shadow_ablation.py`; `CONTACT_METRIC_RECONCILIATION_AND_SHADOW_ABLATION.md` | implemented_with_assumptions | The paper does not publish an ablation protocol or causal optimizer evidence. |
| A_STAGE7_1_MORPHOLOGY_TARGET_001 | Stage 7.1 diagnostic | P1 | Compare raw source metric thumb targets with robot-length reconstructed targets without replacing either the formal source target or the accepted warm-start artifact. | `warm_start_audit.py`; `WARM_START_FIDELITY_AND_REACHABILITY_AUDIT.md` | implemented_with_assumptions | The paper does not publish a morphology-normalized reachability diagnostic. |
| A_STAGE7_1_WORKSPACE_SAMPLE_001 | Stage 7.1 diagnostic | P1 | Use deterministic 4096-point Sobol thumb joint-limit samples; report nearest distance and sampled convex-hull membership as a diagnostic approximation, never a global reachability proof. | `warm_start_audit.py`; `WARM_START_FIDELITY_AND_REACHABILITY_AUDIT.md` | implemented_with_assumptions | The paper does not publish a workspace sampler or global IK certificate. |
| A_STAGE7_1_FINAL_ATTRIBUTION_001 | Stage 7.1 audit | P1 | Keep formal bone, canonical keypoint, Stage 8/Stage 9 `E_IM`, and contact proxy metrics separate when attributing warm versus final error; contact proxy is not ground truth. | `warm_start_audit.py`; `WARM_START_FIDELITY_AND_REACHABILITY_AUDIT.md` | implemented_with_assumptions | The paper does not define this cross-stage attribution report. |

## Stage 9.3.3 shadow assumptions

| ID | Boundary | Priority | Convention | Evidence | Status | Open point |
|---|---|---:|---|---|---|---|
| A_SHADOW_NUMERICAL_EQUIVALENCE_001 | diagnostic replay | P0 | Derive field tolerances from predeclared float64 floors and `20x` independent-repeat noise, with hard caps; never fit tolerance to the official-shadow difference. | `shadow_equivalence.py`; `shadow_equivalence_contract.json` | implemented_with_assumptions | The paper does not publish a replay-equivalence protocol. |
| A_SHADOW_CONTEXT_BINDING_001 | diagnostic replay | P0 | Bind source, warm, graph, object, collision samples, profiles, canonical SDF, and official previous-final state per frame; reject inconsistent provenance. | `shadow_equivalence.py`; `shadow_context_binding.json` | implemented_with_assumptions | Accepted artifact provenance contains a manifest/runtime commit mismatch in the current run. |
| A_SHADOW_PROFILE_ISOLATION_001 | diagnostic ablation | P0 | Change only the named margin, QuerySet mode, or projection objective; preserve paper weights, bounds, samples, solver profile, and previous state. | `shadow_profile_isolation_audit.json` | implemented_with_assumptions | The paper does not specify shadow profiles. |
| A_LONG_FINGER_ERROR_ATTRIBUTION_001 | diagnostic attribution | P1 | Report index/middle/ring RMSE, Ebone, EIM, contact proxy, counterfactual states, and constraint/gradient evidence without claiming ground-truth contact. | `shadow_equivalence.py`; shadow reports | implemented_with_assumptions | The paper does not publish this causal decomposition. |
| A_SHADOW_REGULARIZATION_ABLATION_001 | diagnostic attribution | P1 | Keep regularization attribution read-only and separate from the formal Eq. (8) objective; do not rank incompatible total objectives. | `shadow_equivalence.py` | implemented_with_assumptions | A paper-exact regularization ablation is not specified. |

## Stage 9.3.4 provenance and causal assumptions

| ID | Boundary | Priority | Convention | Evidence | Status | Open point |
|---|---|---:|---|---|---|---|
| A_SOLVER_EFFECTIVE_PROVENANCE_001 | Stage 9.3.4 provenance | P0 | Compare versioned solver-effective code, profiles, input artifacts, and environment metadata; exclude docs/viewers/tests from numerical closure. | `stage9_3_4.py`; provenance diff | implemented_with_assumptions | Historical package wheels are not preserved by the Stage 10 manifest. |
| A_HISTORICAL_CURRENT_LANE_SEPARATION_001 | Stage 9.3.4 lanes | P0 | Historical exact replay is independent and may be unavailable; it never blocks the new current-lineage baseline. | historical lane manifest and environment audit | implemented_with_assumptions | Exact historical environment may need external reconstruction. |
| A_REFINEMENT_MULTISTART_001 | Stage 9.3.4 multistart | P1 | Keep objective, constraints, solver profile, and strict acceptance fixed while varying only declared initialization seeds and comparing frozen-initial-QuerySet/native phases. | `multistart_results_per_seed.json` | implemented_with_assumptions | The frozen QuerySet hook is an engineering diagnostic, not a paper-specified method. |
| A_BASE_SEED_ABLATION_001 | Stage 9.3.4 base seed | P1 | Use SE(3)-only, no-scale, positive-determinant Kabsch fits and report warm geometry separately from final solves. | `base_seed_warm_geometry.csv` | implemented_with_assumptions | The paper does not specify base-seed fitting weights. |
| A_BASE_SEED_PRIOR_REFERENCE_002 | Stage 9.3.4 initialization | P1 | Treat base fitting as an initialization diagnostic and do not present it as evidence for the formal optimizer basin without a seed-and-prior run. | base-seed manifest | implemented_with_assumptions | A paper-exact initialization prior is not disclosed. |
| A_CAUSAL_BRANCH_ROLLOUT_001 | Stage 9.3.4 rollout | P0 | Require accepted, full-512, multi-frame branch evidence before ranking a causal candidate; otherwise fail closed. | branch rollout manifest | implemented_with_assumptions | No candidate branch is admitted by this bounded run. |
| A_STAGE9_4_HUMAN_DECISION_GATE_001 | Stage 9.4 routing | P0 | Keep `ENTER_STAGE9_4=NO`, `HUMAN_DECISION_REQUIRED=YES`, and `STOP_AFTER_STAGE9_3_4=TRUE` until the complete causal bundle is reviewed. | `stage9_4_readiness.json` | implemented_with_assumptions | Human review is required for any next-stage choice. |

## Stage 9.3.5 projection and causal-closure assumptions

| ID | Boundary | Priority | Convention | Evidence | Status | Open point |
|---|---|---:|---|---|---|---|
| A_STAGE9_3_5_LINEAGE_LOCK_001 | Stage 9.3.5 inputs | P0 | Bind every diagnostic report to the current-lineage manifest, Stage 10 manifest, selected frames, robot, canonical SDF, and full 512-point surface; fail closed on mismatch. | `stage9_3_5.py`; `input_identity_and_lineage.json` | implemented_with_assumptions | The paper does not publish this diagnostic lineage contract. |
| A_STAGE9_3_5_STATE_METRIC_001 | Stage 9.3.5 projection | P0 | Keep `projection_state_metric` separate from Eq. (8)-(9), QuerySet slack, formal solver status, and accepted trajectories. | `projection_state_metric.json` | implemented_with_assumptions | The paper does not specify a projection state metric. |
| A_STAGE9_3_5_FULL512_001 | Stage 9.3.5 feasibility | P0 | Evaluate warm, final, path, and projection states against all 512 canonical collision samples; no reduced-grid proxy may pass the gate. | `warm_final_path_feasibility.json`; `projection_independent_validation.json` | implemented_with_assumptions | The paper does not publish a causal feasibility scan. |
| A_STAGE9_3_5_BRANCH_GATE_001 | Stage 9.3.5 branch | P0 | Run a branch rollout only after the declared long-finger improvement, full-512, strict feasibility, and multi-frame gate passes; otherwise report not required. | `branch_gate.json`; `branch_rollout_manifest.json` | implemented_with_assumptions | No paper-specified causal branch or improvement threshold exists. |
| A_PROJECTION_STATE_METRIC_001 | Stage 9.3.5 projection | P0 | Keep the warm-centred projection metric versioned and separate from Eq. (8)-(9), formal solver status, and accepted trajectories. | `projection_state_metric.json`; `projection_solver_results.json` | implemented_with_assumptions | The paper does not specify a projection state metric. |
| A_WARM_FINAL_PATH_FEASIBILITY_001 | Stage 9.3.5 path | P0 | Use the SO(3)-geodesic warm-to-final path, at least 1001 alphas, full-512 signed distance, all feasible intervals, and refined boundaries without a monotonicity assumption. | `warm_final_path_feasibility.json`; `warm_final_path_feasibility.csv` | implemented_with_assumptions | The paper does not publish a path-feasibility diagnostic. |
| A_PROJECTION_FEASIBILITY_RESTORATION_001 | Stage 9.3.5 projection | P0 | Treat solver feasibility and independent canonical validation as separate gates; never accept status 9, reduced samples, or a candidate without full-512 residual checks. | `projection_independent_validation.json`; `projection_solver_results.json` | implemented_with_assumptions | The paper does not specify this diagnostic restoration contract. |
| A_OBJECTIVE_PATH_ATTRIBUTION_001 | Stage 9.3.5 attribution | P1 | Decompose the common formal objective at warm, final, projection, and counterfactual endpoints and report directional/path and variable-group attribution without fabricating multipliers. | `objective_endpoint_decomposition.json`; `objective_directional_attribution.json`; `objective_variable_group_attribution.json` | implemented_with_assumptions | The paper does not publish a causal objective attribution. |
| A_CONSTRAINT_PRESSURE_DIAGNOSTIC_001 | Stage 9.3.5 constraints | P1 | Report per-sample, per-link, per-finger, and interaction pressure from canonical full-512 samples; label it diagnostic pressure, not a dual multiplier or contact ground truth. | `constraint_attribution.json`; `interaction_constraint_joint_attribution.json` | implemented_with_assumptions | The paper does not publish a constraint-pressure decomposition. |
| A_PROJECTION_BRANCH_ROLLOUT_001 | Stage 9.3.5 branch | P0 | Run projection-informed branch rollouts only after strict full-512 feasibility, long-finger improvement, motion-fraction, and multi-frame gates pass; otherwise fail closed. | `branch_gate.json`; `branch_rollout_manifest.json` | implemented_with_assumptions | No paper-specified causal branch or improvement threshold exists. |
| A_STAGE9_3_5_HUMAN_DECISION_GATE_001 | Stage 9.3.5 routing | P0 | Keep `ENTER_STAGE9_4=NO`, `HUMAN_DECISION_REQUIRED=YES`, and `STOP_AFTER_STAGE9_3_5=TRUE` until the complete causal bundle and immutability report receive human review. | `stage9_4_readiness.json`; `official_artifact_immutability.json` | implemented_with_assumptions | Human review is required for any next-stage choice. |
| A_STAGE9_4_ONE_SHOT_CLOSURE_001 | Stage 9.4 closure | P0 | Keep projection diagnostic-only, execute only fixed C0--C7 profiles on `(0,10,30,36,39)`, select one root cause, and permit at most one faithful repair before the full 60-frame gate. | `stage9_one_shot_summary.json`; `stage9_final_decision.json` | implemented_with_assumptions | The paper does not specify this causal-ablation or repair protocol. |
| A_STAGE9_4_EQ9_Q_MEMBERSHIP_001 | Stage 9.4 repair | P0 | Interpret Eq. (9) temporal `q` as finger joint correction and retain base position/rotation priors as separate terms; previous final is remapped to the current seed chart. | `formal_regularization_code_map.json`; `scipy_slsqp_active_set_contact_rich_v3_fixed.yaml` | implemented_with_assumptions | The paper notation does not publish the repository coordinate-vector expansion. |

## Derived sign proxy for open GRAB objects

The frozen quality lane uses `hybrid_original_distance_proxy_sign_v1` as
paper-unspecified geometry engineering. Candidate selection is identity,
deterministic local repair, then fixed 256-axis voxel marching-cubes fallback;
convex hull is not an accepted proxy. The original mesh is immutable and
remains the source for visualization, object samples, closest point, unsigned
magnitude, contact-position target, and provenance. Only the sign comes from a
strict watertight derived proxy. Boundary-loop, synthetic-patch, 20k deviation,
source-contact, and active-QuerySet evidence are persisted under the experiment
geometry directory.

The current G3 retry proves proxy validity but routes to
`SIGN_PROXY_CONTACT_REGION_CONFLICT` because active QuerySet samples intersect
the fixed original-boundary exclusion zone. This is a formal fail-closed result,
not evidence that the open mesh can be silently accepted and not an A–E
completion claim.

## Wuji Hand2 three-clip boundary

The W1/W2/W3 suite is fixed to subject `s1` and native 60-frame windows. It
uses a generic target-hand registry with 20 DoF and `r_wrist`; no Wuji-only
loss or solver branch is introduced. The formal collision surface excludes
visual soft pads. Independent full-surface validation uses a fresh strict
reference backend and reports GRAB semantic contacts as `DATASET_PROXY` only.

The benchmark demonstrates within-subject multi-object behavior, not the
paper's original hardware equivalence, cross-subject generalization, or
real-time performance.

## W2.3 sequential finalization assumptions

`wuji_continuous_sequential_v1` is an engineering candidate derived from the
full-state continuous profile. Its only solver-semantic difference is
`window.fallback_enabled=false`; metadata and scope differences are explicit.
The production path is audited separately from the diagnostic five-frame
window, and all new artifacts are isolated under the W2.3 output root.

`R_pen(0 mm)` is retained as a sensitive signed-distance diagnostic while
`R_pen(2 mm)` is the paper hard threshold. A W3 0.90 to 0.95 zero-threshold
change is therefore reported as a shallow numerical/mesh warning when the
2 mm gate and maximum-depth bound pass. The window oracle uses local frame 34
as a fixed anchor and treats future states as hints; unresolved window solver
status cannot block the sequential recommendation.

## Final-refinement performance boundary

`wuji_continuous_sequential_fast_exact_v1` is a performance-candidate execution
profile. It keeps CPU float64, objective/constraint definitions, collision samples,
strict active-set semantics, recovery, determinism, and the independent final
full-surface audit unchanged. Cache/batch/affinity instrumentation is engineering
only; the paper does not specify SciPy SLSQP, the 672-sample Wuji surface,
active-set round scheduling, or thread policy. It remains non-default and
`author_exact: unresolved` until explicit parity gates pass.

## P4 compiled winding execution boundary

The v4 compiled generalized-winding handle and certified FD-probe reuse are
paper-unspecified engineering. They preserve float64, the generalized-winding
definition, thresholds, collision samples, solver, and independent full audit.
Near-threshold and uncertified probes retain exact reference fallback; v4 is
not recommended or selected for Stage-12 absent qualification evidence.

## P2 analytic SDF execution boundary

`wuji_continuous_sequential_fast_exact_v2` is engineering-only and non-default.

`wuji_continuous_sequential_fast_exact_v3_compiled_cpu` is likewise an
engineering-only experimental backend. It accelerates only ambiguous spatial
FD closest-point probes; the paper does not prescribe a compiled backend.
It preserves CPU float64, formal collision sample identity, solver budgets, and
the final independent full-surface audit. The paper does not prescribe this BVH,
spatial-gradient, or Lipschitz-cache implementation; `author_exact` remains unresolved.

## Generic solver-feasibility restoration boundary

Recoverable SLSQP status-8/9 slack restoration is paper-unspecified solver
engineering. It preserves the original objective, physical-unit constraints,
QuerySet, bounds, tolerance, float64 execution, and independent full audit. It
is forbidden from selecting behavior by dataset, object, or sample; author
exactness remains unresolved.
