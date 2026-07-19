# Assumptions and unpublished details

This register intentionally records missing information instead of silently selecting values.
Each identifier is referenced from `docs/PAPER_FIDELITY.yaml` where it affects traceability.

| ID | Category | Impact | Provisional handling | Evidence | Status | Resolution criteria |
| --- | --- | --- | --- | --- | --- | --- |
| A_HAND_FRAME_001 | coordinate convention | P0 | Keep frame conversion blocked; document wrist-centered wording only. | Sec. 3.2, Eq. 1 | pending_author_confirmation | Author-provided source wrist-centered frame definition. |
| A_ROBOT_HAND_FRAME_001 | coordinate convention | P0 | Keep robot-frame mapping blocked. | Sec. 3.1-3.2 | pending_author_confirmation | Author-provided robot wrist/base frame convention. |
| A_BASE_PARAMETERIZATION_001 | optimization variable | P1 | Do not choose Euler, axis-angle, or quaternion. | Appendix A.1, Eq. 9 | pending_author_confirmation | Exact `q_base` rotation coordinates and constraints. |
| A_FIRST_FRAME_INITIALIZATION_001 | initialization | P1 | Require an explicit first-frame seed before implementation. | Sec. 3.2, Eq. 2 | pending_author_confirmation | First-frame warm-start rule or released code. |
| A_OBJECT_SAMPLING_001 | geometry | P1 | Configure 50 samples but do not choose temporal reuse semantics. | Sec. 3.3, Table 3 | pending_author_confirmation | Per-frame/per-sequence/per-object sampling rule. |
| A_OBJECT_SAMPLING_METHOD_001 | geometry | P1 | Do not select a surface sampler or seed. | Sec. 3.3 | pending_author_confirmation | Sampling algorithm, seed, and determinism policy. |
| A_DELAUNAY_BACKEND_001 | geometry | P0 | Leave backend and flags unimplemented. | Sec. 3.3, Eq. 4 | pending_author_confirmation | Library, dimensionality, flags, and version. |
| A_DELAUNAY_DEGENERACY_001 | geometry | P1 | Fail fast on degenerate input until policy is known. | Sec. 3.3 | pending_author_confirmation | Coplanar, cospherical, and duplicate-point handling. |
| A_SOLVER_001 | optimization | P0 | Do not pick an optimizer. | Sec. 3.4, Eq. 8 | pending_author_confirmation | Exact constrained solver and variable parameterization. |
| A_SOLVER_TERMINATION_001 | optimization | P0 | Keep iteration/tolerance/line search null. | Sec. 3.4 and Table 3 boundary | not_provided | Released implementation or author response. |
| A_JOINT_LIMIT_001 | robot constraints | P0 | Record Eq. 8 constraint scope without inventing joint-limit equations. | Sec. 3.4 | pending_author_confirmation | Exact joint-limit and self-collision implementation. |
| A_COLLISION_QUERY_SET_001 | collision | P0 | Keep `Q_t` construction blocked. | Sec. 3.4, Eq. 8 | pending_author_confirmation | Hand-object pair query construction. |
| A_SIGNED_DISTANCE_BACKEND_001 | collision | P0 | Do not select an SDF/mesh query backend. | Sec. 3.4 and Appendix A.3 | pending_author_confirmation | Signed-distance convention, gradients, and mesh preprocessing. |
| A_HAND_SURFACE_SAMPLES_001 | metric | P1 | Keep robot surface sample count null. | Appendix A.3, Eq. 12 | not_provided | Released metric code or explicit sample rule. |
| A_CONTACTPOSE_THRESHOLD_001 | metric | P1 | Keep intensity threshold null; preserve all other attribution rules. | Appendix A.3 | not_provided | Contact intensity threshold from authors or ContactPose code. |
| A_TABLE1_TARGET_HAND_001 | experiment | P1 | Report Table 1 without naming an unverified target hand. | Table 1 | pending_author_confirmation | Target hand identity and asset/version. |
| A_RL_TRACKED_LINKS_001 | RL observation | P1 | Keep tracked link list null. | Sec. 4 and Appendix A.5.2 | not_provided | Exact list/order of tracked links. |
| A_RL_AXIS_POINTS_001 | RL observation | P1 | Keep axis-point spatial offsets null; record six points only. | Appendix A.5.2 and Table 4 | not_provided | Exact six-point construction and offsets. |
| A_RL_SIMULATOR_001 | RL infrastructure | P0 | Do not install or select a simulator in Stage 0/1. | Appendix A.5 | pending_author_confirmation | Simulator, version, and physics solver settings. |
| A_RL_PD_GAINS_001 | RL infrastructure | P1 | Keep low-level gains null. | Appendix A.5.5 | not_provided | Gains or actuator configuration used for results. |
| A_PPO_UNLISTED_PARAMS_001 | RL optimization | P1 | Keep clip/value/gradient values null. | Appendix A.5.6 | not_provided | Full PPO configuration or released training code. |
| A_PRIVATE_PENSPIN_DATA_001 | dataset | P0 | Treat self-collected Pen-Spin data as unavailable. | Appendix A.4 | blocked | Author release or access procedure for 32 clips. |
| A_WUJI_ASSET_001 | hardware | P0 | Record transfer claim; do not claim hardware reproduction. | Sec. 5.2, Fig. 4 | blocked_missing_asset | Wuji URDF, calibration, controller, and deployment assets. |

