# Open questions for the authors

The following questions are written as a reproducibility request. No author response is assumed.

## P0 — blocks a one-to-one method implementation

1. Could you provide the precise source and robot wrist-centered frame definitions and the
   source-to-robot MediaPipe keypoint mapping (A_HAND_FRAME_001, A_ROBOT_HAND_FRAME_001)?
2. Is there an intended MANO-to-MediaPipe21 semantic and fingertip-anchor profile, including the
   MANO topology/model release? The current adapter records this as A_MANO_MEDIAPIPE_SEMANTICS_001
   and A_MANO_FINGERTIP_VERTICES_001 because the paper accepts MediaPipe-style inputs but does not
   disclose a MANO conversion.
3. Which optimizer, variable parameterization, joint-limit implementation, termination criteria,
   line search, and collision-query set are used for Eq. 8 (A_SOLVER_001,
   A_SOLVER_TERMINATION_001, A_JOINT_LIMIT_001, A_COLLISION_QUERY_SET_001)?
4. Which signed-distance implementation and gradient convention are used for penetration
   constraints (A_SIGNED_DISTANCE_BACKEND_001)?
5. What simulator, version, physics solver, and actuator model were used for Tables 4–6
   (A_RL_SIMULATOR_001)?
6. Can the 32-clip MoCap Pen-Spin dataset and Wuji deployment assets/calibration be released or
   accessed under a reproducible license (A_PRIVATE_PENSPIN_DATA_001, A_WUJI_ASSET_001)?

7. For the target robot hand, are the 21 semantic anchors intended to be joint centers, link
   origins, fingertip markers, or another published profile? Stage 4 records the Arti-MANO profile
   explicitly as A_ROBOT_KEYPOINT_ANCHORS_001 and A_ARTIMANO_KEYPOINT_MAPPING_001.
8. Is the URDF root/palm frame the intended paper robot hand frame, and should the fixed fingertip
   visual spheres participate in later collision queries (A_ROBOT_BASE_FRAME_001,
   A_ARTIMANO_COLLISION_COVERAGE_001)?

9. What are the exact source and robot wrist-frame axes, and does “wrist-centered” remove only
   translation or also the wrist rotation (A_BONE_DIRECTION_FRAME_001)?
10. Which directed bone edges and adjacent pairs define $A_B$, and are wrist-to-MCP edges included
    (A_BONE_PAIR_SET_001)?

## P1 — affects numerical results

1. Are the 50 object samples drawn once per object, per sequence, or per frame, and which surface
   sampler/seed is used (A_OBJECT_SAMPLING_001, A_OBJECT_SAMPLING_METHOD_001,
   A_OBJECT_SAMPLING_SEED_001, A_OBJECT_SAMPLE_TEMPORAL_REUSE_001)? The Stage 6 engineering
   profile uses deterministic area-weighted triangles, PCG64 seed 20260720, and fixed
   object-local face+barycentric anchors.
2. Which Delaunay backend and flags are used, and how are duplicate/coplanar/cospherical inputs
   handled (A_DELAUNAY_BACKEND_001, A_DELAUNAY_DEGENERACY_001)?
3. How is the first warm-start frame initialized, and how is `q_base` rotation parameterized
   (A_FIRST_FRAME_INITIALIZATION_001, A_BASE_PARAMETERIZATION_001)?
4. For the displayed warm-start objective, should the base be optimized despite local-frame
   non-observability, and how should the scene base seed be calibrated from the hand frames
   (A_WARMSTART_BASE_OBSERVABILITY_001, A_BASE_SEED_ALIGNMENT_001)?
5. Are raw joint radians, direct URDF bounds, native-frame temporal terms, and no $Δt$ normalization
   intended for Eq. (2) (A_WARMSTART_COORDINATES_001, A_WARMSTART_JOINT_LIMITS_001,
   A_WARMSTART_TIME_DISCRETIZATION_001)?
6. What robot hand and asset version produced Table 1 (A_TABLE1_TARGET_HAND_001)?
7. What are the exact ContactPose intensity threshold, robot surface sample count, six object
   axis-point construction, and tracked-link list (A_CONTACTPOSE_THRESHOLD_001,
   A_HAND_SURFACE_SAMPLES_001, A_RL_AXIS_POINTS_001, A_RL_TRACKED_LINKS_001)?

## P2 — affects performance or engineering details

1. What are the low-level PD gains, action residual limits, PPO clip ratio, value-loss coefficient,
   and maximum gradient norm (A_RL_PD_GAINS_001, A_PPO_UNLISTED_PARAMS_001)?
2. What exact random-number generator and scheduling policy are used for each Table 5 entry?

## GRAB adapter questions

1. Is the GRAB source scene frame and object row-vector convention intended to be the official
   downstream coordinate contract, including the source MANO translation used for the wrist pose
   (`A_GRAB_SCENE_FRAME_001`, `A_GRAB_WRIST_FRAME_001`)?
2. Should every sequence use its personalized `vtemp`, and is there a released rule for handling a
   missing or incompatible personalized template (`A_GRAB_PERSONALIZED_VTEMP_001`)?
3. The adapter now uses the official `otaheri/GRAB/tools/utils.py` `contact_ids` table for
   semantic labels. Does downstream paper code intend a different aggregation of those labels
   (`A_GRAB_CONTACT_MAPPING_001`)?
4. Should the table be treated as a support surface in later interaction graphs, and is the
   filename-derived Stage 5 sequence ID compatible with the released manifest
   (`A_GRAB_TABLE_ROLE_001`, `A_GRAB_SEQUENCE_ID_001`)?

## P3 — documentation completeness

1. Which exact source assets and rendering scripts generated Figures 1–5?
2. Which release/version of each baseline was used, including Mink, OmniRetarget, DexPilot, and
   GeoRT?
