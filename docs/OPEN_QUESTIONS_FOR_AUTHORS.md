# Open questions for the authors

The following questions are written as a reproducibility request. No author response is assumed.

## P0 — blocks a one-to-one method implementation

1. Could you provide the precise source and robot wrist-centered frame definitions and the
   source-to-robot MediaPipe keypoint mapping (A_HAND_FRAME_001, A_ROBOT_HAND_FRAME_001)?
2. Which optimizer, variable parameterization, joint-limit implementation, termination criteria,
   line search, and collision-query set are used for Eq. 8 (A_SOLVER_001,
   A_SOLVER_TERMINATION_001, A_JOINT_LIMIT_001, A_COLLISION_QUERY_SET_001)?
3. Which signed-distance implementation and gradient convention are used for penetration
   constraints (A_SIGNED_DISTANCE_BACKEND_001)?
4. What simulator, version, physics solver, and actuator model were used for Tables 4–6
   (A_RL_SIMULATOR_001)?
5. Can the 32-clip MoCap Pen-Spin dataset and Wuji deployment assets/calibration be released or
   accessed under a reproducible license (A_PRIVATE_PENSPIN_DATA_001, A_WUJI_ASSET_001)?

## P1 — affects numerical results

1. Are the 50 object samples drawn once per object, per sequence, or per frame, and which surface
   sampler/seed is used (A_OBJECT_SAMPLING_001, A_OBJECT_SAMPLING_METHOD_001)?
2. Which Delaunay backend and flags are used, and how are duplicate/coplanar/cospherical inputs
   handled (A_DELAUNAY_BACKEND_001, A_DELAUNAY_DEGENERACY_001)?
3. How is the first warm-start frame initialized, and how is `q_base` rotation parameterized
   (A_FIRST_FRAME_INITIALIZATION_001, A_BASE_PARAMETERIZATION_001)?
4. What robot hand and asset version produced Table 1 (A_TABLE1_TARGET_HAND_001)?
5. What are the exact ContactPose intensity threshold, robot surface sample count, six object
   axis-point construction, and tracked-link list (A_CONTACTPOSE_THRESHOLD_001,
   A_HAND_SURFACE_SAMPLES_001, A_RL_AXIS_POINTS_001, A_RL_TRACKED_LINKS_001)?

## P2 — affects performance or engineering details

1. What are the low-level PD gains, action residual limits, PPO clip ratio, value-loss coefficient,
   and maximum gradient norm (A_RL_PD_GAINS_001, A_PPO_UNLISTED_PARAMS_001)?
2. What exact random-number generator and scheduling policy are used for each Table 5 entry?

## P3 — documentation completeness

1. Which exact source assets and rendering scripts generated Figures 1–5?
2. Which release/version of each baseline was used, including Mink, OmniRetarget, DexPilot, and
   GeoRT?

