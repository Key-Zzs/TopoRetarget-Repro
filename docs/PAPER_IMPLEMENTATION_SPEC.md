# TopoRetarget implementation specification

This is a research implementation contract, not an algorithm implementation. It separates facts
stated by the paper from choices that remain blocked.

## Inputs and outputs

The input is a human-hand trajectory $P^h_{1:N}$ of MediaPipe-style 21 keypoints, an object pose
trajectory $q^o_{1:N}$, and an object mesh $M$. The output is a robot base-pose trajectory
$q^{base}_{1:N}$ and robot joint trajectory $q^\theta_{1:N}$. Exact source/robot frame conventions
remain assumptions `A_HAND_FRAME_001` and `A_ROBOT_HAND_FRAME_001`.

The repository now has bounded source-hand and target-hand infrastructure before this paper input boundary:
`mano16_smplx`/MANO geometry → explicit `mediapipe21` semantics. It preserves scene-frame data and
records `A_MANO_MEDIAPIPE_SEMANTICS_001`, `A_MANO_FINGERTIP_VERTICES_001`, and the mapping-profile
hash. Stage 4 additionally defines `RobotHandSpec`, differentiable URDF FK, and an explicit
Arti-MANO RH/LH target anchor profile, recording `A_ROBOT_KEYPOINT_ANCHORS_001`,
`A_ARTIMANO_KEYPOINT_MAPPING_001`, `A_ROBOT_BASE_FRAME_001`, `A_ARTIMANO_COLLISION_COVERAGE_001`,
and `A_ARTIMANO_DOF_ORDER_001`. These are repository infrastructure, not a disclosed retargeting
method or MANO-to-robot conversion.

## Initialization

For each non-terminal finger keypoint, form a unit bone direction in each wrist-centered hand
frame. For adjacent bone pairs $\mathcal A_B$, implement Equation 1:

$$E_{bone}(q)=\sum_{(k,l)\in\mathcal A_B}\left\|(d^r_k(q)-d^r_l(q))-(d^s_k-d^s_l)\right\|_2^2.$$

The warm start is Equation 2:

$$\tilde q^r_t=\arg\min_q\;\lambda_{warm}E_{bone}(q)+\lambda_{smooth}\|q-\tilde q^r_{t-1}\|_2^2.$$

The first-frame seed, keypoint mapping, and optimizer are not provided.

## Stage 7 implementation boundary

The repository implements the Eq. (1)-Eq. (2) initialization as
`toporetarget.retarget`. The default profile uses semantic full finger chains,
local keypoint-derived frames, qpos-only optimization, neutral first-frame
initialization, URDF bounds, native contiguous time, and a post-solver base seed.
These details are assumptions, not paper-exact claims. The independent artifact
schema is `toporetarget.warm_start.v1`; it is intended as input to a future
interaction-aware stage.

## Interaction mesh

At each frame, concatenate the 21 human/robot hand points with $N_o=50$ object surface samples:

$$V^s_t=[P^h_t;O_t],\qquad V^r_t(q)=[P^r_t(q);O_t].$$

Run one source-side Delaunay tetrahedralization on $V^s_t$ and retain its edge connectivity for
both graphs. The bounded implementation uses non-incremental `scipy.spatial.Delaunay` with the
explicit `Qbb Qc Qz Q12` profile and a centroid/bounding-box-diagonal conditioning transform
only for Qhull. For each directed edge, compute source-derived weights:

$$\tilde w_{ij,t}=\exp(-\kappa\|v^s_{i,t}-v^s_{j,t}\|_2^2),\qquad
w_{ij,t}=\frac{\tilde w_{ij,t}}{\sum_{j'\in\mathcal N_t(i)}\tilde w_{ij',t}},$$

with $\kappa=30$. The paper does not disclose the surface sampler, seed, Delaunay backend,
degeneracy policy, or zero-neighbor handling; these remain explicit assumptions in
`docs/ASSUMPTIONS.md`.

## Laplacian refinement

For any vertex set $V$, define:

$$\Delta_t(V)_i=\sum_{j\in\mathcal N_t(i)}w_{ij,t}(v_i-v_j).$$

The interaction-mesh loss is:

$$E_{IM}(q)=\frac{1}{71}\sum_{i=1}^{71}\|\Delta_t(V^r_t(q))_i-\Delta_t(V^s_t)_i\|_2^2.$$

Stage 8 evaluates this expression on frozen Stage 7 qpos/base values. It reuses the exact source
edges, source weights, and 50 object points; it records qpos Jacobians and bounded base
perturbation diagnostics, but does not mutate qpos/base or invoke Eq. 8.

The final constrained problem is Equation 8: minimize the interaction-mesh, bone, regularization,
and slack terms subject to signed-distance soft/hard bounds. Appendix A.1 expands regularization:

$$E_{reg}(q;q^{r,*}_{t-1})=\lambda_{reg}\|q-q^{r,*}_{t-1}\|_2^2+
\lambda_{base,pos}\|q^{base}_{pos}\|_2^2+\lambda_{base,rot}\|q^{base}_{rot}\|_2^2.$$

The locked values are in `configs/paper/retarget.yaml`: $\lambda_{IM}=500$, $\lambda_{warm}=1$,
$\lambda_{bone}=0.1$, $\lambda_{smooth}=\lambda_{reg}=2.5$, $\lambda_{base,pos}=100$,
$\lambda_{base,rot}=1$, soft tolerance $\tau=0.001$ m, hard bound $b=0.030$ m, and slack
penalty $w_s=10^5$. The solver, $Q_t$, SDF backend, and joint-limit realization are blocked.

## RL tracking contract

The reference clip contains finger joints, object pose in the robot base frame, and $L$ tracked
hand-link positions. The residual action is $q^{\theta,tar}_t=q^{\theta,ref}_{k_t}+a_t$.
Observation combines proprioception, object axis points, and current/lookahead reference features;
lookahead offsets are 1, 3, and 5 control steps. Reference-state reset samples $k_0$ uniformly.

The reward uses a Gaussian kernel $\psi(e;\sigma)=\exp(-\|e/\sigma\|^2)$:

$$r_t=w_{obj}r_{obj}+w_{link}r_{link}+w_{joint}r_{joint}+w_{smooth}r_{smooth}.$$

The paper locks weights $(8,1,1,-0.01)$ and scales $(0.04,0.025,0.1)$ for object, links, and
joints. Termination, domain-randomization, and PPO values are transcribed in `rl.yaml`.
Tracked links, six axis-point geometry, residual limits, low-level gains, simulator, and omitted
PPO settings remain null or registered assumptions.

## Metrics, datasets, and baselines

Contact precision and alignment are Equations 10–11; penetration depth and the fraction above
$\delta=0.002$ m are Equation 12. ContactPose attribution uses sigmoid normalization, an
unspecified threshold, nearest assignment to 20 hand bones, and 10 vertices per link. The data
and baseline dependencies are listed in `PAPER_FIDELITY.yaml` and the reproduction notes.

The bounded GRAB Stage 5 adapter is repository input infrastructure, not a module specified by the
TopoRetarget paper. Its official GRAB `contact_ids` semantic labels, lazy index, per-sequence cache,
and diagnostic viewer are engineering extensions at the dataset boundary; they do not alter
Equations 1--12, the interaction graph, penetration optimization, or the RL/PPO contract. The
adapter's fresh real-data closeout passes for the bounded clips when the external MANO model files
are supplied explicitly; those files remain runtime assets and are not represented by a synthetic or
neutral replacement.

## Limitations and extension boundary

The paper reports weaker handling of virtual contacts. MANO-to-MediaPipe21 source adaptation is
implemented as the bounded Stage 3 adapter, and Arti-MANO robot mapping/FK is implemented only as
the bounded Stage 4 target-hand interface; it does not convert MANO/MediaPipe points to robot qpos.
Delaunay and Laplacian graph construction are now implemented for the bounded Stage 8 scope, and
the bounded Stage 9 SDF-constrained refinement is implemented with explicit assumptions. RL/PPO,
baseline code, and non-paper extensions remain intentionally outside these stages.
No module in this repository pretends that those algorithms are already implemented.
