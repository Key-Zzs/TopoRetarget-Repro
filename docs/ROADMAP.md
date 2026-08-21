# Roadmap

This roadmap describes durable research states, not an experiment diary.
Run-specific receipts remain in ignored local storage and detailed stage/RL
documentation.

## Foundation

The stable foundation is canonical HOI data, semantic MANO conversion,
target-hand kinematics, interaction-aware retargeting, geometry/SDF validation,
manifest-bound exports, and browser-based evidence. Arti-MANO and Wuji Hand2
Beta1 remain the supported target-hand lanes.

## Stage16 raw-mocap replay overlay (implemented)

The authoritative IsaacLab replay can now show recorded PhysX actual poses,
the original provenance-resolved HOCap MANO/object ghosts, and a distinct
geometric-retarget reference ghost in one world frame/timeline. The raw layer
is visual-only and coordinate/time alignment is deterministic and fail-closed.
It supports object-local fingertip diagnostics but makes no PPO, reward,
physics, controller, or reference change. See [raw-mocap replay
overlay](rl/RAW_MOCAP_REPLAY_OVERLAY.md).

## Stage16 contact timing, angular twist, PF, and DF (implemented)

The mainline evidence order is now:

```text
Raw Mocap
    -> Geometric Retarget
    -> Physical Functionality
     + Demonstration Fidelity
```

The additive `SR_dynamic` receipt remains immutable. The offline contracts
separate raw-MANO, retarget-reference, and PhysX contact timing; audit trace
omega against pose-derived omega; and keep physical completion (`PF`) separate
from demonstration fidelity (`DF`).

The angular semantics closeout identifies the historical trace field as
world-frame instantaneous PhysX COM angular velocity and adopts
`Stage16ActualAngularVelocityAuthorityV2`: actual omega is derived from saved
actual pose using the same control-rate SO(3)-log estimator as Reference
Kinematics V2. V4/170650 changes from 2/20 under the legacy trace field to
20/20 under comparable semantics, without rewriting traces or tuning the
inherited threshold.

The raw grasp review shows that Strict V4 is a reward-specific named-finger to
robot-tip target, not a validated functional human-grasp binary. The additive
`RawHumanGraspReadinessProfileV1` reports all-surface, multi-region, topology,
and coupling layers. For 170105, any-surface contact occurs just before LIFT,
while multi-region and Strict-V4 readiness occur after LIFT; functional raw
readiness remains `NOT_IDENTIFIABLE`. Contact-timing attribution is therefore
still `INCONCLUSIVE`, now at medium confidence and explicitly profile-based.
See [actual angular velocity semantics](rl/ACTUAL_ANGULAR_VELOCITY_SEMANTICS.md),
[raw human grasp readiness authority](rl/RAW_HUMAN_GRASP_READINESS_AUTHORITY.md),
[contact timing attribution](rl/CONTACT_TIMING_LAYER_ATTRIBUTION.md),
[angular-twist audit](rl/ANGULAR_TWIST_AUDIT.md), and
[PF/DF](rl/PHYSICAL_FUNCTIONALITY_AND_DEMONSTRATION_FIDELITY.md).

## Stage16 170650 physical-HOI closure and generic profile (CLOSED)

V4/`hocap_170650` is formally `ACCEPTED_STAGE16_PHYSICAL_HOI`: PF, pose,
linear, Angular-Authority-V2, causality, and geometry all pass 20/20. The
lineage is frozen as a physical-HOI data source/positive control and requires
no further PPO or policy adaptation.

`HumanObjectCouplingContactProfileV1` now describes raw contact geometry,
regions, geometric topology, `T_H^-1 T_O`, relative motion, and continuous
coupling for `hocap_170105` and `hocap_170650`. Its cross-layer status is
`PROFILE_PARTIALLY_VALIDATED` because actual contact points/normals and exact
slip are absent. The next and only generic refinement family is
`SOURCE_PROFILE_TRACKING` at medium confidence; support transfer remains an
outcome metric. No refinement or training is started. See [170650
acceptance](rl/STAGE16_170650_ACCEPTANCE.md), [coupling contact
profile](rl/HUMAN_OBJECT_COUPLING_CONTACT_PROFILE.md), and [generic refinement
target](rl/GENERIC_PHYSICAL_REFINEMENT_TARGET.md).

## Stage16-D causal zero-g milestone (CLOSED)

`CAUSAL_ZERO_G_MILESTONE_COMPLETE`

Stage16-D freezes a simplified Isaac/PhysX reference-tracking baseline:

```text
robot action -> hand-object contact -> object dynamics
```

It is physically causal under this frozen contract: gravity is zero, support is
absent, external object guidance is absent, and rollout-time object-state and
wrist-root writes are forbidden. It is not physically realistic, real-world
calibrated, or full-gravity validation.

### Frozen method state

```text
Aggregate V3
  -> STABLE_BASELINE / global default (aggregate_v3)

Strict Per-Finger V4
  -> EXPERIMENTAL_PARTIAL / explicit opt-in (strict_per_finger_v4)
```

V3 is the stable reference-gated aggregate fingertip pair-force objective.
V4 implements source-side MANO contact semantics and strict independent
per-finger force credit. It improves selected interaction-fidelity,
free-flight, or twist diagnostics, but did not consistently exceed V3 on both
clips' physics qualification; it is therefore not the global default.

### Frozen infrastructure

- Reference Kinematics V2
- PPO-26D Isaac Lab backend
- unified V3/V4 contact-reward configuration
- Source Contact Semantics
- Evaluation Suite V2
- full hand-object pair telemetry
- simulation-data export and replay diagnostics

Historical V1/V2/V3/V4 artifacts remain available through provenance-aware
compatibility mapping. The closeout never reinterprets a historical V4 artifact
as V3 merely because a newer configuration now defaults to V3.

## Causal Physical PPO gravity curriculum (EXECUTION)

`feature/ppo-physical` contains the physical bootstrap contracts, Contact-ready
RSI V2, source-support feasibility evidence, and the staged gravity/friction
curriculum. P1 uses bounded full-gravity true-PhysX diagnostics to construct
named safe reset banks without guidance, support injection, or rollout writes.
P2 never substitutes a generic plane/table when source support is unavailable.

The explicit virtual wrist's C4 rotational controller repair is complete:
PhysX ignored the generated USD's per-body hand-gravity opinions after
reduced-coordinate articulation import, so the production spawn now applies
the equivalent runtime articulation override. This preserves gravity-on task
objects and does not authorize PPO, reward, action, or physical-qualification
changes. See [wrist controller root cause](rl/WRIST_ROTATIONAL_CONTROLLER_ROOT_CAUSE.md).

The current execution route runs four independent frozen lineages (V3/V4 ×
both clips) continuously from zero-g through C0--C4. Completion of each fixed
sample budget promotes the next stage. Saturation, optimization health,
interaction, twist, penetration, reference geometry, and Evaluation Suite V2
are final diagnostics, never PPO curriculum stop gates. See [Physics
curriculum](rl/PHYSICS_CURRICULUM.md).

The frozen V3 `hocap_170105` C1 saturation gate also has durable pre-gate
instrumentation; it remains a diagnostic/reproducibility interface and does
not authorize C2 or alter the physical route. See [C1 saturation
instrumentation](rl/C1_SATURATION_INSTRUMENTATION.md).

Its historical reproduction remains attribution evidence only. The retained
0.98/0.25 saturation thresholds now emit telemetry warnings and do not prevent
C1--C4 continuation.
The bounded P3-C1.2 PPO optimization attribution is recorded in [C1 PPO
optimization attribution](rl/C1_PPO_OPTIMIZATION_ATTRIBUTION.md); an
`INCONCLUSIVE` result due to missing exact PPO-batch evidence remains fail-closed
and does not authorize formal P3 continuation.

The C0 contact-skill-collapse audit localizes the first transient loss to PPO
update 3 / 122,880 samples and identifies the frame0-only training reset as the
primary cause. A frozen uniform-RSI `[0,320]` counterfactual retains 10/10
deterministic frame-0 contact and lift through update 6, while reward,
controller, reference, action, and runtime-write contracts remain unchanged.
C0 physical training therefore defaults to uniform RSI; formal evaluation
remains frame0. The bounded V3/hocap_170105 continuation completed C0 and C1
from the exact U6 state: both endpoints retain 10/10 frame0 contact, but C0
and C1 endpoint lift is 0/10, with C1 run at 0.25g / 1.75x friction. This is
not authorization for C2--C4 or four-lineage reruns. See [contact-skill
collapse localization](rl/CONTACT_SKILL_COLLAPSE.md).

The follow-up offline grasp/lift localization consumes the 46 new saved
checkpoints/exact batches and 480 frame0 traces without retraining. It fixes
the C0 grasp/lift transition to U25 -> U26: U25 is 10/10 persistent grasp and
lift, while U26 is 0/10 with only late grazing contact. Frozen U26
APPROACH/contact/GRASP restarts also remain 0/10 lift, whereas the U25 GRASP
control remains 10/10. The fail-closed conclusion is
`PPO_OPTIMIZATION_FORGETTING_PRIMARY`, not a reward shortcut or controller
regression. The only permitted next step is
`NEXT_CONTACT_SKILL_POLICY_PRESERVATION_ABLATION`; do not promote the C0
endpoint or start C2--C4. See [contact-skill collapse
localization](rl/CONTACT_SKILL_COLLAPSE.md).

The exact-batch policy-preservation ablation selected an opt-in 0.50x actor-LR
candidate after a paired actor-only/critic-baseline shadow replay. That
single-update result retained 10/10 frame-0 grasp/lift and the U25 GRASP reset,
but did not establish live training preservation. The completed 26-update,
1,048,576-sample V3/`hocap_170105` full-C0 validation instead loses both grasp
and lift at U17 / 696,320 samples (0/10 thereafter and 0/20 at endpoint), while
the frozen 1.0x lineage remains 10/10 at U25 and first collapses at U26. The
actual classification is `CANDIDATE_REGRESSION` and
`STATUS=SHADOW_ONLY_NOT_SUFFICIENT`. Do not switch the production default or
start C1; the only next action is
`NEXT_UPDATE_DEPTH_POLICY_PRESERVATION_ABLATION`. See [full C0 longitudinal
validation](rl/CONTACT_PRESERVING_FULL_C0_VALIDATION.md).

This remains a historical C0 optimization-preservation result, not the current
physical-program main action. The frozen-source C0--C4 gravity/friction sweep
has now received isolated-process timeout/terminal-capture repair and the
authorized minimal-adaptation decision tree. All four C4 receipts are
technically complete. The historical V4/170650 `SR_dynamic V1` result was
2/20; the later comparable-semantics Authority-V2 requalification supersedes
that result for physical-HOI acceptance with PF/DF 20/20. V3/170650 recovered
through C2 but failed the full C3 budget, and V3/170105 C1 plus V4/170105 C4
exhausted their budgets without lift recovery. No further PPO/reward/LR sweep
is authorized; accepted V4/170650 is frozen and 170105 proceeds only through
the selected future object-agnostic profile-tracking refinement. See
[full-gravity capability closure](rl/FULL_GRAVITY_CAPABILITY_CLOSURE.md).

### Support resolution reconstruction (implemented, not promoted)

The source-first resolver, stable-pre-contact planar inference, finite runtime
proxy, geometry audit, and object-only full-gravity PhysX A/B are implemented
for `hocap_170105` and `hocap_170650`. Neither clip has recoverable source
support, so both use an explicitly labeled inferred plane. The proxy receives
continuous contact and approximately `mg`, while object position and quaternion
pose remain stable in the nominal object-only runs. The support contract is
therefore qualified with runtime transfer deferred by the existing hand-object
geometry blocker; it is not an authorization to add a table to the RL
environment or advance P3/G3/P4. See [Support resolution](physics/SUPPORT_RESOLUTION.md).

## Next causal physical stage

P3-B.5 attributes the C2 failure to reset geometry: selected safe-bank states
already fail the frozen geometry gate at frame 0 under all frozen A/B/C/D
counterfactuals. The next permitted work is an explicit repair of the C2 absolute-geometry
failure under the frozen causal contract. The required order after that repair
is:

```text
Contact-ready RSI V2
    ↓
Support Feasibility
    ↓
Gravity + Friction Curriculum
    ↓
Full-gravity / zero-guidance qualification
    ↓
Multi-Clip
```

P3-B.6 completed the full 321-frame physical mask, finite-support RSI bank,
dynamic reset, and joint zero-replay receipts. Both clips remain
`P3_RESTART_BLOCKED_REFERENCE_GEOMETRY`; PPO was not started. See [P3-B.6
physical scene and RSI requalification](rl/PHYSICAL_SCENE_RSI_REQUALIFICATION.md).

External guidance or data-H2R remains an assisted fallback after this causal
physics route; it is not a replacement for it.

## Documentation entry points

- [Dexplore-style multiplicative reward and RSE](rl/DEXPLORE_STYLE_MULTIPLICATIVE_REWARD_RSE.md)
- [Stage16-D causal zero-g milestone](stages/STAGE16D_CAUSAL_ZERO_G_MILESTONE.md)
- [Stage 16 Physical Bootstrap](stages/STAGE16_PHYSICAL_BOOTSTRAP.md)
- [Physics curriculum](rl/PHYSICS_CURRICULUM.md)
- [Hand gravity control abstraction](rl/HAND_GRAVITY_CONTROL_ABSTRACTION.md)
- [Wrist rotational controller root cause](rl/WRIST_ROTATIONAL_CONTROLLER_ROOT_CAUSE.md)
- [Stage16 full-gravity causal status](stages/STAGE16_FULL_GRAVITY_CAUSAL.md)
- [Stage 16-D physics contract](stages/STAGE16D_PHYSICS_CONSISTENT_RETARGETING.md)
- [Reference Kinematics V2 contract](rl/REFERENCE_KINEMATICS_CONTRACT.md)
- [PPO-26D contract](rl/REFERENCE_TRACKING_PPO_26D.md)
- [C1 PPO optimization attribution](rl/C1_PPO_OPTIMIZATION_ATTRIBUTION.md)
- [Contact-preserving full C0 validation](rl/CONTACT_PRESERVING_FULL_C0_VALIDATION.md)
- [Frozen source policy gravity sweep](rl/FROZEN_SOURCE_POLICY_GRAVITY_SWEEP.md)
- [Actual angular velocity semantics](rl/ACTUAL_ANGULAR_VELOCITY_SEMANTICS.md)
- [Raw human grasp readiness authority](rl/RAW_HUMAN_GRASP_READINESS_AUTHORITY.md)
- [Reference-gated contact reward V3](rl/REFERENCE_GATED_CONTACT_REWARD.md)
- [Strict per-finger contact reward V4](rl/STRICT_PER_FINGER_CONTACT_REWARD.md)
- [Source contact semantics](rl/SOURCE_CONTACT_SEMANTICS.md)
- [Evaluation Suite V2](rl/EVALUATION_SUITE_V2.md)
- [Paper fidelity policy](PAPER_FIDELITY.md)
## P3-B.7 restart contract

P3-B.7 distinguishes an immutable geometric reference (diagnostic soft
target) from a physically valid hard reset and the actual PPO trajectory
(both hard gates).  A failed reference-wide geometry audit no longer blocks
training by itself; absent safe early table-supported resets does.

## Stage16 grouped-reward/RSE bounded refinement

The opt-in grouped multiplicative reward and reference-scoped exploration
passed offline and no-step runtime gates, then completed the preregistered ten
V4/`hocap_170105`/C4 updates (409,600 samples). U10 improved lift from 0/10 to
6/10, but PF remained 0/10 because persistent multi-contact followed LIFT;
Confirm20 was not triggered. The classification is
`MULTIPLICATIVE_RSE_REFINEMENT_PARTIAL`. Accepted V4/170650 remained an offline
positive control with no PPO. The only next action is
`NEXT_DIAGNOSE_MULTIPLICATIVE_RSE_RESIDUAL_FAILURE`, not additional tuning.
