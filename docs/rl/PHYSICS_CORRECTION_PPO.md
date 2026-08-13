# Physics-Correction PPO

The historical physics-correction demonstration/BC lane retains the frozen
764-D observation and 26-D action contracts. Its old
`PhysicsConsistentRetargetedTrajectoryV1` prerequisite does not govern the
separate D.5-R0+ reference-residual PPO-26D lane: that lane is authorized by
its independent Gate A and trains directly against the original factor-8
reference, not a CEM or corrected-trajectory demonstration.

## Entry and training contract

The demonstration builder requires both the 20-replica trajectory report and
the independent geometry gate to pass. Its split unit is a complete trajectory
(80 percent train, 20 percent validation), so frames from one rollout cannot
cross the split. Behavior cloning initializes the actor only, uses no critic
pseudo-labels, and exposes no future state.

Single-clip PPO uses 4096 environments, rollout length 16, a 764-to-26 actor,
GAE 0.95, gamma 0.99, clip 0.2, four epochs, 32 minibatches, learning rate
`1e-4`, and at most one frozen fallback to `5e-5`. The fixed sample ladder is
1,048,576; 4,194,304; 16,777,216; and 67,108,864 samples, with at most two
seeds per clip. Formal selection reloads a checkpoint and evaluates 20
independent deterministic frame-0 episodes for 321 steps.

Two-clip PPO requires both single-clip policies to pass. It uses balanced clip
sampling, one shared controller/reward/normalizer, no clip ID in the
observation, and no clip-specific parameter. The three fixed initializations
are scratch, the 170105 single policy, and the 170650 single policy.

## Current result

The metric implementation itself is validated. It queries all 21 allowed hand
proxy/object proxy pairs with the exact runtime convex geometry, a fixed
`python-fcl==0.7.0.11` backend, and the frozen
`RuntimeCollisionProxyPenetrationV1` contract. Formal p95 is computed only from
contact-active per-frame-worst samples; the all-frame p95 is diagnostic.

Both corrected trajectories pass the absolute max/p95 limits but fail the
source-relative max and p95 limits. `170105` also remains 15/20 after the one
authorized terminal-tail repair; its only authorized global fallback regressed
to 12/20. `170650` remains 20/20 empirically, but that cannot override its
failed geometry gate. The demonstration dataset was not created; BC and
single-clip PPO did not run; samples are zero and best/last checkpoints do not
exist. Two-clip PPO, V2 export, and sensitivity also did not run. These are gate
outcomes, not training failures.

The implemented entry points are:

```bash
conda run -n toporetarget-rl python scripts/rl/isaaclab/build_stage16d_demonstrations.py \
  --qualification .local/reports/stage16d_metric_qualification_and_ppo/trajectory_requalification_170105.json \
  --qualification .local/reports/stage16d_metric_qualification_and_ppo/trajectory_requalification_170650.json \
  --geometry .local/reports/stage16d_metric_qualification_and_ppo/geometry_qualification_170105_terminal_refined.json \
  --geometry .local/reports/stage16d_metric_qualification_and_ppo/geometry_qualification_170650.json \
  --output .local/reports/stage16d_metric_qualification_and_ppo/demonstration_manifest.json

conda run -n toporetarget-rl python scripts/rl/isaaclab/train_stage16d_single_ppo.py \
  --clip hocap_170105 \
  --qualification .local/reports/stage16d_metric_qualification_and_ppo/trajectory_requalification_170105.json \
  --geometry .local/reports/stage16d_metric_qualification_and_ppo/geometry_qualification_170105_terminal_refined.json \
  --bc-output .local/reports/stage16d_metric_qualification_and_ppo/bc_training_170105.json \
  --output .local/reports/stage16d_metric_qualification_and_ppo/ppo_training_170105.json

conda run -n toporetarget-rl python scripts/rl/isaaclab/qualify_stage16d_ppo.py \
  --training .local/reports/stage16d_metric_qualification_and_ppo/ppo_training_170105.json \
  --output .local/reports/stage16d_metric_qualification_and_ppo/ppo_evaluation_170105.json

conda run -n toporetarget-rl python scripts/rl/isaaclab/train_stage16d_two_clip_ppo.py \
  --evaluation .local/reports/stage16d_metric_qualification_and_ppo/ppo_evaluation_170105.json \
  --evaluation .local/reports/stage16d_metric_qualification_and_ppo/ppo_evaluation_170650.json \
  --output .local/reports/stage16d_metric_qualification_and_ppo/two_clip_ppo.json
```

Today every command above exits with its explicit `NOT_RUN` or
`NOT_AUTHORIZED` record. None silently starts workers or creates a checkpoint.

## Phase 1 precondition for any future correction

An object-twist term is only meaningful when reference pose, timestamps, and
stored linear/world-angular twist agree under the declared finite-difference
and quaternion-frame convention.  Stage 16-D's frozen factor-8 references do
not currently meet that precondition.  Consequently, this document is a
future-correction design record, not authority to edit the reward: the current
evidence-based decision is `PHASE3_OBJECT_TWIST_REWARD_NOT_RECOMMENDED`.

Any reconsideration must first version and validate the reference repair, then
complete fresh bounded RSI state-quality and gravity/support counterfactual
diagnostics.  It must preserve the causal 26-D `env.step(action)` pathway and
cannot use contact shaping, contact termination, curriculum physics, attachment,
or external H2R actions as a substitute for that evidence.

## D.4R2 superseding entry decision

`STAGE16D_GEOMETRY_GATE_REVISION_BLOCKED` now precedes all commands above.
V1 contact-preserving attainability was not demonstrated, and no legal stable
dynamic-contact floor exists for V2. Online geometry qualification, G1/G2,
demonstration export, BC, environment scaling, single PPO, two-clip PPO, and V2
export are all `NOT_RUN_GATE_BLOCKED`; samples and checkpoints remain zero.

## D.4R3 PPO authorization result

The newer stable free-object calibration also failed before formal20:
`STAGE16D_STABLE_FREE_OBJECT_GRASP_CALIBRATION_BLOCKED`. This is a calibration
contract failure, not a PPO training failure. BC and PPO workers were not
started; samples and checkpoints remain zero. Two-clip PPO cannot start unless
both single-clip policies independently qualify.

## D.5-R0 superseding PPO gate

The preceding sections preserve the historical D.4 entry decision, but it is
superseded for `Stage16DReferenceResidualAction26DV1`. The training reference
is the original factor-8 Stage16-D reference, never a corrected yellow-object
path or CEM output. Gate A is the only entry gate. It intentionally does not
evaluate terminal contact, terminal stability, final success, exact
hand-object penetration, or old CEM qualification; those are Gate C post-PPO
diagnostics.

The protocol is `TOPORETARGET_PPO_REPRODUCTION_WITH_26D_WRIST_ADAPTATION`.
It retains the paper tracking reward/PPO, adds wrist tracking because the
virtual wrist is controllable, and labels the 6-D wrist residual, explicit
serial 3P+3R wrist, factor-8 timing, and IsaacLab backend as engineering
adaptations. See [REFERENCE_TRACKING_PPO_26D.md](REFERENCE_TRACKING_PPO_26D.md).

## D.5-R5 to D.7 continuation contract

The active 170650 L0 policy has 1,024,000 samples and is
`STAGE16D_PPO26D_L0_COMPLETE_NOT_YET_QUALIFIED`. D.5-R6A resumes its complete
actor, critic, optimizer, observation normalization, RNG state, and cumulative
sample counter to 4,194,304 samples without changing reward, observation,
reference, RSI, physics, controller, action scale, network, LR, or target KL.
It saves bounded milestone checkpoints and evaluates each with the frozen
development seeds. KL early-stop telemetry records requested/actual epochs and
minibatches plus per-epoch/minibatch KL; wrist-translation, wrist-rotation, and
finger group deltas are diagnostic only.

Post-4M branch selection is predeclared: R6B for at least two strong learning
indicators, R6C when RSI terminal contact is at least 0.60 but frame-zero is at
most 0.20 with a 0.40 gap, and plateau/update diagnosis before R6D. A possible
update bottleneck permits exactly one 1M probe changing LR from `1e-4` to
`5e-5`; no KL/clip/reward/physics/action change is coupled to it. Reward V2,
if authorized, adds only bounded non-saturating object progress and resets the
critic/optimizer while preserving the V1 actor and normalization.

R7 is the first use of the 20 unseen frame-zero formal seeds. It is where
terminal contact/stability, causality, self/inter-finger geometry, exact active
runtime-proxy hand-object geometry, action bounds, and no-hidden-write checks
become accepting gates. A Gate C failure is
`STAGE16D_170650_PPO_TRAINED_NOT_PHYSICS_QUALIFIED`, not `PPO_NOT_AUTHORIZED`.
R8 uses the global contract selected on 170650 with a fresh 170105 policy; D.6
requires both single clips to physics-qualify; D.7 labels only qualified
episodes as `PhysicsQualifiedIsaacTrajectory`.
