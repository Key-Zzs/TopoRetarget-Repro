# Physics-Correction PPO

The Stage 16-D PPO lane retains the frozen 764-D observation and 26-D action
contracts. It can start only from a formally qualified
`PhysicsConsistentRetargetedTrajectoryV1`; an empirical optimized seed is not
training authorization.

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

## D.4R2 superseding entry decision

`STAGE16D_GEOMETRY_GATE_REVISION_BLOCKED` now precedes all commands above.
V1 contact-preserving attainability was not demonstrated, and no legal stable
dynamic-contact floor exists for V2. Online geometry qualification, G1/G2,
demonstration export, BC, environment scaling, single PPO, two-clip PPO, and V2
export are all `NOT_RUN_GATE_BLOCKED`; samples and checkpoints remain zero.
