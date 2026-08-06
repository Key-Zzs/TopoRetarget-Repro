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

Both formal geometry gates are blocked. The demonstration dataset was not
created; BC and single-clip PPO did not run; samples are zero and best/last
checkpoints do not exist. Two-clip PPO and the sensitivity audit also did not
run. These are gate outcomes, not training failures.

The implemented entry points are:

```bash
conda run -n toporetarget-rl python scripts/rl/isaaclab/build_stage16d_demonstrations.py \
  --qualification .local/reports/stage16d_physics_consistent_retargeting/trajectory_qualification_170105.json \
  --qualification .local/reports/stage16d_physics_consistent_retargeting/trajectory_qualification_170650.json \
  --geometry .local/reports/stage16d_physics_consistent_retargeting/geometry_audit_170105_v3.json \
  --geometry .local/reports/stage16d_physics_consistent_retargeting/geometry_audit_170650_v3.json \
  --output .local/reports/stage16d_physics_consistent_retargeting/demonstration_manifest.json

conda run -n toporetarget-rl python scripts/rl/isaaclab/train_stage16d_single_ppo.py \
  --clip hocap_170105 \
  --qualification .local/reports/stage16d_physics_consistent_retargeting/trajectory_qualification_170105.json \
  --geometry .local/reports/stage16d_physics_consistent_retargeting/geometry_audit_170105_v3.json \
  --bc-output .local/reports/stage16d_physics_consistent_retargeting/bc_training_170105.json \
  --output .local/reports/stage16d_physics_consistent_retargeting/ppo_training_170105.json

conda run -n toporetarget-rl python scripts/rl/isaaclab/qualify_stage16d_ppo.py \
  --training .local/reports/stage16d_physics_consistent_retargeting/ppo_training_170105.json \
  --output .local/reports/stage16d_physics_consistent_retargeting/ppo_evaluation_170105.json

conda run -n toporetarget-rl python scripts/rl/isaaclab/train_stage16d_two_clip_ppo.py \
  --evaluation .local/reports/stage16d_physics_consistent_retargeting/ppo_evaluation_170105.json \
  --evaluation .local/reports/stage16d_physics_consistent_retargeting/ppo_evaluation_170650.json \
  --output .local/reports/stage16d_physics_consistent_retargeting/two_clip_ppo.json
```

Today every command above exits with its explicit `NOT_RUN` or
`NOT_AUTHORIZED` record. None silently starts workers or creates a checkpoint.
