# Stage 16-C.2 Isaac Lab DirectRLEnv contract

`IsaacWorldWristFingerDirectRLEnv` is an `ENGINEERING_EXTENSION` for the
frozen Isaac Sim 5.1.0 / Isaac Lab v2.3.2 lane. It is not a paper-method,
hardware-control, sim-to-real, oracle, or PPO result.

## Frozen contract

- Two immutable world-wrist references: `hocap_170105`, `hocap_170650`; 41
  frames at 20 Hz, with 20 canonical Wuji finger joints and 16 tracked links.
- One 26-D action: `[0:3]` local wrist translation at 0.01 m, `[3:6]` local
  SO(3) logarithm residual at 5 degrees, and `[6:26]` canonical finger residuals
  scaled to 10 percent of each range. The canonical-to-Isaac mapping is explicit.
- Global Cartesian wrist wrench: translational stiffness/damping 250 N/m/1.0,
  rotational 2 Nm/rad/0.5, force limit 25 N, torque limit 1.5 Nm, and reference
  twist feed-forward 1.0. Finger drive settings are stated in the control YAML.
- Zero gravity, no ground or support, active object mass 0.05 kg. Observations
  are world-frame at the documented 0/1/3/5 reference offsets; root quaternion
  is Isaac's `wxyz`, and root velocities are world-frame.
- Formal termination is 5 cm object position and axis error, 45 degree object
  orientation error, and the pre-existing wrist 20 cm/90 degree safety bound.
  Failure takes precedence over final-frame success.

During formal rollout the task applies wrench and finger targets only. Object
and wrist root state are written only on reset; the runtime contract reports
`object_rollout_state_writes=0` and
`wrist_root_state_writes_during_step=0`. The C.3 kinematic-object probe is a
separate, explicitly non-formal diagnostic and reports its own state writes.

## C.2 result

Real RTX 5080 / CUDA PhysX smoke reports under
`.local/reports/stage16c2_c5_isaaclab/` are all
`STAGE16C2_DIRECT_RL_ENV_VALIDATED`:

| Run | Scope | Result |
| --- | --- | --- |
| `c2_smoke_1env.json` | 1 env, 1000 steps | finite, valid lifecycle/reset evidence |
| `c2_smoke_1env_alternating.json` | 1 env, both clips, 1000 steps | finite and both clips selected |
| `c2_smoke_128env.json` | balanced 128 envs, 1000 steps | finite CUDA observations/actions and unique action rows |

Use the bounded smoke runner only with the process-scoped EULA authorization:

```bash
conda run -n toporetarget-isaaclab env OMNI_KIT_ACCEPT_EULA=YES \
  python scripts/rl/isaaclab/smoke_stage16c2_direct_env.py \
  --num-envs 128 --steps 1000 --balanced-clips --accept-eula
```

## Gate boundary

C.3 is `STAGE16C3_SEMANTIC_QUALIFICATION_PARTIAL`, not a qualified PhysX
semantic/contact result. It records diagnostic wrist errors above 2 cm/10
degrees and no direct all-hand contact-pair/impulse proof. C.4 vector
qualification and C.5 oracle are therefore `NOT_RUN_GATE_BLOCKED_BY_C3`; C.6
PPO is not authorized, with zero samples and zero checkpoints.
