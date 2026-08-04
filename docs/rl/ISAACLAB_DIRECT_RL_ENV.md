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
- The wrist target is interpolated at every 120 Hz physics substep from the
  preserved 20 Hz keys (Hermite translation, shortest-arc SLERP rotation).
  Wrenches are refreshed through Isaac Lab's instantaneous composer every
  substep. Finite impedance and computed-wrench profiles are implementation
  candidates, not a C.3-qualified controller.
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

C.3R2 contact telemetry is observational: two object-centric one-body views
filter the same 21 collision-bearing hand bodies, rather than reading 21
hand-centric views. `off`, `aggregate`, and `diagnostic` have no reward or
control effect. The contact-enabled scene uses USD cloning (`clone_in_fabric`
false) because Fabric-cloned ContactSensor views fail to resolve at 128 envs
in this frozen Isaac Sim 5.1 runtime.

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

C.3R2 completed `C3_CONTACT_READOUT_VALIDATED` with real 1/128-env CUDA PhysX
child-process evidence. C3-0 then completed
`C3_REFERENCE_OR_FRAME_CONTRACT_VALIDATED`: its diagnostic target is the
canonical-URDF FK derived from source wrist pose plus finger state. The stored
link field remains unchanged and continues to serve its existing
observation/reward contract.

The generic D6 wrapper exposes zero D6 tensor joints in the live GPU contract;
therefore the permitted explicit serial 3P+3R articulation was qualified. It
has six GPU tensor joints, a fixed virtual anchor, no Cartesian-wrench
fallback, and no real arm. Under the original 41-key timing, its PD,
full-articulation computed-torque and bounded-preview paths all fail; that
C3R4 result remains immutable historical evidence.

The user then explicitly authorized one global reference retiming. C3R5
preserves both source NPZ hashes and every source key while deriving a shared
factor-8, 321-sample view at the same 20 Hz runtime cadence. With unchanged
gains, effort bounds, 26-D action, 764-D observation and gates, the shared
`high_authority_bounded` explicit-wrist profile passes both clips. C3-1 through
C3-5, all 26 action bases, task contact causality, reset/termination
classification and zero formal wrist/object rollout writes pass. The current
status is `STAGE16C3_SEMANTIC_QUALIFICATION_VALIDATED`, which opens C.4 only.
C.4 then completes all formal 128/512/1024/2048/4096 aggregate-contact counts
with clean exits, finite tensors, exact action/observation/contact shapes, and
zero formal wrist/object rollout writes. The result is
`STAGE16C4_GPU_VECTOR_BACKEND_VALIDATED`; 4096 environments measure 700.35
samples/s, 3731 MiB process-VRAM peak, and zero contact warnings under shared
GPU load. The selected rollout geometry is 4096 environments, length 16, four
shards, and 65536 samples/update. This validates vector-backend capacity and
stability, not linear scaling or training-optimal throughput. C.5A then
validates the candidate-state contract and O0 allocation/isolation at
1/32/96/144 candidates, but its required natural no-clone baseline exceeds hard
caps. It is therefore `STAGE16C5A_BLOCKED_PHYSX_REPLICATION_BASELINE_NONDETERMINISM`:
O1, history replay, benchmark, C5B and C.6/PPO are not run, and tolerance is not
softened. See `ISAACLAB_CONTACT_CAUSALITY.md`, `ISAACLAB_WRIST_DYNAMICS.md`,
`ISAACLAB_STATE_REPLICATION.md`, and the ignored C5A report bundle.
