# Reference-tracking MDP

At `k_t`, action is a residual finger command and the low-level target is `clamp(q_ref[k_t] + a_t)`. Observation concatenates `q`, `qdot`, previous action, current six axis points, then reference q/axis/link features at `[0, 1, 3, 5]`; reference features are never noised or delayed.

Reference reset samples `k0` uniformly. `k_t = min(k0 + t, N - 1)`. Timeout and all other termination classes are mutually exclusive. Success is only the final reference frame without an earlier failure.

Training uses `dt=0.01`, decimation 5, 20 Hz controls/references, 400-step maximum episodes, and 40 control-step PPO rollouts. See `configs/rl/stage16/paper_protocol.yaml`.
