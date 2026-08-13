# Reference-tracking MDP

At `k_t`, action is a residual finger command and the low-level target is `clamp(q_ref[k_t] + a_t)`. Observation concatenates `q`, `qdot`, previous action, current six axis points, then reference q/axis/link features at `[0, 1, 3, 5]`; reference features are never noised or delayed.

Reference reset samples `k0` uniformly. `k_t = min(k0 + t, N - 1)`. Timeout and all other termination classes are mutually exclusive. Success is only the final reference frame without an earlier failure.

Training uses `dt=0.01`, decimation 5, 20 Hz controls/references, 400-step maximum episodes, and 40 control-step PPO rollouts. See `configs/rl/stage16/paper_protocol.yaml`.

## Stage 16-B world wrist-and-finger extension

`WORLD_WRIST_FINGER_TRACKING_PROTOCOL` is explicitly an `ENGINEERING_EXTENSION`.
Its action is `a[0:6]` (local wrist translation/rotation residual in
`[-1, 1]`) plus `a[6:26]` (the original 20 finger residuals). The selected
global scale is 10 mm, 5 degrees, and 10% of each finger joint range. The
low-level wrist target is `T_world_wrist_ref[k+1] * exp(a_wrist)` and applies
a finite, clipped wrench rather than directly writing wrist pose. The free
object's pose and velocity are initialized only at reset; formal rollouts do
not teleport it.

The 764D observation has current world-wrist error/twist, finger state,
previous 26D action, current object state, and `[0,1,3,5]` references in both
world and wrist-relative coordinates. Reward keeps object/link/finger terms
and adds wrist position/rotation tracking and action smoothness. Existing
object Table-4 termination is preserved; 20 cm wrist-position and 90 degree
wrist-orientation safety limits are explicit engineering safety gates.

The Stage-16B oracle is clone-only contact-aware CEM over a true H-by-26
action sequence with the same 26D bounds at H=1/5/10. Receding-horizon
execution applies only the first action. It is a controllability gate, never
a PPO result.
