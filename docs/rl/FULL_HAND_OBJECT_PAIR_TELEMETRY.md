# Full Hand--Object Pair Telemetry

The Stage 16-D R2 audit can capture the complete active-object force matrix for
the named 21 hand collision bodies. This is diagnostic telemetry, not a new
reward, observation, action, controller input, state write, attachment, or
force.

Enable it only for an existing frozen Formal20 re-export:

```bash
python scripts/rl/isaaclab/evaluate_stage16d_ppo26d.py \
  --capture-all-frame-zero-replicas \
  --capture-full-hand-object-pair-telemetry \
  --frame-zero-replicas 20 --rsi-replicas 0
```

| Field | Shape | Contract |
| --- | --- | --- |
| `replica_hand_object_pair_force_world` | `[321,20,21,3]`, `float32` | World-frame force on the fixed active object from each named hand collision body, in N. |
| `replica_hand_object_pair_presence` | `[321,20,21]`, `bool` | Exact nonzero pair-force presence, not a reconstructed threshold. |
| `replica_hand_object_pair_force_valid` | `[321,20]`, `bool` | Frame 0 is invalid because it has no post-physics sample; frames 1--320 are valid. |
| `hand_body_names`, `hand_body_indices`, `hand_body_groups` | `[21]` | Stable collision-body manifest. |
| `hand_collision_shape_mapping` | scalar JSON | Collision body/shape provenance. |

The object-side sensor is filtered to hand-body pairs of the active clip. It
therefore excludes inactive objects and does not treat a self-contact as an
object pair. Qualification rejects nonfinite valid samples, bad frame-zero
validity, wrong `[321,20,21,3]` shape, duplicate/missing body mapping, wrong
active clip, or any state-write/hidden-force contamination.

`r_wrist` is an asset base/root collision body. The current Wuji asset has no
separately named palm collision body, so reports call it
`WRIST_BASE_CONTACT_BODY` rather than claiming a palm substitute.

Use the offline qualifier after the trace is written:

```bash
python scripts/evaluation/qualify_stage16d_full_hand_pair_telemetry.py \
  --trace .../trace_full_pair.npz --evaluation .../full_pair_r2_evaluation.json \
  --clip hocap_170105 --output .../qualification.json
```
