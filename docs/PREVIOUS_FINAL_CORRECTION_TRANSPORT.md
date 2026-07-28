# Previous-Final Correction Transport

The optimizer chart is `scene_local_seed_delta_exp_left`:

```text
c_base = [p_final - p_warm, Log(R_final R_warm^T)]
R(c, warm) = Exp(c_rot) R_warm
p(c, warm) = p_warm + c_pos
```

For the next warm frame, the accepted previous correction is decoded through
the current warm pose. Finger correction is transported as
`clip(q_warm_t + (q_final_{t-1} - q_warm_{t-1}), q_min, q_max)`.
The clamp count and source frame IDs are persisted. Frame zero uses the warm
state and has no temporal term. No future final state or source MANO pose is
used as an optimizer initialization.

`encode_base_correction` and `decode_base_correction` are inverse-tested at
float64 tolerance `1e-10` for the `r_wrist` scene base frame.
