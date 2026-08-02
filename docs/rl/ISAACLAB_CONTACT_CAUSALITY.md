# Isaac Lab contact-causality gate

The C.3 scene config resolves one filtered `ContactSensor` for every one of the
21 C.1 collision-bearing hand bodies. Each sensor targets exactly one hand body
and filters only the two frozen HO-Cap objects; virtual tips are intentionally
excluded because they have no collision mesh.

This is inventory and API-contract evidence, not contact causality. The
all-hand runtime collection attempt exits without writing a trace when the 21
filtered views are read. Contact point and friction buffers were disabled for
that multi-sensor path, and no force/impulse trace was produced. Consequently:

- contact causality is `NOT_PROVEN`;
- object motion is not used as a proxy for contact;
- C.3 cannot pass even if the wrist gate were to pass;
- C.4/C.5/PPO remain blocked.

The failure is recorded at
`.local/reports/stage16c3_repair_c5_oracle/contact_capture_status.json`.
