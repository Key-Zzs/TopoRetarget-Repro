# Stage 16-D Phase 3 Object-Dynamics Reward

Phase 3 is authorized only after the V2 reference, twist observability,
meaningful terminal residual, contact-not-primary, and physics-integrity gates
pass.  It is a bounded single-clip experiment: `hocap_170650` only.

## Frozen reward

`TopoRetargetReferenceTrackingReward26DV2` retains every V1 pose, link,
finger, wrist, and 26-D smoothness term.  It adds only

```text
e_v     = ||v_actual_world - v_ref_world_v2||_2
r_v     = exp(-(e_v / sigma_v)^2)
e_omega = ||omega_actual_world - omega_ref_world_v2||_2
r_omega = exp(-(e_omega / sigma_omega)^2)
```

The frozen conservative weights are `w_v=0.5`, `w_omega=0.5`; their combined
maximum contribution is 1.0 and the contract rejects any combined weight above
2.0.  The frozen scale artifact records the V2 pooled reference statistics and
terminal-dynamics provenance.  Contact, terminal,
penetration, guidance, gravity, and clip-specific rewards remain prohibited.
The terms match signed world twists, never speed magnitude alone.

## Policy and initialization

The frozen 764-D actor observation already contains the current 6-D object
twist; no future actual state and no observation-dimension change is allowed.
V2 reference twist is supplied to the reward backend, which asserts V2
metadata.  Training initializes actor and observation normalization from the
V1 L0 checkpoint while resetting critic and optimizer; `reward_v2_samples` is
a distinct counter.

## Bounded protocol

Run a real host-GPU probe and choose capacity from bounded PPO smoke evidence.
Then run P1 (at least 1,048,576 Reward-V2 samples), development evaluation at
each roughly-1M checkpoint, and at most 4,194,304 samples before deciding
whether the frozen reward is effective.  Extension to 16,777,216 samples is
permitted only after the 4M effectiveness rule passes.  Formal holdout results
are post-selection evaluation only.

The sample limits are hard caps.  Full updates use the frozen 40-step rollout;
when a target leaves fewer than 40 aligned control steps per environment, the
last PPO update uses exactly that shorter aligned rollout so the recorded
`reward_v2_samples` reaches the target without overshooting it.

Evaluation keeps source-relative geometry as a diagnostic.  Absolute runtime
geometry safety, contact causality, no hidden control, terminal contact, and
terminal stability remain independent gates.
