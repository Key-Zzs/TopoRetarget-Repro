# C1 Action Saturation Attribution

The Stage16 P3 `C1` failure receipt for `aggregate_v3/hocap_170105` records
`PPO26D_ACTION_SATURATION_FAIL_FAST: phase=before_update fraction=0.260371`.
This is not a joint-limit, target-clamp, force, torque, or actuator metric.

The exact gate is evaluated before `optimizer.step()` over a collected rollout:

\[
S = \frac{1}{T N 26}\sum_{t,e,d}
\mathbb{1}\left[\left|\tanh(\mu_\theta(\hat o_{t,e}))_d\right| \ge 0.98\right].
\]

Here `mu_theta` is the actor's unbounded Gaussian location, `hat o` is the
frozen-normalizer observation, and the policy's deterministic action is its
tanh-squashed mean. The gate fails strictly when `S > 0.25`.

The downstream action path is separate from this metric:

```text
actor location -> tanh-normalized 26D action
-> wrist (3 x 0.01 m, 3 x 5 deg) / finger (20 x 10% joint range) residual
-> reference-centred SE(3) wrist and finger targets
-> joint-limit clamp / explicit virtual-wrist target
-> articulation target, applied torque, actual state
```

Only the first line is measured by this C1 gate. A joint clamp or an actuator
limit can be a downstream consequence, but cannot be inferred from `0.260371`.

The final failed collection has `T=24`, `N=1024`, and 26 dimensions, giving a
denominator of 638,976. The formatted receipt corresponds to 166,371 counted
elements (the receipt prints six decimal places).

The complete C1 ledger shows deterministic policy-mean saturation increasing
across the 25 retained full 40-step rollouts from `0.018080` to `0.207148`.
The 24-step tail is `0.260371`. Therefore the documented gate trigger is
`POLICY_OUTPUT_SATURATION_PRIMARY`; it must not be re-described as a physical
actuator saturation.

The historical run did not persist the C1 pre-failure actor, normalizer, RNG,
per-step actions, command targets, clamp flags, contact forces, or actuator
limits. A C0 predecessor checkpoint is not an acceptable replacement. Thus
the following are deliberately **inconclusive**, not negative findings:

- residual action authority;
- reference-centred joint-limit clamp;
- actuator/controller response saturation;
- task-phase/contact-load concentration;
- C0-versus-C1 physics contribution; and
- whether the 24-step estimator itself is primary.

Future authorized C1 continuations use
[C1 saturation instrumentation](C1_SATURATION_INSTRUMENTATION.md): detached
action-time receipts are flushed before the unchanged hard gate, so a failure
preserves the policy and downstream action pipeline rather than reconstructing
them from a predecessor checkpoint.

Run the read-only extractor after the immutable receipts exist:

```bash
PYTHONPATH=src conda run -n toporetarget-rl \
  python scripts/rl/isaaclab/attribute_stage16_c1_action_saturation.py
```

It writes only untracked evidence under
`.local/reports/stage16_c1_action_saturation_attribution/`; it does not start
Isaac, restore a policy, run PPO, or mutate a checkpoint. The next action is
to fix diagnostic persistence before any future authorized C1 attempt. It is
not implemented by this attribution pass, and the saturation threshold remains
unchanged.
