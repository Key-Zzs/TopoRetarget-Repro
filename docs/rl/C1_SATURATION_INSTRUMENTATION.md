# C1 Saturation Instrumentation

`Stage16SaturationInstrumentationV1` is a detached, reusable receipt layer for
the frozen PPO-26D C1 gate. It does not change the policy, action bounds,
normalizer semantics, controller, curriculum, reward, or threshold.

For every collected rollout it writes a lightweight summary. A rolling buffer
retains four full receipts; a gate failure preserves its full actor-mean,
sampled-action, scaled-residual, pre/post-safety command, actuator target,
joint response, phase, contact, and object-error tensors together with the
previous full rollout.

The persistence ordering is fixed:

```text
collect -> summarize -> persist pre-gate state -> evaluate 0.25 gate -> update or stop
```

The authority metric remains exactly
`count(abs(tanh(actor_location)) >= 0.98) / (T * N * 26)`, failing strictly
above `0.25`. The action-time actor mean is authoritative; a later re-forward
over mutable vector-environment buffers is diagnostic only.

On a hard failure, the receipt includes pre-gate actor, critic, optimizer,
normalizer, Python/NumPy/Torch RNG state, curriculum identity, trigger rollout,
and prior full rollout. No optimizer update has occurred at that point.
