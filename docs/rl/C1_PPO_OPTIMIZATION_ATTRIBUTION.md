# C1 PPO optimization attribution

Stage16 P3-C1.2 is a bounded, offline attribution pass for the V3
`hocap_170105` C1 saturation reproduction. It consumes immutable C1.1 receipts
and must not start Isaac, run PPO, or change reward, optimization, action, or
geometry contracts.

The required causal distinction is between state-distribution shift, actor
parameter drift, observation-normalizer drift, advantage/reward pressure, PPO
update instability, policy-distribution effects, tanh parameterization, critic
scaling, and residual-authority exhaustion. The pre-registered machine-readable
decision contract is `P3C12OptimizationAttributionDecisionV1`.

The auditor is fail-closed. Raw observations are required for a fixed probe;
the exact PPO batch is required for actor-mean gradient pressure, clipping
pressure, advantage buckets, and leave-one-reward-term-out GAE; a complete
frozen world-state bundle is required for an action counterfactual. Downstream
actor outputs or aggregate reward metrics must not be substituted for these
inputs.

Run the bounded pass with:

```bash
PYTHONPATH=src conda run -n toporetarget-rl \
  python scripts/rl/isaaclab/attribute_stage16_c1_ppo_optimization.py
```

The report is written to
`.local/reports/stage16_p3_c1_2_ppo_optimization_attribution/`. An
`INCONCLUSIVE` result is valid when the causal inputs are absent; it is not
permission to retrain or tune the next suspected contract.
