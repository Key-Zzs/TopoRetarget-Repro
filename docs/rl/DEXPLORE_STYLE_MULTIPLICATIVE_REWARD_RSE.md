# Stage16 Dexplore-Style Multiplicative Reward and RSE

This Stage16 method is inspired by Dexplore's semantic reward grouping and
soft-reference exploration. It is not an exact port and does not vendor or
copy Dexplore's runtime. Stage16 keeps its existing simulator, fixed-wrist
controller, reference, action, PPO hyperparameters, physical gates, and
uniform reference-state initialization (RSI).

## Scope and frozen boundary

The opt-in configuration is
`configs/rl/stage16/stage16_dexplore_reward_rse_v1.yaml`. The default remains
`legacy_additive`; existing jobs therefore preserve branch-point reward
semantics unless `grouped_multiplicative_v1` is selected explicitly.

The experiment freezes C4 full gravity, nominal friction, the active table,
hand and virtual-wrist gravity off, the repaired fixed-wrist runtime, V4
strict per-finger contact, Reference Kinematics V2, the 26D action, and all PPO
hyperparameters. There is no phase-dependent switch, fixed grasp frame,
profile reward, per-object weight, or material adjustment.

## Grouped multiplicative reward

The implementation first aggregates related errors within four semantic
groups, maps each group to `(0,1]`, and then multiplies groups in log space:

```text
R_obj   = exp(-E_obj)
R_hand  = exp(-w_scope(D_ref) E_hand)
R_int   = 0.5 (R_contact_v4 + R_prox)
R_reg   = exp(-0.01 smoothness_26d)

R_total = exp(log(clamp(R_obj, eps, 1))
            + log(clamp(R_hand, eps, 1))
            + log(clamp(R_int, eps, 1))
            + log(clamp(R_reg, eps, 1)))
eps     = 1e-12
```

`E_obj` is the weighted mean of existing normalized object-axis, linear-twist,
and angular-twist squared errors. `E_hand` is the weighted mean of existing
normalized tracked-link, finger-joint, wrist-position, and wrist-orientation
squared errors. All group exponents are frozen at one.

The outer product provides a soft AND: improving one group cannot compensate
for an arbitrarily poor group. Internal aggregation remains smooth so the
method does not multiply every scalar term independently.

## Interaction semantics

When the frozen reference-contact mask expects one or more fingertips,
`R_contact_v4` is the unchanged strict V4 contact reward. Actual fingertip
distance is computed against the full triangle surface of the unchanged
visual object mesh. For expected fingertips,

```text
excess_i = max(d_actual_i - 0.03 m, 0)
R_prox   = exp(-(1 / 0.03 m) mean_expected(excess_i))
R_int    = 0.5 (R_contact_v4 + R_prox)
```

When no contact is expected, contact and proximity both equal one. This avoids
rewarding unnecessary contact and does not introduce an APPROACH/GRASP/LIFT
branch. Reference distance is the authority for hand-scope weighting; current
actual distance is used only for the interaction proximity term.

## Reference-scoped exploration

Reference distance softly controls hand tracking:

```text
D_ref   = min reference fingertip-to-object-surface distance
w_scope = clip(D_ref / 0.20 m, 0, 1)
```

Near the object, hand tracking is relaxed so exploration can resolve contact;
far from the object, the original kinematic target remains strong.

Adaptive termination uses counters initialized to `N_fail=N_total=1`:

```text
kappa       = clip(N_fail / N_total, 0.5, 1)
T_g(kappa)  = kappa T_g_base
```

Only primary RSE deviation terminations increment `N_fail`; both those failures
and normal completions increment `N_total`. Technical failures are excluded.
The normalized object, hand, and expected-interaction deviations share this
adaptive threshold. Existing safety termination remains independent.

## Dexplore versus this Stage16 adaptation

The motivating Dexplore design ties adaptive kappa to its Start initialization
contract. Stage16 deliberately does not adopt that coupling: training reset
indices remain uniformly sampled from `[0,320]`, because prior Stage16 evidence
showed frame0-only training collapses contact skill. Deterministic evaluation
still starts at frame 0 and runs the complete trajectory. This separation is a
core compatibility requirement, not an implementation detail.

## Gates and bounded result

Offline validation must show finite bounded rewards, counterfactual soft-AND
ordering, a non-pathological accepted 170650 distribution, no reward collapse,
and RSE sanity. A no-optimizer-step Isaac run then checks reward scale, finite
returns/advantages, finite gradients, and byte-unchanged parameters and
optimizer state.

Those gates passed. The authorized V4/170105/C4 run used the frozen healthy
source actor, 1,024 environments, 40 rollout steps, and at most ten updates.
Every update received deterministic frame0 Eval10. No update reached the 10/10
PF trigger for Confirm20. U10 produced lift 6/10 and DF pose/linear/angular V2
of 7/10, 6/10, and 6/10, but PF remained 0/10 because persistent multi-contact
began after LIFT. The result is `MULTIPLICATIVE_RSE_REFINEMENT_PARTIAL`, not an
accepted physical-HOI checkpoint. The only next action is residual-failure
diagnosis, not more tuning.

## PF V2 correction and symmetric continuation

The first PF V2 stop was invalid: the evaluator masked the independent table
ContactSensor with the frame-zero-invalid hand-object pair-force stream. The
historical accepted 170650 traces retain a recorded reset support sample before
release, so the corrected audit is
`PF_V1_PRELIFT_GATE_PARTIALLY_OVERCONSTRAINED`, not
`PF_V2_SEMANTICS_INVALID`. The V2 support-proxy rule is frozen and still makes
no exact wrench-transfer or surface-slip claim.

With the identical grouped multiplicative reward, RSE, C4 physics, PPO
hyperparameters, and uniform RSI contract, U10/170105 received one new update:
U11 reached PF V2 Eval10 10/10 and its same-checkpoint Confirm20 gives PF V2,
physical lift, causal lift, support transfer, sustained coupling, and all three
DF dimensions at 20/20. PF V1 deliberately remains 0/20 because its historical
pre-reference-LIFT timing gate was not changed.

The separate historical-170650 experimental actor received the full ten-update
409,600-sample budget. Its best observed U2 Eval20 remains 20/20 on PF V1,
PF V2, lift, support transfer, and DF. The continuation is non-monotonic: U8
PF V2=0/10, followed by U10 Eval10 recovery to 10/10. This instability does
not overwrite or demote the frozen historical accepted 170650 actor. It also
does not identify an RSE-specific effect because reward aggregation and RSE
were changed together. The only next action is no-tuning diagnosis of that
continuation instability.
