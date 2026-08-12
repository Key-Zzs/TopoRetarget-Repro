# 严格逐指 Contact Reward V4

## 范围

`StrictPerFingerContactRewardV4` 是 Stage 16-D 的因果 PPO contact contract。它相对冻结的
V3 只改变一个方法变量：V3 使用 reference 3 cm proximity 与 aggregate named-tip force，V4 使用
source-confirmed 的逐指语义与彼此独立的 named-tip force。

```text
Reward V4 = Reward V2 + r_contact_v4
```

V4 不继承或追加 V3 的 aggregate contact term。object/link/finger/wrist tracking、26-D
smoothness、signed object linear/angular twist、action、764-D observation、controller、physics、
PPO architecture 与 hyperparameters 都保持 V2/V3 冻结值。

## Source mask

在 runtime frame `t` 与固定 finger order `thumb/index/middle/ring/pinky` 中，immutable source
mask 只在 `SOURCE_CONTACT_CONFIRMED` 或 `SOURCE_CONTACT_PERSISTENT` 时令 `m_src[f,t]=1`。
`SOURCE_CONTACT_PROBABLE`、`SOURCE_CONTACT_TRANSITION`、`SOURCE_PROXIMITY_ONLY`、
`SOURCE_NO_CONTACT` 与 ambiguous state 全部为零。

source authority 是 `SourcePerFingerContactEvidenceV1`；其 native contact semantics 和固定的
`41 key -> 321 control frame` mapping 只读。mask 绝不从 robot contact 重新生成，也不随 policy
结果改变。每个 clip 固化 `strict_source_contact_mask[T,5]`、source classes 与 finger order。

## 精确力与奖励

named Wuji tips 是 `r_thumb_distal`、`r_index_finger_distal`、`r_middle_finger_distal`、
`r_ring_finger_distal` 与 `r_pinky_distal`。某个 required finger 只可读取自身 filtered
active-object PhysX pair force：

```text
F[f,t] = || force(tip_f -> active_object)[t] ||_2

r_cf[t] = 0                                      pair presence 为 false
                                                   或 F[f,t] <= numerical floor
          exp(-lambda_tip / (F[f,t] + epsilon))  其他情况

K[t] = sum_f m_src[f,t]
r_contact_v4[t] = 0                               K[t] = 0
                  w_c / K[t] * sum_f m_src[f,t] * r_cf[t]  其他情况
```

`w_c=1.0`、`epsilon=1e-5 N` 与 numerical floor 已冻结。只要 valid pair 不存在或只有 noise force，
就没有奖励。net body force、whole-hand force、same-finger group sum、five-finger aggregate force，
以及没有有效 pair force 的 contact presence 都不能替代它。

按 `K[t]` 归一化使 contact term 的上界不随 source-required finger 数量变化；更关键的是，任何
thumb/index/ring/pinky 的 force 都不能为缺失的 required middle-finger contact 记分。

## Calibration 与信息流

`lambda_tip` 在 PPO 前冻结一次：从两段 clip 的 V1 Formal20 exact pair-force telemetry 中，收集
source-required、named tip、valid pair presence 且 positive 的 force，取 pooled median。报告会记录每个
clip 与 finger 的覆盖量。它不使用 V3/V4 的结果，对所有 clip/finger 共享，且没有逐 clip/finger scale。

reward 只能读取当前 actual pair-force/presence 与冻结的 source-side mask。actor 仍是 764-D
observation，不能获得 future actual contact、force、object state 或 success signal。

## 因果与评测边界

V4 没有 object guidance force/torque、object pose/velocity/angular-velocity write、attachment、
suction、teleport、rollout reset、hidden support、gravity/friction curriculum、contact-loss
termination、terminal reward、penetration reward、Multi-Clip PPO 或 data-H2R。

development 只使用冻结的 development seeds；Formal20 是未见过的 deterministic frame-zero holdout。
除 Evaluation Suite V2 外，评测还报告 source-tip recall、persistent recall、fully-missing contact、
cross-finger compensation、same-finger non-tip substitution、no-tip/no-hand flight、recontact、twist、
geometry 与逐指 force-farming diagnostics。
