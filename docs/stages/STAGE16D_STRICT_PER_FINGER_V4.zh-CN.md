# Stage 16-D Strict Per-Finger V4

## 状态与问题

Strict Per-Finger V4 是当前 Stage 16-D 的单变量因果 PPO experiment。它检验：将 V3 的
proximity/aggregate contact term 替换为 source-confirmed independent-finger contact 后，能否提升
interaction fidelity，同时不损害冻结的 kinematic、physics 与 absolute-geometry gates。

因果链保持不变：

```text
policy action -> robot dynamics -> hand-object contact -> PhysX object dynamics
```

不使用 external object control 是 qualification requirement，不是 reward bonus。

## 冻结输入与公平初始化

V4 input freeze 对 `SourcePerFingerContactEvidenceV1`、raw MANO/object provenance、Reference
Kinematics V2、V1 Formal20 exact pair-force telemetry、V3-selected contract/checkpoint lineage、formal
seed manifests 以及 physics/action/observation/controller contracts 进行 hash。任何不一致都属于 input
provenance drift，必须阻止 V4。

每个 clip 只从自身 V3-matched V1-L0 actor 与 observation normalizer 开始；critic 和 optimizer
全新初始化。禁止 cross-clip transfer，以及 V3/V2 checkpoint warm start。actor 保持 764-D，继续通过
相同 26-D reference residual action 作用。

## 必需评测协议

两个 clip 都训练到至少 4,194,304 samples，并保存约 1M、2M、3M、4M development candidates。
development 选择为词典序：qualified/physics success、persistent 与 source-tip recall、较低
cross-finger compensation/flight、stability 与 twist、tracking error，最后选择更早 checkpoint。total
reward 不是主要选择标准。

每个 selected checkpoint 都进行未见 Formal20 evaluation。比较保留 V1、V3、V4 结果；source-contact
interaction success 是独立 qualification dimension，不能静默重定义 Evaluation Suite V2 success。
每 clip 的 4M effectiveness gate 可授权但不强制 8M/12M/16M continuation。

## V4 之后的路线

若 V4 在两个 clip 的 interaction fidelity 与 causal-physics guards 上通过验证，则冻结 causal
contact contract，只继续 Contact-ready RSI V2、support feasibility，然后 gravity 与 friction
curriculum。只有该因果路线不足时才考虑 external guidance 或 data-H2R。
