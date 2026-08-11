# Reference-Gated Contact Reward

## 范围

`TopoRetargetReferenceTrackingReward26DV3` 是 Stage 16-D 的 causal PPO reward contract，
且只改变一个方法变量：

```text
Reward V3 = Reward V2 + r_contact
```

它不包含 object force/torque、pose/velocity write、attachment、suction、support、trajectory
servo、contact-loss termination、terminal reward、penetration reward、curriculum、observation
feature、action scale 或 PPO architecture change。

## 冻结信号

五指固定顺序为 `thumb`、`index`、`middle`、`ring`、`pinky`，采用共享的 Evaluation Suite V2
distal-link root engineering fingertip landmark。reference-only unsigned distance 到 active
reference object visual mesh 定义：

```text
m_f = 1[D(x_ref,f, O_ref) < 0.03 m]
S_contact = sum_f m_f * ||F_f,active-object||_2
r_contact = 1.0 * exp(-lambda_c / (S_contact + 1e-5))
```

没有任何 active `m_f` 时，`r_contact` 显式为零。2 cm 仅作 diagnostic；V3 不 sweep 或自适应
3 cm threshold。visual mesh 用于 unsigned proximity 时不要求 watertight；若使用 collision proxy，
必须明确标为 approximation。

`F_f,active-object` 只能是当前 PhysX 从该 fingertip body 到 active manipulated object 的 force
vector。net fingertip force、self-collision、palm/other-body、inactive-object、support 与 scene
contact 都不能替代。runtime sensor 是 object-side filtered force matrix，五个 column 按名称冻结，
两条 clip 完全共享。

## Scale 与信息流

`lambda_c` 在训练前由两条 V1 formal trace pooled 的 exact positive-contact `S_contact` median
冻结；至少需要 100 个 sample。只保存 aggregate force 与 pair presence 的 trace 无法安全分解出五个
pair-force magnitude，因此必须阻断 PPO。

reward 可读取当前 pair force 和 current/future reference target。actor 保持 764-D observation，
不得加入 future actual force/contact、success label 或 future actual object state。
