# Eq. (9) temporal scope 解释

Q3 保留两个冻结的工程 profile，用于动态 GRAB 的配对比较：

- `scipy_slsqp_active_set_contact_rich_v2`：literal full-state temporal regularization，
  `paper_consistent=true`，但 author-exact 语义仍 unresolved。
- `scipy_slsqp_active_set_contact_rich_v3_fixed`：decomposed finger temporal term 加 base
  priors，`paper_consistent=true`，author-exact 仍 unresolved；该 profile 的 quality-neutral
  结论必须来自冻结动态 unit 的 paired evidence。

Warm profile 仅作 reference。静态 ContactPose 不用于判断 temporal preference；最多做一次
等价性检查。只有两个 profile 都通过 optimizer、feasibility、full-surface audit、data
integrity 和 provenance gate 时，才允许给出 preference；否则写
`NO_EMPIRICAL_PREFERENCE`。

当前 ContactPose selection 为 `Q1_CONTACTPOSE_SELECTION_BLOCKED`，没有 freeze 或 Q3 baseline，
所以当前不存在 Eq. (9) 的 empirical preference。
