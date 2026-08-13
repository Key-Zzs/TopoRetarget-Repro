# Stage16-C.5A-R3 接触拓扑与鲁棒 Oracle 交接

## 最终状态

`STAGE16C5_CONTACT_ORACLE_BLOCKED`。R3 已完成要求的诊断并实现 robust-oracle
contract，但没有在物理层面通过 Isaac Lab Oracle。两个冻结 selected action trace 的 C5C
20-replica gate 都失败；C5B、C.6/PPO、C.7、checkpoint 与真实机器人工作均未获授权。PPO
仍为 0 samples、0 checkpoints。

当前 C3/C4 结论保持有效且未改动：C3 使用 factor-8 派生的 321-sample reference、20 Hz
control、120 Hz physics、decimation 6、26-D action、764-D observation 与
`finite_virtual_6d_wrist_actuator_v1`；C4 是已验证的 4096-environment
aggregate-contact benchmark，吞吐 700.35 samples/s。本 R3 serial diagnostic 不改变这些
结论，也不把 virtual wrist 变成真实机械臂。

## 冻结范围与实现

R3 保留 reference/action hash、asset、physics contract、reward、termination、controller、
effort limit、mass/friction、solver setting 与 hard cap。禁止直接 object pose/root write、
wrist root write、hidden force 和 teleport。

实现新增精确 T0--T5 topology matrix、纯 Python classification/sharding contract、可测试的
robust statistical selector 与 parent/worker runner。runner 对每个 replica 使用一个 environment
并从 frame zero fresh reset；没有 candidate-state restore 或 cross-shard state transfer。因为
natural simultaneous-contact population 不是有效的 deterministic pool，serial 形式是刻意的。

## 拓扑证据

每个 cell 在冻结 factor-8 输入下运行 20 trials。`raw` 是 direct-state fingerprint comparison；
`derived` 是 task metric comparison。

| ID | Natural contact topology | Raw | Derived | 结果 |
| --- | --- | --- | --- | --- |
| T0 | 1 scene / 1 active | stable | stable | PASS |
| T1 | 33 scenes / 1 active，32 个 no-contact dummies | stable | stable | PASS |
| T2 | 33 scenes / 33 simultaneous contacts | divergent | divergent | FAIL |
| T3 | 33 scenes / staggered starts | stable | stable | PASS，仅诊断 |
| T4 | 33 contacts: 1x33 | divergent | divergent | FAIL |
| T4 | 33 contacts: 2x16/17 | divergent | divergent | FAIL |
| T4 | 33 contacts: 4x8/8/8/9 | divergent | divergent | FAIL |
| T5 | 96 contacts: 1x96 | divergent | divergent | FAIL |
| T5 | 96 contacts: 4x24 | divergent | divergent | FAIL |
| T5 | 96 contacts: 8x12 | divergent | divergent | FAIL |

分类为 `TRUE_CONTACT_SOLVER_NONDETERMINISM`。T0 通过，但没有一个 natural-contact shard
通过；T3 仅改变 candidate start time，不能作为 candidate-pool repair 的证据。因此在冻结测试中，
`SINGLE_SCENE_CONTACT_BATCHING_FAILURE` 与 `HARNESS_METRIC_FAILURE` 都被排除。

## Robust contract 与 C5C

`RobustOracleContractV1` 允许每 candidate 运行 1/4/8 个独立 replicas，使用
`mean_cost + 1.0 * population_std`（同时报告 alpha=0.8 的 upper CVaR），并按 failure probability、
CVaR formal-gate violation、worst normalized margin、mean object error、mean rotation error、
contact stability、action smoothness、effort、candidate ID 的字典序选择。C5C 恰好评估 20 个
独立 frame-zero replicas，保持未改变的 gate：position <= 0.02 m、rotation <= 10 deg、
axis <= 0.03 m、success >= 90%、final reach >= 90%。

| Clip | C5C 结果 | 原因 |
| --- | --- | --- |
| `hocap_170105` | success 与 final reach 均为 0/20；position 0.00827 m、axis 0.04665 m、rotation 47.48 deg | `FAILURE_OBJECT_ORIENTATION` |
| `hocap_170650` | success 与 final reach 均为 0/20；mean position 0.03841 m、axis 0.05492 m、rotation 26.66 deg | `FAILURE_OBJECT_AXIS_POINT` |

两个 qualification report 都记录正式 execution-state write 为 `object=0`、`wrist=0`；无
hidden force/teleport，也没有 action/reference mutation。因此 robust ranking 已实现但未在物理层面
qualified，不能豁免这些 C5C failures。

## Serial 资源测量

runner 测量规定的 32/96 candidate count 与 1/4/8 replicas，但每个 rollout 都在 fresh one-environment
Isaac worker 中调用。最终 machine-readable report 记录全部六格的 effective rollouts、wall time、
effective rollouts/s、GPU samples/VRAM 和 IPC overhead。这些是 safe serial-dispatch resource
measurements，不是 simultaneous contact-batch throughput，更不能替代 C4 的
4096-environment benchmark。

| Candidates x replicas | Clip | Effective rollouts | Wall time (s) | Rollouts/s | VRAM peak (MiB) | IPC (s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 32 x 1 | `hocap_170650` | 32 | 133.322 | 0.240021 | 4469 | 0.824 |
| 32 x 4 | `hocap_170105` | 128 | 555.057 | 0.230607 | 4472 | 0.858 |
| 32 x 8 | `hocap_170105` | 256 | 1108.492 | 0.230944 | 4723 | 0.818 |
| 96 x 1 | `hocap_170650` | 96 | 394.084 | 0.243603 | 4469 | 0.809 |
| 96 x 4 | `hocap_170105` | 384 | 1662.453 | 0.230984 | 4426 | 0.771 |
| 96 x 8 | `hocap_170105` | 768 | 3308.348 | 0.232140 | 4726 | 0.859 |

每一行都报告 no hidden execution-state writes。六格全部只使用 preexisting selected contact trace；
没有一格是 CEM/Oracle optimization episode。

## 证据与复现

以下 ignored local outputs 是证据权威：

- `.local/reports/stage16c5a_r3_contact_topology_final/contact_topology_diagnosis.json`
- `.local/reports/stage16c5a_r3_robust_oracle_final_retry1/robust_oracle_report.json`

同一目录下的 worker reports 保留每个 topology 及 replica 的精确结果。首次 robust 输出目录在
pre-runtime import/configuration defect 后被保留；`_retry1` 是不会覆盖既有证据的 authoritative
completed retry。

聚焦 CPU contract 的测试命令：

```bash
env PYTHONPATH=src conda run --no-capture-output -n toporetarget-isaaclab \
  pytest -q tests/rl/isaaclab/test_stage16c5_replication.py \
  tests/rl/isaaclab/test_stage16c5a_r3_topology_robust.py
```

## 下一步权限边界

后续 goal 只有在显式冻结范围并重新 qualification 后，才能提出新的 physical repair。它不能静默
放宽 tolerance、改变 controller/reference/solver、写 object/wrist state，或把 robust statistics
当作失败 physical task 的替代品。在此之前，C5B 与 PPO 保持停止。
