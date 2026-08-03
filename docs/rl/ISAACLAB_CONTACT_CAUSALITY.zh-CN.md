# Isaac Lab 接触读取与因果门禁

Stage 16-C.3R2 替换了不安全的旧设计：旧设计会在任务进程中构造并读取 21 个
hand-side filtered view。当前安装的 Isaac Lab 2.3.2 `ContactSensorCfg` 将
`force_matrix_w` 暴露为 `[environment, sensor body, filter shape, xyz]`，且其源码明确说明：
filtered contact 需要每个 environment 只有一个 sensor primitive。

运行时因此只保留两个 object-side view：

- `Object170105`：一个 object body，过滤全部 21 个 C.1 collision-bearing hand body；
- `Object170650`：相同的一个 body / 21 filter 合约。

这不是 21 次 Python sensor 读取。`aggregate` 保留 object-net force、impulse 与
pair presence；`diagnostic` 额外保留原始 filtered body-pair force matrix。两种模式都不改变
reward 或 control，也不伪造 contact point、normal、tangential force 或 point-level force。
记录采用有界 latest-only transport（4096 samples），因此高环境数诊断不会无限累积 Python
telemetry。

启用接触的 profile 使用 GPU physics replication 下的 USD cloning；Fabric cloning 被关闭，
因为真实 Isaac Sim 5.1 contact view 在 128 environment 下无法解析 replicated body。这是工程运行时
选择，不是物理、硬件或 sim-to-real 声明。

## C.3R2 读取结果

`scripts/rl/isaaclab/probe_stage16c3_contact_api.py` 在独立 child process 中运行每个 probe，并写入
flush 后的阶段事件、stdout、stderr、exit code 和 tensor shape。汇总位于
`.local/reports/stage16c3r2_c5/contact/c3_contact_readout_summary.json`。

真实 RTX 5080 / CUDA PhysX 结果为 `C3_CONTACT_READOUT_VALIDATED`：

- 已围栏的 raw-PhysX 1-env no-contact fixture 在 1000 physics step 内读取到零 force matrix；
- 1-env 单指 preload fixture 对两个 HO-Cap object 都产生有限、非零的 contact，并包含请求的 distal
  filter slot；
- 1-env random action 完成 1002 physics step 且 clean exit；
- 128-env aggregate random action 完成 1002 physics step 且 clean exit，force matrix 为有限的
  `[128, 1, 21, 3]`。

preload 的 state write 被明确限制为 probe rollout 之前的 C.1 fixture setup；普通 DirectRLEnv rollout
仍不会写入 wrist root 或 object state。这只验证读取能力。C3-0 reference/frame gate 现已通过，
但后续 C3R3/R4 的 computed-torque 与 bounded-preview 路径均未通过冻结 wrist gate。最终结果为
`C3_WRIST_ACTUATION_ARCHITECTURE_BLOCKED`；任务级 contact–momentum causality 仍为 not run，
不能从 preload fixture 推断。
