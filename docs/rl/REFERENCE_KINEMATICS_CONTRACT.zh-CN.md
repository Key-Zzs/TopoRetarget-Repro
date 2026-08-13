# Stage 16-D Reference Kinematics Contract

## 版本化 reference artifact

`reference_kinematics_version=1` 是不可变的历史 artifact。
`reference_kinematics_version=2` 是独立、带 hash 的 factor-8 321-key
control-grid materialization。V2 在 metadata 中记录 V1 parent 和 native
source hash，绝不覆盖任一输入。

## 时间与 pose 语义

native reference 有 41 个 key，间隔 0.05 s。factor-8 control reference 有
321 个 key，同样间隔 0.05 s，因此保留 native duration。native key 位于
`0, 8, ..., 320`，必须精确保留。translation 用 shape-preserving cubic
Hermite trajectory 插值。quaternion 是归一化、sign-continuous 的 `wxyz`
active right-handed rotation，并采用 shortest-arc SLERP。

## Twist 语义

`*_twist_world_ref[..., :3]` 是从 materialized pose 和 timestamp grid
导出的带符号 world-frame linear velocity。`[..., 3:]` 是由 SO(3) relative
rotation log 得到的 world-frame angular velocity。内部样本采用 centered
derivative，端点采用匹配的二阶单侧 derivative。body-frame angular velocity
是单独命名的 conversion，不能静默代替 world-frame field。

qualification 必须证明 timestamp 单调、factor-8 key preservation、
quaternion validity/sign continuity、finite twist、linear/SO(3) integral
consistency 和预期 factor-8 derivative scaling。terminal reference twist
是描述性监督，不能为了满足 terminal-stability gate 而被写成零。

## Consumer

V1 policy 保留 V1 provenance。V2 consumer 必须在加载 V2 twist 前 assert
`reference_kinematics_version == 2`。metadata 而非文件名约定是权威版本检查。
