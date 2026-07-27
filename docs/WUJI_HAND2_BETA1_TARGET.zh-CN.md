# Wuji Hand2 Beta1 目标手集成

仓库将 Wuji Hand2 Beta1 作为通用 target-hand 实例纳入。当前属于 W0/W1 基础设施和有界验证，
不声称复现原始 Wuji 硬件、部署标定或论文级 zero-shot transfer。

## 资产边界

vendor bundle 位于 `third_party/robot_hands/wuji_hand2_beta1/`，只包含批准的 Hand2 Beta1
body 资产：左右手 URDF、左右手 MJCF、其引用的 STL mesh，以及上游 MIT license。STEP/USD、ROS
URDF 变体、ROS/package 文件和其他上游目录均排除。详见
[`WUJI_HAND2_ASSET_PROVENANCE.md`](WUJI_HAND2_ASSET_PROVENANCE.md)。

上游是 `wuji-technology/wuji-description`，请求 ref 为 `release/v2026.7.23`，本地通过已有
remote-tracking ref 解析到 commit `2b57d2621caed4e65207bb767ba25fc8eaec0881`。同版本号 tag
指向不同 commit，没有替换使用。

## 注册实例

| Robot ID | side | root | links | joints | actuated DoF | fixed joints |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `wuji_hand2_beta1_rh` | right | `r_wrist` | 26 | 25 | 20 | 5 |
| `wuji_hand2_beta1_lh` | left | `l_wrist` | 26 | 25 | 20 | 5 |

两个实例都由 YAML 和 `RobotHandRegistry` 加载。公开 qpos 顺序显式位于
`configs/robots/joint_orders/`，不会从 XML 顺序推断，也不会填充为历史 Arti-MANO 的宽度。
左右手的轴和 limits 保留为独立的上游事实。

## 通用集成与边界

使用 F0 的 `RobotHandSpec`/`RobotHandModel`/`RobotHandRegistry` contract，包含通用 URDF/FK、
qpos bounds、anchors、geometry、Jacobian、MediaPipe-21 tip site 语义，以及独立 visual、URDF
collision、MJCF collision profile。没有 `WujiHand2Beta1Adapter`、Wuji 专用 retarget pipeline、
Wuji 专用 solver、Stage 7/8/9 的 robot-name conditional 或硬件控制代码。

有界 smoke 使用 airplane canonical cache 的 `[240,243)` 三帧，验证 Stage 7 warm-start、Stage 8
图和 Eq. (7)、collision QuerySet、通用 Stage 9 objective/constraint/Jacobian construction；不
运行正式 Stage 9 optimizer，也不生成 Wuji final trajectory。

W2 是 main 上的完整 Wuji 重定向里程碑（至少三个 watertight clip）；S1 是
`develop/pene-loss` 上首先用 Arti-MANO 验证的通用 SDF penetration loss；I1 再用 Arti-MANO 与
Wuji 做 baseline/SDF 联合验证；W3 为 export，R0/R1 为 MJCF playback/PD 和 PPO tracking。
