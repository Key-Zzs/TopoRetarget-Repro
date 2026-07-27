# W0/W1 Wuji Hand2 Beta1 集成

W0/W1 将批准的 Wuji Hand2 Beta1 左右手 body 资产加入 `main`，并通过 F0 通用 target-hand
contract 注册，同时显式分离 import provenance、语义、visual surface、URDF collision、MJCF
collision 和未来 simulator metadata。

已完成：确定性的 MIT 资产 vendor/hash；左右手 generic spec、qpos-order、anchors、surface 和
独立 collision profile；不依赖 MuJoCo 的 URDF/MJCF consistency 与 manifest check；Stage 7/8
不再硬编码 22 DoF；CLI 两种 robot-loading 形式；airplane `[240,243)` 的 Stage 7、Stage 8、
collision QuerySet 和 Stage 9 objective/constraint/Jacobian construction smoke；中英文文档同步。

`.local/reports/wuji_hand2/` 中的证据显示 warm-start qpos `[3,20]`、Stage 8 Jacobian
`[3,213,20]`、672 个 collision sample，Stage 9 报告明确 `optimization_performed=false`。
canonical source、object samples、上游 checkout、历史 worktree 和 `develop/pene-loss` worktree
均未修改。

W0/W1 不包含 Wuji 专用 adapter/solver、penetration-loss branch、MuJoCo playback、PPO、硬件标定、
完整多 clip 重定向或原始硬件复现声明。W2 至少需要三个 watertight clip。
