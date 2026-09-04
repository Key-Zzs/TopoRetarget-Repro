# OakInk2 HTML Viewer V2

OakInk2 Viewer V2 替换已废弃的 O1R2-C 手写 WebGL camera 路径。它采用稳定的
Ref2Dex MeshCat 架构：命名 scene node 与唯一 renderer camera；但最终 HTML 不依赖
Ref2Dex 或 MeshCat runtime。每个 HTML 都是 self-contained，不需要 CDN 或网络。

唯一 scene frame 是 `SCENE_WORLD_MANO_ROOT_RELATIVE`。MANO 顶点、闭合/开放面、
21 个关节与逐帧 object transform 全部由 Python 预计算。浏览器不解析 quaternion，
不执行 MANO FK/skinning，不处理 beta 或 `center_idx`。hand 与 skeleton 的 model
matrix 是 identity；object 使用 Python 预计算的逐帧 scene model。

`ViewerCameraStateV1` 是唯一 camera authority。`FRONT`、`OBLIQUE`、`SIDE` 是使用
O1R2 可信矩阵的确定性状态。左键拖动围绕默认 `FOCUS_INTERACTION` 或可选
`FOCUS_HAND` 修改 camera yaw/pitch；滚轮只修改 camera distance；`RESET CAMERA`
恢复 `OBLIQUE`。设计上不支持 pan。orbit 与 zoom 永远不改变 hand vertices、joints
或 object scene pose。

真实 visibility mode 包括 `HAND ONLY`、`HAND + OBJECT`、`SKELETON ONLY` 与
`HAND + SKELETON + OBJECT`。播放与 slider 保留两条冻结 episode 各自 180 个精确
source mocap frame ID。切 frame 与播放时保留当前 camera。source/canonical 是
identity 状态说明，不是伪造切换按钮。

生成并认证 authoritative same-two package：

```bash
PYTHONPATH=src conda run -n ref2dex-oakink \
  python scripts/data/run_oakink2_o1r2d.py --action all
```

只生成一条 authoritative episode、不运行浏览器认证：

```bash
PYTHONPATH=src conda run -n ref2dex-oakink \
  python scripts/data/run_oakink2_o1r2d.py \
  --action generate --review dev_01
```

生成阶段可通过 `--frame-ids` 指定 source-frame timeline，但必须包含冻结 primary
frame；该选项不是 O1R2-D 认证路径。

认证通过 Chrome DevTools input domain 发送真实 mouse/pointer callback 与 wheel
event，覆盖水平、垂直、对角、反向、连续十次 drag，以及 zoom、reset、preset
切换、orbit sweep、旋转后切 frame 和播放。报告记录 `gl.readPixels`、landmark
rank/depth、不可变的 3D pairwise distance、scene fingerprint、object model 与
hand-object anchor。机器 PASS 不能代替两条 episode 的人工验收；两条都批准前
`O5_ALLOWED=NO`。
