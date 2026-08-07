# Stable Dynamic Contact Calibration

`StableFreeObjectGraspCalibrationV1` is an engineering calibration, not a task
trajectory. It asks whether the frozen Wuji hand, controller, runtime collision
proxies, PhysX parameters, and 26D action interface can establish and hold a
free object using real multi-sided contact. It never writes object or wrist
state after the 321-step schedule starts and uses no ground, support, hidden
force, attachment, source trajectory, or corrected trajectory.

The object-canonical initializer uses convex-proxy centroid, PCA/OBB axes,
extents, support directions, and a fail-closed python-fcl reset refinement.
Data-derived topology is thumb/index for `hocap_170105` and
thumb/index/pinky for `hocap_170650`; the extraction algorithm is shared.
C1 freezes -6/0/+6 mm offsets × closure 0.5/1.0. Unique C2 adds only -10/+10
mm with the same closure grid.

All 20 development candidates ran four replicas × 321 steps. No candidate
passed contact, topology, terminal hold, terminal twist, and exact geometry
together. The result is `STAGE16D_STABLE_GRASP_CALIBRATION_BLOCKED` with stop
marker `STAGE16D_STABLE_FREE_OBJECT_GRASP_CALIBRATION_BLOCKED`. Formal20 was
not authorized. Therefore `EmpiricalStableDynamicContactReferenceV1` and V2
were not created. If one is established later, it remains empirical engineering
evidence—not physical truth or a mathematical lower bound.

## IsaacLab simulation-trace replay

`scripts/rl/isaaclab/replay_stage16d_simulation_trace.py` provides a stable,
read-only viewer for every calibration/fix trace that follows the recorded
Stage 16-D trace contract. It renders the exact 21 authored runtime convex hand
proxies and the selected object proxy at their recorded poses. Contacting hand
proxies turn red; the object turns orange for nonzero exact penetration and red
above 3 mm. The terminal prints the active contact groups, force, object speed,
penetration, finite flag, and reason code for each frame.

This is an IsaacLab/Isaac Sim visualization of recorded simulation data, not a
new PhysX rollout. Current calibration traces do not contain robot joint states,
so the viewer intentionally does not synthesize a full visual-mesh hand. That
keeps the replay aligned with the collision/contact geometry used by the formal
metric.

The same viewer also accepts the nominal 321-frame trace exported by
`PhysicsConsistentRetargetedTrajectoryV1`. For that schema it derives contact
groups from the recorded per-body contact mask, reports the matching trajectory
qualification JSON, and can overlay the recorded source object as a translucent
cyan ghost. It never broadcasts nominal contact telemetry onto the other robust
replicas, so corrected traces expose replica 0 only.

Run the current `hocap_170105` trace continuously at half speed:

```bash
cd /home/deepcybo/workspace/dex/retarget/TopoRetarget-Repro
DISPLAY=:1 XAUTHORITY=/run/user/1000/gdm/Xauthority \
conda run --no-capture-output -n toporetarget-isaaclab \
env OMNI_KIT_ACCEPT_EULA=YES \
python scripts/rl/isaaclab/replay_stage16d_simulation_trace.py \
  --trace .local/reports/stage16d_stable_grasp_geometry_ppo/calibration_dev_hocap_170105_c2_ce94ed85254f.npz \
  --geometry .local/reports/stage16d_stable_grasp_geometry_ppo/calibration_dev_hocap_170105_c2_ce94ed85254f_geometry.npz \
  --replica 0 --fps 20 --speed 0.5 --loop --accept-eula
```

Freeze a suspicious frame for inspection (close the IsaacLab window to exit):

```bash
DISPLAY=:1 XAUTHORITY=/run/user/1000/gdm/Xauthority \
conda run --no-capture-output -n toporetarget-isaaclab \
env OMNI_KIT_ACCEPT_EULA=YES \
python scripts/rl/isaaclab/replay_stage16d_simulation_trace.py \
  --trace .local/reports/stage16d_stable_grasp_geometry_ppo/calibration_dev_hocap_170105_c2_ce94ed85254f.npz \
  --geometry .local/reports/stage16d_stable_grasp_geometry_ppo/calibration_dev_hocap_170105_c2_ce94ed85254f_geometry.npz \
  --replica 0 --frame 250 --accept-eula
```

For a new repair, pass its trace and matching optional `*_geometry.npz`. The
object id is inferred from a filename containing `hocap_170105` or
`hocap_170650`; use `--object` when a custom filename does not contain one.

Replay the complete current `hocap_170105` corrected trajectory with its source
object ghost:

```bash
cd /home/deepcybo/workspace/dex/retarget/TopoRetarget-Repro
DISPLAY=:1 XAUTHORITY=/run/user/1000/gdm/Xauthority \
conda run --no-capture-output -n toporetarget-isaaclab \
env OMNI_KIT_ACCEPT_EULA=YES \
python scripts/rl/isaaclab/replay_stage16d_simulation_trace.py \
  --trace .local/reports/stage16d_physics_consistent_retargeting/trajectory_trace_170105_v3.npz \
  --source-trace .local/reports/stage16d_physics_consistent_retargeting/source_trace_170105.npz \
  --qualification .local/reports/stage16d_physics_consistent_retargeting/trajectory_qualification_170105_v3.json \
  --object hocap_170105 --fps 20 --speed 0.5 --loop --accept-eula
```

This trajectory is currently labelled
`STAGE16D_TRAJECTORY_QUALIFICATION_BLOCKED`; replay availability does not change
that qualification result.
