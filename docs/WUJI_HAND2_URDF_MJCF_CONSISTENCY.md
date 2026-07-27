# Wuji Hand2 Beta1 URDF/MJCF consistency

The URDF is the generic differentiable/reference FK source. The MJCF is the simulation-facing
joint, actuator, tip-site, and collision-source contract. The two files are audited independently
and then compared; neither silently replaces the other.

Compared invariants are the root link, 20 actuated joints, per-joint axes and limits, five tip
sites, link transforms at neutral/midpoint/ten deterministic random qpos samples, anchor positions,
rotation determinants, mesh references, 21 MJCF collision geometries, and ten contact-exclude pairs.
The backend-free audit is in `src/toporetarget/robots/simulation.py` and is part of
`toporetarget robots validate`; it does not require MuJoCo or run physics.

| Side | Joint order | Tip/site match | Max rotation error | Max translation error | Max anchor error |
| --- | --- | --- | ---: | ---: | ---: |
| RH | pass | pass | `1.0061e-6 rad` | `7.6159e-8 m` | `9.3232e-7 m` |
| LH | pass | pass | `1.1555e-6 rad` | `8.8662e-8 m` | `8.7834e-7 m` |

The audit tolerance is `2e-6 rad` for rotations and `1e-6 m` for translations/anchors. All
determinants are positive and all checked values are finite.

The URDF reports 21 visual and 21 collision geometry instances per side. The five fixed tip links
have no URDF visual/collision geometry; this expected upstream structure is represented by MJCF
sites and soft-pad visual meshes. MJCF formal collision uses 21 convex-hull geometries and excludes
ten proximal-to-wrist pairs. These policies are recorded in `configs/robots/collision/`.

This is a kinematic and asset audit, not MuJoCo playback, PD validation, force/torque calibration,
or hardware acceptance. R0 remains the future playback milestone.
