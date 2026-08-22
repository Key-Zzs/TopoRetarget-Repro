# Raw Mocap Replay Overlay

`replay_physical_hoi_trace.py` can display three strictly distinct
visual layers in one Stage16 world frame and one recorded replay timeline:

- **PHYSX ACTUAL** is the recorded collision-body/object pose from the replay
  trace. It is always visible.
- **RAW MOCAP** is the original HOCap right-MANO surface and original HOCap
  object mesh/pose reconstructed from the frozen world-reference provenance.
- **RETARGET REFERENCE** is the geometric robot reference: link-point ghost
  plus its reference object. It is not raw MANO.

All ghosts are USD visual primitives only. They do not receive a collider,
rigid body, gravity, contact API, sensor, or a physics step. The replay itself
writes recorded visual transforms and calls `render()` only.

## Provenance, coordinates, and time

The resolver follows `world_wrist.stage16.npz` metadata to its canonical HOCap
artifact, then follows the canonical provenance to `meta.yaml`, `poses_m.npy`,
`poses_o.npy`, subject MANO betas, and the selected raw object mesh. It does
not infer raw data from a replay trace and does not hard-code a sequence.

Raw HOCap world and canonical Scene are the same frozen coordinate convention.
The deterministic audit checks raw object and MANO-root values against their
selected canonical counterparts before enabling the ghost. The report calls
this identity `T_stage16_world<-raw_mocap`; it is not an eye-fit offset. A
raw/geometric-reference object difference is preserved as a diagnostic, not
misclassified as a coordinate failure.

Replay `reference_index` is the timing authority. Its complete range and the
world-reference key count identify the per-trace retiming scale. Raw object
translation is linearly interpolated and its rotation uses shortest-arc SLERP;
MANO root rotations use SO(3) interpolation. PCA coefficients/translations
retain the source audit's linear interpolation semantics.

## Commands

The currently accepted positive C4 Formal20 scene is:

```bash
OMNI_KIT_ACCEPT_EULA=YES conda run -n toporetarget-isaaclab python \
  scripts/rl/isaaclab/replay_physical_hoi_trace.py --accept-eula --loop \
  --trace .local/sim_data/stage16_full_gravity_capability_closure/formal20/v4_hocap_170650/episode_000.npz \
  --object hocap_170650 \
  --mocap-similarity-output .local/reports/stage16_raw_mocap_replay_overlay/hocap_170650/similarity.json
```

The representative 170105 C4 failure is:

```bash
OMNI_KIT_ACCEPT_EULA=YES conda run -n toporetarget-isaaclab python \
  scripts/rl/isaaclab/replay_physical_hoi_trace.py --accept-eula --loop \
  --trace .local/sim_data/stage16_frozen_source_policy_gravity_sweep/v3/hocap_170105/c4/episode_00.npz \
  --object hocap_170105 \
  --mocap-similarity-output .local/reports/stage16_raw_mocap_replay_overlay/hocap_170105/similarity.json
```

`--no-reference-ghost` retains its existing meaning: hide the geometric
reference only. `--no-mocap-ghost` hides the raw MANO, raw object, and tips.
`--no-mocap-object` keeps raw MANO/tips while hiding its object. Use
`--no-reference-object` to keep only the retarget-hand link ghost. With
`--require-mocap-ghost`, missing immutable source provenance or a local MANO
asset fails closed; otherwise ACTUAL replay remains usable and reports
`RAW_MOCAP_GHOST_UNAVAILABLE=<reason>`.

The raw object defaults to its full frozen source mesh. For faster rendering,
`--mocap-object-low-poly` selects a deterministic 2,000-face **display-only**
mesh; `--mocap-object-max-faces N` selects another face budget (minimum 4).
The source mesh and raw pose remain immutable; similarity diagnostics and
recorded replay-transform evidence still use their full-resolution sources.
The startup line reports `object_display_faces=shown/raw`.

## Live ghost visibility in IsaacLab

In the non-headless replay, the `Replay Ghost Visibility` floating panel lets
you show or hide **Raw MOCAP** and **Retarget reference** independently while
the replay continues. The matching shortcuts are `M` and `R`. They author only
the USD visibility of `/World/Replay/Mocap` and `/World/Replay/Reference`; they
do not restart playback or modify the recorded transforms, physics, collision
proxies, or diagnostics. Start with the default ghost options so the desired
layer is constructed; a layer explicitly disabled with `--no-mocap-ghost` or
`--no-reference-ghost` remains unavailable to the live controls.

When a layer is hidden, the replay also skips its per-frame USD transform,
marker, and vertex-buffer writes. Revealing it resumes updates at the current
replay frame; no hidden catch-up work is accumulated.

The raw MANO surface now uses opacity `0.18`, its object uses `0.20`, and its
fingertip markers use `0.65` so the actual replay stays visually dominant.

The optional similarity JSON is explicitly `RAW_MOCAP_VS_ACTUAL`. It records
object translation/rotation differences and each morphology-aware,
object-local MANO-tip versus robot-distal-body support-point distance. It does
not compute a MANO-to-robot mesh vertex distance.
