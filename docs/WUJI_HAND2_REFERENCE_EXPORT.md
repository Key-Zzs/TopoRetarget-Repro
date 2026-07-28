# Wuji Hand2 reference export

Reference bundles live under
`.local/experiments/wuji_hand2_grab3_v1/exports/` with one directory per
clip. They contain NPZ and Zarr arrays, metadata, provenance, validation, and
metrics. Export is read-only: it does not call the solver, and the exported
`qpos` and `base_pose_scene` arrays must be exact matches to the final artifact.

The bundle describes the 20D qpos order, joint limits, robot asset hashes,
anchor/collision profiles, source hashes, graph hash, solver profile, object
poses, timestamps, and accepted mask.
