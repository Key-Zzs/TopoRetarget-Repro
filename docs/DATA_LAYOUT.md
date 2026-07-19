# External data layout

The configured root is expected to contain registered dataset directories:

```text
<storage-root>/<registered-alias>/data/**
```

The registry allowlist is in `configs/datasets/registry.yaml`. The resolver checks only those
first-level aliases, reports alias ambiguity, and searches directories below `data/` to a bounded
depth (default 4) without following symlinks. Non-dataset directories such as backups, outputs,
models, checkpoints, and temporary folders are ignored.

Stage 3 derived caches are local outputs under `.local/cache/hoi/` and are not raw dataset data.
For example, `cubemedium_inspect_1_rh_f000000_f000060_mp21.zarr` preserves the Stage 2B source
cache and adds a scene-frame `mediapipe21` hand track; it does not represent a full GRAB conversion.
