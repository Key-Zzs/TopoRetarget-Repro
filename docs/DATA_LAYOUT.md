# External data layout

The configured root is expected to contain registered dataset directories:

```text
<storage-root>/<registered-alias>/data/**
```

The registry allowlist is in `configs/datasets/registry.yaml`. The resolver checks only those
first-level aliases, reports alias ambiguity, and searches directories below `data/` to a bounded
depth (default 4) without following symlinks. Non-dataset directories such as backups, outputs,
models, checkpoints, and temporary folders are ignored.
