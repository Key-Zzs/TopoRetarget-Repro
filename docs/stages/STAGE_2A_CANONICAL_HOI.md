# Stage 2A — canonical HOI interface

## Objective

Define one lossless, robot-independent HOI contract with explicit coordinate semantics, lazy
sequence loading, opt-in cache storage, comparison metrics, and headless raw/canonical views.

## Implemented files

- `src/toporetarget/data/schema.py`
- `src/toporetarget/data/adapters/base.py`
- `src/toporetarget/data/storage.py`
- `src/toporetarget/data/synthetic.py`
- `src/toporetarget/geometry/se3.py`
- `src/toporetarget/geometry/frames.py`
- `src/toporetarget/viz/comparison.py`
- `src/toporetarget/viz/errors.py`
- `src/toporetarget/viz/matplotlib_viewer.py`
- `src/toporetarget/cli/data.py`
- `tests/unit/` and `tests/integration/test_synthetic_hoi_roundtrip.py` Stage 2A tests

## Commands and boundaries

```bash
toporetarget data make-synthetic --output .local/cache/hoi/synthetic_demo.zarr
toporetarget data inspect --input .local/cache/hoi/synthetic_demo.zarr --frame 0
toporetarget data compare --dataset synthetic --sequence demo \
  --canonical .local/cache/hoi/synthetic_demo.zarr --layout side-by-side --frame 0 \
  --output .local/reports/stage2a/synthetic_side_by_side.png \
  --error-json .local/reports/stage2a/synthetic_side_by_side.json
```

`FrameRange` is contiguous and half-open. `native_fps` is metadata only. There is no temporal
resampling, no display/data stride confusion, no spatial sampling, no robot model, no source-data
modification, and no full-dataset conversion command. Cache writes require an explicit `--output`.

## Tests and definition of done

The test suite covers schema variants, irregular timestamps, SE(3) round-trips, optional Zarr
round-trips, import-time filesystem behavior, known perturbation metrics, and headless PNG output.
Stage 2A is complete when those tests and the repository lint/type/paper checks pass and this
snapshot is placed in the Git index without a commit.

Stage 2B is not started in this snapshot. Known limitations are that the GRAB reader, real MANO
backend, and licensed-data validation are intentionally deferred to Stage 2B; no MANO-to-
MediaPipe mapping, robot model, Delaunay, SDF, optimization, PPO, or SPIDER is included.
