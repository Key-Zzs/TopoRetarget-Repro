# Stage 2B — minimal GRAB inspection

## Objective

Read one explicitly selected GRAB NPZ, reconstruct one selected hand through a replaceable backend,
load the object mesh and pose, preserve native time semantics, and validate a raw-to-canonical Zarr
round-trip without modifying source data.

## Implementation files

- `src/toporetarget/data/readers/grab.py`
- `src/toporetarget/data/adapters/grab_inspect.py`
- `src/toporetarget/data/mano_backends/base.py`
- `src/toporetarget/data/mano_backends/smplx_backend.py`
- `tests/unit/test_grab_parser.py`
- `tests/unit/test_grab_adapter_with_fake_mano.py`
- `tests/integration/test_grab_fixture_roundtrip.py`
- `tests/licensed_data/test_grab_real_sequence.py`
- `docs/GRAB_INSPECTION.md`

## Local validation

- Real sequence ID: `cubemedium_inspect_1`
- Selected hand: right
- Requested clip: `[0, 60)`
- Native FPS: `120.0`
- Intended cache: `.local/cache/hoi/grab/cubemedium_inspect_1_rh_f000000_f000060.zarr`
- `describe`: passed; GRAB fields and object/personalized-vtemp resources resolved.
- Fake-MANO fixture: parser, frame range, source-vertex preservation, Zarr round-trip, PNG layouts,
  error metrics, and source non-modification passed.
- Real visualizations and error reports: generated for `cubemedium_inspect_1`, right hand, frames
  `[0, 60)` using the user-provided MANO root through `smplx`.
- Source integrity: recorded in `.local/reports/stage2b/source_integrity.json`; unchanged is true.

## Blocker and definition of done

The user supplied a usable `MANO_RIGHT.pkl`/`MANO_LEFT.pkl` root. Stage 2B is **complete for the
bounded inspection scope**: the real cache, side-by-side/overlay images, JSON/CSV metrics, and
numerical real-data tolerance gate all passed. This does not claim full-dataset conversion,
MANO-to-MediaPipe mapping, or later retargeting stages.
