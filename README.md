# TopoRetarget-Repro

[中文说明](README.zh-CN.md)

TopoRetarget-Repro is an unofficial, independent, paper-traceable reproduction scaffold for
*TopoRetarget: Interaction-Preserving Retargeting for Dexterous Manipulation*. It is intended to
start with GRAB to Arti-MANO and later support multiple HOI datasets and arbitrary URDF/MJCF
dexterous hands.

## Status

- Stage 0 complete: repository scaffold, configuration, read-only dataset discovery, and local Arti-MANO importer.
- Stage 1 complete: complete 16-page paper audit, parameter provenance, assumptions, and fidelity checker.
- Stage 2A complete: canonical HOI schema, explicit coordinate semantics, opt-in Zarr storage,
  deterministic synthetic data, error metrics, and headless comparison visualization.
- Stage 2B not started: the real-data GRAB inspection adapter is the next bounded step.

This repository does not implement the TopoRetarget retargeting algorithm, robot interfaces,
MANO-to-MediaPipe mapping, numerical optimization, Delaunay/SDF, RL/PPO, or baselines. Stage 2A
does not convert a full dataset and has no robot dependency.

## Data and local assets

The repository does not contain GRAB, OakInk, OakInk2, ContactPose, TACO, HO-Cap, ARCTIC, DexYCB,
MANO, or SMPL-X. Put external data under a local storage root using:

```text
<storage-root>/<registered-dataset-alias>/data/**
```

Machine-specific paths belong in ignored `.local/config.yaml` or environment variables. Start from
[`configs/paths.example.yaml`](configs/paths.example.yaml) and [`.env.example`](.env.example).

## Commands

```bash
python -m pip install -e ".[dev]"
toporetarget --help
toporetarget data --help
toporetarget data make-synthetic --output .local/cache/hoi/synthetic_demo.zarr
toporetarget data inspect --input .local/cache/hoi/synthetic_demo.zarr --frame 0
toporetarget doctor datasets --root "$REF2DEX_STORAGE_ROOT" --max-depth 4
toporetarget assets import-artimano --source-root "$MANIPTRANS_ROOT" --destination .local/assets/artimano
toporetarget doctor assets
toporetarget doctor paper
toporetarget doctor all
```

The dataset doctor performs read-only, allowlisted, bounded directory discovery and ignores
unregistered storage directories. The Arti-MANO command imports the complete URDF/mesh tree from
ManipTrans into `.local/assets/artimano/`; the result is not tracked by Git. Paper traceability is
checked with `python scripts/check_paper_fidelity.py`.

## Development

Run:

```bash
ruff check .
ruff format --check .
mypy src
pytest -q
python scripts/check_paper_fidelity.py
git diff --check
```

See [`docs/ROADMAP.md`](docs/ROADMAP.md), [`docs/PAPER_FIDELITY.md`](docs/PAPER_FIDELITY.md), and
[`docs/LICENSE_AND_DATA_POLICY.md`](docs/LICENSE_AND_DATA_POLICY.md). The existing repository
license is preserved in [`LICENSE`](LICENSE). Cite the TopoRetarget paper and ManipTrans when
using the corresponding research or local Arti-MANO source.

The canonical interface is documented in [`docs/HOI_DATA_INTERFACE.md`](docs/HOI_DATA_INTERFACE.md)
and coordinate semantics in [`docs/COORDINATE_CONVENTIONS.md`](docs/COORDINATE_CONVENTIONS.md).
