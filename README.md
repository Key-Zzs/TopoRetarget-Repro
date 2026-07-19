# TopoRetarget-Repro

[中文说明](README.zh-CN.md)

TopoRetarget-Repro is an unofficial, independent, paper-traceable reproduction scaffold for
*TopoRetarget: Interaction-Preserving Retargeting for Dexterous Manipulation*. It is intended to
start with GRAB to Arti-MANO and later support multiple HOI datasets and arbitrary URDF/MJCF
dexterous hands.

## Status

- Stage 0 complete: repository scaffold, configuration, read-only dataset discovery, and local Arti-MANO importer.
- Stage 1 complete: complete 16-page paper audit, parameter provenance, assumptions, and fidelity checker.
- Stage 2+ not started.

This repository does not currently implement the TopoRetarget retargeting algorithm, MANO loading,
GRAB adapters, numerical optimization, Delaunay/SDF, RL/PPO, or baselines.

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
