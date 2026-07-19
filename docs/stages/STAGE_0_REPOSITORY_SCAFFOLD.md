# Stage 0 — repository scaffold

## Completed items

- Added a `src/toporetarget` package and Typer CLI.
- Added precedence-aware external path configuration.
- Added allowlisted, bounded, symlink-safe dataset discovery with JSON reports.
- Added Arti-MANO import with URDF mesh validation, hashes, manifest, dry-run, force, and atomic replacement.
- Added English/Chinese README and roadmap, data/license policy, CI, tests, and third-party references.
- Preserved the existing GNU GPLv3 `LICENSE`.

## Path and data policy

Tracked files contain no machine-specific absolute dataset or upstream paths. The local
`.local/config.yaml`, `.local/reports/`, `.local/cache/`, and `.local/assets/` are ignored. Raw HOI
data is never copied, unpacked, modified, or symlinked into this repository.

## Arti-MANO policy

Only `ManipTrans/maniptrans_envs/assets/mano_urdf/` is imported locally. The complete URDF and
visual/collision mesh tree is retained; ManipTrans Python code and the rest of the repository are
not copied. The ignored manifest records upstream commit, source license hash, and per-file hashes.

## Commands executed

The scaffold was checked with `git rev-parse`, `git status`, `git remote`, `git branch`, `git log`,
the local Arti-MANO importer, dataset doctor, paper doctor, `pytest`, Ruff, mypy, and
`git diff --check`. Exact final results are recorded in the handoff response and reproduction log.

## Deferred work

MANO model loading, dataset adapters, Delaunay/SDF/optimization, baselines, RL, hardware
deployment, and paper result reproduction belong to later roadmap stages.

