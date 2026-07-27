# Stage 10 — bounded GRAB to Arti-MANO workflow

Stage 10 is an orchestration layer over the audited Stage 5–9 CLIs. It does not
replace or silently alter the canonical converter, warm-start solver, interaction
graph, or constrained refinement.

## Contract

The workflow accepts one explicit GRAB sequence, one hand, and one 60-frame native
window. With `--auto-contact-window`, the selector ranks complete windows using the
official semantic `contact.object` mapping, a minimum contact-frame ratio of `0.5`,
strict watertight object geometry, and a source-hand/object median-distance gate of
`0.02 m`. The selected window and all rejected candidates are written before any
downstream artifact is produced.

The DAG is:

`resolve source → canonicalize → validate → MediaPipe21 provenance → keypoint validation → mesh audit → object samples → warm start → interaction graph → frozen interaction evaluation → final refinement → independent validation/audit → semantic sanity → review bundle → manifest`.

Every node records its implementation version, dependency hashes, configuration
hashes, output hashes, duration, and reuse/invalidation reason. Raw GRAB NPZ and
external MANO files are read-only runtime inputs.

## Example

```bash
PYTHONNOUSERSITE=1 ~/miniconda3/envs/topo-retarget/bin/python -m toporetarget \
  workflow run-grab \
  --sequence s1/airplane_lift \
  --index .local/index/grab \
  --hand right --robot artimano_rh \
  --auto-contact-window --window-length 60 \
  --mano-model-root /path/to/MANO \
  --asset-root third_party/robot_hands/artimano \
  --run-root .local/runs/stage10 \
  --manual-acceptance .local/reports/stage9/manual_acceptance.json
```

Planning is solver-free:

```bash
toporetarget workflow plan-grab \
  --sequence s1/airplane_lift --index .local/index/grab \
  --hand right --robot artimano_rh --start-frame 240 --end-frame 300 \
  --window-length 60 --output .local/reports/stage10/plan.json
```

The end-to-end run remains `pending_human_acceptance` after machine validation. A
human must inspect the generated named frames and fill the acceptance file; Codex
does not write a human pass.

When a node fails, the run remains failed and retains its completed upstream
artifacts. The per-run reports include `input_audit.json`, `artifact_reuse.json`,
`invalidation_tests.json`, `performance.json`, `determinism.json`,
`semantic_sanity.json`, `source_integrity.json`, and `stage10_summary.json`.
`determinism.json` remains explicitly pending until a successful final artifact
can be compared with a full fresh repeat; a failed refinement is never promoted
to a reference trajectory.

## Outputs

Each run is under `.local/runs/stage10/<sequence>__<hand>__<robot>__<range>/` and
contains `plan.json`, `status.json`, `manifest.json`, per-node logs/cache records,
canonical/warm-start/graph/final artifacts, validation reports, review PNGs/GIF,
and optional `robot_reference.v1` exports. A reference export is read-only with
respect to the source and records its final-artifact hash and workflow provenance.

The stable report copies for the current workspace are under
`.local/reports/stage10/`, including `input_audit.json` and
`workflow_plan.json`. These are ignored local evidence and are not Git inputs.

Stage 10 does not implement Eq. 10–12, ContactPose metrics, baselines, PPO/RL,
physics, or full-dataset conversion.

## Accepted bounded milestone

The current accepted run is `s1/airplane_lift`, right hand, `artimano_rh`,
global frames `[240,300)`. Its run root is
`.local/runs/stage10_reference_runtime/s1__airplane_lift__right__artimano_rh__f000240_f000300`.
The Stage 9.2 final artifact is referenced by hash, not copied or recomputed;
the manifest records every Stage 5–9 node as reused and zero Stage 9 solver
invocations. Runtime mode is `reference`, with performance debt still open.
