# Workflow resume, cache, and provenance

`workflow run-grab --resume` reuses a node only when its cache record has the same
content signature, every output exists, every output hash matches, and its recorded
validation status is passing. A changed upstream artifact changes downstream
signatures. `--force-stage NODE` reruns that node and all downstream nodes; use
`--force-output` when an existing Stage 5–9 artifact must be replaced explicitly.

The run manifest records:

- repository commit and dirty-worktree state;
- Python executable, platform, package versions, and CUDA visibility;
- source sequence/path and before/after source SHA-256;
- official contact mapping identity and selected frame range;
- all Stage 5–9 profile/config hashes;
- artifact paths and hashes;
- node signatures, dependencies, cache reuse, invalidation reason, and timing;
- semantic sanity, cross-stage identity, coverage, and manual-review paths.

The source hash is checked after the final node. A source change fails the run; the
workflow never edits raw GRAB, MANO, or robot asset files. Generated artifacts and
reports belong under ignored `.local/` paths.

## Diagnostics

```bash
toporetarget workflow status --run .local/runs/stage10/<run>/manifest.json
toporetarget workflow validate --run .local/runs/stage10/<run>/manifest.json
toporetarget workflow review-template --run .local/runs/stage10/<run>/manifest.json
```

The cache is intentionally conservative. A failed validation report, missing
artifact, changed profile, changed request/window, or changed implementation
version prevents reuse rather than silently accepting a stale result.

## Stage 9.1 solver-profile resume

Select the contact-rich profile explicitly with
`--refinement-solver-profile scipy_slsqp_active_set_contact_rich_v2`. The
selected profile ID and YAML hash are included in the Stage 9 node signature,
manifest, final artifact metadata, and v2-specific final filename. Therefore a
v1 final artifact cannot satisfy a v2 Stage 9 node. The solver-profile change
removes only `final_refinement` and its dependent validation/audit/review/manifest
nodes from reuse; Stage 5-8 nodes retain their signatures and artifacts.

The resume record must preserve the original contact-rich window and the same
Stage 5-8 input hashes, then run the full 60-frame v2 refinement and its
independent repeat. Stage 10 math and acceptance gates are unchanged. Solver
termination and the deferred stationarity policy remain explicit
paper-undisclosed assumptions.

## Stage 9.2 frame checkpoints

The final refinement CLI has a separate frame-level checkpoint path described in
[`REFINEMENT_CHECKPOINT_AND_RESUME.md`](REFINEMENT_CHECKPOINT_AND_RESUME.md).
Its input/profile hashes are part of the checkpoint manifest and are checked
before resume; a workflow cache record cannot substitute for a strict accepted
frame checkpoint. The execution policy and SLSQP solver policy remain separate.
