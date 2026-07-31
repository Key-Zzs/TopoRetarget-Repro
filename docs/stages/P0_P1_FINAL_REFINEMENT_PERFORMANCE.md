# P0/P1 Final Refinement Performance

P0 pauses the final queue, inventories only evidence-matched Stage-12 workers,
and validates immutable checkpoint lineage before any worker termination. A
legacy worker lacking a safe checkpoint is `SIGSTOP`-ed by exact PGID and is
recoverable only through `SIGCONT`; no broad Python process kill is permitted.

P1 introduces a candidate exact execution profile and diagnostics. It must not
promote that profile until five real frames, reference parity, scheduler A/B/C,
and artifact-integrity gates have produced measured evidence. Full Stage-12
resume remains a separate operator-review action.
