# Stage 9.2 checkpoints and resume

`toporetarget.final_retarget_checkpoint.v1` stores one atomic NPZ per strict-
accepted frame under:

```text
.local/cache/retarget/final_checkpoints/<run-id>/
  manifest.json  progress.json  frames/  temporary/  logs/
```

Each frame contains source/global index and timestamp, qpos/base/correction,
keypoints/link poses/collision points, QuerySet IDs and reasons, slack,
active/full signed distances and residuals, objective components, solver and
execution profile identities, acceptance fields, and the previous/current
checkpoint hashes. Temporary files are fsynced and atomically renamed. Failed
or status-9 frames never enter the contiguous accepted chain.

Start a bounded run:

```bash
python -m toporetarget retarget refine \
  --canonical "$CANONICAL" --warm-start "$WARM_START" --graph "$GRAPH" \
  --collision-samples "$ROBOT_SURFACE" \
  --solver-profile scipy_slsqp_active_set_contact_rich_v2 \
  --execution-profile cached_checkpoint_cpu_float64_v1 \
  --checkpoint-root .local/cache/retarget/final_checkpoints/$RUN_ID \
  --max-wall-time 1200 --progress-json .local/reports/stage9_performance/progress.json \
  --output .local/cache/retarget/final/$RUN_ID.zarr
```

`--max-wall-time` is soft: it is checked before a new frame and at safe outer
boundaries, never by killing an active SLSQP solve. The completed accepted frame
is saved, progress is `paused`, and the status includes the next frame and a
resume command. `--stop-after-frame N` has the same paused semantics after the
inclusive local frame N.

Resume rejects input, QuerySet, solver, execution-profile, or frame-range
mismatches. It uses only the last contiguous chain; an orphan such as frame 4
after 0,1,2 is reported and is never used as temporal state. Frame k+1 receives
the previous final base/qpos from frame k, not a fresh Stage 7 previous-state
reference.

Inspect and assemble independently:

```bash
python -m toporetarget retarget checkpoint-status --checkpoint-root "$ROOT"
python -m toporetarget retarget validate-checkpoints --checkpoint-root "$ROOT" --report "$REPORT"
python -m toporetarget retarget assemble-refinement --checkpoint-root "$ROOT" --output "$FINAL"
python -m toporetarget retarget compare-refinement-runs --left "$FRESH" --right "$RESUMED" --report "$COMPARE"
```

Assembly requires a complete hash chain, reconstructs ragged QuerySet/slack
arrays, validates the final artifact, and does not overwrite an existing
artifact without `--force`.
