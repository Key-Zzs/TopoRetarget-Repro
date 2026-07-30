# S1.1 signal-rich GRAB evaluation

S1.1 is the bounded diagnostic lane for deciding whether the fixed dense SDF
penetration-loss profile deserves a signal-rich follow-up. It is isolated from
the frozen two-clip S1 run and from the paused G3/G4 archives.

The fixed profile is `dense_squared_hinge_deadzone1mm_v2`: positive-outside
signed distance, 1 mm dead zone, squared excess hinge, `lambda_sdf=0.1`, the
existing 512-sample Arti collision surface, the solver-only convex-hull backend
inside optimization, and the strict reference triangle-winding backend for
validation. E0 is always the baseline and is run before S1.

## Selection contract

The raw GRAB pool is enumerated in lexicographic path order. A candidate must
have a right hand, native 120 FPS, at least 60 frames, the required MANO
parameters, finite object pose, and a strict watertight/orientable object mesh.
The 60-frame window is selected from source contact labels, hand-object
proximity, and source continuity only. G1, G2, G3, and G4 are excluded before
eligibility; no E0, S1, or derived solver result can select or reselect a row.

The deterministic shortlist expands in the recorded order `40 -> 80 -> all`
only when fewer than three clips pass the 12-frame E0 penetration-active gate.
The stress set, when valid, contains exactly three clips, at least three
objects, and at least two subjects. If the gate cannot freeze that set, the
formal status is `S1_1_INSUFFICIENT_PENETRATION_ACTIVE_CLIPS` and no full S1
run is attempted.

## Commands

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=src python -m toporetarget workflow \
  run-s1-signal-rich-evaluation \
  --config configs/experiments/s1_1_signal_rich_grab_v1.yaml \
  --experiment-root .local/experiments/s1_1_signal_rich_grab_v1 \
  --max-wall-time 1800 --resume
```

The workflow writes source scans and selection manifests, per-probe E0
artifacts, frozen-set locks, backend consistency reports, full E0/S1 artifacts,
per-clip/link/finger CSVs, a self-contained HTML dashboard, hashes, and one
`reports/final_decision.json`. The existing
`.local/experiments/s1_sdf_penetration_loss_v1/` tree is read-only input for the
G1 control diagnosis and is never overwritten.

## Decision semantics

`S1_CONDITIONALLY_ACCEPTED_FOR_PENETRATION_ACTIVE_CASES` is allowed only after
the fast/reference backend gate, strict 60-frame solver/audit completion, and
the fixed improvement gate pass. Backend disagreement routes to
`S1_1_ROUTE_TO_S1_2_BACKEND_STUDY`; insufficient active clips routes to the
insufficient-signal status; otherwise S1 development stops. The global default
remains E0, and these results do not establish paper-level contact-retention
improvement or ground-truth contact quality.

