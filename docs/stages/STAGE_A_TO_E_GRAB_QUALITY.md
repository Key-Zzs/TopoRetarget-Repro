# Stage A–E GRAB Quality Workflow

Stage A freezes the prescribed G1–G4 units and runs/reuses the retained paper
baseline artifacts. Stage B builds deterministic Arti-MANO visual contact
regions. Stage C runs paper warm, seed-only morphology candidates, and the two
fixed position-prior diagnostics. Stage D scores the predeclared contact grid
on source-only selected frames, then evaluates the retained candidates over all
60 frames. Stage E assembles the fixed 2×2 matrix and applies hard,
regression, improvement, and Pareto gates.

Use `toporetarget quality status` for the current report. The experiment root is
`.local/experiments/grab_artimano_quality_v1/`, with `selection/`, `baseline/`,
`surface_contact/`, `morphology_warm/`, `contact_final/`, `matrix_2x2/`,
`reports/`, `html/`, and checkpoint directories. All report and HTML outputs
are regenerated from the frozen manifest and remain outside Git tracking.

The workflow ends with a machine-readable status, five stage flags,
`RECOMMENDED_PROFILE`, `MANUAL_ACCEPTANCE_REQUIRED=NO`, and
`CONTACTPOSE_STATUS=DEFERRED`. Conclusions are limited to the s1
within-subject multi-object development benchmark.
