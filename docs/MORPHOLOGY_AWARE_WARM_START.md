# Morphology-Aware Warm Start

The baseline `paper_warm` is preserved. `morphology_seed_only_v1` creates a
fixed candidate set: paper/previous warm, robot-length reconstruction, thumb
workspace-nearest, previous morphology seed, and two deterministic
perturbations. Robot-length targets use source canonical bone directions with
robot link lengths; human bone lengths are never copied into the robot.

Every candidate is selected by solver success, bounds, the official Eq. (2)
objective, and a deterministic candidate-ID tie break. Thumb error or contact
proxy is not a selection objective. The final solver remains the same Eq. (1)–
(2) solver, so this branch is a faithful engineering initialization extension:
`paper_objective_unchanged=true` and `paper_solver_initialization_extension=true`.

`morphology_position_prior_v1` is a separate paper-external diagnostic. Its
position residual is normalized by a fixed robot characteristic length and its
declared weights are only `0.1` and `1.0`; weights are never tuned after seeing
results. A failed morphology gate leaves M* as diagnostic-only and does not
replace the baseline.
