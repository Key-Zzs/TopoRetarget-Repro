# Stage 9.3.4 provenance-rebased multistart and causal ablation

Stage 9.3.4 is an audit-only experiment layer for the official
`s1/airplane_lift`, right-hand, `artimano_rh` window `[240,300)`. It does not
change Eq. (1)--(9), paper weights, the Stage 7 warm artifact, the Stage 8
interaction graph, accepted Stage 9.2 artifacts, the Stage 10 manifest, or
manual acceptance.

## Lane contract

The historical lane resolves the recorded solver commit in a detached
worktree and records whether its exact Python/wheel environment is available.
If it is not, the lane emits `HISTORICAL_EXACT_REPLAY_UNAVAILABLE`; the
current lane continues independently and never substitutes the current
environment for the historical one.

The current lane creates a new 60-frame run from the Stage 10 canonical,
warm-start, graph, and collision-sample artifacts. It uses the formal solver
profile, strict acceptance, checkpoint/resume, and the persisted 512-sample
`reference_winding_v1` audit. Outputs are diagnostic and live only under
`.local/runs/stage9_3_4_*`.

After the baseline, `run-current-baseline-repeats` replays the selected
diagnostic frames three times from frame 0 in separate output roots. The
replays use the same current lineage and are summarized in
`current_baseline_repeat.json`; they are reproducibility evidence, not an
accepted reference artifact.

## Controlled diagnostics

The bounded multistart runner uses the existing objective and constraints with
the native QuerySet. It records requested seeds, acceptance, solver status,
objective, QuerySet count, full-SDF minimum, and per-finger metrics. The
frozen-initial-QuerySet phase uses the solver's immutable initial QuerySet hook
and is kept separate from the native QuerySet phase; this hook is an
engineering diagnostic, not a paper-specified method.

Base-seed diagnostics are SE(3)-only Kabsch fits with `det(R)=+1`, no scale,
and declared point groups/weights. Warm geometry is recorded separately from
the final formal solves, which run under both `initialization-only` and
`seed-and-prior` protocols. Margin and full-512 profiles are diagnostic
QuerySet isolations. Projection profiles are not ranked against the formal
objective unless an equivalent solver invocation exists. SciPy SLSQP multipliers are
not fabricated; gradient and link-attribution reports remain `NOT_RUN` when
the persisted result contract cannot support them.

Contact retention fields are signed-distance proxies, not ground-truth contact
labels. A strict solver pass, `COMPLETE_WITH_WARNINGS`, or proxy improvement is
not paper-level causal proof.

## Route policy

The only admissible final route is written to
`.local/reports/stage9_3_4/stage9_4_readiness.json` and repeated in the summary
and HTML. A passing current baseline is necessary but insufficient. Missing
historical replay, unsolved projection branches, absent reliable multipliers,
or missing multi-frame branch-rollout evidence keep `ENTER_STAGE9_4` false and
require human review. The frozen initial-QuerySet hook is implemented and
executed, but remains an engineering diagnostic rather than a paper-specified
method.

All changes remain unstaged; the historical worktree is detached and must not
be committed.

## Stage 9.3.5 handoff boundary

Stage 9.3.5 continues only from the current-lineage baseline and the Stage 10
manifest recorded above. Its warm-to-final path scan, projection states,
counterfactuals, objective decomposition, and constraint-pressure reports are
diagnostic state analyses. They do not claim a paper-specified causal
mechanism, do not modify the formal objective or solver profiles, and cannot
authorize Stage 9.4 without a separately declared accepted multi-frame branch
rollout.
