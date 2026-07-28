# W2.2 Wuji Continuous Retargeting Closeout

W2.2 is the bounded diagnostic closeout for the frozen W1/W2/W3 Wuji Hand2
Beta1 RH suite. It does not replace, rewrite, or export over any formal
baseline or continuous trajectory.

## Result

The closeout status is:

`WUJI_CONTINUOUS_PROFILE_NOT_RECOMMENDED_WINDOW_FALLBACK_FAILED`

The profile remains an engineering extension for offline reference-generation
experiments only. `author_exact=unresolved`, `paper_method=false`, RL and
realtime readiness are false, and no cross-subject claim is made.

The formal 180-frame W1/W2/W3 artifacts pass the numerical, full-surface
collision, bounds, and continuity checks. W2 has 13 absolute q-step
transitions; all 13 are classified as `SOURCE_OR_WARM_DRIVEN`, with zero
correction-driven transitions, zero mixed/inconclusive transitions, and zero
jump-and-return transitions. The maximum correction q step is
`0.007226704582365295` rad.

## B0/B1/B2 attribution

The bounded ablation uses seven fixed windows and the same QuerySet, collision
profile, paper weights, and `maxiter=100` solver budget:

- B0: frozen baseline, warm reset, no transport or correction temporal term;
- B1: previous-final transport only, with the B0 frame objective;
- B2: transport plus the bounded correction temporal term, with retry/window
  disabled in isolated mode.

All 210 expected isolated/operational rows are present. Some bounded solver
rows fail (`B1=3`, `B2=16`), so the causal ablation conclusion is
`ABLATION_INCONCLUSIVE_DUE_TO_SOLVER_FAILURE`; this is preserved as evidence,
not converted into a benefit claim.

## Five-frame fallback

The synthetic deterministic fixture passes routing, checkpoint/resume, and
center-only commit checks. The real W3 shadow uses fixed global
`[441,446)`, local `[34,39)`, anchor local `34`, center local `35`, and future
hints for locals `36..38`. Each frame has an independent QuerySet, slack
vector, and hash. The repeated run is deterministic and leaves formal hashes
unchanged.

The real joint SLSQP returns `status=4` (`Inequality constraints
incompatible`), and the center fails the continuity thresholds. Therefore the
window fallback gate fails. The quality gate also records a W3 penetration-rate
regression from `0.90` to `0.95`; this independently prevents recommendation.

## Reproduction and evidence

Run the full closeout from the repository root:

```bash
export PYTHONNOUSERSITE=1 PYTHONPATH=src
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
/home/deepcybo/miniconda3/envs/topo-retarget/bin/python scripts/wuji_continuity_closeout.py \
  --root .local/experiments/wuji_hand2_continuous_v1 \
  --baseline-root .local/experiments/wuji_hand2_grab3_v1 \
  --suite configs/experiments/wuji_hand2_continuous_v1.yaml
```

All closeout outputs are under
`.local/experiments/wuji_hand2_continuous_v1/closeout_v1/`, especially:

- `w2_qstep_attribution/`;
- `bounded_ablation/` and its 42 solver checkpoints;
- `window_fallback/real_w3_shadow.json`;
- `reports/recommendation_gate.json` and `reports/artifact_integrity.json`;
- `html/` and `screenshots/`.

The closeout command is diagnostic-only. It does not use `git add`, commit,
push, reset, clean, or tag, and it does not modify the sibling
`TopoRetarget-Repro-pene-loss` worktree.
