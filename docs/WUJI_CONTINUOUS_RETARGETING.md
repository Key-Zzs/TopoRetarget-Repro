# Wuji Continuous Retargeting

`wuji_continuous_full_state_v1` is an engineering extension for the fixed
Wuji Hand2 Beta1 RH W1/W2/W3 suite. The historical
`scipy_slsqp_active_set_contact_rich_v3_fixed` profile and its artifacts are
immutable and remain the paper-core reference.

The extension preserves Eq. (8), collision constraints, query semantics,
bounds, slack, base priors, and full-surface audits. It adds previous-final
correction transport, a chart-consistent full-state temporal correction term,
continuity acceptance, deterministic retry, and a bounded five-frame
receding-horizon fallback. It never filters a finished trajectory.

The profile is not the paper method: `paper_method=false`,
`engineering_extension=true`, and `author_exact=unresolved`. It is currently
validated only on the fixed `s1` airplane, apple, and alarm-clock windows; it
is not a cross-subject or RL-readiness result.

Run the fixed suite with:

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=src \
  /home/deepcybo/miniconda3/envs/topo-retarget/bin/python -m toporetarget workflow run-grab-suite \
  --suite configs/experiments/wuji_hand2_grab3_v1.yaml \
  --grab-root /mnt/nas/storage/Ref2Dex_storage/GRAB/data/GRAB \
  --index .local/index/grab \
  --mano-model-root /mnt/nas/storage/Ref2Dex_storage/shared_assets/body_models/mano \
  --robot wuji_hand2_beta1_rh \
  --solver-profile wuji_continuous_full_state_v1 \
  --experiment-root .local/experiments/wuji_hand2_continuous_v1 \
  --resume --evaluate --export-reference --generate-html
```

All generated experiment evidence is isolated under
`.local/experiments/wuji_hand2_continuous_v1/`.

## W2.2 closeout status

The bounded closeout is documented in
[`stages/W2_2_WUJI_CONTINUITY_CLOSEOUT.md`](stages/W2_2_WUJI_CONTINUITY_CLOSEOUT.md)
and writes only to
`.local/experiments/wuji_hand2_continuous_v1/closeout_v1/`. Its current status
is `WUJI_CONTINUOUS_PROFILE_NOT_RECOMMENDED_WINDOW_FALLBACK_FAILED`: formal
W1/W2/W3 trajectories pass their numerical and continuity gates, and W2's 13
absolute q-step transitions are warm/source-driven, but the real W3 five-frame
shadow returns SLSQP status 4 and fails the center continuity gate. W3's
penetration rate also regresses from 0.90 to 0.95. The profile is therefore
not recommended for offline reference generation until those gates are
resolved.

## W2.3 sequential finalization

W2.3 adds the derived `wuji_continuous_sequential_v1` candidate. Its only
solver-semantic difference from the full-state profile is that the
five-frame fallback is disabled on the production sequential path. The
window remains an isolated, nonblocking diagnostic shadow with a fixed-left
anchor and normalized coordinates. Run the bounded finalization with:

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=src \
  /home/deepcybo/miniconda3/bin/python scripts/wuji_w2_3_finalization.py
```

The complete audit, replay, penetration thresholds, oracle, exports, HTML,
and integrity evidence is under
`.local/experiments/wuji_hand2_continuous_v1/w2_3_finalization/`; see
[`stages/W2_3_WUJI_SEQUENTIAL_FINALIZATION.md`](stages/W2_3_WUJI_SEQUENTIAL_FINALIZATION.md).
The candidate remains offline-only with `RL_READY=NO`,
`REALTIME_READY=NO`, `CROSS_SUBJECT_VALIDATED=NO`, and
`AUTHOR_EXACT=UNRESOLVED`.
