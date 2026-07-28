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
