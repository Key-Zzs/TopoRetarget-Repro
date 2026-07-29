# Wuji Hand2 GRAB retargeting

This benchmark composes the generic TopoRetarget data, Stage 7 warm-start,
source-only Stage 8 interaction graph, and Stage 9 constrained refinement
components for `wuji_hand2_beta1_rh`.

The frozen units are all subject `s1`: `airplane_lift` frames `[240,300)`,
`apple_eat_1` frames `[212,272)`, and `alarmclock_lift` frames `[407,467)`.
The target has 20 qpos dimensions and a tracked `r_wrist` root. Its formal
collision surface is separate from visual soft-pad meshes and is frozen by the
Wuji collision profile.

Run from the repository root:

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=src \
/home/deepcybo/miniconda3/envs/topo-retarget/bin/python -m toporetarget workflow run-grab-suite \
  --suite configs/experiments/wuji_hand2_grab3_v1.yaml \
  --grab-root /mnt/nas/storage/Ref2Dex_storage/GRAB/data/GRAB \
  --index .local/index/grab \
  --mano-model-root /mnt/nas/storage/Ref2Dex_storage/shared_assets/body_models/mano \
  --robot wuji_hand2_beta1_rh \
  --solver-profile scipy_slsqp_active_set_contact_rich_v3_fixed \
  --experiment-root .local/experiments/wuji_hand2_grab3_v1 \
  --resume --max-wall-time 1800 --evaluate --export-reference --generate-html
```

The current run is an offline reference runtime, not a real-time controller.
GRAB semantic-contact metrics are explicitly `DATASET_PROXY` and are not
ContactPose ground truth. Banana/open mesh data, SDF penetration loss,
morphology priors, and RL are outside this benchmark.

The follow-up W2.3 sequential finalization is isolated from this baseline
under `.local/experiments/wuji_hand2_continuous_v1/w2_3_finalization/` and is
documented in [`stages/W2_3_WUJI_SEQUENTIAL_FINALIZATION.md`](stages/W2_3_WUJI_SEQUENTIAL_FINALIZATION.md).
It never overwrites this baseline export.
