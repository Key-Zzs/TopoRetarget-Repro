# Stage 16 — Reference-Tracking PPO

Stage 13 (additional dataset adapters), Stage 14 (robot plugin matrix), and Stage 15 (complete baseline matrix) are deliberately `DEFERRED`. This branch implements only the TopoRetarget reference-tracking PPO protocol from Appendix A.5.

The protocol is paper-exact where public: base-frame reference quantities, residual finger-joint action, the observation layout and offsets, Table 4 reward/termination, Table 5 randomization ranges, and Table 6 network/PPO values. The simulator, tracked-link list, axis offsets, action scale, gains, and unlisted PPO fields remain explicit assumptions in the Stage 16 ledger.

Run environment audit:

```bash
conda env create -f environment.stage16.yml
conda run -n toporetarget-rl python scripts/rl/audit_stage16_environment.py
conda run -n toporetarget-rl python scripts/rl/qualify_stage16_environment.py
```

Only a provenance-complete dynamic `Stage16ReferenceClip` is eligible for training. The implementation refuses static ContactPose samples. The currently installed MuJoCo backend is an isolated CPU correctness backend; it is never reported as the author-exact backend or as a 4096-environment reproduction.

Results, generated references, videos, checkpoints, reports, and build products are ignored under `.local/`.
The qualification command exercises E0--E4 only against a synthetic neutral Wuji asset reference;
it is a simulator/contract check, never a HOCap reference or policy result.

## Accepted HOCap functional protocol

For an accepted Stage-12 final, first materialize an ignored `RobotReferenceV2`,
the 20 Hz Stage-16 clip, and a derived local OBJ collision mesh. The raw HOCap
dataset remains read-only under its configured storage root.

```bash
conda run -n toporetarget-rl python scripts/rl/materialize_stage12_hocap_reference.py \
  --final-trajectory "$RUN/final/final_refinement_fast_exact_v2_r1/final_retarget.zarr" \
  --canonical "$RUN/canonical/canonical_hoi_v2.zarr" \
  --checkpoint-manifest "$RUN/checkpoints/final_refinement_fast_exact_v2_r1/manifest.json" \
  --robot-reference-output .local/stage16_reference_tracking_ppo/references/clip.robot_reference.zarr \
  --object-mesh-output .local/stage16_reference_tracking_ppo/objects/clip.obj \
  --report .local/reports/stage16_reference_tracking_ppo/accepted_reference_clip.json
conda run -n toporetarget-rl python scripts/rl/export_reference_clips.py \
  --robot-reference .local/stage16_reference_tracking_ppo/references/clip.robot_reference.zarr \
  --output .local/stage16_reference_tracking_ppo/references/clip.stage16.npz
```

`qualify_hocap_pd.py`, `train_hocap_reference_ppo.py`, and
`evaluate_hocap_reference_policy.py` run the bounded functional protocol.
They use the per-object derived collision mesh. Because these accepted inputs
are wrist-relative, the CPU runtime explicitly uses no synthetic ground,
zero gravity, and an explicitly disabled absolute-height termination; this is
an engineering assumption, not an author-exact simulator claim. Two accepted
clips demonstrate functionality but are not the paper's HOCap-32 evaluation.
