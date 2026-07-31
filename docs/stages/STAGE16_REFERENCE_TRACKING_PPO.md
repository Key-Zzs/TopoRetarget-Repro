# Stage 16 — Reference-Tracking PPO

Stage 13 (additional dataset adapters), Stage 14 (robot plugin matrix), and Stage 15 (complete baseline matrix) are deliberately `DEFERRED`. This branch implements only the TopoRetarget reference-tracking PPO protocol from Appendix A.5.

The protocol is paper-exact where public: base-frame reference quantities, residual finger-joint action, the observation layout and offsets, Table 4 reward/termination, Table 5 randomization ranges, and Table 6 network/PPO values. The simulator, tracked-link list, axis offsets, action scale, gains, and unlisted PPO fields remain explicit assumptions in the Stage 16 ledger.

Run environment audit:

```bash
conda env create -f environment.stage16.yml
conda run -n toporetarget-rl python scripts/rl/audit_stage16_environment.py
```

Only a provenance-complete dynamic `Stage16ReferenceClip` is eligible for training. The implementation refuses static ContactPose samples. The currently installed MuJoCo backend is an isolated CPU correctness backend; it is never reported as the author-exact backend or as a 4096-environment reproduction.

Results, generated references, videos, checkpoints, reports, and build products are ignored under `.local/`.
