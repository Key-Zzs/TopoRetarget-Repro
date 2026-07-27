# W0/W1 Wuji Hand2 Beta1 integration

## Objective

Add the approved Wuji Hand2 Beta1 RH/LH body assets to `main` and register them through the F0
generic target-hand contract while keeping import provenance, semantics, visual surfaces, URDF
collision, MJCF collision, and future simulator metadata explicit.

## Done

- vendored the pinned MIT subset with deterministic source/per-file/manifest hashes;
- added RH/LH generic specs, qpos-order profiles, anchors, surface profiles, and separate collision
  profiles;
- added backend-free URDF/MJCF consistency and generic manifest checks;
- widened Stage 7 warm-start and Stage 8 evaluation from historical 22-DoF assumptions;
- verified both CLI robot-loading forms;
- ran a bounded airplane window `[240,243)` through Stage 7, Stage 8, collision QuerySet creation,
  and Stage 9 objective/constraint/Jacobian construction;
- updated bilingual provenance, semantic, consistency, fidelity, notices, log, and roadmap docs.

Evidence is ignored under `.local/reports/wuji_hand2/`: warm-start qpos `[3,20]`, Stage 8 Jacobian
`[3,213,20]`, 672 collision samples, and a finite Stage 9 construction report with
`optimization_performed=false`. The source cache, object samples, upstream checkout, historical
worktree, and `develop/pene-loss` worktree were not modified.

## Non-goals

W0/W1 does not add a Wuji-specific adapter or solver, penetration-loss branch, MuJoCo playback,
PPO, hardware calibration, full multi-clip retargeting, or an original-hardware reproduction claim.
W2 requires at least three watertight clips before a full Wuji retargeting claim.
