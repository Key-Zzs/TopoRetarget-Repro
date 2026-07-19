# Paper fidelity audit

## 1. Paper identity and PDF hash

The audited source is *TopoRetarget: Interaction-Preserving Retargeting for Dexterous
Manipulation*, arXiv `2606.16272` v2. The local PDF is `docs/TopoRetarget.pdf`, 16 pages,
SHA-256 `21c06a125430854dcff0d778283963b7fe107c8dfa79e3982639a80c21b206ab`.

## 2. Scope of the strict reproduction

Stage 1 audits the complete PDF, including Sections 3.1–3.4, Section 4, Sections 5–7, Equations
1–12, Figures 1–5, Tables 1–6, and Appendices A.1–A.5. The machine-readable index is
[`PAPER_FIDELITY.yaml`](PAPER_FIDELITY.yaml). The numerical method, simulator, datasets, and
baselines are not implemented in this stage.

## 3. Equation-to-code traceability

Equations 1–2 map to future warm-start code, Equations 3–7 to future interaction-graph and
Laplacian code, Equation 8 to future constrained refinement, Equation 9 to regularization, and
Equations 10–12 to future ContactPose and penetration metrics. Each entry records its PDF page,
known values, unknowns, assumptions, and future implementation/test targets in the YAML manifest.

## 4. Table-to-config traceability

Retargeting values from Table 3 are in `configs/paper/retarget.yaml`; MDP/reward and termination
values from Table 4 and RL/domain-randomization/PPO values from Tables 5–6 are in
`configs/paper/rl.yaml`. ContactPose metric definitions are in `configs/paper/metrics.yaml` and
baseline adaptations are in `configs/paper/baselines.yaml`. The reproduction notes transcribe all
rows of Tables 1–6 rather than referring to them only by name.

## 5. Figure reproduction plan

`docs/reproduction/figure1.md` through `figure5.md` record each figure's purpose, required data
and code, blocker, and expected output. No synthetic figure is presented as a paper reproduction.

## 6. Dataset dependencies

The paper uses ContactPose (25 of 28 grasps), Ho-cap (32 clips), and a self-collected MoCap
Pen-Spin set (32 clips, average 12.4 s). These are external and are never copied into this
repository. Stage 0 only discovers registered dataset paths under the configured storage root.

## 7. Baseline dependencies

OmniRetarget, Mink, DexPilot, and GeoRT are recorded in `configs/paper/baselines.yaml`. The
OmniRetarget dexterous adaptation follows Appendix A.2: MediaPipe keypoints replace humanoid
keypoints, the wrist/base is optimized rather than fixed, and collision uses full hand geometry.
No upstream repository is copied in this stage.

## 8. Unpublished implementation details

The solver, Delaunay backend/flags, SDF backend, first-frame seed, coordinate-frame details,
ContactPose intensity threshold, robot surface sampling, tracked links, axis-point geometry,
simulator/physics settings, low-level gains, and unlisted PPO values are explicitly registered in
[`ASSUMPTIONS.md`](ASSUMPTIONS.md). Configuration leaves undisclosed values null.

## 9. Current blockers

The private Pen-Spin data, Wuji deployment assets, target hand identity for Table 1, and several
solver/geometry/RL details are unavailable from the paper. These blockers prevent result-level
reproduction and are not silently resolved.

## 10. Definition of method-complete

Method-complete means all publicly specified method equations, configurations, constraints,
metrics, and evaluation code are implemented and tested, with every remaining assumption
explicitly resolved or marked as a deliberate extension. This repository is not method-complete.

## 11. Definition of result-complete

Result-complete additionally requires the same datasets, private trajectories, robot assets,
hardware, simulator, seeds, and experimental conditions needed to reproduce the reported numbers.
This repository is not result-complete.

