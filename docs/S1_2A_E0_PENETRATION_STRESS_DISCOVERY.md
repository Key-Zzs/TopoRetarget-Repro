# S1.2A E0 penetration stress discovery

S1.2A is a paper-external diagnostic lane. It discovers GRAB clips where the
right-hand `artimano_rh` E0 retargeting result has an actual collision signal,
then compares the unchanged E0 baseline with the frozen
`dense_squared_hinge_deadzone1mm_v2` loss at `lambda=0.1` and a 1 mm dead zone.

G1 (`s1/airplane_lift`) is a source-MANO penetration control whose E0 robot
collision surface did not carry the signal. G2 (`s1/apple_eat_1`) was weak.
G3 (`s1/banana_lift`) remains an open-mesh/sign-semantics dispute, and G4
(`s1/alarmclock_lift`) remains a solver/contact dispute. They are excluded
before source eligibility and cannot be reintroduced by experiment results.

The workflow enumerates the raw GRAB pool, retains only complete right-hand,
native-120-FPS, watertight/orientable-object candidates, and records every
failure. Stage 7 warm starts use only fixed source frames `[first, middle,
last]`; at most 40 warm-passing candidates enter the short E0 probe. E0 probe
failures are classified and do not stop the scan. The top three clips are
frozen by the lexicographic E0 robot penetration key
`frames>1mm, mean excess, max, active links, sequence id`.

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=src conda run -n topo-retarget python -m toporetarget workflow \
  run-s1-2a-stress-discovery \
  --config configs/experiments/s1_2a_e0_penetration_stress_v1.yaml \
  --experiment-root .local/experiments/s1_2a_e0_penetration_stress_v1 \
  --resume
```

All derived files are under `.local/experiments/s1_2a_e0_penetration_stress_v1/`.
The frozen set is `selection/stress_selection.lock`; full comparison artifacts
are under `stress/<clip_id>/E0/` and `stress/<clip_id>/S1/`. The fast convex-hull
backend is audited only against the reference triangle-winding backend on the
E0 active region. A fast-backend miss is reported and never fixed by changing
the loss.

This stress set is not a GRAB benchmark, does not establish a global default,
and does not establish ground-truth contact-retention improvement. A passing
automatic decision can only authorize a later bounded lambda study; it cannot
change the frozen formulation or selection.
