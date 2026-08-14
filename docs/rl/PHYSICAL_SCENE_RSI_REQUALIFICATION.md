# P3-B.6 physical scene and RSI requalification

P3-B.6 is a fail-closed qualification stage. It scans all 321 frames of both
HOCap clips, reconstructs all 21 Wuji collision bodies from the V2 wrist pose
and finger state, queries the exact `python-fcl` runtime manifest, and evaluates
the finite inferred table as a static kinematic actor. Tracked points and the
reference ghost are diagnostics only; neither is formal collision geometry.

The generated `PhysicalReferenceValidityMaskV1` contains, per frame:

- reference/source semantic class and support state;
- H-O, H-T, O-T, and inter-finger penetration/separation metrics;
- validity and explicit failure reasons;
- the reference pose, finger state, object twist, and source interval.

`PhysicalSafeRSIBankV1` is built from the full bank using the frozen geometry
gates and support causality. It contains no historical blacklist. The offline
bank is evidence for reset selection, not an authorization to start PPO.

## Runtime qualification

Dynamic reset uses Isaac Lab 5.1 / GPU PhysX with 1g gravity, nominal friction,
active finite table actors, zero residual actions, and no guidance, attachment,
table movement, or rollout object/wrist-root state writes. Each candidate uses
four replicas and a 20-control-step horizon. Joint zero replay starts at the
earliest physically valid PRE_CONTACT frame and is authorized only if all 320
steps complete without a runtime termination.

The current receipt is:

| Clip | Offline physical bank | Dynamic 4×20 safe states | Joint replay |
| --- | ---: | ---: | --- |
| `hocap_170105` | 162 | 66 | not authorized; `FAILURE_JOINT_LIMIT` at step 4 |
| `hocap_170650` | 102 | 2 | not authorized; `FAILURE_JOINT_LIMIT` at step 5 |

The dynamic passes have zero object/wrist-root rollout writes and continuous
support contact. Dynamic failures are runtime joint-limit failures, not table
falling or object displacement. Both full reference trajectories nevertheless
fail the formal active H-O p95 gate; `hocap_170105` also fails the formal H-T
gate. Therefore the only decision is:

```text
P3_RESTART_BLOCKED_REFERENCE_GEOMETRY
```

PPO gravity training was not started. The next permitted step is
`REFERENCE_GEOMETRY_REPAIR`; expanding a blacklist or moving the table is not
an acceptable repair.

## Reproduction

```bash
PYTHONPATH=src conda run --no-capture-output -n toporetarget-isaaclab \
  python scripts/physics/validate_physical_scene_rsi.py --phase offline

PYTHONPATH=src conda run --no-capture-output -n toporetarget-isaaclab \
  python scripts/physics/validate_physical_scene_rsi.py --phase dynamic \
  --clip hocap_170105 --dynamic-start 0 --dynamic-max-states 32 \
  --dynamic-steps 20 --accept-eula

PYTHONPATH=src conda run --no-capture-output -n toporetarget-isaaclab \
  python scripts/physics/validate_physical_scene_rsi.py --phase joint \
  --clip hocap_170105 --joint-replicas 4 --accept-eula

PYTHONPATH=src python scripts/physics/finalize_physical_scene_rsi.py
```

The dynamic phase is run one clip/chunk per Isaac process. The complete ignored
receipt is under `.local/reports/stage16_p3b6_scene_rsi_requalification/`:
`physical_reference_validity_mask.parquet`, `.npz` masks, safe banks, dynamic
qualification reports, joint traces, screenshots, `p3_restart_decision.json`,
and `handoff.md`.
