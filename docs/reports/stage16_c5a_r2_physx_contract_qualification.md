# Stage 16-C.5A-R2 PhysX contract qualification

## Verdict

`BLOCKED_NO_GPU_PHYSX_CONTRACT`: no candidate in the frozen G0--G5 matrix
passed the S2 contact-onset replication gate.  The selected-contract file
therefore remains null, and C3-P, C4-P, natural-baseline O1, C5B, C5C, and
PPO are not authorized to run.

This is a physical-contract block, not a configuration-application failure:
every executed candidate reported matching requested config and live USD
attributes, finite observations, no early termination, and zero hidden
object/wrist execution-state writes.

## Frozen prerequisites

- Current branch/head was a descendant of `3457396`; the R1 diagnosis was
  retained as a frozen input.
- P0 audited the requested config and the live `/physicsScene`, robot, and
  both object prims.  G0 is TGS on `cuda:0`, 4--8 position iterations, 1--2
  velocity iterations, actor 8/2, enhanced determinism off, no Fabric clone.
- P1 verified that the installed Isaac Lab / Isaac Sim stack exposes and maps
  `enable_enhanced_determinism`, solver iteration attributes, and
  `solve_articulation_contact_last` to live USD.
- P2 froze the baseline/reference/controller/physics hashes and the complete
  six-GPU-plus-one-CPU matrix before any candidate run.  No candidate was
  added afterwards.

The machine-readable freeze and audit evidence is in
`.local/reports/stage16c5a_r2_physx_contract/`, including
`frozen_inputs.json`, `current_physx_contract.json`,
`physx_determinism_api_audit.json`, and
`physx_contract_candidate_matrix.json`.

## Reproduction commands

Run the pre-flight audit with the Isaac environment, then the frozen-matrix
qualifier for one candidate and stage:

```bash
conda run -n toporetarget-isaaclab env OMNI_KIT_ACCEPT_EULA=YES \
  python scripts/rl/isaaclab/audit_stage16c5_physx_determinism_api.py \
  --accept-eula --runtime-only \
  --output-dir .local/reports/stage16c5a_r2_physx_contract

conda run -n toporetarget-isaaclab env OMNI_KIT_ACCEPT_EULA=YES \
  python scripts/rl/isaaclab/qualify_stage16c5_physx_contract.py \
  --accept-eula \
  --matrix .local/reports/stage16c5a_r2_physx_contract/physx_contract_candidate_matrix.json \
  --candidate-id G0 \
  --frames .local/reports/stage16c5a_state_replication/replication_test_frames.json \
  --stage S2 --trials 20 \
  --output .local/reports/stage16c5a_r2_physx_contract/candidates/G0/S2_contact_onset.json
```

Both scripts refuse to overwrite an existing report.  The qualifier returns a
report for a physics-gate failure so the frozen matrix can continue; its JSON
`result` remains the only pass/fail authority.

## Real runtime qualification

Each GPU candidate passed S0 (one environment, 100 no-contact steps) and S1
(two clips, pre-contact, 20 trials each).  Each then failed S2 under the same
two-clip, 20-trial contact-onset procedure (40 samples per candidate).

| Candidate | Authorized difference from G0 | S2 result | Dominant S2 evidence |
| --- | --- | --- | --- |
| G0 | Frozen historical control | FAIL | angular velocity max 1.277e-2 (cap 1e-2); reward max 2.626e-3 (cap 1e-3) |
| G1 | Enhanced determinism | FAIL | Same S2 failure class as G0 |
| G2 | G1 + position iterations 16 | FAIL | angular velocity max 1.705e-2; reward max 2.035e-3 |
| G3 | G2 + velocity iterations 4 | FAIL | angular velocity max 1.680e-2; reward max 2.033e-3 |
| G4 | Position/velocity iterations 32/4 | FAIL | angular velocity max 3.835e-2; reward max 3.737e-3 |
| G5 | G2 + articulation contact last | FAIL | angular velocity max 9.262e-3 but quaternion/reward caps still fail (1.381e-3 / 2.037e-3) |
| C0 | CPU-only G0 diagnostic | FAIL | angular velocity max 7.186e-2; CPU result is diagnostic only |

Every report records PID/PGID, runtime device, the requested contract, actual
USD values, and write-audit counters.  The consolidated ledger is
`.local/reports/stage16c5a_r2_physx_contract/physics_contract_selection.json`.

## Why the block is real

The failure starts only at contact onset.  It persists after the documented
enhanced-determinism flag, multiple bounded solver-iteration settings, and
the documented contact-last ordering.  The CPU diagnostic also fails (and is
worse), so this result does not support a claim that merely changing GPU
solver flags resolves the issue.

The qualification contract forbids both tolerance softening and hidden object
or wrist state writes.  Neither is an acceptable repair because either would
turn an unresolved contact discrepancy into an apparent replay success.

## Repair path

Do not activate a candidate or proceed to C3-P/C4-P/O1.  A subsequent goal
must first authorize a new, explicitly frozen candidate family after an
evidence-backed investigation of the PhysX contact batching/scene topology
contract.  It must retain the same source/reference/controller hashes,
re-run P0--P2, and re-run the full S0--S5 gate.  Potential future dimensions
must be justified by the installed APIs; they cannot be silently folded into
this completed matrix.

## Scope stop

No C5B, C5C, PPO implementation, training, or run occurred.  No PPO-facing
configuration was enabled.  This report records a fail-closed qualification
outcome rather than a proxy success.
