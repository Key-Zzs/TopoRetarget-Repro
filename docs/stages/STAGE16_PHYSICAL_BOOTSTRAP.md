# Stage 16 Physical Bootstrap (P0–P2)

This document defines the bridge from the closed Stage16-D causal zero-gravity
baseline to a possible physical PPO stage. It is intentionally not a PPO,
gravity-curriculum, or friction-curriculum result.

The machine-readable contracts are:

- `configs/rl/stage16/stage16_physical_bootstrap.yaml`
- `configs/rl/stage16/stage16_contact_ready_rsi_v2.yaml`
- `configs/rl/stage16/stage16_p3_entry_gate_v1.yaml`

The materialization command is
`scripts/evaluation/run_stage16_physical_p0_p2.py`. Its reports are local,
ignored diagnostic evidence under `.local/reports/stage16_physical_p0_p2`.

## P0: frozen physical boundary

P0 keeps the Stage16-D parent, Reference Kinematics V2, PPO-26D action
interface, current nominal asset/contact configuration, Source Contact
Semantics, and Evaluation Suite V2 traceable by hash. The intended gravity is
`(0, 0, -9.81) m/s²`, labelled `EARTH_NOMINAL_ENGINEERING_TARGET`; it is not a
claim of source calibration.

The contract forbids external guidance, rollout object-state writes, and
rollout wrist-root writes. Object and wrist state writes remain reset-only.
Unknown mode strings and malformed provenance fail closed.

## P1: Contact-ready RSI V2

RSI V2 classifies every V2 runtime index from three sources:

1. Source Contact Semantics confirmed/persistent labels are the contact truth.
2. The retargeted Wuji link-to-object-axis distance is recorded as geometry
   evidence; it cannot manufacture contact.
3. V2 object twist distinguishes manipulation from a terminal hold.

The classes are `PRE_CONTACT`, `NEAR_CONTACT`, `CONTACT_READY`,
`PERSISTENT_CONTACT`, `MANIPULATION`, `TERMINAL_HOLD`, and `AMBIGUOUS`.
`PRE_CONTACT` and `AMBIGUOUS` are explicitly invalid reset states. The old V3
three-centimetre reward mask is forbidden as RSI truth.

The bounded gravity diagnostic uses real Isaac/PhysX with nominal gravity,
zero policy residual plus reference following, four replicas per state, and
20 control steps (one second). It records contact timing/persistence,
pre-contact displacement and velocities, joint/finite/catastrophic outcomes,
and rollout-write counters. It does not train PPO or add a support mesh.

Only a state for which every replica passes the predeclared engineering
thresholds can enter a safe bank. The only initial P3 banks are
`CONTACT_READY_SAFE`, `PERSISTENT_SAFE`, and `MANIPULATION_SAFE`;
`NEAR_CONTACT_SAFE` and `TERMINAL_SAFE` remain diagnostic-only by default.

## P2: support feasibility

Support uses the following evidence hierarchy:

1. explicit source scene/support asset;
2. recoverable source scene geometry with provenance;
3. hand-support evidence from the true-PhysX P1 diagnostic;
4. otherwise `SUPPORT_UNKNOWN`.

An infinite ground plane, generic table, fixture, attachment, or hidden
support is never an automatic fallback. In particular, a source that lacks a
recoverable support asset cannot authorize a frame-zero full-gravity reset.

If P1 provides contact-ready safe banks but source support remains unavailable,
P2 may authorize only `CONTACT_READY_ONLY_VALIDATED`. That is a constrained
reset policy, not evidence that the source object was supported by the hand at
frame zero.

## P3 decision contract

P3 is an entry decision only. It never starts a trainer. The frozen gates are:

- G0 provenance and parent contracts;
- G1 Contact-ready RSI V2;
- G2 source-support or constrained contact-ready feasibility;
- G3 current absolute hand-object and inter-finger geometry gates;
- G4 controller/actuator and joint-limit safety for the authorized reset bank;
- G5 zero guidance, no hidden support, and no prohibited rollout writes.

No historical zero-gravity geometry artifact may be relabelled as a
full-gravity G3 pass. If G3 has not been run against the current physical
state, P3 is `P3_BLOCKED_TECHNICAL`, even when P0/P1/P2 otherwise pass. A
`P3_READY_WITH_CONSTRAINTS` decision, when all gates genuinely pass, still
limits resets to the named safe banks and never permits frame-zero
full-gravity reproduction without source-backed support.

## Reproduction boundary

Run P0/P1/P2 only after their input reports exist:

```bash
conda run -n toporetarget-rl python scripts/evaluation/run_stage16_physical_p0_p2.py p0
conda run -n toporetarget-rl python scripts/evaluation/run_stage16_physical_p0_p2.py build-p1-banks
# Run the bounded fresh Isaac workers, then merge their COMPLETE receipts.
conda run -n toporetarget-rl python scripts/evaluation/run_stage16_physical_p0_p2.py run-p1-diagnostics
conda run -n toporetarget-rl python scripts/evaluation/run_stage16_physical_p0_p2.py merge-p1-diagnostics
conda run -n toporetarget-rl python scripts/evaluation/run_stage16_physical_p0_p2.py finalize-p1
conda run -n toporetarget-rl python scripts/evaluation/run_stage16_physical_p0_p2.py finalize-p2
conda run -n toporetarget-rl python scripts/evaluation/run_stage16_physical_p0_p2.py finalize
```

The first P3 PPO run, any gravity curriculum, any friction curriculum, and
any P4 human decision require separate authorization.
