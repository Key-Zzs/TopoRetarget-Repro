# Per-Finger Reference/Actual Contact Audit

`scripts/evaluation/audit_stage16d_per_finger_contact.py` is the original
offline, read-only audit for a completed Stage 16-D Reward V3 Formal20 trace.
It does not launch IsaacLab, train PPO, or alter the reference, reward,
controller, or physics contracts.

The successor R2 audit uses [Reference Contact Contract V2](REFERENCE_CONTACT_CONTRACT.md)
and [full hand--object pair telemetry](FULL_HAND_OBJECT_PAIR_TELEMETRY.md).
It preserves this document's V3 mask and five-tip trace interpretation, while
making source-supported contact, <=2 cm geometric candidate contact, and 2--3
cm proximity-only evidence explicit.

The audit preserves the frozen V3 primary reference mask
`distance_m < 0.03`.  It additionally labels `distance_m <= 0.02` as strong
reference evidence and `0.02 < distance_m < 0.03` as proximity-only ambiguous
evidence.  The latter is diagnostic only; it is not a reward replacement.

Actual contact is the recorded V3 per-fingertip actual-contact boolean together
with the recorded pair-force validity mask.  The audit never invents a force
threshold or derives finger force from an aggregate contact value.

The report includes per-episode/finger recall, persistent missing windows,
force concentration, aggregate compensation, and time-resolved free-flight
re-catch evidence.  Palm substitution is reported only when a matching
time-resolved wrist-object pair contact source exists; otherwise it remains
unavailable.

Run it against the default frozen inputs with:

```bash
conda run -n toporetarget-rl python scripts/evaluation/audit_stage16d_per_finger_contact.py
```

The output is intentionally under `.local/reports/` and is not a tracked
experiment result. A per-finger reward design is not authorized merely by a
visual floating finger: it needs source-supported expected contact, persistent
actual loss, aggregate compensation, and linked physics degradation.
