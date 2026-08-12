# Source Contact Semantics

`SourcePerFingerContactEvidenceV1` is the source authority for the final
Stage 16-D cross-embodiment contact audit. It is offline and diagnostic-only:
it does not replace the frozen Reward V3 3 cm mask or alter PPO, a checkpoint,
RSI, controller, reference, or physics.

For the selected HOCap clips, it reconstructs the raw right MANO surface from
`poses_m.npy`, subject-specific calibration betas, and `MANO_RIGHT.pkl`. It
queries every MANO surface vertex against the selected raw object mesh's exact
triangles in the raw HOCap world/object pose convention. A raw source frame is
`SOURCE_CONTACT_CONFIRMED` only when all of the following hold:

- minimum surface-to-triangle distance is at most 2 mm;
- a connected component of at least three MANO vertices lies within 5 mm; and
- the condition persists for at least two native 30 Hz frames.

The audit stores 1/2/5 mm threshold sensitivity. `SOURCE_CONTACT_PROBABLE`
meets the geometric/component rule but not native persistence;
`SOURCE_CONTACT_TRANSITION` is an explicit state change; `SOURCE_PROXIMITY_ONLY`
is at most 10 mm without the robust component evidence; all other samples are
`SOURCE_NO_CONTACT`.

MANO thumb/index/middle/ring/pinky/palm regions are derived from MANO v1.2 LBS
joint-chain weights. Low-margin webbing vertices become
`boundary_ambiguous`, rather than being silently credited to a finger. Segment
labels use the same joint influences and rest-chain longitudinal geometry; a
tip surface is a model-derived terminal quantile, never a hard-coded vertex
list.

The native source has exactly 41 selected keys. It maps to the existing 321
control frames at factor 8: exact keys retain their class, only adjacent
confirmed keys fill a `SOURCE_CONTACT_PERSISTENT` interval, two no-contact
keys fill no contact, and every other interval remains transition.

Run the final read-only materialization with:

```bash
python scripts/evaluation/finalize_stage16d_source_contact_semantics.py
```

The report is written under
`.local/reports/stage16d_source_contact_semantics_final_audit/`.
