# Reference Contact Contract V2

`ReferenceContactContractV2` is an offline, diagnostic-only interpretation of
the frozen Stage 16-D Reward V3 reference mask. It does not change Reward V3,
PPO, the checkpoint, the controller, or any simulator setting.

## Frozen boundary

The historical V3 primary mask remains exactly:

```text
reference_fingertip_to_object_distance_m < 0.03
```

V2 records that mask as `historical_v3_primary_mask`; it must never be fed back
to an already trained V3 policy. V2 uses the following interpretation:

| Evidence class | Rule | Meaning |
| --- | --- | --- |
| `NO_CONTACT_EXPECTED` | distance >= 3 cm and no source/topology support | Reference allows the finger to float. |
| `PROXIMITY_ONLY_AMBIGUOUS` | 2 cm < distance < 3 cm and no support | Historical V3 proximity could be too broad. |
| `GEOMETRIC_STRONG_CONTACT_CANDIDATE` | distance <= 2 cm, no source support | A strong geometric candidate, not source-confirmed contact. |
| `SOURCE_SUPPORTED_CONTACT` | explicit, derived-source, or topology support | Contact is expected even if geometry is farther than 2 cm. |
| `REFERENCE_CONTACT_EVIDENCE_CONFLICT` | source/topology support but distance > 5 cm | Do not silently choose either signal; investigate provenance. |

`SOURCE_EXPLICIT`, `SOURCE_DERIVED`, and `TOPOLOGY_DERIVED` take precedence
over geometric proximity. A V2 expectation window is persistent only when it
lasts at least three 50 ms control steps.

## Current HOCap R2 limitation

The frozen inputs for `hocap_170105` and `hocap_170650` include the historical
five-tip distance field, but no verified source per-finger contact annotation,
source hand--object field mapped to the Wuji tips, or source-derived topology
signal. Consequently their <=2 cm samples are reported as
`GEOMETRIC_STRONG_CONTACT_CANDIDATE`, never promoted to confirmed source
contact.

Persistent floating on those samples is an important V4 hypothesis, but is not
authorization to replace V3. Capture or map source/topology evidence first,
rerun the same Formal20 audit, then consider a separately versioned V4 ablation.

## Materialization

```bash
python scripts/evaluation/prepare_stage16d_contact_contract_v2.py
```

It writes ignored local evidence under
`.local/reports/stage16d_contact_contract_v2_audit/`, verifies the selected
checkpoint, Formal20 seeds, references, physics provenance, and frozen V3
mask, and fails closed on any drift.
