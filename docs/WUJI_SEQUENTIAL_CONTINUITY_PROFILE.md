# Wuji Sequential Continuity Profile

`wuji_continuous_sequential_v1` is a separately named W2.3 candidate derived
from `wuji_continuous_full_state_v1`. It is intended only for offline reference
generation on the frozen W1/W2/W3 `s1` clips.

## Profile split

| Field | Full-state profile | Sequential profile | Difference allowed |
| --- | --- | --- | --- |
| profile id | `wuji_continuous_full_state_v1` | `wuji_continuous_sequential_v1` | metadata identity |
| production window fallback | enabled | disabled | yes; the only solver-semantic difference |
| temporal/objective/constraint fields | unchanged | unchanged | no other semantic difference |
| role | accepted engineering profile | recommended candidate | metadata |
| realtime / RL / cross-subject | no | no | explicit scope metadata |
| author exact | unresolved | unresolved | explicit provenance |

The structural audit records exactly one semantic difference:
`window.fallback_enabled: true -> false`. The sequential production path ends
after propagated, trust-region, and deterministic multi-start attempts. The
five-frame window remains an experimental diagnostic shadow and cannot alter a
formal trajectory or the recommendation gate.

## Evidence

- Formal W1/W2/W3 trajectories remain 60 frames and have zero production-window
  invocations.
- The bounded selected replay covers 21 frames across W1/W2/W3, including retry
  paths and deterministic numeric-equivalence tolerances.
- W1 full replay is recorded as bounded by the historical approximately
  1977.7-second runtime; it was not started under the controlled budget.
- New output is isolated under
  `.local/experiments/wuji_hand2_continuous_v1/w2_3_finalization/`.

## Recommendation

The current gate is
`WUJI_CONTINUOUS_SEQUENTIAL_PROFILE_RECOMMENDED_WITH_SECONDARY_PENETRATION_WARNING`.
The recommendation is limited to `offline_reference_generation`; it is not a
claim of author-exact reproduction, RL readiness, realtime readiness, or
cross-subject validity.

Run or resume the bounded evidence command with:

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=src \
  /home/deepcybo/miniconda3/bin/python scripts/wuji_w2_3_finalization.py
```

Machine-readable evidence is in `w2_3_finalization/reports/`, with the
compatibility summary at `reports/w2_3_summary.json`.
