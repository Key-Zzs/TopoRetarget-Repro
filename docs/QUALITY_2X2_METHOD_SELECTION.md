# Automatic 2×2 Method Selection

The matrix is:

| ID | Warm | Final |
| --- | --- | --- |
| E0 | paper warm | development base final |
| E1 | M* morphology candidate | development base final |
| E2 | paper warm | C* contact candidate |
| E3 | M* morphology candidate | C* contact candidate |

Each entry is bound to the same four clips, native frames, object samples,
collision samples, canonical SDF, tolerance, and acceptance policy. Existing
identical artifacts may be reused by hash; no equivalent solve is silently
duplicated.

Selection first requires complete 60/60 strict acceptance, q/slack bounds,
full-512, zero frames over 2 mm penetration, artifact completeness, source
integrity, and deterministic selected-frame repeat. Relative to E0, contact F1
must not regress by more than 10%, morphology RMSE by more than 0.5 mm, or
continuity by more than 20%; runtime p95 is capped at 3× unless quality gain is
explicit. A passing extension must improve at least three clips and macro F1 by
five percentage points while preserving finger and thumb gates.

The final selection is Pareto-based over contact F1/precision/alignment,
morphology RMSE, penetration, continuity, and runtime. Ties are broken by
profile ID. If no extension passes, `RECOMMENDED_PROFILE=E0` and
`EXTENSIONS_REJECTED=true` are the valid result.
