# Wuji Multi-Threshold Penetration Audit

W2.3 re-audits the immutable baseline and formal continuous artifacts using
the persisted 672-point collision surface and signed distance. `R_pen(0 mm)`
counts every negative signed distance and is a sensitive numerical/mesh
diagnostic. The paper hard threshold is `R_pen(2 mm)`; the 0.25, 0.5, and 1 mm
columns are engineering diagnostics.

## Rates and depth

| Unit/profile | Rpen0 | Rpen0.25 | Rpen0.5 | Rpen1 | Rpen2 | Max depth (mm) | p95 (mm) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| W1 baseline | 0.0167 | 0.0167 | 0.0167 | 0 | 0 | 0.8620 | 0 |
| W1 continuous | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| W2 baseline | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 | 1.0806 | 1.0087 |
| W2 continuous | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 | 1.0175 | 1.0130 |
| W3 baseline | 0.9000 | 0.8000 | 0.6833 | 0.5333 | 0 | 1.0747 | 1.0611 |
| W3 continuous | 0.9500 | 0.9167 | 0.7833 | 0.6167 | 0 | 1.0191 | 1.0117 |

The 2 mm hard gate passes for all three clips: continuous `R_pen(2 mm)` is
not worse than baseline, maximum depth remains below 2 mm, and full-surface
and unqueried audits pass. W3 fails only the separately reported secondary
rate/depth warning, so the sequential recommendation remains valid with a
warning.

## W3 0.90 -> 0.95 interpretation

At threshold 0 mm the continuous rate changes from `0.90` to `0.95`, a delta
of `+0.05`. At the paper threshold the rate remains `0`; maximum depth improves
from `1.0747` mm to `1.0191` mm. The affected link groups are
`r_index_finger_distal`, `r_index_finger_middle`, `r_middle_finger_distal`,
and `r_thumb_distal`. The classification is
`SHALLOW_NUMERIC_PENETRATION_ONLY`: it is a shallow signed-distance diagnostic,
not a paper-threshold failure.

Raw rows, per-link aggregation, and machine-readable thresholds are in
`w2_3_finalization/penetration_audit/` and
`w2_3_finalization/reports/penetration_reaudit.{json,csv}`.
