# Fast/reference SDF consistency

S1.1 keeps two SDF roles separate:

| Role | Backend | Allowed use |
|---|---|---|
| optimizer inner loop | `convex_hull_exact_solver_only` | objective/gradient callback after the existing solver-side contract is built |
| independent validation | `reference_triangle_winding` | full-surface audit, penetration metrics, backend gate, and final decision |

On frozen stress clips the audit compares finite signed distances, sign
agreement, absolute error, correlation, penetration-active recall/precision and
gradient-normal cosine. The required gate is sign agreement at least 0.99,
reference penetration-above-1-mm recall at least 0.95, correlation at least
0.95, gradient cosine at least 0.90 when gradients are available, and all
finite. Any failed clip fails the aggregate gate and routes to
`S1_1_ROUTE_TO_S1_2_BACKEND_STUDY`; the fast backend can never substitute for
the reference validation path.

Per-clip JSON/CSV/HTML artifacts live under
`backend_consistency/`; the aggregate is
`reports/fast_reference_consistency.json`.

