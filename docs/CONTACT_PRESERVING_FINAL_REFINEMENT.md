# Contact-Preserving Final Refinement

Contact candidates use source semantic labels, source MANO surface proxies,
object-local closest points, and the Arti-MANO visual surface profile. For each
active region, the position residual is normalized by 10 mm and uses Huber
`delta=1`; direction residual is `1-dot(u_robot,u_source)` after clipping the
dot product to `[-1,1]`. Region losses are means, so clips with more labels do
not receive more weight automatically.

The fixed grid is P1 `(0.25,0)`, P2 `(1,0)`, P3 `(4,0)`, PD1 `(1,0.1)`, and
PD2 `(1,0.5)` for position and direction weights. These are
`paper_method=false` and `paper_external_extension=true`. The existing Eq. (8),
bone/interaction terms, q/slack bounds, penetration constraints, QuerySet,
full-512 audit, and solver tolerance are not weakened.

GRAB contacts are proxies, not ground truth. A semantic label with implausible
source surface distance reduces confidence and records a virtual-contact
warning; it is never silently promoted to a hard constraint. Candidates that do
not pass all hard/regression/improvement gates remain visible as rejected
diagnostics, and C* is never presented as paper-exact.
