# Solver feasibility restoration

`restore_slack_feasibility` is a generic initialization repair for recoverable
SLSQP status 8/9 exits. It holds the returned primary q/base variables fixed
and re-evaluates the unchanged physical-unit active constraints.

For every active soft row it computes the minimum original slack necessary to
satisfy that row. Hard residuals must already be non-negative, all values must
be finite, and the required slack must fit in the pre-existing slack bounds.
Only then is a slack coordinate advanced with `nextafter(required, +inf)` when
needed. This avoids arbitrary interior padding while making the exact
floating-point inequality representable. The reconstructed state is merely the
initial point for the existing original-objective/original-constraint solver;
normal optimizer convergence, active-set convergence, bounds, and one
independent full-surface audit remain mandatory.

This is solver backend engineering, not a paper objective or constraint change,
and is not author-exact. It has no dataset, object, selection, or sample
conditional.
