# Certified Hierarchical Winding

No hierarchical winding path is implemented in this revision.  A BVH traversal
may only replace an exact leaf reduction if it supplies a proven bound on each
node contribution and proves the remaining interval lies entirely above or
below the configured winding threshold.  No heuristic, ray parity, or
unproven approximate fast winding is eligible for v4.

Status: `CERTIFIED_HIERARCHICAL_WINDING_NOT_PROVEN`.
