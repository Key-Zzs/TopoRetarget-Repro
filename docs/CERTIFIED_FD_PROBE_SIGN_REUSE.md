# Certified FD-Probe Sign Reuse

For an ambiguous spatial finite-difference base point `x`, v4 first obtains an
exact signed distance `phi(x)`.  For a probe at distance `h`, its sign is
reused only when `abs(phi(x)) > h + safety_margin`.  Since signed distance is
1-Lipschitz, that condition proves every such probe remains on the same side
of the zero level set.  The probe cannot write to the trajectory sign cache.

Every point failing the certificate, including a surface crossing, enters the
exact compiled winding path and then the near-threshold Python reference
fallback if required.  The rule is separate from the existing cross-iterate
Lipschitz cache certificate.
