# S1.3 Backend Profiling

Cold context construction and warm query timing are reported separately. The
performance gate requires a 3x median batch-SDF gain and a 2x callback gain.
No extrapolated timing is accepted: a backend that is exact but fails either
gate remains unavailable for T4 discovery.
