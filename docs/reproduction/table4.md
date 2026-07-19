# Table 4 — Reference-tracking MDP

Transcribed from Appendix A.5.4, PDF p. 14.

| Term/condition | Expression or setting | Weight/limit |
| --- | --- | --- |
| Object | $\psi(\frac16\sum_{m=1}^6\|u_m-u_m^{ref}\|_2;0.04)$ | 8.0 |
| Link position | $\frac1L\sum_{\ell=1}^L\psi(\|p_\ell-p_\ell^{ref}\|_2;0.025)$ | 1.0 |
| Joint position | normalized joint error kernel, scale 0.1 | 1.0 |
| Action smoothness | $\|a_t-a_{t-1}\|_2^2+\|a_t-2a_{t-1}+a_{t-2}\|_2^2$ | -0.01 |
| Timeout | Episode reaches 20 s | — |
| Object unstable | Height < 0.06 m; linear velocity > 10 m/s; angular velocity > 500 rad/s | — |
| Object position error | Error > 0.05 m | — |
| Object orientation error | Error > 45 degree | — |
| Object axis-point error | Any of six errors > 0.05 m | — |

Tracked-link identity and axis-point spatial construction are not provided.

