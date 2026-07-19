# Table 5 — Domain randomization ranges

Transcribed row by row from Appendix A.5.5, PDF p. 15.

| Randomized quantity | Range/setting | Mode |
| --- | --- | --- |
| Joint-position observation noise | $N(0,0.02)$ rad | Step obs. |
| Joint-velocity observation noise | $N(0,0.05)$ rad/s | Step obs. |
| Object axis-point position noise | $N(0,0.002)$ m | Step obs. |
| Object axis-point orientation noise | $N(0,0.01)$ rad | Step obs. |
| Observation delay | 0–2 control steps | Step obs. |
| Reference reset: finger joints | $U[-0.02,0.02]$ rad | Reset |
| Reference reset: object position | $U[-0.005,0.005]$ m | Reset |
| Reference reset: object orientation | Axis-angle, angle $U[-0.03,0.03]$ rad | Reset |
| Object COM offset | [-0.003, 0.003] m | Startup |
| Robot friction scale | [0.7, 1.3] | Startup |
| Robot collision-geometry scale | [0.9, 1.1] | Startup |
| Object mass/inertia scale | [0.4, 1.6] | Startup |
| PD stiffness scale | [0.75, 1.5], log-uniform | Startup |
| PD damping scale | [0.5, 2.0], log-uniform | Startup |
| Joint damping scale | [0.3, 3.0], log-uniform | Startup |
| Joint armature scale | [0.75, 1.3] | Startup |
| Joint friction-loss scale | [0.5, 2.0] | Startup |
| Encoder bias | [-0.01, 0.01] rad | Startup |
| Robot link inertia scale | [0.4, 1.5] | Startup |
| Robot link mass scale | [0.4, 1.5] | Startup |
| External object force | $U[-0.25,0.25]$ N, impulse every 0.6–1.8 s | Rollout |
| External object torque | $U[-0.00375,0.00375]$ N m | Rollout |

