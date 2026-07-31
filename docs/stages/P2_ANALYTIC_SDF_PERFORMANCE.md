# P2 Analytic SDF Performance

P2 freezes `wuji_continuous_sequential_fast_exact_v1` and introduces
`wuji_continuous_sequential_fast_exact_v2`. It validates exact closest point/SDF
gradients, chain-rule constraint Jacobians, spatial-FD fallback, sign-cache safety,
determinism, and fixed five-frame qualification. It does not resume Stage-12 or alter
any legacy `SIGSTOP` worker.
