# Table 3 — Selected retargeting parameter settings

Transcribed from Appendix A.1, PDF p. 12. These values are locked in `configs/paper/retarget.yaml`.

| Parameter | Value | Description/unit |
| --- | ---: | --- |
| $N_o$ | 50 | Object surface samples / samples |
| $\lambda_{IM}$ | 500 | Interaction-mesh weight / dimensionless |
| $\kappa$ | 30 | Distance-decay factor / inverse meter |
| $\lambda_{warm}$ | 1 | Initialization bone-direction weight |
| $\lambda_{bone}$ | 0.1 | Refinement bone-direction weight |
| $\lambda_{smooth}$ | 2.5 | Initialization temporal smoothness |
| $\lambda_{reg}$ | 2.5 | Refinement temporal regularization |
| $\lambda_{base,pos}$ | 100 | Floating-base translation prior |
| $\lambda_{base,rot}$ | 1 | Floating-base rotation prior |
| $\tau$ | 0.001 m | Soft penetration tolerance |
| $b$ | 0.030 m | Hard penetration backstop |
| $w_s$ | $1.0\times10^5$ | Slack penalty |

The paper does not provide optimizer, iteration, tolerance, line-search, Delaunay-flag, or robot
surface collision-sample settings; those remain null rather than inferred.

