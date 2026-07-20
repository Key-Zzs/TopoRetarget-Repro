# Sequential warm-start optimization

Stage 7 solves the initialization problem independently of object geometry.
For `t=0` the initial q is Arti-MANO neutral q and only the bone residual is
used. For `t>0`, the previous frame's successfully solved warm-start q is the
temporal reference:

```math
\tilde q_t=\arg\min_q\lambda_{warm}E_{bone}(q)+
\lambda_{smooth}\|q-\tilde q_{t-1}\|_2^2.
```

The weights are loaded from `configs/paper/retarget.yaml` (`lambda_warm=1` and
`lambda_smooth=2.5`); the YAML aliases preserve the existing Stage 6 paper
parameter names without creating a second source of values.

## Residual and solver

The numerical residual is:

```text
sqrt(lambda_warm) * vec(pair_residuals)
sqrt(lambda_smooth) * (q - previous_q)       # t > 0 only
```

The reported paper objective is computed independently as the squared norm of
these residual blocks. SciPy's `least_squares` reports half that value through
`result.cost`, so the artifact stores both the paper objective and the solver
status and never labels the half-cost as Eq. (2).

The default `paper_repro_scipy_trf` profile uses float64,
`scipy.optimize.least_squares(method="trf")`, direct URDF lower/upper bounds,
and a Torch-autograd Jacobian of the differentiable Stage 4 FK and frame path.
The engineering tolerances and maximum evaluations are profile data, not paper
facts. Strict mode stops on the first failed frame; diagnostic mode records a
false valid mask and never copies the previous q as a fake solution.

The optimized coordinates are raw 22-joint radians. No range normalization,
sigmoid, PCA, latent pose, or per-joint weighting is applied. The input frame
sequence is contiguous native data: timestamps and 120 FPS are preserved, no
resampling occurs, and the temporal weight is not dt-normalized.

## Base seed and artifact

After q solving, the full initialization state is formed from canonical frames:

```math
T^S_{B,t}=T^S_{H_s,t}(T^B_{H_r,t}(q_t))^{-1}.
```

This base seed is outside Eq. (1) and Eq. (2); it is an explicit Stage 7
alignment strategy that preserves source scene motion for later stages. The
alignment identity and translation/rotation errors are stored in provenance.

`toporetarget.warm_start.v1` is an independent Zarr artifact under
`.local/cache/retarget/warm_start/`. It contains qpos, base pose, source and
robot canonical frames, FK anchors, directions, adjacent features, pair
residuals, per-frame objectives, temporal terms, statuses/evaluations, bounds
margins, timestamps, hashes, and assumptions. It never writes the source HOI
Zarr and it does not contain object samples or SDF results.

Artifacts are schema-checked, atomically published, and not overwritten unless
`--force` is supplied. Source/profile hashes are retained so a consumer can
reject an incompatible warm-start.

## Commands

```bash
toporetarget retarget warm-start \
  --canonical "$GRAB_CACHE" --hand right --robot artimano_rh \
  --start-frame 0 --end-frame 60 \
  --frame-profile canonical_keypoint_wrist_v1 \
  --bone-profile mediapipe21_full_finger_chain_v1 \
  --solver-profile paper_repro_scipy_trf \
  --output .local/cache/retarget/warm_start/right.zarr

toporetarget retarget validate-warm-start \
  --canonical "$GRAB_CACHE" --warm-start .local/cache/retarget/warm_start/right.zarr \
  --report .local/reports/stage7/right_validation.json \
  --csv .local/reports/stage7/right_validation.csv
```

The artifact is an initialization trajectory for a future interaction-aware
stage, not a final TopoRetarget reference and not a physical-feasibility claim.

Stage 8 reads this artifact without modifying qpos or base pose. Its source graph,
directed weights, and object samples are separate derived artifacts; Eq. (7) evaluation
does not start the Stage 9 optimizer.
