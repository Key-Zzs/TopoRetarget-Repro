# Faithful Reproduction Finalization

This stage closes the bounded `GRAB/s1/airplane_lift`, right-hand,
`artimano_rh`, global-frame `[240,300)` reproduction after the Stage 9
implementation audit. It does not add another Stage 9.3.x diagnostic.

## Decision

The available numerical/model-assisted evidence and the completed human review
accept **case A**:

- the fixed and old current-lineage results have no visible relative
  degradation across the 60-frame contact sheet;
- the largest old-to-fixed robot-keypoint displacement is `0.654 mm`;
- the largest base translation difference is `0.276 mm`, and the largest base
  rotation difference is `0.00621 rad`;
- all 60 fixed frames have optimizer status zero, strict acceptance, and
  full-512 hard/soft audit passes;
- maximum raw penetration is zero for old and fixed;
- per-finger fixed-minus-old RMSE changes are `+0.0267 mm` thumb,
  `+0.0336 mm` index, `-0.0206 mm` middle, `+0.0785 mm` ring, and
  `+0.0423 mm` pinky.

The old Stage 9 quality gate reported `REPAIR_CANDIDATE_REJECTED` because it
required at least `1.5392 mm` long-finger improvement. That is an improvement
gate, not evidence of a visible regression. The finalization claim is
therefore semantic correction with quality neutrality, not quality
improvement.

The fixed trajectory is not uniformly smoother: its maximum base translation
and `q` steps are slightly lower, while its maximum rotation step and base
jerk are slightly higher. All remain sub-millimetre/small-angle motions and no
visible discontinuity was found, so the supported conclusion is continuity
neutral rather than improved.

Absolute contact retention remains limited in both old and fixed trajectories.
The source-label-conditioned robot visual-surface proxy places the old/fixed
middle distal surface about `13 mm` from the corresponding source-labeled
object region. This proxy is not contact ground truth, but it prevents a false
claim that the fix improved contact. The fixed profile is accepted only as
quality-neutral relative to the old result.

## Profile classification

The authoritative classification is
[`configs/retarget/finalization/faithful_reproduction_profiles.yaml`](../configs/retarget/finalization/faithful_reproduction_profiles.yaml):

- `scipy_slsqp_active_set_contact_rich_v2` is
  `historical_accepted`, non-faithful, and retained as the legacy engineering
  comparison. Its known deviation is that the base correction is included in
  temporal regularization.
- `scipy_slsqp_active_set_contact_rich_v3_fixed` is the canonical
  paper-faithful profile with `validated_quality_neutral` status. It is not
  claimed to improve numerical quality.

## Paper-fidelity conclusion

Projection is not a paper method. It is diagnostic-only, closed, and is not an
accepted reference.

The Eq. (9) implementation error was that the six-dimensional floating-base
correction was included in the temporal `q` term while base translation and
rotation were already controlled by their separate priors. The v3 fixed
profile applies temporal regularization only to finger `q` and retains all
paper weights and the separate floating-base priors.

V3 fixed is now the canonical faithful baseline. The v2 result remains the
historical/non-faithful engineering comparison because it preserves prior
accepted behavior and is useful for regression.

## Versioned Stage 10 candidate

The new candidate is isolated under:

```text
.local/runs/stage10_faithful_regularization_fix_v1/
  s1__airplane_lift__right__artimano_rh__f000240_f000300__faithful_regularization_fix_v1/
```

It contains a new manifest, NPZ and Zarr robot references generated from the
fixed artifact, the four-state HTML, visual/numerical audit, old-vs-new
comparison, profile classification, paper-fidelity statement, and manual
acceptance template. The historical Stage 10 manifest and references are not
modified. The root `INDEX.json` identifies this suffixed run as the only
authoritative candidate and marks any earlier pre-human manifest as
non-authoritative.

## Human acceptance and final status

The human reviewer played the 60-frame four-state bundle, reviewed local
frames `0, 9, 10, 12, 25, 27, 29, 30, 36, 39, 59`, and recorded case A with
all required visual checks passing. The validated record is bundled as
`review/manual_acceptance.json`. Finalization was executed with:

```bash
toporetarget workflow finalize-faithful-reproduction \
  --manual-acceptance /tmp/manual_acceptance.json
```

The authoritative versioned manifest and root `INDEX.json` now report
`FAITHFUL_REPRODUCTION_FINALIZED_CASE_A`, `human_manual_acceptance=pass`, and
`canonical_faithful_profile=scipy_slsqp_active_set_contact_rich_v3_fixed`.

The validator retains all three declared branches:

- A requires every visual check to be true and promotes v3 fixed to the
  canonical faithful quality-neutral profile.
- B requires at least one recorded regression plus a rationale and evidence
  frame. V3 fixed remains paper-faithful but is explicitly not production
  recommended; v2 remains the non-faithful historical engineering comparison.
- C requires every visual check to be true plus an improvement rationale and
  evidence frame. It records visible improvement in this versioned export
  without overwriting historical Stage 10.
