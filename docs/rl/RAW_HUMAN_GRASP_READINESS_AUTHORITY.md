# Stage16 Raw Human Grasp Readiness Authority

`RawHumanGraspReadinessProfileV1` is an offline, morphology-aware description
of contact between the provenance-resolved raw HOCap MANO surface and the raw
selected object. It deliberately does not reduce the evidence to an unvalidated
functional-grasp boolean.

## Strict V4 is reward-specific

Strict V4 does not measure only MANO tips. Its source-side evidence uses MANO
v1.2 LBS-derived named finger regions over the complete finger surfaces. A
source region is confirmed by a minimum distance at most `2 mm`, a connected
component of at least three vertices within `5 mm`, and two native 30 Hz
frames. Confirmed/persistent source fingers are mapped to same-named robot
distal-tip pair-force reward targets at runtime.

That mapping is valid for `StrictPerFingerContactRewardV4`, but it was not
validated as an authority for force-bearing human grasp readiness:

```text
STRICT_V4_MASK_FUNCTIONAL_GRASP_AUTHORITY=NOT_SUPPORTED
STRICT_V4_IS_MANO_TIP_ONLY=NO
```

## Profile layers

The profile reports, without outcome tuning:

- persistent contact anywhere on all 778 MANO vertices;
- persistent robust contact by thumb, index, middle, ring, pinky, and palm;
- contact by tip, distal, middle, proximal, and palm surface segments;
- persistent multi-region and thumb/non-thumb topology;
- a geometric opposing-normal/separation diagnostic;
- continuous relative translation and rotation curves.

The object mesh is queried by exact triangle distance. Persistence preserves
the source duration of `2/30 s`; at the 20 Hz runtime representation this is
two frames. The raw object meshes are not watertight by edge incidence, so
triangle-normal opposition remains a geometric diagnostic and is not promoted
to force closure. Raw force and a preregistered coupled-motion threshold are
unavailable.

```text
DECISION=MULTIPLE_AUTHORITIES_REQUIRED_NO_SINGLE_BINARY
FUNCTIONAL_RAW_READY=NOT_IDENTIFIABLE
CONFIDENCE=HIGH
FORCE_CLOSURE_CLAIMED=NO
```

## Frozen event profile

| Clip | Any surface | Multi-region | Strict V4 target | Opposing topology | Thumb/non-thumb | LIFT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 170105 | 182 | 190 | 192 | 197 | NOT_IDENTIFIABLE | 184 |
| 170650 | 109 | 136 | 136 | 140 | 155 | 184 |

Frames are runtime indices at 20 Hz; the report preserves raw-source seconds
in a separate field. For 170105, any-surface contact appears `0.10 s` before
LIFT, but persistent multi-region contact appears `0.30 s` after LIFT. This
resolves the earlier Strict-V4-only raw interpretation only partially: the
profile is richer, but it still does not authorize a functional binary.

The frame profiles, region authority, event receipts, comparison table, and
replay questions are generated under:

```text
.local/reports/stage16_angular_semantics_and_raw_grasp_authority/raw_grasp_authority/
```
