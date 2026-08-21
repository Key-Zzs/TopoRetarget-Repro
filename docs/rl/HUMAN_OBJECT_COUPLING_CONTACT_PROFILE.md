# Human-Object Coupling Contact Profile

`HumanObjectCouplingContactProfileV1` is a reusable offline descriptor of how
a hand and object enter and maintain interaction. It is not a reward, a
success gate, or a human functional-grasp label.

## Contract

For hand/root pose `T_W_H(t)` and object pose `T_W_O(t)`, the profile stores

```text
T_H_O(t) = inverse(T_W_H(t)) T_W_O(t).
```

It derives all angular quantities from poses with the Reference Kinematics V2
centered SO(3)-log estimator. Relative linear and angular speed, windowed
relative-pose variation, hand/object motion magnitude, and the diagnostic
dimensionless ratios

```text
C_v = ||d p_H_O / dt|| / (||v_H|| + ||v_O|| + epsilon)
C_w = ||omega_H_O|| / (||omega_H|| + ||omega_O|| + epsilon)
```

remain continuous; no outcome-fitted coupling threshold is defined. These
ratios are scale diagnostics and are not constrained to `[0, 1]` because
rotating hand-frame lever-arm motion can exceed the sum of origin translation
speeds.

Raw-human rates use raw source seconds; retarget and actual rates use runtime
control seconds. Cross-layer comparisons retain both clocks and align by
reference phase/runtime mapping. Absolute rates from different clocks are not
silently treated as the same time scale.

Each source frame also preserves minimum hand/object surface distance,
near-contact count/fraction, connected components, MANO LBS-derived regions
and segments, contact spread, normal opposition, Strict-V4 reward target,
any-surface contact, multi-region contact, and geometric opposing topology as
separate fields.

```text
RAW_HUMAN_FUNCTIONAL_GRASP_BINARY_REQUIRED=NO
FORCE_CLOSURE_CLAIMED=NO
OUTCOME_TUNED=NO
```

## Authority boundaries

Raw HOCap uses provenance-resolved MANO/object geometry, the existing
raw-to-Stage16 transform/time map, `ManoSurfaceRegionMapV1`, and frozen
SourcePerFingerContactEvidenceV1 distances. The object meshes are not
watertight, so normal opposition is named only
`GEOMETRIC_CONTACT_TOPOLOGY`.

Retarget uses `ReferenceContactContractV2` named-tip distance geometry and
reference wrist/object poses. Existing PhysX traces provide wrist/object poses,
all-hand and named-tip contact presence, support contact, and lift. They do not
contain object-local contact points/normals or exact surface-relative slip;
therefore actual topology and exact slip remain `NOT_IDENTIFIABLE`.

## Frozen HOCap events

| Event | hocap_170105 | hocap_170650 |
| --- | ---: | ---: |
| any-surface | 182 | 109 |
| multi-region | 190 | 136 |
| opposing topology | 197 | 140 |
| Strict V4 reward target | 192 | 136 |
| LIFT | 184 | 184 |

The 170105 source interaction is gradual across LIFT, whereas 170650 has a
large pre-LIFT preparation margin. These are event descriptors, not
ground-truth grasp labels.

## Open-dataset path

HOCap, GRAB, OakInk/OakInk2, ContactPose, and other open HOI sources follow the
same object-agnostic path:

```text
raw hand/object
    -> HumanObjectCouplingContactProfileV1
    -> geometric retarget
    -> PhysX rollout
    -> PF + DF
    -> accept, or generic source-profile physical refinement
```

This path requires neither per-object reward/friction tuning nor a manually
labeled grasp frame.

The preregistered contract is
`configs/evaluation/stage16_human_object_coupling_profile_v1.yaml`; the reusable
implementation is
`src/toporetarget/evaluation/human_object_interaction_profile.py`. Generated
per-frame evidence remains ignored under
`.local/reports/stage16_170650_closure_and_human_object_profile/profile/`.
