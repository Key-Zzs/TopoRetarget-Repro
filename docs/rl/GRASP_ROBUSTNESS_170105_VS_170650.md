# V4 170105 vs 170650 Grasp Robustness Diagnostic

This diagnostic compares frozen V4 C4 evidence only: the 20-replica
`hocap_170650` Formal20 record and the 10-replica `hocap_170105` C4 record. It
does not train, roll out a new policy, alter an asset, or select a per-object
physics/reward setting.

## Three evidence layers

Each episode is read through the existing Stage16 raw-mocap overlay authority:

```text
source raw MANO/object contact expectation
    -> retarget robot distal-body topology in object coordinates
    -> recorded PhysX actual contact, force-vector, and twist telemetry
```

The source MANO layer is the authority for expected contact topology. The
retarget layer is assessed with robot distal-body positions expressed in the
object frame. Recorded actual contact and force telemetry establish what PhysX
observed; they do not supply missing contact normals or points.

## Dynamic and contact measures

The comparison reports per phase (`APPROACH`, `CONTACT`, `GRASP`, `LIFT`):
contact count, persistent per-finger contact, semantic grasp-and-lift,
table/object contact, source-versus-retarget topology error, and relative
hand/object twist. Persistent multi-finger contact must begin on or before the
reference LIFT phase to support a timing-success claim.

The frozen traces contain world-frame pair-force vectors but not contact points,
contact normals, nor PhysX's effective material-combine mode. Consequently,
exact normal/tangential force decomposition, tangential slip,
friction-cone utilization, grasp-wrench margin, and effective hand-object or
table-object friction are `NOT_IDENTIFIABLE`. A labeled
`TIP_OBJECT_RELATIVE_SPEED_PROXY` is diagnostic only and cannot justify a
friction-primary conclusion.

## Decision rule

The diagnostic emits exactly one primary root cause, confidence, and next
action. `FRICTION_MARGIN_PRIMARY` requires exact normal/tangential force and
effective friction evidence; `NORMAL_FORCE_CLOSURE_PRIMARY` additionally
requires contact geometry and inertia. If those conditions are absent, the
result stays `NOT_IDENTIFIABLE` or `INCONCLUSIVE` rather than inventing a
friction explanation.

For the frozen V4 comparison, the retained result is
`CONTACT_TIMING_PRIMARY`: 170105's persistent multi-finger contact begins after
the reference LIFT onset, unlike 170650. The sole follow-up named by the
receipt is `NEXT_CONTACT_TIMING_PHYSICAL_REFINEMENT`; this document does not
authorize that follow-up or any new PPO work.

The detailed, ignored receipt is written under:

```text
.local/reports/stage16_dynamic_physical_qualification_and_grasp_diagnostic/
```
