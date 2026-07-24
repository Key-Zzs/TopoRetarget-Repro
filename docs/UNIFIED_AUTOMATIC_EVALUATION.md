# Unified Automatic Evaluation

The registry is `toporetarget.metric_registry.v1`. Every metric has an ID, definition, unit,
direction, applicability, semantics class, required inputs, missing-data behavior, and aggregation
rule. The implementation is in `src/toporetarget/metrics/`.

`contact_precision_eq10`, `contact_alignment_eq11`, `max_penetration_eq12`, and
`penetration_rate_2mm_eq12` follow Appendix A.3 of the local paper copy. The ContactPose formulas
require official native contact attribution and object-relative source/robot positions/directions.
Zero-length alignment vectors and empty contact sets fail closed. Angles are degrees in reports;
positions and penetration are millimetres. The 2 mm rate is a frame fraction for dynamic units and
a per-unit static result for ContactPose.

GRAB IDs are deliberately `*_proxy`: GRAB semantic object-vertex labels do not provide the
ContactPose source in-contact bone-segment attribution. They may diagnose retention and region drift,
but they cannot be called paper Eq. (10)/(11) or ground truth.

Automatic gates are data, solver, metric-completeness, and provenance gates. Missing values are
`N/A` with a reason, never zero. The dashboard is generated only from the manifest-bound result
table and includes failed rows rather than cherry-picking successful units.
