# Stage Q1–Q3: Multi-Dataset Benchmark, Unified Evaluation, Frozen Baseline

## Scope

This stage answers whether the current MANO → Arti-MANO pipeline transfers across contact-rich
objects and subjects, whether the Stage 7 morphology gap recurs, whether Stage 9 changes interaction,
and whether the two Eq. (9) temporal scopes differ on dynamic trajectories. It does not change the
method or claim the paper's full ContactPose experiment, RL, physics, or real-time performance.

## Commands

```bash
export PYTHONNOUSERSITE=1 PYTHONPATH=src
python -m toporetarget benchmark inspect-datasets --grab-root "$GRAB_ROOT" \
  --contactpose-root "$CONTACTPOSE_ROOT" --output .local/benchmarks/hoi_benchmark_v1/dataset_audit.json
python -m toporetarget benchmark select --config configs/benchmarks/hoi_benchmark_v1.yaml
python -m toporetarget benchmark freeze
python -m toporetarget benchmark run --profiles warm,scipy_slsqp_active_set_contact_rich_v2,scipy_slsqp_active_set_contact_rich_v3_fixed --resume
python -m toporetarget benchmark evaluate --metric-registry configs/metrics/hoi_metrics_v1.yaml --html
```

## Completion status

The automated status is one of `Q1_Q2_Q3_BENCHMARK_COMPLETE`,
`Q1_SELECTION_BLOCKED`, `Q1_CONTACTPOSE_SELECTION_BLOCKED`, `Q2_METRICS_BLOCKED`, `Q3_BASELINE_EXECUTION_BLOCKED`, or
`Q1_Q2_Q3_COMPLETE_WITH_RECORDED_BASELINE_FAILURES`. The last status is allowed only for real
solver/data/runtime failures and never for an undersized selection.

The current local run is `Q1_CONTACTPOSE_SELECTION_BLOCKED`: the GRAB probe selected the fixed
clip plus three additional valid clips from 16 evaluated entries, while all 110 ContactPose
candidate annotation records lacked recognized official contact attribution. No selection
manifest or baseline was created after this gate.

Q4 morphology-aware warm-start, Q5 surface contact proxies, Q6 contact-aware extension, and Q7
cross-trajectory automatic profile selection remain explicitly unstarted.
