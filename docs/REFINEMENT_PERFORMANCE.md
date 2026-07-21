# Stage 9.2 refinement performance and profiling

Stage 9.2 is an execution-path optimization layer around the frozen Stage 9
Eq. (8)-(9) contract. It does not change paper weights, signed-distance signs,
the 512 collision samples, the v2 solver profile, or strict status-9 rejection.

The default execution profile is
`configs/retarget/refinement_execution/cached_checkpoint_cpu_float64_v1.yaml`:
CPU float64, exact-x common-forward reuse, persistent per-run SDF resources,
batched collision-point Jacobians, and independent 512-point audits outside
inner SLSQP callbacks. The execution profile is deliberately separate from the
solver profile.

The profile also records `seed_delta_normalized_v1`. SciPy receives an explicit
diagonal normalized vector for conditioning, but every callback maps back to the
raw `[delta_p, delta_omega, q, slack]` vector before evaluating Eq. (8)-(9), and
all results/checkpoints remain raw metres, radians, and joint radians. This is an
invertible execution transform, not a hidden objective or coordinate change.

The reference SDF builds one exact triangle-AABB tree per run. AABB lower bounds
only prune triangles that cannot beat the current exact leaf distance; the leaf
closest-point formula and independent winding/sign audit are unchanged.

## Profiling

Profile only the fixed frames selected from the Stage 9.1 reports:

```bash
python -m toporetarget retarget profile-refinement \
  --canonical "$CANONICAL" --warm-start "$WARM_START" --graph "$GRAPH" \
  --robot artimano_rh --collision-samples "$ROBOT_SURFACE" \
  --frames 240 --frames 238 --frames 29 \
  --solver-profile scipy_slsqp_active_set_contact_rich_v2 \
  --execution-profile cached_checkpoint_cpu_float64_v1 \
  --output-root .local/reports/stage9_performance
```

The command writes callback counts, exact-x cache statistics, internal timers,
`cprofile/`, bounded `torch_profiler/` key-average smoke output,
`execution_path_audit.json`,
`benchmark_frames.json`, `bottleneck_summary.json`, and per-class
`profile_*.json`. `py-spy` is reported as unavailable when it is not installed;
the command never installs or attaches a system profiler automatically.

The timer report distinguishes frame wall time, SLSQP time, objective and
constraint callback counts, FK/keypoint/interaction/bone work, solver SDF,
collision Jacobian, full 512-point audit, and checkpoint/report I/O. Full audit
counts must scale with initialization plus active-set outer rounds and final
acceptance, not with callback count.

## Runtime gates

The preferred engineering gate is median frame solve time below 10 s, p95 below
20 s, and a strict 60-frame solve below 20 min. A reference-runtime result may
be reported only below 20 s median, below 40 s p95, below 40 min for 60 frames,
with reliable checkpoint/resume and every frame strict-accepted. The completed
run meets the reference-runtime minimum gate; its status is
`STAGE9_2_COMPLETE_REFERENCE_RUNTIME`. The preferred single-frame gate remains
unmet, so Stage 10 remains blocked.

These are local engineering gates, not paper claims. The paper's reported
4.70 ms/frame is retained only as a reference point.

The initial unscaled contact-rich profile reached status 9 at 164.32 s. The v3
execution profile adds the exact analytic URDF spatial Jacobian, strict reference
recovery for primary status 9, and a persistent reference-SDF AABB tree with leaf
size 512. Its first 60-frame run is strict accepted with median `10.766 s`, p95
`38.711 s`, and `1104.827 s` total solve time; the deterministic repeat is
median `10.773 s`, p95 `39.052 s`, and `1107.368 s`. Both runs pass checkpoint
chain validation and independent full-surface validation (`60 x 512`, maximum
signed-distance error `2.50e-16 m`). Full persisted arrays compare exactly after
excluding `solve_time_s` and documented metadata. The reference-runtime minimum
gate passes; the preferred single-frame median/p95 gate remains unmet, so Stage 10
is still blocked. Evidence is recorded under `.local/reports/stage9_performance/`.
