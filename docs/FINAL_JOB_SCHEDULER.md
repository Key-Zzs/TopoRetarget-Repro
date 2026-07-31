# Final Job Scheduler

Stage-12 final-refinement work is fail-closed. `toporetarget jobs pause-final`
creates `.local/control/final_jobs/PAUSED`; a launcher checks it before it consumes
a final-job queue item and records `PAUSED_BY_OPERATOR_CONTROL` rather than a
solver failure. `status-final` is read-only and `drain-final` never kills a
legacy worker implicitly. `resume-final` intentionally refuses implicit resume.

The versioned default is `configs/runtime/final_refinement_cpu_v1.yaml`:
one worker, one BLAS thread, one Torch intra/inter-op thread, and no affinity.
The worker cap is based on physical CPU cores, never logical threads. A future
two-worker recommendation requires a controlled A/B/C benchmark and must keep
single-frame latency within 25% of the one-worker baseline.

Every new job appends JSONL heartbeats beneath `.local/runtime/final_jobs/`.
The heartbeat is operational evidence only and never replaces an accepted-frame
atomic checkpoint.
