# Stage 16-C.5B PhysX oracle gate

C.3/C.4 are validated, but the Oracle evaluator is **not authorized**. Its
immediate prerequisite, Stage 16-C.5A state replication, is
`STAGE16C5A_BLOCKED_PHYSX_REPLICATION_BASELINE_NONDETERMINISM`: the natural
no-clone 20x8 baseline violates frozen hard caps before O1 snapshot/restore
qualification. C5A's CUDA candidate-state contract and O0 allocation/isolation
at 1/32/96/144 candidates are valid, but do not prove state replication,
candidate rollout independence, controllability, or evaluator throughput.

No C5B Oracle episode, CEM, formal 20-episode evaluation, policy training, PPO
sample, or checkpoint has run. The future evaluator must be a separately
authorized PhysX experiment, not a conversion of MuJoCo Oracle evidence. It
must retain the frozen factor-8 references, action bounds, free-object dynamics,
no-ground scene, formal termination, and no direct object pose writes.

Unblock order is strict: reproduce a passing natural no-clone baseline under
the same frozen hard caps; qualify O1 tensor clone at every required phase;
use deterministic history replay only if a tensor-clone contact mismatch
remains; then separately authorize C5B. This file does not authorize any of
those operations.
