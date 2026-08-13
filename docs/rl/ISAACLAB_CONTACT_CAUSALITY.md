# Isaac Lab contact-readout and causality gates

Stage 16-C.3R2 replaces the unsafe design that constructed and read 21
hand-side filtered views in the task process. The installed Isaac Lab 2.3.2
`ContactSensorCfg` exposes `force_matrix_w` as
`[environment, sensor body, filter shape, xyz]`, and its own source documents
that filtered contacts require a single sensor primitive per environment.

The runtime therefore has exactly two object-side views:

- `Object170105`: one object body filtered to all 21 C.1 collision-bearing
  hand bodies;
- `Object170650`: the same one-body/21-filter contract.

This is not 21 Python sensor reads. `aggregate` retains object-net force,
impulse, and pair presence. `diagnostic` additionally retains the raw filtered
body-pair force matrix. Neither mode affects reward or control, and neither
fabricates contact points, normals, tangential forces, or per-point force.
Records use bounded latest-only transport (4096 samples), so high-environment
diagnostics cannot accumulate unbounded Python telemetry.

The contact-enabled profile uses USD cloning with GPU physics replication;
Fabric cloning is disabled because the real Isaac Sim 5.1 contact view cannot
resolve replicated bodies at 128 environments. This is an engineering runtime
choice, not a physics, hardware, or sim-to-real claim.

## C.3R2 readout result

`scripts/rl/isaaclab/probe_stage16c3_contact_api.py` runs every probe in a
separate child process and writes flushed stage events, stdout, stderr, exit
code, and tensor shapes. The aggregate summary is
`.local/reports/stage16c3r2_c5/contact/c3_contact_readout_summary.json`.

The real RTX 5080 / CUDA PhysX result is
`C3_CONTACT_READOUT_VALIDATED`:

- a fenced raw-PhysX 1-env no-contact fixture read zero force matrices for
  1000 physics steps;
- 1-env single-finger preload fixtures yielded finite nonzero contact for both
  HO-Cap objects and included the requested distal filter slot;
- 1-env random action ran 1002 physics steps cleanly;
- 128-env aggregate random action ran 1002 physics steps cleanly with finite
  `[128, 1, 21, 3]` force matrices.

The preload state writes are explicitly limited to C.1 fixture setup before
the probe rollout; normal DirectRLEnv rollout still writes neither wrist root
nor object state. This validates readout capability only. Under the original
timing, C3R3/R4 computed-torque and bounded-preview paths failed the frozen
wrist gate, so task-level causality was correctly not run in that closeout and
must not be inferred from preload fixtures.

## C3R5 task-level causality result

After the user separately authorized one shared factor-8 reference retiming,
`qualify_stage16c3_retimed_contact_causality.py` ran a collision-disabled
baseline and both complete 320-interval task references in isolated processes.
The baseline produced exactly zero net contact force and zero object delta-v.
Each clip then produced at least one finite nonzero object-side force/impulse
sample coincident with a subsequent finite delta-v or delta-omega. Both wrist
tracking gates pass, and the runtime contracts record zero wrist-root writes,
zero formal object rollout writes, and no hidden object force.

The current result is `C3_CONTACT_CAUSALITY_VALIDATED`. Because only aggregate
object force is available, angular causality remains explicitly approximate;
the report does not fabricate a contact point or point force. Evidence is
`.local/reports/stage16c3r5_reference_retiming_c4/contact_causality_scale8.json`.
Together with C3-0 through C3-4, this supports the current stage status
`STAGE16C3_SEMANTIC_QUALIFICATION_VALIDATED`.
