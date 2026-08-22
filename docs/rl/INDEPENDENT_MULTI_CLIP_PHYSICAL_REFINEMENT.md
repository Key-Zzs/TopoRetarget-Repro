# Independent Multi-Clip Physical Refinement

`scripts/rl/isaaclab/run_physical_refinement_batch.py` is the batch-facing
entrypoint for a five-clip HOCap pilot.  Its role is deliberately narrow: it
freezes a held-out selection before outcomes, binds one shared method contract,
and owns independent lineage/status/timing receipts.  It delegates physical
work to declared production authorities; it never copies their implementation.

## Immutable selection and method contracts

`prepare` scans only `meta.yaml`, `poses_m.npy`, `poses_o.npy`, subject MANO
availability, object meshes, frame shapes/endpoints, and FPS.  It excludes the
development clips `hocap_170105` and `hocap_170650`, stratifies deterministically
by primary object and subject using seed `20260822`, and writes:

- `selection/candidate_pool.csv`
- `selection/excluded_development_clips.csv`
- `selection/held_out_5_manifest.{json,yaml}`
- `selection/selection_receipt.json`

The manifest includes raw hashes, ranks, duration, and `manifest_sha256`; its
validation rejects development leakage, an altered hash, or recorded outcome
selection. A candidate discovered corrupt before downstream work may only be
replaced through a new, explicit deterministic manifest; it is never swapped
after a physical result.

The single batch-wide method contract fixes full gravity, nominal friction,
grouped multiplicative reward, RSE, PF V2/DF evaluation, evaluate-first, and a
pre-registered upper bound of 15 PPO updates. Training RSI is declared as the
runtime reference's valid index domain, rather than a fixed `[0,320]` literal.

## Independent lineages and durable state

Every clip has distinct actor, critic, optimizer, normalizer, and deterministic
RNG roots. The guard rejects any shared lineage identifier. A stage receipt has
input/output hashes, UTC endpoints, monotonic wall time, productive and
technical-retry time, cache hit, retries, and exit code. JSON receipts are
published through temp-file-plus-rename, so a completed stage can be resumed
without being run again.

The intended transition is:

```text
SELECTED -> RAW_VALIDATED -> RETARGETED -> SOURCE_POLICY_READY -> SUPPORT_READY
         -> FROZEN_EVAL_DONE -> ACCEPTED_FROZEN
                              -> PPO_CANDIDATE -> ACCEPTED_AFTER_REFINEMENT
                                               -> PPO_BUDGET_EXHAUSTED
```

An accepted frozen policy has zero PPO updates. A candidate Eval10 pass must be
confirmed on Confirm20, and a Confirm20 pass stops that clip immediately.

## Authority preflight

The batch requires explicit support for each held-out clip from these
authorities: retarget, source policy, support, frozen evaluation, physical
refinement, qualification, and trace export. This is material: current
production commands restrict their clip argument to `hocap_170105` and
`hocap_170650`, and their source checkpoints are development lineage assets.

Run:

```bash
PYTHONPATH=src conda run -n toporetarget-rl \
  python scripts/rl/isaaclab/run_physical_refinement_batch.py prepare
PYTHONPATH=src conda run -n toporetarget-rl \
  python scripts/rl/isaaclab/run_physical_refinement_batch.py validate-config
```

If an authority is absent, `execute` writes one `PIPELINE_INVALID` receipt per
clip, records the exact unsupported stages, and records `PPO_UPDATES=0`. That
result is a capability block, not an experimental physical failure; it must not
be summarized as frozen or final success rate.

When every authority explicitly supports the frozen manifest, the same entry
point may be supplied a declared authority manifest. The execution engine must
then invoke only those commands and publish the full raw-to-final timing,
PF/DF tables, failure taxonomy, aggregate bottleneck analysis, and replay
commands under `.local/reports/independent_multiclip_hocap_pilot/`.
