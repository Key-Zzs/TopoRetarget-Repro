# S1.2A E0 penetration-stress discovery v2

Discovery v2 selects stress cases from E0 only.  It first audits every
source-valid candidate, then runs a one-frame E0 probe for every eligible
60-frame window, followed by a Top-40 five-frame confirmation and a Top-10
full E0 pass.  Candidate-level failures are retained and do not stop the
global scan.

Coverage is a hard gate: at least 500 successful probes or 70% successful
coverage is required before claiming a corpus-level Top-3.  The three cases are
ranked only by complete 60-frame E0 reference-winding audits and then locked
before any S1 solve.  S1 can therefore neither select nor replace a case.

E0 and S1 use the same validated inner backend, warm starts, graph, QuerySet,
solver settings, temporal implementation, and reference audit.  The global
default remains E0 whatever the stress-case outcome.  The actual run protocol
and checkpointed evidence are implemented by
`.local/tools/run_t4_discovery_v2.py` and written below
`.local/experiments/pene_loss_temporal_sync_and_stress_v2/t4_discovery/`.
