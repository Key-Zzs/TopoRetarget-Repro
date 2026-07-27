# GRAB Arti-MANO Quality A–E Experiment

This experiment is a frozen, four-clip development benchmark for the
right-handed `artimano_rh` target. The clips are `s1/airplane_lift [240,300)`,
`s1/apple_eat_1 [212,272)`, `s1/banana_lift [1658,1718)`, and
`s1/alarmclock_lift [407,467)`, all at native 120 FPS. All four units belong to
subject `s1`; the only supported claim is **within-subject multi-object
development benchmark**, not cross-subject generalization.

The entry point is:

```bash
PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/deepcybo/miniconda3/envs/topo-retarget/bin/python -m toporetarget \
  quality run-a-to-e \
  --config configs/experiments/grab_artimano_quality_v1.yaml \
  --resume --max-wall-time 1800 --generate-html
```

Outputs are restricted to `.local/experiments/grab_artimano_quality_v1/`.
`selection/selection.lock` forbids sequence replacement, frame changes, result-
based reselection, and skipped frames. Every record carries source, MANO,
personalized-vtemp, object, Arti-MANO, solver, metric, frame, and timestamp
lineage. ContactPose is intentionally deferred (`contactpose_status=deferred`);
its adapter is required for a future extension but is not a blocker here.

## A–E boundaries

Stage A runs the existing paper-core warm start and both retained Eq. (9)
interpretations: `scipy_slsqp_active_set_contact_rich_v2` and
`scipy_slsqp_active_set_contact_rich_v3_fixed`. `author_exact` remains
`unresolved`; neither interpretation is renamed or discarded. Strict acceptance,
full-512 audits, collision query sets, solver tolerances, and paper weights stay
unchanged.

Stage B creates visual surface contact proxies through deterministic, versioned
Arti-MANO link-region mappings. Skeleton anchors, visual contact proxies, and
collision samples are separate representations. Visual normals are semantic
directions, not claims of absolute outward orientation.

Stage C keeps `paper_warm` and tests `morphology_seed_only_v1` candidates. The
seed-only branch changes only initialization and still selects by the official
Eq. (2) objective. `morphology_position_prior_v1` is separately labelled
paper-external, dimensionless, and diagnostic.

Stage D declares exactly P1/P2/P3/PD1/PD2. GRAB contact scores are
`metric_semantics=DATASET_PROXY`, never ContactPose Eq. (10)/(11). If the
contact objective is not accepted by all hard and regression gates, C* remains a
diagnostic-only record and the paper-core baseline remains eligible.

Stage E evaluates E0–E3 with hard gates, regression gates, and a deterministic
Pareto rule. There is no hidden scalar score. Recommending E0 is a valid result.

## Artifact and integrity policy

Raw GRAB, MANO, personalized vtemp, object meshes, Arti-MANO assets, and old
Stage 5–10 artifacts are read-only. No `.local` artifact is tracked. The command
does not stage, commit, push, tag, release, reset, or clean the checkout.

The generated dashboard and four HTML files are diagnostic views. They do not
establish physical contact truth, cross-subject generalization, RL, physics,
sim-to-real, online control, production runtime, or paper-runtime equivalence.

## Current local execution status

The frozen selection and Arti-MANO surface artifact were built successfully.
Surface validation includes deterministic sample reconstruction, FK round-trip,
and analytic-versus-finite-difference Jacobian checks. G1 was reused only from
matching historical artifacts. G2 completed both
`scipy_slsqp_active_set_contact_rich_v2` and
`scipy_slsqp_active_set_contact_rich_v3_fixed` at 60/60 strict-accepted frames.
The initial run was blocked at G3 before final solving because the source
banana object mesh is open (`boundary_edge_count=66`, `watertight=false`,
`sign_reliability=open_surface`) and the strict raw signed-distance backend
rejected it. The resumed run and its stricter routing are recorded below; see
`reports/failure_report.json` for the current formal status.

## 2026-07-24 derived sign-proxy resolution and strict routing

The open banana mesh was audited without changing its bytes, hash, or mtime.
The selected geometry profile is
`hybrid_original_distance_proxy_sign_v1`; G1, G2, and G4 select identity, while
G3 selects `candidate_1_local_repair`. The G3 proxy is used only for sign; all
original closest-point, unsigned-distance, object-sample, visualization, and
contact-position semantics remain source-mesh semantics.

The G3 proxy passed watertightness, winding, boundary, non-manifold,
positive-volume, 20,000-point bidirectional surface deviation, bbox, and patch
area gates. Source semantic contact tips have zero boundary-exclusion and
synthetic-patch conflicts. The formal Stage 9 retry nevertheless fails closed
with `SIGN_PROXY_CONTACT_REGION_CONFLICT`: the first active QuerySet contains
three samples in the original boundary exclusion zone, two of which are nearest
synthetic patch faces. No margin, frame, trajectory, profile, or raw mesh was
changed to bypass this gate. Therefore this local run is formally
`GRAB_QUALITY_A_TO_E_BLOCKED`, not A–E complete, and must not be reported as
unblocked or as a recommended trajectory.

The geometry audit is at
`.local/experiments/grab_artimano_quality_v1/geometry/banana_original_vs_proxy.html`
with first/middle/last PNG snapshots in the same directory. The formal status
is `GRAB_QUALITY_A_TO_E_BLOCKED` and `hard_blocker` is
`SIGN_PROXY_CONTACT_REGION_CONFLICT`; G1/G2/G4 geometry identity evidence and
valid historical downstream artifacts remain reusable, while G3 final and its
downstream C–E artifacts remain invalidated.
