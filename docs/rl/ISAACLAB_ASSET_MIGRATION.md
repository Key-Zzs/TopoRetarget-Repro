# Stage 16-C.1 Isaac Lab asset migration

Stage 16-C.1 is an engineering asset qualification. It is downstream of the
validated C.0 platform, but upstream of any custom `DirectRLEnv`, reward,
termination, PhysX oracle, or PPO. Generated USDs and reports are ignored under
`.local/`; the repository tracks only source contracts, recipes, tests, and
documentation.

## EULA scope

The user explicitly authorized `OMNI_KIT_ACCEPT_EULA=YES` for Isaac Sim
processes in this task. The import and smoke CLIs additionally require
`--accept-eula`; they set the variable only inside that process. The reusable
bootstrap never accepts the license. No privacy or telemetry consent is
included.

## Wuji Hand2 Beta1

- Kinematic source: frozen vendored right-hand URDF at Wuji release
  `release/v2026.7.23`, commit
  `2b57d2621caed4e65207bb767ba25fc8eaec0881`.
- Runtime source: the exact official upstream USD bundle supplied through
  `--upstream-root`; the tracked recipe contains no machine-specific default.
- Contract: floating `r_wrist`, 26 bodies, 20 revolute joints, source axes and
  limits, explicit semantic mapping, and 16 tracked links.
- Collision: the upstream high-poly collision payload exceeded bounded GPU
  cooking time. The accepted deterministic fallback preserves visual geometry
  and creates support-direction convex proxies for 21 collision bodies, with
  at most 61 vertices per body. Self collision is disabled for this smoke.

## HO-Cap objects

The frozen objects are exactly `hocap_170105` and `hocap_170650`. Their source
OBJ hashes, unit scale, identity origin, mass, COM, principal inertia, friction,
zero gravity, no ground, and no support are fail-closed config fields. Original
OBJ meshes remain the visual geometry. Both use the same `convex_hull_v1`
strategy with one deterministic proxy; the manifests record vertex/triangle
counts and maximum support-gap deviation over 256 directions.

Mass/inertia are engineering nominal cross-backend values. Physical provenance
is unresolved, so these assets support simulator-functionality testing only,
not calibrated real dynamics or sim-to-real claims.

## Commands

```bash
conda run -n toporetarget-isaaclab \
  python scripts/rl/isaaclab/validate_stage16c1_assets.py

conda run -n toporetarget-isaaclab \
  python scripts/rl/isaaclab/import_wuji_hand2.py \
  --upstream-root /home/deepcybo/workspace/dex/wuji-description \
  --accept-eula

conda run -n toporetarget-isaaclab \
  python scripts/rl/isaaclab/import_hocap_objects.py --accept-eula

conda run -n toporetarget-isaaclab \
  python scripts/rl/isaaclab/smoke_stage16c1_assets.py \
  --object hocap_170105 --num-envs 128 --steps 1000 --accept-eula
```

Repeat the last command for `hocap_170650`; use `--num-envs 1` for individual
joint response and `--contact --steps 100` for the bounded contact smoke.

## Acceptance

The real RTX 5080/CUDA PhysX runs resolve 20/20 actuated joints and 16/16
tracked links, match source joint limits within `1.2e-7 rad`, keep both free
objects stationary for 1000 zero-gravity steps, and produce finite contact
motion without object pose control. Both 128-env runs use unique environment
origins, CUDA tensors with `[128,20]` joint and `[128,13]` object state shapes,
and a declared even-environment reset with `0.0 m` final reset error.

The host has no active display. Headless numerical validation is complete, but
interactive viewer/screenshot review is recorded as a soft visual limitation.
This does not weaken any asset, dynamics, contact, CUDA, or vectorization hard
gate, and it does not authorize Stage 16-C.2 or later work.
