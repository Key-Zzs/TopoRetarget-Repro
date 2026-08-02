# Stage 16-C.0 Isaac Lab environment setup

Stage 16-C.0 qualifies an isolated GPU simulation platform. Stage 16-C.1 uses
that platform for asset migration only. Neither stage implements a Stage-16
`DirectRLEnv`, runs a PhysX oracle, or starts PPO.

## Frozen stack

The stack selected on 2026-08-02 is the latest stable, non-beta Isaac Lab
release:

| Component | Frozen value |
|---|---|
| Installation | Isaac Sim `all` pip bundle + editable Isaac Lab source packages |
| Python | 3.11.15 (Isaac Sim ABI: 3.11) |
| NumPy | 1.26.0 |
| PyYAML | 6.0.2 |
| Isaac Sim | 5.1.0 |
| Isaac Lab | `v2.3.2` / `37ddf626871758333d6ed89cf64ad702aef127d0` |
| PyTorch | 2.7.0, CUDA 12.8 wheel |
| Torchvision | 0.22.0, CUDA 12.8 wheel |

Isaac Lab 3.0 was still a beta and its release notes warned about breaking
changes. Isaac Lab v2.3.2 is the latest stable release and supports Isaac Sim
5.1. NVIDIA now labels Isaac Sim 5.1 unsupported, so this exact pair is frozen
for reproduction and receives no claim of continuing vendor support. The
checked official sources are frozen in
`configs/rl/stage16/isaaclab_platform.yaml`.

## Host gate

Isaac Sim 5.1 documents Ubuntu 22.04/24.04, at least 32 GiB RAM, an RTX 4080
or newer with 16 GB VRAM, and Linux driver 580.65.06. Pip installation also
requires glibc 2.35 or newer. The qualification verifier records the actual
host and never upgrades the driver, kernel, glibc, or system CUDA. Its report
retains raw output or explicit command-not-found evidence for date, uname,
OS release, ldd, both NVIDIA-SMI forms, nvcc, Python, lscpu, free, df, lsblk,
Vulkan, GLX, and display/session variables.

## Bootstrap

Inspect all fixed commands without changing the machine:

```bash
bash scripts/bootstrap_stage16_isaaclab_env.sh --dry-run
```

Create the isolated environment and official source checkout:

```bash
bash scripts/bootstrap_stage16_isaaclab_env.sh \
  --env-name toporetarget-isaaclab \
  --external-root .local/external/IsaacLab
```

The bootstrap is fail-closed. It does not use `sudo`, delete an existing Conda
environment, overwrite a non-empty external directory, modify the official
checkout, or accept an NVIDIA EULA. A pre-existing checkout must have the
expected official remote, exact commit, and a clean worktree.

The bootstrap installs the six Python packages from the frozen official source
checkout in editable mode. It intentionally does not call
`isaaclab.sh --install`: that helper performs an unrelated VS Code setup step
which imports `isaacsim` and can trigger the first-run EULA prompt. Skipping
that editor-only step keeps environment installation non-interactive without
changing the checked-out source.

The optional multi-gigabyte `extscache` bundle is not preinstalled. C.0 uses
the official `all` bundle and lets Kit resolve only extensions required by the
bounded smoke runs. Runtime extension resolution remains a hard network/cache
gate and is recorded as `NETWORK_OR_ASSET_CACHE_FAILURE` if it cannot finish.
The bootstrap installs the official Torch cu128 wheels before Isaac Sim so
Isaac Sim resolves those exact CUDA builds instead of downloading a default
wheel and replacing it later. It also pins `setuptools==80.9.0`: the frozen
Isaac Lab release depends on `flatdict==4.0.1`, whose legacy build imports
`pkg_resources` (removed from setuptools 81 and newer). The bootstrap builds
that fixed package without isolation after pinning setuptools, before Isaac
Lab dependency resolution. The notebook-only
dependency chain is constrained to `ipython==8.37.0`, `onnx==1.21.0`,
`psutil==5.9.8`, and `typing_extensions==4.12.2` so it does not override Isaac
Sim kernel pins.
The official packages retain one upstream metadata conflict: Isaac Lab pins
`starlette==0.49.1`, while Isaac Sim pins FastAPI 0.115.7, which declares
`starlette<0.46.0`. The verifier records `pip check` verbatim rather than
claiming complete dependency consistency.

## Platform verification

Static manifest, host, source, and command validation:

```bash
conda run -n toporetarget-isaaclab \
  python scripts/verify_stage16_isaaclab_platform.py --phase static
```

Finite headless empty-scene and GPU PhysX smoke:

```bash
conda run -n toporetarget-isaaclab \
  python scripts/verify_stage16_isaaclab_platform.py \
  --phase empty-scene --steps 1000
```

Finite local-only primitive spawn adapted from the frozen official tutorial:

```bash
conda run -n toporetarget-isaaclab \
  python scripts/verify_stage16_isaaclab_platform.py \
  --phase primitive --steps 1000
```

The result records the official `spawn_prims.py` path and SHA256. The bounded
adapter omits its remote table asset but preserves ground, light, and rigid
primitive creation.

Official task smoke and 128-environment vector qualification:

```bash
conda run -n toporetarget-isaaclab \
  python scripts/verify_stage16_isaaclab_platform.py \
  --phase full --steps 1000
```

After explicit user authorization, add `--accept-eula`. The completed run
passed all hard gates and records
`STAGE16C0_ISAACLAB_PLATFORM_VALIDATED_WITH_LIMITATIONS`:

```bash
conda run -n toporetarget-isaaclab \
  python scripts/verify_stage16_isaaclab_platform.py \
  --phase full --steps 1000 --accept-eula
```

Optional 512-environment platform smoke:

```bash
conda run -n toporetarget-isaaclab \
  python scripts/verify_stage16_isaaclab_platform.py \
  --phase vector --num-envs 512 --steps 1000
```

Interactive viewer smoke when a real display is available:

```bash
conda run -n toporetarget-isaaclab \
  python scripts/verify_stage16_isaaclab_platform.py \
  --phase viewer --steps 1000 --viewer
```

Reports are written only below
`.local/reports/stage16c_isaaclab_platform/` and are not committed.

## EULA boundary

The first Isaac Sim startup may display the NVIDIA Omniverse License Agreement.
The reusable bootstrap never answers this prompt or sets an acceptance
environment variable. The committed platform config records the user's
explicit authorization for this task. `--accept-eula` sets the official
`OMNI_KIT_ACCEPT_EULA=YES` value only for that verifier/import process; it does
not grant privacy or telemetry consent. Without an authorization record, the
verifier still fails closed as `ISAACLAB_EULA_ACCEPTANCE_REQUIRED`.

## Completion boundary

C.0 requires a real Isaac Sim import, finite empty-scene run, Isaac Lab import,
official task reset/step, CUDA tensors, GPU PhysX without CPU fallback, headless
execution, and a truly parallel 128-environment smoke. Viewer failure is a soft
limitation; every other listed runtime gate is hard. Only a validated C.0 may
authorize Stage 16-C.1 asset migration.

Stage 16-C.1 commands, provenance, collision fallback, and runtime gates are
documented in [ISAACLAB_ASSET_MIGRATION.md](ISAACLAB_ASSET_MIGRATION.md).
