# Stage 16-C.0 Isaac Lab environment setup

Stage 16-C.0 qualifies an isolated GPU simulation platform. It does **not**
migrate Wuji Hand2 Beta1 or HO-Cap objects, implement a Stage-16 `DirectRLEnv`,
run a PhysX oracle, or start PPO.

## Frozen stack

The stack selected on 2026-08-01 is the latest stable, non-beta Isaac Lab
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
5.1. The checked official sources are frozen in
`configs/rl/stage16/isaaclab_platform.yaml`.

## Host gate

Isaac Sim 5.1 documents Ubuntu 22.04/24.04, at least 32 GiB RAM, an RTX 4080
or newer with 16 GB VRAM, and Linux driver 580.65.06. Pip installation also
requires glibc 2.35 or newer. The qualification verifier records the actual
host and never upgrades the driver, kernel, glibc, or system CUDA.

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
wheel and replacing it later.

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

Official task smoke and 128-environment vector qualification:

```bash
conda run -n toporetarget-isaaclab \
  python scripts/verify_stage16_isaaclab_platform.py \
  --phase full --steps 1000
```

With the committed configuration this command stops before importing Isaac Sim
and records `ISAACLAB_EULA_ACCEPTANCE_REQUIRED`. It is a deliberate fail-closed
qualification command, not an instruction to accept the agreement.

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
environment variable. The committed platform config records no user
authorization. Runtime qualification therefore stops before first import and
reports `ISAACLAB_EULA_ACCEPTANCE_REQUIRED`. Only after a future explicit user
authorization is recorded in the platform config may `--accept-eula` set the
official `OMNI_KIT_ACCEPT_EULA=YES` value for that verifier process. An
unaccepted agreement is reported as
`ISAACLAB_EULA_ACCEPTANCE_REQUIRED`; it is never represented as a runtime pass.

## Completion boundary

C.0 requires a real Isaac Sim import, finite empty-scene run, Isaac Lab import,
official task reset/step, CUDA tensors, GPU PhysX without CPU fallback, headless
execution, and a truly parallel 128-environment smoke. Viewer failure is a soft
limitation; every other listed runtime gate is hard. Only a validated C.0 may
authorize Stage 16-C.1 asset migration.
