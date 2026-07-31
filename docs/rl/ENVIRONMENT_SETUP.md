# Stage 16 environment setup

The reproducible environment is `toporetarget-rl`, defined by
[`environment.stage16.yml`](../../environment.stage16.yml) and the direct pins in
[`requirements-stage16.txt`](../../requirements-stage16.txt). No `sudo` or system-wide
installation is required.

```bash
conda env create -f environment.stage16.yml
# or, from a checkout where the environment may already exist:
bash scripts/bootstrap_stage16_env.sh
conda run -n toporetarget-rl python -c \
  'import mujoco, numpy, scipy, torch; print(mujoco.__version__, numpy.__version__, scipy.__version__, torch.__version__)'
```

The validated local inventory on 2026-07-31 was Python 3.12.13, MuJoCo 3.3.6, NumPy 2.5.1,
SciPy 1.18.0, Torch 2.13.0+cu130, Matplotlib 3.11.1, Pillow 12.3.0, Zarr 2.18.7,
Numcodecs 0.15.1, pytest 9.1.1, Ruff 0.16.1, and mypy 2.3.0. The torch wheel exposes CUDA
metadata, but the host had no usable NVIDIA driver; Stage 16 therefore uses MuJoCo CPU
correctness mode. This is an environment fact, not a paper-simulator claim.

Headless inspection uses MuJoCo offscreen rendering when an EGL/OSMesa context is available.
If both fail, `visualize_hocap_policy_mujoco.py` writes a numerical fallback PNG and HTML
dashboard and records `PASS_WITH_LIMITATION`; it does not fabricate geometry screenshots.
Interactive mode requires a working GLFW/X11 display and uses `mujoco.viewer`.

All external dataset/model paths are explicit. The two current references and OBJ meshes are
under ignored `.local/stage16_reference_tracking_ppo/`; raw NAS data is never copied into the
repository. Generated scenes, runs, checkpoints, PNG/MP4, and logs remain ignored under
`.local/`.
