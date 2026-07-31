# Simulator backend decision

Audit order is installed GPU vectorized backend, Wuji-compatible backend, MJX, then MuJoCo CPU correctness mode. The local audit found an RTX 5080 and CUDA PyTorch but no installed Isaac Gym/Lab, MJX/JAX, Warp, RL-Games, RSL-RL, SKRL, or Wuji simulator package. The isolated `toporetarget-rl` environment provides MuJoCo 3.3.6 and PyTorch 2.13.0+cu130.

Therefore the selected backend is `mujoco_cpu_reference`. It verifies free-object dynamics, contact, fixed base, residual command wiring, and deterministic reset. It cannot meet the paper's 4096 parallel-environment throughput and is not author-exact. Unsupported Table-5 mutations are surfaced in capability reports, never silently omitted.
