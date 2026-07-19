# Table 6 — PPO training and network hyperparameters

Transcribed row by row from Appendix A.5.6, PDF p. 16.

| Parameter | Value |
| --- | --- |
| Parallel environments | 4096 |
| Simulation step | 0.01 s |
| Decimation | 5 |
| RL control frequency | 20 Hz |
| Reference trajectory frequency | 20 Hz |
| Episode length | 20 s (400 control steps) |
| Rollout length | 40 control steps/environment |
| Samples per PPO iteration | 163,840 |
| Actor network | MLP [512, 256, 128], ELU |
| Critic network | MLP [512, 512, 256, 128], ELU |
| Observation normalization | Enabled |
| Action distribution | Softplus Gaussian |
| Optimizer | Adam |
| Learning rate | $1\times10^{-4}$ |
| PPO epochs/minibatches | 4 / 32 |
| Entropy coefficient | 0.001 |
| Discount $\gamma$ | 0.99 |
| GAE $\lambda$ | 0.95 |

Clip ratio, value coefficient, and maximum gradient norm are not provided.

