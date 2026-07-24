# 统一自动评价

Q1–Q3 的自动评价通过 metric registry 将指标分成四类：

- `PAPER_EXACT`：ContactPose Appendix A.3 Eq. (10)–(12)，只有官方 attribution 与所需输入齐全时才计算。
- `DATASET_PROXY`：GRAB 的 source/robot contact proxy，只能作为 proxy，不得写成 paper-exact。
- `GENERIC_GEOMETRIC`：适用于两个 dataset 的几何、trajectory 和约束摘要。
- `ENGINEERING_DIAGNOSTIC`：solver status、runtime、strict acceptance 等工程诊断。

Eq. (10) 以 mm 汇报 contact precision；Eq. (11) 以 degree 汇报 alignment，并拒绝零长度
bone segment；Eq. (12) 使用 signed distance，报告最大 penetration、超过 2 mm 的 frame
fraction 和最小 signed distance。静态 ContactPose 不伪造 temporal frames，所有 temporal
metric 返回 `NOT_APPLICABLE`。

评价只读取 manifest-bound run artifact。缺失输入返回 `N/A` 和原因，不填零；profile 和
dataset 分别做等权 macro mean/median。当前本地 run 在 ContactPose selection gate 前阻塞，
因此 Eq. (10)–(12) 的实现 smoke test 通过，但没有真实 benchmark result-level metric。
