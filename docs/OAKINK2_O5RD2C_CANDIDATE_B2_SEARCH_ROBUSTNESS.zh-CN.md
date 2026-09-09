# OakInk2 O5R-D2C Candidate B2 Search Robustness

## 结论

O5R-D2C 在既有 106 个已消费 DEV1 development frames 范围内完成，结果为：

```text
CANDIDATE_B2_ALIGNMENT_GATE=PASS
CANDIDATE_B2_EXECUTION_DEVELOPMENT_GATE=PASS
RETARGET_OBJECTIVE_V2_DESIGN_STATUS=FROZEN_READY_FOR_INDEPENDENT_VALIDATION
SELECTED_EXECUTION_CONTRACT=S1_DETERMINISTIC_MULTI_START
```

本阶段没有运行 SparseValidationV2、WindowValidationV2、DEV2 frame0、DEV2 full、DEV1 full rerun/refinement 或任何下游 physical/policy stage。独立验证必须留给 O5R-D2D。

## Candidate B2 alignment

第一道 gate 只 replay 冻结的 B1 25 个 Q0/Q1 状态，不运行 optimization：

| 分类 | 数量 |
| --- | ---: |
| `B2_ADMISSIBLE_AND_PRIMARY_BETTER` | 17 |
| `B2_ADMISSIBLE_PRIMARY_EQUIVALENT` | 7 |
| `B2_TRUE_CONSTRAINT_VIOLATION` | 1 |

24 个 clear low-`E_IM` 且满足 true constraints 的 Q1 全部被 Candidate B2 接纳或 primary 优先，比例为 24/24。prediction-correction-only rejection 为 0。唯一 true violation 是 ordinal 2222 的 bone-direction violation；它不是 prediction-correction wall。

因此 Candidate B2 语义在本阶段锁定：

```text
H_IM = max(E_IM / 1e-4 - 1, 0)^2
prediction correction = soft diagnostic/preference
actual Semantic V1 continuity = hard authority
```

## D2B root cause

D2B 的 HIGH 18/20 与 LOW 19/20 不是“找不到可行终态”：

- ordinal 2358 的 primary/secondary 均 independently feasible，且 `E_IM <= 1e-4`；
- ordinal 2367 的 secondary independently feasible，且 `E_IM <= 1e-4`；
- ordinal 217 的 primary/secondary independently feasible，且旧状态本身 interaction-valid；
- 三帧都因为 SLSQP `status=9` 被旧 execution 丢弃，随后 valid `q_old` fallback 又被标成 technical failure。

因此主要机制是 `CANDIDATE_RETENTION_FAILURE + INVALID_FALLBACK`。预算耗尽是触发信号，不是 scientific invalidity 本身。

## Generic execution contracts

两个候选都在 development run 之前冻结：

| Search | Seeds | Sequential | Top-K | HIGH/MID/LOW |
| --- | --- | --- | ---: | --- |
| S1 | old/rest/midpoint | NO | 1 | 全部 PASS |
| S2 | S1 + previous-refined transported | YES | 1 | 全部 PASS |

两者共同使用：

- exact Eq.7 per-vertex mass 的 asset-derived contributor ranking；
- robot DOF name 推导的 generic semantic finger block；
- deterministic contributor `least_squares/trf` probes，`max_nfev=24`；
- Candidate B2 whole-hinge primary，`maxiter=8`；
- secondary fidelity polish，`maxiter=8`；
- finite + independent hard-validity terminal screening，不把 optimizer success flag 当 scientific validity；
- `q_primary_valid` retention；polish 破坏 interaction 时 reject polish；
- `q_old[t]` 永远在 candidate pool，valid fallback 算 technical completion。

S3/top-K=2 未实现：旧失败已经存在 interaction-valid top-1 terminals，没有证据支持增加第二 contributor block。

S1 与 S2 都通过后，按预定 runtime、nfev、simplicity tie-break 选择 S1。S2 的 sequential path 已实现并在同一 60-frame development set 上通过，但没有因任务标题而强行选择更慢、更复杂的 contract。

## Development 结果

| Stratum | Technical | Old p95 `E_IM` | New p95 `E_IM` | true continuity / wrist / bone / collision / joint limits |
| --- | ---: | ---: | ---: | --- |
| HIGH | 20/20 | 1.943759878993e-4 | 7.106302056692e-5 | PASS |
| MID | 20/20 | 1.379943803003e-4 | 6.296368745394e-5 | PASS |
| LOW | 20/20 | 7.329716722495e-5 | 7.418477715252e-5 | PASS |

LOW 中所有旧 interaction-valid 状态仍 interaction-valid。60 帧全部先获得 interaction-valid primary；60 次 secondary polish 中有 2 次因 interaction regression 被拒绝并保留 primary。选中的 60-frame 结果没有需要使用 baseline fallback，但 fallback invariant 已由测试覆盖。

HIGH/MID/LOW 的中心连续 3-frame representative sub-window 各运行 3 次，selected seeds、blocks、retention decisions、final q/base 与 p95 在 `1e-8` tolerance 内一致，determinism PASS。

B1 25-frame reachability cross-check 为 25/25 technical/feasible，median gap closure 约为 1.0；B1 best-observed Q1 从未作为 seed。

## Frozen contracts

真实 hash 以以下文件为 authority：

```text
.local/reports/oakink2_o5rd2c_candidate_b2_search_v1/frozen_method/
  retarget_objective_v2_contract.json
  retarget_objective_v2_contract.sha256
  execution_contract_v2.json
  execution_contract_v2.sha256
```

## CLI

以下入口的 `--help` 已实际运行：

```bash
conda run -n toporetarget-rl \
  python scripts/data/run_oakink2_o5rd2c.py --help
```

分步命令包括：

```bash
conda run -n toporetarget-rl python scripts/data/run_oakink2_o5rd2c.py preflight
conda run -n toporetarget-rl python scripts/data/run_oakink2_o5rd2c.py replay-q0-q1-alignment
conda run -n toporetarget-rl python scripts/data/run_oakink2_o5rd2c.py audit-d2b-failures
conda run -n toporetarget-rl python scripts/data/run_oakink2_o5rd2c.py compare-b1-b2-search
conda run -n toporetarget-rl python scripts/data/run_oakink2_o5rd2c.py run-search-s1
conda run -n toporetarget-rl python scripts/data/run_oakink2_o5rd2c.py run-search-s2
conda run -n toporetarget-rl python scripts/data/run_oakink2_o5rd2c.py run-search-s3
conda run -n toporetarget-rl python scripts/data/run_oakink2_o5rd2c.py run-development-gates
conda run -n toporetarget-rl python scripts/data/run_oakink2_o5rd2c.py run-determinism
conda run -n toporetarget-rl python scripts/data/run_oakink2_o5rd2c.py run-q1-reachability
conda run -n toporetarget-rl python scripts/data/run_oakink2_o5rd2c.py freeze-objective-v2
conda run -n toporetarget-rl python scripts/data/run_oakink2_o5rd2c.py freeze-execution-v2
conda run -n toporetarget-rl python scripts/data/run_oakink2_o5rd2c.py generate-future-certification-plan
```

这些 action 中不存在 independent validation、DEV2 或 DEV1 full solve 入口。

## 下一阶段

```text
NEXT=O5R-D2D_OBJECTIVE_V2_INDEPENDENT_CERTIFICATION
```

O5R-D2D 才能在 development exclusion ledger 零重叠前提下冻结新的 SparseValidationV2 与 WindowValidationV2，并严格按 SparseGate → WindowGate → DEV2 frame0 x3 → DEV2 240 once 的顺序继续。
