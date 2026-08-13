# Source Contact Semantics

`SourcePerFingerContactEvidenceV1` 是 Stage 16-D 跨具身 contact semantics 的 source authority。
它不改动冻结的 V3，但其 confirmed/persistent-confirmed runtime mapping 是单独版本化的 Strict
Per-Finger V4 reward 的 mandatory source mask；它从不改动 checkpoint、RSI、controller、reference
或 physics。

对选定 HOCap clip，审计从 `poses_m.npy`、subject-specific calibration betas 与
`MANO_RIGHT.pkl` 重建原始右手 MANO surface，并在原始 HOCap world/object pose 约定中，对选定
原始 object mesh 的精确 triangles 查询每个 MANO vertex。一个原始 source frame 只有同时满足以下
条件才是 `SOURCE_CONTACT_CONFIRMED`：最小 surface-to-triangle 距离不超过 2 mm、5 mm 内至少
3 个 MANO vertices 构成 connected component，并持续至少 2 个原始 30 Hz frames。

审计保存 1/2/5 mm threshold sensitivity。`SOURCE_CONTACT_PROBABLE` 满足几何/component
条件但未满足 native persistence；`SOURCE_CONTACT_TRANSITION` 是明确状态变化；
`SOURCE_PROXIMITY_ONLY` 在 10 mm 内但没有 robust component evidence；其余为
`SOURCE_NO_CONTACT`。

MANO thumb/index/middle/ring/pinky/palm regions 由 MANO v1.2 LBS joint-chain weights 推导。
low-margin webbing vertices 归为 `boundary_ambiguous`，不会被静默计入任何 finger。segment 使用同一
joint influence 与 rest-chain longitudinal geometry；tip surface 为 model-derived terminal quantile，
没有 hard-coded vertex list。

native source 恰有 41 个选定 keys，按 factor 8 映射到已有的 321 control frames：精确 keys 保留
自身 class；只有相邻 confirmed keys 才填充 `SOURCE_CONTACT_PERSISTENT` interval；两个 no-contact
keys 填 no contact；其他 interval 都保持 transition。

V4 第一版只将 `SOURCE_CONTACT_CONFIRMED` 和 `SOURCE_CONTACT_PERSISTENT` 视为 mandatory。probable、
transition、proximity-only、no-contact 和 ambiguous state 在 strict mask 中都明确为零。policy 不能用
actual robot contact 重新生成或改动这项 source-side decision。

```bash
python scripts/evaluation/finalize_stage16d_source_contact_semantics.py
```

报告写入 `.local/reports/stage16d_source_contact_semantics_final_audit/`。
