# Cross-Embodiment Contact Mapping

Stage 16-D 最终 audit 将 source-derived HOCap MANO contact semantics 映射到冻结的 21-body
Wuji Formal20 telemetry。这是 audit mapping，不是 Reward V3 或 physics change。

| Human source region | Strict Wuji evidence | Same-finger group evidence |
| --- | --- | --- |
| thumb | `r_thumb_distal` | 四个有名字的 thumb collision bodies |
| index | `r_index_finger_distal` | 四个有名字的 index collision bodies |
| middle | `r_middle_finger_distal` | 四个有名字的 middle collision bodies |
| ring | `r_ring_finger_distal` | 四个有名字的 ring collision bodies |
| pinky | `r_pinky_distal` | 四个有名字的 pinky collision bodies |

比较三种解释：strict per-finger（source expected finger 需要指定 distal Wuji contact）、
per-finger contact group（strict tip 或同一 digit 其他 named collision body 均可）和 historical frozen
five-tip force-sum aggregate V3。

`r_wrist` 是 wrist/base collision body，不是 palm。报告不会把它记为 palm substitute。cross-finger
contact 与 wrist/base contact 只作为 compensation/unmapped evidence，不能当作 strict per-finger
satisfaction。

最终报告将每个 source-expected finger/replica/control step 标为 strict satisfied、same-finger
substitution、cross-finger compensation、wrist-base-unmapped 或 fully missing，并关联记录的 V3
contact scale/reward 和 object errors/twist residual。既有 Formal20 terminal stability 没有采集为
post-PPO pass/fail signal，因此报告保留该限制，不虚构 physics label。
