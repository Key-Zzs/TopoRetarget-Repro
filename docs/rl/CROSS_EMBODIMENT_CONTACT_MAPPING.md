# Cross-Embodiment Contact Mapping

The final Stage 16-D audit maps source-derived HOCap MANO contact semantics to
the frozen 21-body Wuji Formal20 telemetry. This is an audit mapping, not a
Reward V3 or physics change.

| Human source region | Strict Wuji evidence | Same-finger group evidence |
| --- | --- | --- |
| thumb | `r_thumb_distal` | four named thumb collision bodies |
| index | `r_index_finger_distal` | four named index collision bodies |
| middle | `r_middle_finger_distal` | four named middle collision bodies |
| ring | `r_ring_finger_distal` | four named ring collision bodies |
| pinky | `r_pinky_distal` | four named pinky collision bodies |

The three compared interpretations are:

- strict per-finger: a source-expected finger requires its named distal Wuji
  contact;
- per-finger contact group: the strict tip or another named collision body in
  that same digit is acceptable;
- aggregate V3: the historical, frozen five-tip force sum.

`r_wrist` is a wrist/base collision body, not a palm. No report may count it as
a palm substitute. Cross-finger contact and wrist/base contact are reported as
compensation/unmapped evidence, never as strict per-finger satisfaction.

The final report classifies each source-expected finger/replica/control step as
strictly satisfied, same-finger substitution, cross-finger compensation,
wrist-base-unmapped, or fully missing. It correlates those states with the
recorded V3 contact scale/reward and object errors/twist residuals. Existing
Formal20 terminal stability was not captured as a post-PPO pass/fail signal, so
the report preserves that limitation rather than inventing a physics label.
