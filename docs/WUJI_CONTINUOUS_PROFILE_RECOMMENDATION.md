# Wuji Continuous Profile Recommendation

## Final recommendation

| Gate | Requirement | Result | Pass |
| --- | --- | --- | --- |
| profile split | only the declared window flag differs semantically | `true -> false` | yes |
| formal path | three 60-frame trajectories invoke no production window | count `0` | yes |
| numerical / continuity | accepted, finite, feasible, continuous, bounded corrections | W1/W2/W3 pass | yes |
| quality | no material EIM/Ebone or joint-limit regression | pass | yes |
| penetration hard gate | `R_pen(2 mm)` not worse; max depth <= 2 mm | pass | yes |
| penetration secondary gate | 1 mm rate/depth warning limits | W3 warning | no, warning only |
| selected replay | 21 bounded frames and retry paths equivalent | pass | yes |
| window experimental | window branch independently validated | unresolved nonblocking | no, nonblocking |

The resulting status is
`WUJI_CONTINUOUS_SEQUENTIAL_PROFILE_RECOMMENDED_WITH_SECONDARY_PENETRATION_WARNING`.
The recommended profile is `wuji_continuous_sequential_v1`, scoped only to
`offline_reference_generation`. `RL_READY=NO`, `REALTIME_READY=NO`,
`CROSS_SUBJECT_VALIDATED=NO`, and `AUTHOR_EXACT=UNRESOLVED`.

The window status is reported separately as
`WINDOW_FALLBACK_EXPERIMENTAL_UNRESOLVED_NONBLOCKING`; it cannot promote or
demote the sequential gate. The recommendation JSON and compatibility report
are in `w2_3_finalization/recommendation/` and `w2_3_finalization/reports/`.
