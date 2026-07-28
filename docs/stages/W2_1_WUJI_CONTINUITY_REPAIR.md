# W2.1 Wuji Continuity Repair

W2.1 is the fixed three-clip engineering validation of Wuji Hand2 Beta1 RH:

- W1 `s1/airplane_lift`, global `[240,300)`;
- W2 `s1/apple_eat_1`, global `[212,272)`;
- W3 `s1/alarmclock_lift`, global `[407,467)`.

The baseline is frozen as
`scipy_slsqp_active_set_contact_rich_v3_fixed` with
`trajectory_continuity_guarantee=false`. The new profile is
`wuji_continuous_full_state_v1`; it is an engineering extension and keeps
`author_exact=unresolved`.

The experiment root contains baseline identity/jump reproduction, transport
and anomaly evidence, full-run artifacts, deterministic retries, window
usage, independent collision validation, performance, exports, and HTML
review. Raw GRAB/MANO data, tracked Wuji assets, historical Stage-10 outputs,
and the separate `pene-loss` worktree are outside the new artifact write
scope.
