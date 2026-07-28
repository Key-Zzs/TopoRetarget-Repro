# W2/W3 Wuji full retargeting

W2 is `s1/apple_eat_1` `[212,272)` and W3 is `s1/alarmclock_lift`
`[407,467)`. Both use the same fixed hand, robot, frame/bone profiles,
collision surface, query profile, execution profile, and Stage 9 solver as W1.

The suite runs both units after the frozen selection and object watertight
gate. A soft wall-time pause is resumed from the same hash chain. A solver
status such as SciPy status 9 is never promoted to strict acceptance; any
failure remains in the unit report while the other fixed unit continues.
