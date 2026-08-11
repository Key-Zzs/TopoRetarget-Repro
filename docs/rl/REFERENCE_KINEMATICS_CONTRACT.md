# Stage 16-D Reference Kinematics Contract

## Versioned reference artifacts

`reference_kinematics_version=1` remains an immutable historical artifact.
`reference_kinematics_version=2` is a separate, hash-carrying materialization
for the factor-8 321-key control grid.  V2 records its V1 parent and native
source hashes in metadata; it never overwrites either input.

## Time and pose semantics

The native reference has 41 keys on a 0.05 s grid.  The factor-8 control
reference has 321 keys on the same 0.05 s grid and therefore preserves the
native duration.  Native keys occur at `0, 8, ..., 320` and are preserved
exactly.  Interpolated translation uses a shape-preserving cubic Hermite
trajectory.  Quaternion keys are normalized, sign-continuous `wxyz` active
right-handed rotations, interpolated by shortest-arc SLERP.

## Twist semantics

`*_twist_world_ref[..., :3]` is the signed world-frame linear velocity derived
from the materialized pose and timestamp grid.  `[..., 3:]` is world-frame
angular velocity from the SO(3) relative rotation log.  Interior samples use
centered derivatives; endpoints use the matching second-order one-sided
derivative.  Body-frame angular velocity is a separately named conversion and
is never silently substituted for the world-frame field.

The qualification must prove monotonic timestamps, factor-8 key preservation,
quaternion validity/sign continuity, finite twists, linear and SO(3) integral
consistency, and expected factor-8 derivative scaling.  A terminal reference
twist is descriptive supervision: it must not be zeroed merely to satisfy a
terminal-stability gate.

## Consumers

V1 policies retain their V1 provenance.  A V2 consumer must assert
`reference_kinematics_version == 2` before loading V2 twists.  The metadata,
not a filename convention, is the authoritative version check.
