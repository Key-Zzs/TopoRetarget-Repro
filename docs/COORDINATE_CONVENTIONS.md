# Coordinate conventions

The canonical HOI interface uses mathematical right-handed coordinate frames and column
vectors. Hand side (`left` or `right`) is a semantic label and is independent of frame
handedness; left-hand samples are never implicitly mirrored.

## Frames and notation

- `S` is the stable scene frame for one sequence. It is not required to be a geographic world
  frame. If forward/up axes are not specified by the source, metadata records `unknown`.
- `H_R` and `H_L` are per-frame right- and left-hand wrist frames.
- `O` is an object-local frame. Each mesh is stored once in this frame.

`T^A_B` maps coordinates in frame `B` to frame `A`:

```text
p^A = T^A_B p^B
```

The implementation uses homogeneous transforms, with the final row `[0, 0, 0, 1]` and a
rotation determinant of `+1`:

```text
p^H = (T^S_H)^-1 p^S
p^S = T^S_H p^H
p^O = (T^S_O)^-1 p^S
p^S = T^S_O p^O
```

These operations are implemented in `src/toporetarget/geometry/se3.py` and named wrappers are
available in `src/toporetarget/geometry/frames.py`.

## Storage semantics

Hand vertices and keypoints are primarily stored in `S`; wrist pose is stored as `T^S_H`. Object
vertices remain in `O` and object pose is stored as `T^S_O`. This preserves wrist global trajectory,
scene/object motion, and bimanual relationships. A wrist-relative-only cache would lose those
relationships and cannot be reconstructed without the original wrist pose.

Single-hand sequences still retain `S`; the interface does not force them into a right-wrist frame.
`native_fps` is metadata only. Timestamps are seconds and are the temporal source of truth; no
interpolation, frame-rate conversion, or display-stride mutation is performed.

All positions and meshes use metres, angles use radians, and time uses seconds. If quaternions are
used at an adapter boundary, their order is `xyzw`; rotation matrices are the internal primary
representation.
