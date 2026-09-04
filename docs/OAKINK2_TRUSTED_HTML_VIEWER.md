# OakInk2 HTML Viewer V2

OakInk2 Viewer V2 replaces the deprecated O1R2-C manual WebGL camera path. It
adopts the stable Ref2Dex MeshCat architecture—named scene nodes and one
renderer-owned camera—without adding a runtime dependency on Ref2Dex or
MeshCat. Each generated HTML is self-contained and requires no CDN or network.

The scene frame is `SCENE_WORLD_MANO_ROOT_RELATIVE`. Python precomputes all
MANO vertices, closed/open faces, 21 joints, and per-frame object transforms.
The browser does not decode quaternions, run MANO FK/skinning, apply betas, or
interpret `center_idx`. Hand and skeleton model matrices are identity; the
object uses its Python-precomputed per-frame scene model.

`ViewerCameraStateV1` is the single camera authority. `FRONT`, `OBLIQUE`, and
`SIDE` are deterministic states using the trusted O1R2 matrices. Left-drag
changes camera yaw/pitch around `FOCUS_INTERACTION` (default) or `FOCUS_HAND`;
the wheel changes camera distance; `RESET CAMERA` restores `OBLIQUE`. Pan is
not supported by design. Orbit and zoom never mutate hand vertices, joints, or
the object scene pose.

The real visibility modes are `HAND ONLY`, `HAND + OBJECT`, `SKELETON ONLY`,
and `HAND + SKELETON + OBJECT`. Play/pause and the slider retain each frozen
episode's 180 exact source mocap frame IDs. Camera state persists across frame
changes and playback. Source/canonical is identity metadata, not a fake switch.

Generate and certify the authoritative same-two package:

```bash
PYTHONPATH=src conda run -n ref2dex-oakink \
  python scripts/data/run_oakink2_o1r2d.py --action all
```

Generate one authoritative episode without browser certification:

```bash
PYTHONPATH=src conda run -n ref2dex-oakink \
  python scripts/data/run_oakink2_o1r2d.py \
  --action generate --review dev_01
```

An explicit source-frame timeline can be supplied for generation with
`--frame-ids`; it must include the episode's frozen primary frame. It is not an
O1R2-D certification path.

Certification drives Chrome through the DevTools input domain, producing real
pointer callbacks and wheel events. It covers horizontal, vertical, diagonal,
reverse, and repeated drag; zoom; reset; preset transitions; an orbit sweep;
and frame/playback-under-orbit. `gl.readPixels`, landmark rank/depth, immutable
3D pairwise distances, scene fingerprints, object models, and hand-object
anchors are recorded. Machine PASS does not replace the required two-episode
human review, and O5 remains unavailable until both approvals are recorded.
