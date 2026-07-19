# Canonical HOI data interface

Schema version: `toporetarget.hoi.v1`.

The interface is robot-independent. It supports one hand, bimanual sequences, multiple rigid
objects, articulated objects, and optional contacts without requiring MANO or MediaPipe layouts.
The Stage 5 GRAB implementation adds a lazy metadata index and one explicitly selected sequence
conversion; it still does not provide a full-dataset conversion command.

## Records

| Record | Required content | Shape / dtype | Unit |
| --- | --- | --- | --- |
| `SequenceMetadata` | schema, dataset/sequence IDs, timestamps, scene/source frames, source-to-scene transform, provenance | timestamps `[T]` `float64`; transform `[4,4]` `float64` | seconds |
| `ProvenanceRecord` | source identity, adapter/version, conversion options and no-resampling/no-sampling flags | strings, JSON values | n/a |
| `PoseTrack` | pose, validity, frame names | pose `[T,4,4]` `float64`, valid `[T]` `bool` | metres/radians |
| `MeshDefinition` | local vertices, triangle faces, frame/mesh ID, units | vertices `[V,3]` `float64`, faces `[F,3]` integer | metres |
| `KeypointTrack` | positions, validity, explicit layout name | positions `[T,K,3]` `float64`; valid `[T]` or `[T,K]` | metres |
| `ManoParameterTrack` | optional full source parameters | adapter-defined arrays | axis-angle radians/metres |
| `HandTrack` | ID, `left`/`right` side, wrist pose, optional mesh/vertices/keypoint layouts/parameters | vertices `[T,V,3]` | metres |
| `RigidObjectTrack` | object ID, local mesh, scene pose and validity | mesh once; pose `[T,4,4]` | metres/radians |
| `ArticulatedObjectTrack` | part meshes, part poses, parent/child structure, articulation metadata | one `ArticulatedPartTrack` per part | metres/radians |
| `ContactTrack` | hand/object IDs, source representation, validity and optional labels/associations | adapter-defined arrays | source-defined |

`layout_name` is explicit (`mano16_smplx` with legacy alias `mano16`, `mediapipe21`, or a named
dataset-native layout); no layout is silently reshaped or relabeled. `KeypointTrack` also records
its frame, units, and conversion provenance. Missing contacts are represented by an empty
`contacts` list, not by fabricated zeros.

## Stage 3 target track

The Stage 3 adapter adds `hands[*].keypoint_tracks["mediapipe21"]` with scene-frame primary data,
`[T,21,3]` positions, metre units, fixed semantic names, and an explicit profile hash. It keeps
the original MANO track and all object, mesh, wrist-pose, timestamp, FPS, parameter, and contact
fields. Wrist-frame points are derived temporarily with the stored wrist pose; they are not a
second canonical source of truth. `mediapipe21` describes semantic compatibility only and does
not imply a MediaPipe detector dependency or prediction accuracy.

Validation checks time dimensions, finite and strictly increasing timestamps, proper SE(3), metre
mesh units, integer triangle faces, valid hand sides, and finite valid entries. Invalid masked
entries may be non-finite only where the owning validity mask explicitly marks them invalid.

## Lazy loading and caching

`HOIDatasetAdapter` exposes `describe_sequence`, `load_sequence`, `load_raw_renderable`,
`canonicalize`, and `supported_fields`. `FrameRange(start, end)` is a contiguous half-open
selection and contains no stride or interpolation. Adapters do not scan data at import time.

The optional `save_hoi_sequence` / `load_hoi_sequence` functions use Zarr and preserve semantic
arrays, timestamps, mesh definitions, and provenance. Caching happens only when a caller supplies
an output path; raw source directories are never written. Zarr and Matplotlib are optional extras,
so importing `toporetarget` does not require either package.

## Example

```python
from toporetarget.data.synthetic import SyntheticAdapter
from toporetarget.data.storage import load_hoi_sequence, save_hoi_sequence

adapter = SyntheticAdapter()
sequence = adapter.load_sequence("demo")
sequence.validate()
save_hoi_sequence(sequence, ".local/cache/hoi/synthetic_demo.zarr")
reloaded = load_hoi_sequence(".local/cache/hoi/synthetic_demo.zarr")
assert reloaded.timestamps.tolist() == sequence.timestamps.tolist()
```

The canonical representation performs no temporal resampling, spatial/FPS sampling, MANO mesh
sampling, object surface sampling, or robot-model work.

## Stage 5 GRAB extension

`GrabDatasetAdapter` preserves GRAB's source timestamps/native FPS, personalized `vtemp`, native
MANO vertices, object-local meshes, table/support-surface geometry, and source contact arrays.
Contact modes are `none`, `source`, `binary`, and `semantic`; semantic mode is explicitly
unavailable until a verified GRAB label mapping is supplied. The adapter records source size,
mtime, hashes when requested/available, model and mesh hashes, mapping profile hashes, and the
no-resampling/no-sampling scope in `ProvenanceRecord`.
