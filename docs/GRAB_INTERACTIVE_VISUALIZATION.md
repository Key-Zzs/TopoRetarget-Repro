# GRAB interactive visualization

Stage 5 provides `toporetarget data visualize` and
`src/toporetarget/viz/interactive_hoi_viewer.py` for bounded GRAB debugging. The viewer is a
Matplotlib diagnostic surface; it does not alter canonical arrays or perform retargeting.

## Modes and layouts

- `--mode raw` displays the source sequence directly.
- `--mode canonical` displays a canonical Zarr cache.
- `--mode compare` overlays or places raw and canonical scenes side by side.
- `--layout overlay|side-by-side` controls compare composition.

The viewer can show the scene, object, right wrist, or left wrist reference frame. Hand meshes,
object/table meshes, native joints, MediaPipe21 tracks, skeletons, labels, contact markers, axes,
and each hand side have independent visibility controls where the selected representation supports
them.

## Controls

The window provides a frame slider, first/previous/next/last/play controls, playback-speed slider,
reference-frame radio buttons, raw/canonical visibility, hand-side, mesh, skeleton, label, object,
table, contact, axes, and error toggles. Keyboard shortcuts are `Home`, `Left`, `Right`, `End`, and
`Space` for frame navigation and play/pause. The timer is closed explicitly when the viewer closes;
artists are created once and updated in place so frame changes do not grow the artist list. Missing
hand sides are shown as disabled controls in compare mode.

Example:

```bash
toporetarget data visualize --dataset grab --index .local/index/grab \
  --sequence s7/cubemedium_inspect_1 \
  --canonical .local/cache/hoi/grab/s7/cubemedium_inspect_1/both_f000000_f000060.zarr \
  --mode canonical --reference-frame scene --show-mesh --show-contacts \
  --start-frame 0 --end-frame 60 --output .local/reports/stage5/canonical_first.png

toporetarget data visualize --dataset grab --index .local/index/grab \
  --sequence s7/cubemedium_inspect_1 \
  --canonical .local/cache/hoi/grab/s7/cubemedium_inspect_1/right_f000000_f000060.zarr \
  --mode compare --layout side-by-side --frame 30 \
  --output .local/reports/stage5/compare_side_by_side_middle.png
```

`--show` opens the local GUI when a graphical backend is available. `--output` is headless and
works with `MPLBACKEND=Agg`; PNG snapshots are deterministic. GIF output is available when Pillow
is installed, and MP4 output gives a clear ffmpeg error when ffmpeg is unavailable. In both cases
`--display-stride` only reduces rendered frames; it never changes canonical timestamps or arrays.
Large native meshes remain unchanged in the canonical data but use a deterministic viewer-only
point cap when polygon rendering would make the GUI impractical; this display cap is never written
to cache or used by validation.

`--reference-frame` is the documented spelling. `--reference` remains an accepted compatibility
alias.

The interactive smoke test uses a deterministic synthetic sequence to exercise construction,
callbacks, slider movement, play/pause, reference changes, visibility toggles, stable artist
counts, and timer shutdown. Real GRAB acceptance uses native raw/canonical PNG rendering and the
raw/canonical numerical comparison report.
