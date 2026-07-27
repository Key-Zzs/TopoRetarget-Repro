# Third-party asset policy

This repository distinguishes code, tracked robot-hand assets, and machine-local data.

## Policy

- Arti-MANO is distributed as a tracked vendor snapshot under
  `third_party/robot_hands/artimano/`.
- The snapshot contains `LICENSE`, `SOURCE.yaml`, `NOTICE.md`, `asset_manifest.json`, `urdf/`, and
  `meshes/`. It does not contain ManipTrans Python source, environments, or datasets.
- `SOURCE.yaml` is the canonical provenance record: upstream repository and commit, source path,
  observed license, import timestamp, included/excluded paths, per-file hashes, and the source
  manifest hash.
- The repository root `LICENSE` remains the license for repository-owned code. The vendored
  upstream license is retained separately with the asset.
- No separate license was found below the upstream Arti-MANO asset directory at the F0 audit. If a
  future source checkout contains a separate asset license, the vendor command fails closed with
  `ARTIMANO_LICENSE_DECISION_REQUIRED` until the policy is reviewed.

## Resolution and local boundaries

Normal registry resolution is:

1. explicit `--asset-root`;
2. `TOPORETARGET_ARTIMANO_ASSET_ROOT`;
3. tracked `third_party/robot_hands/artimano/`;
4. legacy `.local/assets/artimano/`, with a deprecation warning.

`ARTIMANO_ASSET_ROOT` remains accepted as a deprecated environment alias. `.local` is reserved
for cache, run, and report artifacts; the legacy asset directory is read only for migration and
historical-artifact compatibility.

## Re-import

Use a pinned ManipTrans checkout and a stable timestamp when a reproducible vendor snapshot is
needed:

```bash
python scripts/vendor_robot_hand_assets.py \
  --source-root /path/to/ManipTrans \
  --destination third_party/robot_hands/artimano \
  --imported-at 2026-07-27T19:00:00+08:00
```

The command performs a license gate, validates URDF mesh references, writes through a temporary
directory, and records deterministic source hashes. It refuses to replace an existing destination
without `--force`.
