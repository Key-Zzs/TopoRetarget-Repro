# License and data policy

The existing repository `LICENSE` is GNU GPL version 3. It is preserved without changing its
license type. F0 adds a tracked Arti-MANO vendor snapshot under
`third_party/robot_hands/artimano/`; its upstream `LICENSE`, `NOTICE.md`, provenance, and file
manifest remain beside the asset. External datasets and MANO/SMPL-X models are not redistributed by
this repository. See [`THIRD_PARTY_ASSET_POLICY.md`](THIRD_PARTY_ASSET_POLICY.md) for the asset
gate and resolution contract; configure machine-local data paths through `.local/config.yaml` or
environment variables.

Only read-only directory discovery is performed under the external storage root. The resolver does
not copy, unpack, parse, modify, or symlink raw data. `.local/` is ignored by Git and is reserved
for machine-specific reports, downloaded paper extraction caches, runs, and the legacy imported
Arti-MANO compatibility tree. New default asset resolution uses the tracked vendor snapshot.
