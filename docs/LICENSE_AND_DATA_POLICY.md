# License and data policy

The existing repository `LICENSE` is GNU GPL version 3. It is preserved without changing its
license type. External datasets, MANO/SMPL-X models, and hardware assets are not redistributed by
this repository. Configure their local paths through `.local/config.yaml` or environment variables.

Only read-only directory discovery is performed under the external storage root. The resolver does
not copy, unpack, parse, modify, or symlink raw data. `.local/` is ignored by Git and is the only
location for machine-specific reports, downloaded paper extraction caches, and imported Arti-MANO
assets.

