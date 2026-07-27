# Wuji Hand2 Beta1 asset provenance

| Field | Value |
| --- | --- |
| Repository | `https://github.com/wuji-technology/wuji-description.git` |
| Requested ref | `release/v2026.7.23` |
| Resolved ref | `refs/remotes/origin/release/v2026.7.23` |
| Resolved commit | `2b57d2621caed4e65207bb767ba25fc8eaec0881` |
| Upstream path | `hand2/hand2_beta1/body` |
| License | MIT, retained in the bundle |
| Import tool | `toporetarget.vendor_robot_hand_assets.v2` |
| Source manifest SHA256 | `69ed702aa29166d3a8992af0cae3904bf12d5247e04ebaa2fa16fc410414dd16` |
| Deterministic imported-at | `2026-07-23T13:53:30Z` |

The requested short ref was not present as a local branch in the upstream checkout. The importer
resolved the existing remote-tracking ref without fetching, switching, or modifying that checkout.
The same-version tag `v2026.7.23` points to `87a23cf76ec3593355dcf4168a6fb9d49c2c3f30`; it was
recorded as distinct and not used.

Imported paths are the root `LICENSE`, `urdf/{left,right}.urdf`, `mjcf/{left,right}.xml`, and all
STL files under `meshes/{left,right}/` referenced by the two approved models. The complete source
path list, per-file hashes, and generated manifest hash are in
`third_party/robot_hands/wuji_hand2_beta1/SOURCE.yaml`.

Excluded paths are the ROS URDF variants, `step/**`, `usd/**`, `CMakeLists.txt`, and `package.xml`.
No STEP/USD/CAD, RViz, ROS2 launch, glove, or Hand2 Beta1 v1 payload was imported.

The bundle retains `LICENSE`, `SOURCE.yaml`, `NOTICE.md`, and `asset_manifest.json`. It contains 57
imported source files plus those generated/notice records. The two model hashes are:

| Side | URDF SHA256 | MJCF SHA256 |
| --- | --- | --- |
| RH | `1ae70be3f5e64532203e599eaa98d2af368d0214be9c949a358b7abaa8b6265a` | `1bc53b7ca6f2eb84fc66ac736027984cd1734fa30d27f9c1f640495258d626f9` |
| LH | `cec0a7eb6a34fd82e200def7b75c1d477fad790b2de903aec58e59991994c471` | `ebaa3b07854c8df1847ffbf54d9d6527e82d838d02329e4c35c6ca667b8fef89` |

This is a provenance record, not a legal opinion. Redistributors must retain the included notices
and independently verify obligations for their distribution context.
