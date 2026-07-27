# Robot configuration boundary

Robot URDF/MJCF files are handled through the generic target-hand contract. F0 tracks the
Arti-MANO RH/LH vendor snapshot under `third_party/robot_hands/artimano/`; other robot assets are
still external and must be registered with explicit provenance. This boundary does not implement
MANO loading or robot retargeting.
