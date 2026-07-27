# Robot configuration boundary

Robot URDF/MJCF files are handled through the generic target-hand contract. F0 tracks the
Arti-MANO RH/LH vendor snapshot under `third_party/robot_hands/artimano/`, and W0/W1 tracks the
approved Wuji Hand2 Beta1 body subset under `third_party/robot_hands/wuji_hand2_beta1/`. Other robot
assets must be registered with explicit provenance. This boundary does not implement MANO loading,
hardware control, or robot retargeting.
