# Stable Dynamic Contact Calibration

`StableFreeObjectGraspCalibrationV1` is an engineering calibration, not a task
trajectory. It asks whether the frozen Wuji hand, controller, runtime collision
proxies, PhysX parameters, and 26D action interface can establish and hold a
free object using real multi-sided contact. It never writes object or wrist
state after the 321-step schedule starts and uses no ground, support, hidden
force, attachment, source trajectory, or corrected trajectory.

The object-canonical initializer uses convex-proxy centroid, PCA/OBB axes,
extents, support directions, and a fail-closed python-fcl reset refinement.
Data-derived topology is thumb/index for `hocap_170105` and
thumb/index/pinky for `hocap_170650`; the extraction algorithm is shared.
C1 freezes -6/0/+6 mm offsets × closure 0.5/1.0. Unique C2 adds only -10/+10
mm with the same closure grid.

All 20 development candidates ran four replicas × 321 steps. No candidate
passed contact, topology, terminal hold, terminal twist, and exact geometry
together. The result is `STAGE16D_STABLE_GRASP_CALIBRATION_BLOCKED` with stop
marker `STAGE16D_STABLE_FREE_OBJECT_GRASP_CALIBRATION_BLOCKED`. Formal20 was
not authorized. Therefore `EmpiricalStableDynamicContactReferenceV1` and V2
were not created. If one is established later, it remains empirical engineering
evidence—not physical truth or a mathematical lower bound.
