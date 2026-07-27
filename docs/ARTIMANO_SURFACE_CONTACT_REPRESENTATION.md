# Arti-MANO Surface Contact Representation

`RobotContactSurfaceProfile`, `RobotContactRegion`, and `RobotContactSample`
define the generic visual contact interface. The first implementation is
`artimano_surface_contact_v1`; future dexterous hands can provide the same
contract without changing the 21-anchor skeleton interface.

The profile has thirteen semantic regions: five distal pads, four middle-
phalanx side regions, `thumb_side`, and three palm regions. Samples store link,
ancestry, local frame, visual mesh identity, face ID, barycentric coordinates,
local point, semantic normal, confidence, collision coverage, skeleton-anchor
association, and GRAB label association.

Sampling is deterministic area-weighted triangle sampling with fixed PCG64
seed. Samples are generated once from neutral visual geometry and transformed by
FK across poses. Open visual meshes are allowed because the profile never uses
inside/outside tests. Low-confidence normals are excluded from direction-loss
claims. Collision surfaces remain the existing Stage 9 collision sample set;
visual fallback is never substituted for collision geometry.

Validation records barycentric sum, point reconstruction, deterministic repeat,
link identity, visual/collision coverage, and FK/Jacobian provenance in
`surface_contact/validation.json`. This is a paper-external quality
representation, not an alteration to the TopoRetarget paper objective.
