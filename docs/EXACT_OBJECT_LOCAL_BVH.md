# Exact Object-Local BVH

`exact_object_local_bvh_v1` is a persistent float64 AABB branch-and-bound tree. Leaves
evaluate exact point-to-triangle closest points; traversal terminates only after every
remaining node lower bound cannot beat the exact best distance. Face ties are stable by
face ID. Object pose changes transform queries into object-local coordinates and never
rebuild the tree. Centroid kNN and voxel distances are not final answers.
