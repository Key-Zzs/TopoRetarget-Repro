# S1.3 Reference-Faithful SDF Backend

S1.3 evaluates `original_mesh_batched_exact_bvh_v1`. It retains one
object-local original triangle mesh and strict generalized-winding sign context
for a run. It neither repairs nor simplifies the mesh and does not change the
paper-external dense SDF loss.

The independent final audit remains `reference_winding_v1`. Backend selection
uses accuracy, determinism, memory, and timing only; S1 outcomes do not choose
the backend. A passing backend does not itself accept the SDF formulation or
avoid a fresh T4 stress discovery.
