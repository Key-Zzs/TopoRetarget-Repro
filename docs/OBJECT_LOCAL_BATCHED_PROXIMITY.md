# Object-Local Batched Proximity

`ObjectLocalProximityContext` keeps the exact triangle AABB tree and strict
winding inputs alive once per mesh hash. Query points move scene-to-object once
per batch; closest points and normals move back only after the exact query.

Its counters make mesh loads, BVH construction, batches, points, and transform
time auditable. The context is not a grid, hull, proxy, or cache of prior-frame
signed distances.
