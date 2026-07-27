# Wuji Hand2 Beta1 semantic mapping

The target uses the repository's `mediapipe21` layout. The mapping is an explicit engineering
contract derived from the pinned URDF joint origins and official MJCF tip sites; it is not claimed
to be an unpublished author-exact mapping.

Each side has 21 anchors: `wrist`, then `thumb_cmc`, `thumb_mcp`, `thumb_ip`, `thumb_tip`, followed
by the four anchors for index, middle, ring, and pinky. MCP/PIP/DIP anchors use named URDF joint
origins. Tip anchors use the fixed URDF tip-link frame and are cross-checked against the matching
official MJCF site. The full profiles are in
`configs/robots/anchors/wuji_hand2_beta1_{rh,lh}_mediapipe21.yaml`.

The public qpos order is finger-major and anatomical:

```text
thumb:  cmc_flex, cmc_abd, mcp, ip
index:  mcp_flex, mcp_abd, pip, dip
middle: mcp_flex, mcp_abd, pip, dip
ring:   mcp_flex, mcp_abd, pip, dip
pinky:  mcp_flex, mcp_abd, pip, dip
```

Exact prefixed names, limits, URDF order, MJCF order, and actuator order are in
`configs/robots/joint_orders/wuji_hand2_beta1_{rh,lh}.yaml`. The actuator order is a separate
simulation-facing name list; it is never silently substituted for public qpos.

RH and LH are independently parsed and registered. LH flex axes are signed opposite to RH axes
where the upstream URDF declares that difference; abduction axes and limits come from the matching
side. The implementation does not mirror RH at runtime.

Tip sites are semantic/surface proxies. Soft-pad STL payloads remain visual and are not formal
collision. Formal collision uses the declared MJCF convex hulls, with the ten upstream contact
excludes recorded explicitly. Contact proxies, source contacts, and signed-distance ground truth
remain separate concepts.
