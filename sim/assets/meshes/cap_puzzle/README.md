# cap_puzzle meshes

Source: Onshape exports (binary STL). **Units: millimetres** — in MuJoCo use `scale="0.001 0.001 0.001"`.

| file | what | bbox (mm) | up axis |
|---|---|---|---|
| `bottom.stl` | bottom part, fixed to the table | 75 × 75 × 37 (z −7…30) | Z |
| `top_cap.stl` | cap screwed onto the bottom — unscrew it first | 75 × 75 × 25 (z −17…8) | Z |
| `inner.stl` | inner (white) part — gripper fingers go inside and spread to pull it out | Ø≈41 × 40.5 (z 0…40.5) | Z |
| `box.stl` | generic box | 100 × 30 × 100 | **Y** (rotate 90° about X for Z-up) |

Task: unscrew `top_cap` from `bottom`, then pull `inner` out of `bottom` by inserting the gripper
jaws and opening them (internal grip). Each part is in its own local frame, not in assembled pose.

Note: meshes are non-convex; for contacts MuJoCo uses the convex hull, so the threads/cavities
will need convex decomposition (e.g. CoACD) or primitive colliders for the real interaction.

```xml
<asset>
  <mesh name="cap_bottom" file="meshes/cap_puzzle/bottom.stl"  scale="0.001 0.001 0.001"/>
  <mesh name="cap_top"    file="meshes/cap_puzzle/top_cap.stl" scale="0.001 0.001 0.001"/>
  <mesh name="cap_inner"  file="meshes/cap_puzzle/inner.stl"   scale="0.001 0.001 0.001"/>
</asset>
```
