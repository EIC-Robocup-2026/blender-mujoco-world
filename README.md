# MuJoCo World — Easy Export

A Blender addon for authoring, baking, and exporting portable MuJoCo environments. It has no Python package dependencies inside Blender. Tested with **Blender 5.2.1 LTS**, **Blender 4.5.14 LTS** and **MuJoCo 3.15.0**; it declares Blender 4.2 as its minimum.

## Install once

1. Open **Edit → Preferences → Add-ons** in Blender.
2. Open the menu at the top right and choose **Install from Disk…**.
3. Choose [`mujoco_world.zip`](mujoco_world.zip), then enable **MuJoCo World — Easy Export**.
4. In the 3D View, press **N** and open the **MuJoCo** tab.

## Make a world

1. Build or import your room at its real size. Blender's **Scene → Units → Unit Scale** is respected; one default Blender unit is one metre.
2. Select a mesh and click one of the four buttons:

   | Button | What happens in simulation |
   | --- | --- |
   | **Fixed furniture** | Stays in place; objects and the robot can collide with it. |
   | **Movable object** | Has weight, falls under gravity and can be dragged or picked up. |
   | **Door** | Swings around its origin. No motor or handle. |
   | **Visual floor** | Visible, with **no collision**. |

3. Leave **Add a real MuJoCo floor plane** enabled. This creates a single analytic plane named `floor`, separately from your decorative floor mesh. Its height is in metres.
4. Click **Export world…** and save `scene.xml` in a new folder.
5. Keep the XML and its `scene_assets/` directory together. The exporter writes a small `export_report.json` with any notes.

Visible mesh objects export by default. Choose **Leave out** to exclude an object, or enable **Export selected objects only**. **Leave out** on a body group Empty also excludes every part inside it. A fixed or visual mesh without faces (guide edges or points) is left out with a note in `export_report.json`. Collision helpers belonging to a bake or adopted world remain exportable while hidden. Blender lights illuminate the static bake; they are not directly converted into MuJoCo lights in a newly authored world.

Folders that start with `//` (the bake folder, a shared asset folder, the export path) are relative to the `.blend` file, so save the file first; until then the add-on refuses them instead of writing next to wherever Blender was started.

Objects linked from another `.blend` file are skipped by the role buttons, **Group selected parts** and **Set hinge at 3D cursor**, because Blender discards edits to linked data on reload. Make a library override first; the MuJoCo object settings are editable on overrides.

Hide such helpers, adopted objects and body group Empties with the eye icon (**Hide in Viewport**). Blender does not update the placement or modifiers of an object that is disabled in viewports (monitor icon) or sits in a collection excluded from the view layer, so the export stops with an error that names those objects instead of writing stale geometry.

A failed export leaves the previous `scene.xml`, `export_report.json` and assets unchanged: files are written next to their targets first and replace them only after the whole world was built. In a shared asset folder, new files it had already added stay behind, but no existing file changes.

For a joined room mesh with one baked atlas, choose **Visual only** in its Object type dropdown. Build separate simple boxes for the walls, furniture tops and legs, choose **Collision helper** in each box's dropdown, and keep those helpers visible in Blender: they switch to wire display and export only invisible fixed collision, so they do not cover the baked appearance. Hiding a helper excludes it entirely.

## An openable door in four steps

1. Model the door **closed**, with its bottom just above the floor. Select only the door leaf, not its frame.
2. Click **Door**. Put the **3D cursor on the hinge edge** using Blender's cursor tool or **Shift + right-click**.
3. Click **Set hinge at 3D cursor**. The geometry stays still; the hinge origin moves. **Vertical (local Z)** is the default direction. Inspect the object's local axes if it has been rotated.
4. Export. In MuJoCo, **double-click the leaf, then Ctrl + right-drag** to apply force. Press Space if simulation is paused.

The default limits let the door turn from 0° (the modelled closed position) to −110°. Change the limits under **Advanced** if the door should open the other way. Limits must contain 0°. Give the hinge enough clearance from the jamb and floor; intersecting geometry cannot swing freely.

## Choose a sensible collision shape

The visible mesh and the invisible collision shape are different. Every exported visual mesh has contact type and mask **0**, and zero mass. The separate collision shape supplies physics and the object's weight.

- **Box:** fastest option for walls, door leaves, tabletop pieces and legs.
- **Sphere:** an enclosing sphere for a ball or apple.
- **Cylinder (local Z):** an enclosing cylinder for a can or bottle.
- **Convex mesh:** uses MuJoCo's convex hull. **Holes and cavities are filled.**

For a table, keep its top and legs as separate fixed mesh objects. For shelves or doorways, use separate boards. A single box around an entire table would block the space underneath. A convex hull around a doorway would block the opening.

For a hollow cup, plate, or another object made of several pieces:

1. Mark the detailed surface **Visual only**.
2. Add separate simple meshes for the bottom and walls, and mark them **Collision helper**.
3. Select the visual and all helpers, then click **Group selected parts as one body**.
4. Select the resulting Empty and choose **Movable object**, its weight, or **Door** and its hinge settings.

All child colliders belong to that one body and move together. Body groups can
contain fixed or hinged child groups. Free bodies stay at the top level. Use
separate collision pieces around an opening instead of a convex hull that fills
it. Parent transforms and hierarchy are preserved. Blender rigid-body
constraints are not translated automatically.

Modifiers, nonuniform scale, mirrored scale and parent-induced shear are evaluated into the visible mesh. Object origins and proper rotation remain in the MuJoCo body; you do not need to Apply Scale first. Primitive collision shapes conservatively enclose the evaluated geometry in that body frame. This avoids losing or double-applying scale.

## Materials and baked lighting

The exporter supports a simple **Principled BSDF Base Color** or **Emission Color**, with either a flat colour or one directly connected **Image Texture** using UV coordinates. Multiple material slots and UV seams are preserved. Textures are written as PNG, including packed or unsaved image buffers.

MuJoCo's native viewer does not run Blender shader nodes. Procedural wood, noise, bump, complex material mixes and Cycles lights need to be baked first. Unsupported nodes produce notes in `export_report.json`; they are not silently presented as an exact shader conversion. Ordinary material colours remain a useful preview.

### Bake static lighting

Finish the materials and lights, save the project, then click **Bake static
lighting**. The default creates one **2048 × 2048** diffuse lighting atlas with
**128 samples** and an **8 px** margin. Set the output folder, resolution, and
samples under **World advanced settings**. Larger sample counts reduce noise.
The bake needs the Cycles add-on and a writable bake folder; both are checked
before the Cycles pass starts. **Ctrl + Z** undoes a bake in one step.

The add-on uses Cycles to bake a Combined pass of diffuse colour and emission,
with direct and indirect light, from fixed visual surfaces. Emissive panels and
screens keep their glow. It excludes movable objects, doors, and their
descendants so their current shadows do not become permanent. It retains the
original geometry and materials for editing and preserves separate physics
colliders. Meshes without faces are skipped and listed in the bake report.

> **Metal and glass are approximations.** A diffuse atlas cannot hold
> view-dependent reflections. The bake treats metallic surfaces as diffuse with
> their base colour (instead of baking them black), and leaves glossy
> reflections and transmission out of the atlas. Glass still lets light into
> the room during the bake.

Bake clones keep the object-level inputs of their sources: object colour, ray
visibility, holdout and shadow catcher, pass index, light and shadow linking,
and custom properties read by **Attribute (Object)** nodes. **Object Info →
Random** cannot be reproduced and produces a note.

**Edit original surfaces** asks for confirmation, then brings the original
surfaces back and deletes the atlas and its packed lightmap; the EXR and PNG in
the bake folder stay on disk. Bake again after changing them. A rebake also
restores any part that it no longer bakes, for example one changed to
**Movable object**. If the atlas object is deleted by hand, the export writes
the hidden sources' own visuals again and adds a note.

The output includes a packed half-float linear **EXR master** and an **AgX PNG**.
The PNG always uses AgX with look None, exposure 0 and gamma 1 on the sRGB
display, whatever the scene's view, look, exposure or Output colour management;
the bake summary and the master image record that profile. Projects baked with
an earlier version took the scene's look and exposure, so a rebake can change
their atlas brightness. The visible static atlas uses an Emission material and
has no contact. Moving objects continue to use runtime lighting. Export the
world after baking; saving the Blender file alone does not update MJCF.

The current Chula project already contains its white lights and procedural
materials. Rebaking uses those authored lights and finishes; the add-on contains
no room-specific layout or colour recipe.

### Keep authored worlds editable

A saved Blender project is the authoring source. Exported XML and its asset
folder are the simulation deliverable. Walkie's room sources and exported assets
live in `walkie-ros/simulation/gazebo/worlds/` and are installed with the
`robot_simulation` package. This add-on stays in its separate repository;
`walkie-ros` supplies robot integration and ROS launches.

For an existing Blender scene accompanied by a matching expanded world XML,
**World advanced settings → Adopt existing MJCF physics** associates same-named
body Empties and geometry meshes with its exact masses, compound colliders,
joints, contact parameters, and defaults. This adopts physics onto existing
geometry; it is not a general MJCF geometry importer. Renamed objects can be
mapped through the Python `adopt_mjcf(..., object_map=...)` API. Missing objects
produce an error rather than dropping colliders.

The migrated Chula Blender projects are already adopted. Existing body placement,
primitive dimensions, supported colours/images, mass, and hinge settings can be
edited. New props and grouped collision helpers can be added. Unchanged imported
physics remains exact. Complex procedural material edits need a new static bake.

Adoption checks the whole XML before it changes anything, so a missing object or
a malformed attribute leaves the previous adoption untouched, and **Adopt** can
be undone with Ctrl + Z. Regenerated meshes keep the source mesh's `inertia`,
`maxhullvert` and `smoothnormal`; a visual mesh without its own inertia mode gets
`inertia="shell"`.

When an adopted body is changed to **Movable object** or **Door**, or adopted
parts are grouped into a new body, its **Weight** is split evenly over the body's
colliding geoms. A body that kept its role and weight, or that has an explicit
`<inertial>`, keeps its source masses.

Copies of adopted objects (**Shift + D**) export under their own Blender names;
their joints, sites and other named children are renamed after the copy. The
object that still has the adopted name keeps the original MJCF names. If several
objects claim one adopted name and none of them has it as its Blender name, the
export stops and asks you to rename the original.

### Image export

Simple flat colours and UV Image Textures are supported for Principled Base
Color and Emission. Reroute nodes are followed; muted links count as
unconnected, and a muted shader or image node falls back to the flat colour
with a note. Packed and unsaved images export as PNG. An image without its own
UV Map node uses the mesh's render-active UV map, the one Blender renders with.

- **Baked float images** use the bake's recorded AgX display profile, keeping the HDR master unclipped. A float image on an object marked **Image already contains lighting** that has no recorded profile follows the scene's view transform, look, exposure and gamma; the note in `export_report.json` names the profile used.
- **Ordinary display PNGs** retain their pixel values. An unmodified 16-bit PNG is copied unchanged (MuJoCo reads it as 8-bit).
- **Other float albedo** (an EXR or an unsaved float buffer) is written as an 8-bit sRGB PNG with a note; values above 1 are clipped and noted.
- **Non-Color or linear byte images** are re-encoded to sRGB in **Match PNG appearance** mode and tagged `colorspace="linear"` in **sRGB decode** mode, so they keep Blender's brightness. Both modes add a note, because colour textures should use the sRGB colour space.
- **UDIM (tiled) and missing images** fall back to the flat material colour with a note that names the object, material and image.

### Why texture colour is explicit

The default **Match PNG appearance** writes `colorspace="linear"` on MuJoCo textures. Here that means *do not decode already display-referred PNG byte values again* in the native viewer. This avoids a saved baked image becoming unexpectedly dark. The native viewer's output is not a complete Blender/AgX colour pipeline, so exact Blender and MuJoCo appearance is not guaranteed.

For a custom renderer that decodes sRGB and applies its own final display transform, choose **sRGB decode (custom renderer)** instead. This decision is explicit rather than depending on a PNG's embedded metadata: Blender 5.2 writes an sRGB chunk into PNGs and 4.5 does not, which would flip MuJoCo's `auto` setting. Adopted exports therefore also write the mode's colour space on retained textures that had none. See MuJoCo's [texture colour-space reference](https://mujoco.readthedocs.io/en/stable/XMLreference.html#asset-texture-colorspace).

## Open the exported scene

In a Python environment with MuJoCo 3.15+ installed:

```bash
python -m mujoco.viewer --mjcf=/absolute/path/to/scene.xml
```

On this Linux machine, the native viewer needs the X11 workaround:

```bash
env -u WAYLAND_DISPLAY XDG_SESSION_TYPE=x11 \
  /home/tawan/walkie-ros/.pixi/envs/default/bin/python -m mujoco.viewer \
  --mjcf=/absolute/path/to/scene.xml
```

This command opens the environment. Robot models, ROS controllers and drive controls must be added separately.

## Walkie and omni wheels

The exported `floor` is an actual **MuJoCo plane**, even if a detailed visual floor lies at the same height. No triangle floor mesh participates in collision. In a robot model, preserve the robot's wheel-capsule/plane contact pairs and anisotropic friction; this exporter does not guess robot-specific wheel contacts.

When merging into the current Walkie model, use its existing collision-mask scheme. Click **World advanced settings → Use Walkie contact masks**, or enter the Chula-world values:

| Setting | Value |
| --- | --- |
| Plane contact type | `1` |
| Plane contact mask | `32` |
| Fixed contact type | `64` |
| Fixed contact mask | `56` |
| Door contact type | `32` |
| Door contact mask | `121` |
| Movable contact type | `32` |
| Movable contact mask | `121` |

`robot_bringup.world_loader` gives every geom that moves with a joint (doors,
movable objects and the parts grouped inside them) the movable masks, so the
preset does the same and an exported world behaves like the merged one. Empty
**Door contact type/mask** fields use the fixed values; projects that applied
the preset before these fields existed need to click it again.

Do not add a second plane named `floor` when merging with a robot scene that already supplies it. Walkie's ROS launch uses the installed `robot_bringup.world_loader` library to combine the exported room with the robot and preserve its four explicit wheel/floor pairs. An isolated exported world defaults to masks `1/1` so ordinary free objects collide normally.

## Scope and verification

This is an environment exporter, not a general Blender physics, URDF or robot exporter. It does not translate Blender rigid-body constraints, animations, Geometry Nodes instances, collection instances, procedural shader graphs, or transparency pipelines. Cycles lighting and procedural colour can be baked into the static atlas. Convert curves/text to mesh and realize instances before export. Hollow movable containers need grouped compound helpers; a single convex collider still fills their cavities.

The included `example.blend` is a small test scene. Additional tests exercise grouped compound bodies, retained imported physics, editable doors/props, and static baking with collision and HDR preservation. `tests/build_smoke_scene.py` generates a fresh export in Blender; `tests/validate_export.py` compiles it in MuJoCo and checks geometry, floor contact policy, preserved weight, texture buffers, selection-only export, portability and the door's torque-driven movement. Results are saved in `tests/output/validation_report.json`; the generated scene is saved as `tests/output/example.blend`, so the tracked `example.blend` is not rewritten. `tests/test_export_fixes_blender.py OUTPUT` and `tests/validate_export_fixes.py OUTPUT [WALKIE_ROS_ROOT]` cover radian angles, inertia settings that merge cleanly with a robot, contact mask formats, body name clashes, excluded body groups, and adopted `fromto` primitives. `tests/test_adopt_blender.py OUTPUT` and `tests/validate_adopt.py OUTPUT` build a small adopted world and cover all-or-nothing adoption, carried mesh inertia, duplicated adopted objects, in-place export, viewport-disabled objects, Weight on adopted bodies, the hinge cursor on shared meshes, failed exports, shared asset folders, and door contact masks. `tests/test_shared_assets_blender.py ROOM OUTPUT` exports both Chula variants into one shared folder; pass a writable copy of the room, because it adopts and exports inside it. `tests/test_review_fixes_blender.py OUTPUT` and `tests/validate_review_fixes.py OUTPUT` cover metal and emission in the bake, the pinned lightmap profile under Override and video output, stale bake marks, bake failure cleanup, faceless meshes, clone shading inputs, the bake's early checks, render-active UV maps, 16-bit/float/UDIM/missing/Non-Color images, reroutes and muted nodes, adopted texture colour spaces, and the operators' linked-data, restore and error handling.

The OBJ/MJCF export uses documented [MuJoCo mesh](https://mujoco.readthedocs.io/en/stable/XMLreference.html#asset-mesh) and [hinge joint](https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-joint) conventions. No external service or asset library is required.

## Share assets between variants

Set **World advanced settings → Shared asset folder** when several exported
worlds should reuse one asset library. Leave it empty for the usual directory
beside each XML. Relative folders are resolved from the saved Blender project.
The Chula low-poly project uses `//scene_assets`; its baked project uses
`//../scene_assets`. Keep the shared library with both XMLs when sharing.

An export never changes a file in the shared folder. A file keeps its usual name
when that name is free or already holds the same bytes; otherwise it is written
as `mesh_<hash>` or `texture_<hash>` after its content, so a world that already
references the old file still loads it unchanged. Files no world uses any more
are not deleted automatically.

To build the installation ZIP from this source project:

```bash
python3 package_addon.py
```
