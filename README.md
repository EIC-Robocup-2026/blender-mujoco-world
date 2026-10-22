# MuJoCo World — Easy Export

A Blender add-on for authoring, baking, and exporting portable MuJoCo environments. It has no Python package dependencies inside Blender. Tested with **Blender 4.5.14 LTS** and **Blender 5.2.1 LTS** and **MuJoCo 3.15.0**. It declares Blender 4.2 as its minimum; Blender 4.2 to 4.4 are not tested.

## Install once

1. Build the installation ZIP from this repository (the ZIP is not committed):

   ```bash
   python3 package_addon.py
   ```

   This writes `dist/mujoco_world-<version>.zip`, for example `dist/mujoco_world-0.2.0.zip`.
2. Open **Edit → Preferences → Add-ons** in Blender.
3. Open the menu at the top right and choose **Install from Disk…**.
4. Choose the ZIP. If **MuJoCo World — Easy Export** is not enabled afterwards, tick it in the add-on list.
5. In the 3D View, press **N** and open the **MuJoCo** tab.

To update, build a new ZIP and install it the same way; Blender replaces the previous copy. The add-on is a legacy add-on (`bl_info`), which Blender 4.2 and later still install from a ZIP.

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
4. Click **Export world…** (or **File → Export → MuJoCo World (.xml)**) and save `scene.xml` in a new folder.
5. Keep the XML and its `scene_assets/` directory together. The exporter writes a small `export_report.json` with any notes; [What an export writes](#what-an-export-writes) lists every file.

Visible mesh objects export by default. Choose **Leave out** to exclude an object, or enable **Export selected objects only**. **Leave out** on a body group Empty also excludes every part inside it. A fixed or visual mesh without faces (guide edges or points) is left out with a note in `export_report.json`. Curves, surfaces, text and metaballs are left out until converted to mesh, and instances are not realized: collection, vertex, face and particle instancing, and Geometry Nodes instances on a mesh (only the host's own mesh exports). Each such visible object gets a note, in ordinary and adopted worlds alike. Collision helpers belonging to a bake or adopted world remain exportable while hidden. Blender lights illuminate the static bake; they are not directly converted into MuJoCo lights in a newly authored world.

Folders that start with `//` (the bake folder, a shared asset folder, the export path) are relative to the `.blend` file, so save the file first; until then the add-on refuses them instead of writing next to wherever Blender was started.

Objects linked from another `.blend` file are skipped by the role buttons, **Group selected parts as one body** and **Set hinge at 3D cursor**, because Blender discards edits to linked data on reload. Make a library override first; the MuJoCo object settings are editable on overrides.

Hide such helpers, adopted objects and body group Empties with the eye icon (**Hide in Viewport**). Blender does not update the placement or modifiers of an object that is disabled in viewports (monitor icon) or sits in a collection excluded from the view layer, so the export stops with an error that names those objects instead of writing stale geometry.

A failed export leaves the previous `scene.xml` and `export_report.json` in place. Files are written next to their targets first and replace them only after the whole world was built, so an export that fails while building, the usual case, changes no existing file. The new asset files then move in first, then the XML and report, and only then are the previous export's unused files removed. So even when moving an asset or replacing the XML itself fails, every file the old XML names still exists, but a file the new export rewrote under the same name, such as an edited mesh, may already hold the new content; export again once the cause is fixed. In a shared asset folder, new files it had already added stay behind, but no existing file changes.

For a joined room mesh with one baked atlas, choose **Visual only** in its Object type dropdown. Build separate simple boxes for the walls, furniture tops and legs, choose **Collision helper** in each box's dropdown, and keep those helpers visible in Blender: they switch to wire display and export only invisible fixed collision, so they do not cover the baked appearance. Hiding a helper excludes it entirely.

## What an export writes

Exporting `<name>.xml` writes these files:

| File | Written by | Contents |
| --- | --- | --- |
| `<name>.xml` | every export | The MuJoCo world. Asset paths are relative to it. |
| `<name>_assets/` beside the XML, or the **Shared asset folder** | every export | OBJ meshes and PNG textures the world references. See [Share assets between variants](#share-assets-between-variants). |
| `<name>_assets/.mujoco_world_manifest.json` | exports into the per-XML asset folder | The files this world wrote and their SHA-256 digests. The next export replaces only these files and removes the unchanged ones it no longer writes. A shared folder gets no manifest. |
| `export_report.json` beside the XML | every export | Add-on and Blender versions, object and triangle counts, the floor plane, the texture mode, per-object details and every note or warning. |
| `OPEN_WORLD.txt` beside the XML | exports of new (not adopted) worlds | How to open the world in MuJoCo's viewer and what it does not contain. |

`export_report.json` and `OPEN_WORLD.txt` have fixed names, so give each exported world its own folder; a second XML in the same folder replaces the first world's report.

**Bake static lighting** writes into its **Bake folder** (default `//baked_lighting`, beside the `.blend` file):

| File | Contents |
| --- | --- |
| `static_lighting_linear.exr` | Half-float linear HDR master of the lighting atlas. It is also packed into the `.blend` file. |
| `static_lightmap.png` | The atlas as an AgX display PNG; the export writes the atlas texture from the same profile. |

The export browser of a saved `.blend` file starts at `//mujoco_export/scene.xml`, and afterwards at the last export.

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

All child colliders belong to that one body and move together. Its **Weight** is
split over the colliders in proportion to their volume (a mesh collider counts
its convex hull), so the centre of mass sits where the material is. Body groups can
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

Metal is found in Principled BSDFs (any Metallic value or link), in Metallic
BSDF nodes, which bake as a Diffuse BSDF of their base colour, and inside node
groups. A node group with metal inside is copied for the bake and the copy is
deleted afterwards, so your groups never change. A Glossy BSDF is not
converted: it contributes no diffuse colour, so a surface made only of one
bakes dark. Use a Principled BSDF with Metallic for metal you want in the
atlas.

Bake clones keep the object-level inputs of their sources: object colour, ray
visibility, holdout and shadow catcher, pass index, light and shadow linking,
and custom properties read by **Attribute (Object)** nodes. **Object Info →
Random** cannot be reproduced and produces a note. For light and shadow
linking, each clone joins the receiver and blocker collections that hold its
source, with the source's include or exclude state, and leaves them again right
after the bake, also when the bake fails. A linking collection linked from a
library cannot be edited, so its sources get a note and their clones are lit as
if they were not linked.

**Edit original surfaces** asks for confirmation, then brings the original
surfaces back and deletes the atlas and its packed lightmap; the EXR and PNG in
the bake folder stay on disk. Bake again after changing them. A rebake also
restores any part that it no longer bakes, for example one changed to
**Movable object**. The atlas references its sources as objects, so renaming
a source after baking is fine. If the atlas object is deleted by hand, or a
source is copied after baking (**Shift + D** copies the bake mark, but the
atlas does not contain the copy), the export writes those hidden sources' own
visuals again and adds a note.

The output includes a packed half-float linear **EXR master** and an **AgX PNG**.
The PNG always uses AgX with look None, exposure 0 and gamma 1 on the sRGB
display, whatever the scene's view, look, exposure or Output colour management;
the bake summary and the master image record that profile. Projects baked with
an earlier version took the scene's look and exposure, so a rebake can change
their atlas brightness. The visible static atlas uses an Emission material and
has no contact. Moving objects continue to use runtime lighting. Export the
world after baking; saving the Blender file alone does not update MJCF.

A rebake uses the lights and materials authored in the project; the add-on
contains no room-specific layout or colour recipe.

### Keep authored worlds editable

A saved Blender project is the authoring source. Exported XML and its asset
folder are the simulation deliverable; keep both under version control next to
the project that uses them. Robot integration and launch files belong to the
robot's own repository, not to this add-on.

For an existing Blender scene accompanied by a matching expanded world XML,
**World advanced settings → Link existing world physics…** associates same-named
body Empties and geometry meshes with its exact masses, compound colliders,
joints, contact parameters, and defaults. This adopts physics onto existing
geometry; it is not a general MJCF geometry importer. Renamed objects can be
mapped through the Python `adopt_mjcf(..., object_map=..., body_map=...,
geom_map=...)` API. `object_map` applies to bodies and geoms alike; `body_map`
and `geom_map` apply to one kind and take precedence, so a body and a geom that
share an MJCF name can go to different objects. Missing objects produce an error
rather than dropping colliders, and so do two MJCF elements that match one
object. Unnamed geoms cannot be matched, except a single unnamed world geom on a
baked atlas.

The XML must be standalone. `<include>`, `<frame>` and `<replicate>` are
refused, because the export would write their contents twice; expand or flatten
them first. Geom types and sizes may come from `<default>` classes. Slide and
ball joints are kept exactly as in the source; the door settings (limits,
damping, axis) apply to a body's hinge, and a joint without a `type` is a hinge,
as in MuJoCo. Asset files are found the way MuJoCo finds them, following the
source compiler's `assetdir`, `meshdir`, `texturedir` and `strippath`, including
cube and skybox textures (`fileright` … `fileback`), height fields and skins.

Once a project is adopted, existing body placement, primitive dimensions,
supported colours/images, mass, and hinge settings can be edited. New props and
grouped collision helpers can be added. Unchanged imported physics remains
exact. Complex procedural material edits need a new static bake.

Adoption checks the whole XML before it changes anything, so a missing object or
a malformed attribute leaves the previous adoption untouched, and **Link
existing world physics…** can be undone with Ctrl + Z. Regenerated meshes keep the source mesh's `inertia`,
`maxhullvert` and `smoothnormal`; a visual mesh without its own inertia mode gets
`inertia="shell"`. Regenerated meshes are named after their Blender object. When
two names become the same in MJCF characters, or a name has none of them (Thai
names, for example), a short hash of the Blender name keeps the mesh names apart.

Asset files are read beside the adopted XML. The adoption records that path
relative to the `.blend` file and also as an absolute path, so a blend moved
later still finds them; if neither exists, the export stops and names the folder
it looked in. The absolute path is stored in the `.blend` file and shows the
folder layout of the machine that adopted the world.

The export copies the texture, height field and skin files the adopted XML
names into the asset folder, so it checks them first. An XML from someone else
could otherwise name any file on your computer, such as an SSH key, through an
absolute or `..` path, and the export would copy it next to your world. Each
copied file must:

- lie, after following symbolic links, inside the adopted XML's folder or inside
  the export's asset folder (a variant that names its shared library, such as
  `baked/scene.xml` naming `../scene_assets`, passes when that library is
  the **Shared asset folder**);
- end in a suffix MuJoCo reads for its element: `.png` for textures, `.png` or
  `.bin` for height fields, `.skn` for skins; a `.png` must hold PNG data.

Other asset elements that name a file, such as `<model>`, stop the export. Source
mesh files are never read or copied, because every mesh is regenerated from the
Blender object. For a world you trust whose files sit elsewhere, turn on **World
advanced settings → Allow adopted assets outside the MJCF folder**. The export
then copies those files and lists each one as a warning in `export_report.json`.
The suffix and PNG checks still apply.

These checks protect you from an XML from someone else, the file you adopt. They
do not make a `.blend` from someone else safe to export. The `.blend` stores the
adopted XML, the path the check takes the world's folder from, and this setting,
so a crafted one can allow any folder without a warning. Any `.blend` can also
point an image at a local file, which an export writes into the asset folder.
Export only `.blend` files you trust.

Blender objects are often modelled with the origin at their base, while MuJoCo
places a primitive at its centre. An adopted box, sphere, cylinder, capsule or
ellipsoid whose object origin is elsewhere keeps that offset: the untouched geom
stays where the source put it, and an edited one moves with its object. Only the
position offset is kept; the geom's orientation follows the object. A stretched
capsule keeps its round caps: the radius follows the narrow axes and the length
the long one. A capsule scaled shorter than it is wide stops the export; use a
sphere or a convex helper there.

The export follows the Blender hierarchy. When an adopted geom's object sits
under a different body than in the source XML, adoption and export both add a
note naming the two bodies, and the geom or body keeps its source class through
an explicit `class` or `childclass`, so its contact parameters do not change. A
moving body that had colliders in the source but has none left gets a warning,
because it would pass through everything. A door whose adopted body has only
slide or ball joints keeps them, with a note that its door settings were not
applied.

When an adopted body is changed to **Movable object** or **Door**, or adopted
parts are grouped into a new body, its **Weight** is split over the body's
colliding geoms in proportion to their volume (a mesh geom counts its convex
hull; if one volume is unknown, the split is even). Each geom gets an explicit
`mass`, so the total stays exact whatever density the source used. A body that
kept its role and weight, or that has an explicit `<inertial>`, keeps its source
masses.

Copies of adopted objects (**Shift + D**) export under their own Blender names;
their joints, sites and other named children are renamed after the copy. The
object that still has the adopted name keeps the original MJCF names. If several
objects claim one adopted name and none of them has it as its Blender name, the
export stops and asks you to rename the original.

New objects in an adopted world export as new bodies, or as new geoms of the
adopted body they are parented to, with the names a new world gives them
(`Lamp_000`, `Lamp_000_free`, `Lamp_000_collision` and so on). A new body group
exports as a body named after its Empty, with the joint `<name>_free` or
`<name>_hinge` when it moves; an adopted body changed to **Movable object** or
**Door** gets that joint name too. When one of these names is already a body,
geom, joint or site name in the adopted world, the export stops and names the
objects to rename, because MuJoCo rejects a repeated name. A renamed adopted
object keeps its MJCF names, so a new object that takes its old Blender name
clashes with it.

### Image export

Simple flat colours and UV Image Textures are supported for Principled Base
Color and Emission. Reroute nodes are followed; muted links count as
unconnected, and a muted shader or image node falls back to the flat colour
with a note. Packed and unsaved images export as PNG. An image without its own
UV Map node uses the mesh's render-active UV map, the one Blender renders with.

- **Baked float images** use the bake's recorded AgX display profile, keeping the HDR master unclipped. A float image on an object marked **Image already contains lighting**, or in an Emission shader, that has no recorded profile follows the scene's view transform, look, exposure and gamma; the note in `export_report.json` names the profile used. Such an HDR image is an EXR, HDR or unsaved float buffer.
- **Ordinary display PNGs** retain their pixel values. An unmodified 16-bit PNG in the sRGB colour space is copied unchanged (MuJoCo reads it as 8-bit), also when it is a lightmap in an Emission shader or on an object marked **Image already contains lighting**: its values are display values already, so it gets no view transform and no HDR note.
- **Other float albedo** (an EXR or an unsaved float buffer) is written as an 8-bit sRGB PNG with a note; values above 1 are clipped and noted.
- **Non-Color or linear byte images** are re-encoded to sRGB in **Match PNG appearance** mode and tagged `colorspace="linear"` in **sRGB decode (custom renderer)** mode, so they keep Blender's brightness. Both modes add a note, because colour textures should use the sRGB colour space.
- **UDIM (tiled) and missing images** fall back to the flat material colour with a note that names the object, material and image.

### Why texture colour is explicit

The default **Match PNG appearance** writes `colorspace="linear"` on MuJoCo textures. Here that means *do not decode already display-referred PNG byte values again* in the native viewer. This avoids a saved baked image becoming unexpectedly dark. The native viewer's output is not a complete Blender/AgX colour pipeline, so exact Blender and MuJoCo appearance is not guaranteed.

For a custom renderer that decodes sRGB and applies its own final display transform, choose **sRGB decode (custom renderer)** instead. This decision is explicit rather than depending on a PNG's embedded metadata: Blender 5.2 writes an sRGB chunk into PNGs and 4.5 does not, which would flip MuJoCo's `auto` setting. Adopted exports therefore also write the mode's colour space on retained textures that had none. See MuJoCo's [texture colour-space reference](https://mujoco.readthedocs.io/en/stable/XMLreference.html#asset-texture-colorspace).

Baked and runtime surfaces still differ in two ways. The atlas PNG uses the AgX
curve, while flat colours and ordinary textures on movable props use the plain
sRGB curve, so props can look more saturated or contrasted than the baked room
around them. And MuJoCo's headlight (ambient 0.3, diffuse 0.4 in a new world)
still lights the baked atlas on top of its **Baked texture brightness** emission
(default 0.2), so the baked shading is multiplied again by light that changes
with the viewing angle, and **Match PNG appearance** cannot hold exactly for
baked surfaces. A higher **Baked texture brightness** lets the baked shading
dominate, at the cost of a brighter room.

## Open the exported scene

In a Python environment with MuJoCo 3.15+ installed:

```bash
python -m mujoco.viewer --mjcf=/absolute/path/to/scene.xml
```

On a Wayland desktop, the native viewer may need to run under X11:

```bash
env -u WAYLAND_DISPLAY XDG_SESSION_TYPE=x11 \
  python -m mujoco.viewer --mjcf=/absolute/path/to/scene.xml
```

This command opens the environment. Robot models, ROS controllers and drive controls must be added separately.

## Use with a robot

The exported `floor` is an actual **MuJoCo plane**, even if a detailed visual floor lies at the same height. No triangle floor mesh participates in collision. When you merge the world into a robot model:

- Keep the single plane named `floor`, and do not add a second one if the robot scene already supplies it.
- Preserve the robot's wheel/plane contact pairs and anisotropic friction; this exporter does not guess robot-specific wheel contacts.
- A new world writes angles in radians (`<compiler angle="radian">`), the convention of robot models generated from URDF; an adopted world keeps its source's angle unit.
- Set the contact type and mask of the plane, fixed, door and movable geoms under **World advanced settings** to match the robot's collision classes. Empty **Door contact type/mask** fields use the fixed values.

An isolated exported world defaults to masks `1/1`, so ordinary free objects collide normally.

For the Walkie robot, **World advanced settings → Use Walkie contact masks** sets its collision classes. Walkie's ROS launch reassigns the same classes when it merges a world with the robot, so an exported world behaves the same on its own and merged; see Walkie's [collision classes](https://eic-robocup-2026.github.io/walkie-ros/guides/simulation-mujoco/#collision-classes). Projects that applied the preset before the door fields existed need to click it again.

## Scope

This is an environment exporter, not a general Blender physics, URDF or robot exporter. It does not translate Blender rigid-body constraints, animations, Geometry Nodes instances, collection instances, procedural shader graphs, or transparency pipelines. Cycles lighting and procedural colour can be baked into the static atlas. Convert curves/text to mesh and realize instances before export. Hollow movable containers need grouped compound helpers; a single convex collider still fills their cavities.

The OBJ/MJCF export uses documented [MuJoCo mesh](https://mujoco.readthedocs.io/en/stable/XMLreference.html#asset-mesh) and [hinge joint](https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-joint) conventions. No external service or asset library is required.

## Share assets between variants

Set **World advanced settings → Shared asset folder** when several exported
worlds should reuse one asset library. Leave it empty for the usual directory
beside each XML. Relative folders are resolved from the saved Blender project.
For example, `room.blend` can use `//scene_assets` and `baked/room_baked.blend`
`//../scene_assets`, so both export into the same folder. Keep the shared
library with both XMLs when sharing.

An export never changes a file in the shared folder. A file keeps its usual name
when that name is free or already holds the same bytes; otherwise it is written
as `mesh_<hash>` or `texture_<hash>` after its content, so a world that already
references the old file still loads it unchanged. Files no world uses any more
are not deleted automatically.

The usual per-XML asset folder belongs to one world. The export lists the files
it wrote, with their SHA-256 digests, in `.mujoco_world_manifest.json` inside
that folder, and the next export removes listed files it no longer writes, such
as the mesh of a renamed or deleted object. A listed file is removed only when
it is still a regular file with the recorded content. A file you edited since,
a file the manifest does not list, and a symbolic link (and whatever it points
to) are left alone, so stale files from an export made before the manifest
recorded digests stay until you delete them.

A per-XML export replaces only the files its manifest lists and creates absent
ones. Any other file already in the folder is not this world's: another world
may use the folder as its shared asset folder, an older add-on version wrote
it, or it is an adopted world's own source file. The export references such a
file as it is when it holds the same bytes, and otherwise writes the new
content beside it as `mesh_<hash>` or `texture_<hash>`. If that name is taken
by other content too, the export stops with an error instead of replacing it.
Within one export, two files of the same name with different content (a
texture and a height field both called `terrain`, say) are kept apart the same
way. No export writes through a symbolic link: a per-XML export replaces a
listed link with its own file and treats any other link as taken, and in a
shared folder a link counts as a taken name.

## Development

The tests run real exports in Blender and compile the results with MuJoCo.
`tests/run_all.sh` runs all of them into a fresh temporary folder outside the
checkout and exits non-zero on any failure:

```bash
BLENDER=/path/to/blender PYTHON=/path/to/python3 tests/run_all.sh
```

[tests/README.md](tests/README.md) lists the requirements, the optional inputs
and what each test covers.
