"""Blender fixture: hollow compound, nested hinge, live imported edits.

Run:
  blender --background --factory-startup --python-exit-code 1 \
      --python tests/test_compound_blender.py -- OUTPUT_DIR [MJCF BLEND]
Then validate the written worlds with MuJoCo:
  python tests/validate_compound.py OUTPUT_DIR

An optional MJCF/BLEND pair of an existing project is adopted and edited
too. That project is opened but never saved, and its exports use a shared
asset folder inside OUTPUT_DIR, so nothing outside OUTPUT_DIR is written.
"""
import json
import math
from pathlib import Path
import sys

import bpy

sys.dont_write_bytecode = True  # Blender ignores PYTHONDONTWRITEBYTECODE.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness  # noqa: E402

out, args = harness.begin("compound", "OUTPUT_DIR [MJCF BLEND]", 1, 3)
if len(args) == 1:
    raise SystemExit("Pass both the MJCF and the BLEND file of the existing project, or neither")
from mujoco_world.exporter import export_world  # noqa: E402
from mujoco_world.project import adopt_mjcf, group_selected  # noqa: E402

bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)
scene = bpy.context.scene
scene.unit_settings.scale_length = 1.0

def box(name, pos, half, role="STATIC", parent=None, angle=0):
    bpy.ops.mesh.primitive_cube_add(size=2, location=pos)
    obj = bpy.context.object
    obj.name, obj.scale = name, half
    obj.rotation_euler.z = angle
    obj.mujoco_world.role = role
    bpy.context.view_layer.update()
    if parent:
        world = obj.matrix_world.copy()
        obj.parent = parent
        obj.matrix_world = world
    return obj

def empty(name, pos, role="STATIC", parent=None):
    obj = bpy.data.objects.new(name, None)
    bpy.context.collection.objects.link(obj)
    obj.location = pos
    obj.mujoco_world.body = True
    obj.mujoco_world.role = role
    obj.parent = parent
    return obj

# A cup keeps an open cavity through thirteen primitive helpers on one free body.
cup = empty("Hollow_cup", (0, 0, 0.01), "FREE")
cup.mujoco_world.mass = 0.4
bpy.ops.mesh.primitive_cylinder_add(vertices=24, radius=0.16, depth=0.04, location=(0, 0, 0.04))
base = bpy.context.object
base.name = "Cup_bottom"
base.mujoco_world.role, base.mujoco_world.collision = "COLLIDER", "CYLINDER"
world = base.matrix_world.copy()
base.parent, base.matrix_world = cup, world
for i in range(12):
    angle = 2 * math.pi * i / 12
    box(f"Cup_rim_{i}", (0.145 * math.cos(angle), 0.145 * math.sin(angle), 0.14), (0.015, 0.043, 0.08), "COLLIDER", cup, angle)
box("Cup_visual", (0, 0, 0.04), (0.16, 0.16, 0.02), "VISUAL", cup)

# Nested body hierarchy: a fixed assembly owns a hinged door at its local pivot.
assembly = empty("Assembly", (1.0, 0.2, 0))
assembly.rotation_euler.z = 0.35
door = empty("Nested_door", (0, 0, 0), "HINGE", assembly)
door.mujoco_world.mass = 3
door.mujoco_world.hinge_start = -85
door.mujoco_world.hinge_min = -110
door.mujoco_world.hinge_max = 0
scene.cursor.location = (1.0, 0.2, 0)
bpy.context.view_layer.update()
panel = box("Door_panel", (0, 0, 0), (0.3, 0.025, 0.8), "STATIC")
panel.parent = door
panel.location = (0.3, 0, 0.85)
floor = box("Decorative_floor", (0, 0, -0.01), (2, 2, 0.01), "VISUAL")
bpy.context.view_layer.update()
report = export_world(bpy.context, str(out / "fresh.xml"))
assert report["compound_bodies"] == 2

# Generic grouping preserves transforms and never joins meshes or fills holes.
bpy.ops.object.select_all(action="DESELECT")
group_parts = [box("Grouped_piece_A", (-1, 0, 0.4), (0.06, 0.06, 0.06)), box("Grouped_piece_B", (-1.2, 0, 0.4), (0.04, 0.04, 0.04), "COLLIDER")]
bpy.context.view_layer.update()
before = [o.matrix_world.copy() for o in group_parts]
for obj in group_parts:
    obj.select_set(True)
group = group_selected(bpy.context)
group.name = "Grouped_prop"
group.mujoco_world.mass = 0.5
bpy.context.view_layer.update()
for old, obj in zip(before, group_parts):
    assert max(abs(old[i][j] - obj.matrix_world[i][j]) for i in range(4) for j in range(4)) < 1e-6
export_world(bpy.context, str(out / "grouped.xml"))

if args:
    bpy.ops.wm.open_mainfile(filepath=args[1])
    adopt_mjcf(bpy.context, args[0])
    # Adoption points the asset folder at the project's own shared folder;
    # keep every export of this test inside the output directory instead.
    bpy.context.scene.mujoco_world.asset_directory = str(out / "shared_assets")
    untouched = export_world(bpy.context, str(out / "adopted.xml"))
    # Select one real free body and edit its mass and placement.
    prop = next(o for o in bpy.context.scene.objects if o.type == "EMPTY" and o.mujoco_world.role == "FREE")
    baseline = json.loads(prop["mjworld_settings_baseline"])
    old_pos = list(prop.location)
    prop.location.x += 0.123
    prop.mujoco_world.mass = baseline["mass"] * 1.5
    hinge = next(o for o in bpy.context.scene.objects if o.type == "EMPTY" and o.mujoco_world.role == "HINGE")
    hinge.mujoco_world.hinge_min = -100
    hinge.mujoco_world.hinge_start = -45
    hinge.mujoco_world.damping = 0.75
    # Change the door's supported flat appearance without touching its frame.
    surface = next(o for o in hinge.children if o.type == "MESH" and o.mujoco_world.role == "STATIC")
    material = surface.material_slots[0].material.copy()
    surface.material_slots[0].material = material
    material.node_tree.nodes.get("Principled BSDF").inputs["Base Color"].default_value = (0.01, 0.6, 0.08, 1)
    new = box("New_prop", (2, -2, 1), (0.05, 0.04, 0.06), "FREE")
    new.mujoco_world.mass = 0.25
    new_door = empty("New_group_door", (2.5, -2, 0), "HINGE")
    new_door.mujoco_world.mass = 2.5
    new_door.mujoco_world.hinge_start = -30
    piece = box("New_door_panel", (0, 0, 0), (.25, .02, .7))
    piece.parent = new_door
    piece.location = (.25, 0, .75)
    bpy.context.view_layer.update()
    export_world(bpy.context, str(out / "edited.xml"))
    (out / "edits.json").write_text(json.dumps({"body": prop.name, "old_position": old_pos, "mass": prop.mujoco_world.mass, "door": hinge.name, "door_geom": json.loads(surface["mjworld_mjcf_geom"])["name"]}, indent=2))

harness.complete(out, "compound", real_project=bool(args))
print("COMPOUND_FIXTURE_OK", str(out))
