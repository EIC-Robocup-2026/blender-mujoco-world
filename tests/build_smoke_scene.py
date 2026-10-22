"""Smoke export of a generated scene: transforms, textures, roles, floor plane options.

Run:
  blender --background --factory-startup --python-exit-code 1 \
      --python tests/build_smoke_scene.py -- OUTPUT_DIR
Then validate the written worlds with MuJoCo:
  python tests/validate_export.py OUTPUT_DIR [WALKIE_ROS_ROOT]

Every scene is generated here; nothing outside OUTPUT_DIR is written. The
generated scene is saved as OUTPUT_DIR/smoke_scene.blend for inspection.
"""
import json
from pathlib import Path
import sys

import bpy

sys.dont_write_bytecode = True  # Blender ignores PYTHONDONTWRITEBYTECODE.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness  # noqa: E402

OUT, _ = harness.begin("smoke")
from mujoco_world.exporter import export_world  # noqa: E402

bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)
bpy.context.scene.unit_settings.system = "METRIC"
bpy.context.scene.unit_settings.scale_length = 0.01


def box(name, pos, scale, role="STATIC", rot=(0, 0, 0)):
    bpy.ops.mesh.primitive_cube_add(size=2, location=pos, rotation=rot)
    obj = bpy.context.object
    obj.name = name
    obj.scale = scale
    obj.mujoco_world.role = role
    return obj


floor = box("Decorative_floor", (0, 0, -1), (250, 200, 1), "VISUAL")
# Include a truly planar mesh as another visual, which must compile with no hull.
bpy.ops.mesh.primitive_plane_add(size=80, location=(-100, 100, 0.1))
flat = bpy.context.object
flat.name = "Flat_decorative_floor"
flat.mujoco_world.role = "VISUAL"
image = bpy.data.images.new("Baked_test", width=32, height=32)
pixels = []
for y in range(32):
    for x in range(32):
        pixels.extend([x / 31, y / 31, 0.4, 1])
image.pixels[:] = pixels
material = bpy.data.materials.new("Baked emission image")
if bpy.app.version < (5, 0, 0):
    material.use_nodes = True  # Blender 5 always uses the node tree.
nodes = material.node_tree.nodes
nodes.clear()
texture = nodes.new("ShaderNodeTexImage")
texture.image = image
emission = nodes.new("ShaderNodeEmission")
output = nodes.new("ShaderNodeOutputMaterial")
material.node_tree.links.new(texture.outputs["Color"], emission.inputs["Color"])
material.node_tree.links.new(emission.outputs[0], output.inputs["Surface"])
flat.data.materials.append(material)

hdr = bpy.data.images.new("Unsaved_HDR_light_bake", width=16, height=16, float_buffer=True)
hdr_pixels = []
for y in range(16):
    for x in range(16):
        value = [0.1, 1, 4, 16][x // 4]
        hdr_pixels.extend([value, value, value, 1])
hdr.pixels[:] = hdr_pixels
hdr_material = bpy.data.materials.new("HDR bake emission")
if bpy.app.version < (5, 0, 0):
    hdr_material.use_nodes = True
hdr_nodes = hdr_material.node_tree.nodes
hdr_nodes.clear()
hdr_texture = hdr_nodes.new("ShaderNodeTexImage")
hdr_texture.image = hdr
hdr_emission = hdr_nodes.new("ShaderNodeEmission")
hdr_output = hdr_nodes.new("ShaderNodeOutputMaterial")
hdr_material.node_tree.links.new(hdr_texture.outputs["Color"], hdr_emission.inputs["Color"])
hdr_material.node_tree.links.new(hdr_emission.outputs[0], hdr_output.inputs["Surface"])
hdr_patch = box("HDR_visual_patch", (100, -100, 0.2), (30, 30, 0.1), "VISUAL")
hdr_patch.data.materials.append(hdr_material)
bpy.context.scene.view_settings.view_transform = "AgX"
bpy.context.scene.view_settings.exposure = -0.7

wall = box("Fixed_wall", (0, 190, 100), (250, 10, 100))
helper = box("Collision_helper", (200, 80, 40), (10, 30, 40), "COLLIDER")
helper.data.materials.append(material)  # Must not export its image or visible mesh.
assert helper.display_type == "WIRE"
door = box("Door", (0, -100, 100), (35, 2, 100), "HINGE")
# Shift the geometry, not its world origin; hinge stays at x=0 along the left edge.
for v in door.data.vertices:
    v.co.x += 1
door.mujoco_world.mass = 6

prop = box("Free_mirrored_box", (-80, 0, 35), (-12, 18, 10), "FREE", rot=(0.2, -0.3, 0.6))
prop.mujoco_world.mass = 0.8
parent = bpy.data.objects.new("Scaled_parent", None)
bpy.context.collection.objects.link(parent)
parent.location = (80, 0, 25)
parent.scale = (2, 0.7, 1.4)
parent.rotation_euler = (0.1, 0.2, 0.5)
shear = box("Parent_shear_box", (0, 0, 0), (10, 14, 12), "FREE", rot=(0.3, 0.4, 0.2))
shear.parent = parent
shear.location = (0, 0, 30)
shear.mujoco_world.mass = 1.2

bpy.ops.mesh.primitive_uv_sphere_add(segments=12, ring_count=6, radius=15, location=(0, 90, 40))
sphere = bpy.context.object
sphere.name = "Free_sphere"
sphere.scale = (1.2, 0.8, 1.0)
sphere.mujoco_world.role = "FREE"
sphere.mujoco_world.collision = "SPHERE"
bpy.ops.mesh.primitive_cylinder_add(vertices=12, radius=12, depth=30, location=(70, 95, 45), rotation=(0.2, 0.4, 0))
cylinder = bpy.context.object
cylinder.name = "Free_cylinder"
cylinder.scale = (1.3, 0.7, 1.0)
cylinder.mujoco_world.role = "FREE"
cylinder.mujoco_world.collision = "CYLINDER"
convex = box("Convex_prop", (-80, 80, 40), (12, 12, 12), "FREE")
convex.mujoco_world.collision = "CONVEX"
ignored = box("Excluded_helper", (300, 0, 100), (10, 10, 10), "EXCLUDE")
hidden = box("Hidden_helper", (300, 100, 100), (10, 10, 10))
hidden.hide_set(True)

bpy.context.view_layer.update()
expected = {}
depsgraph = bpy.context.evaluated_depsgraph_get()
for obj in bpy.context.scene.objects:
    if obj.type == "MESH" and obj.visible_get() and obj.mujoco_world.role != "EXCLUDE":
        evaluated = obj.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh()
        expected[obj.name] = [list(evaluated.matrix_world @ v.co * 0.01) for v in mesh.vertices]
        evaluated.to_mesh_clear()

out = OUT
report = export_world(bpy.context, str(out / "scene.xml"))
(out / "source_buffer_check.json").write_text(json.dumps({"hdr_max": max(hdr.pixels), "hdr_filepath": hdr.filepath}))
# Independent reference through Blender's display-render save path.
image_settings = bpy.context.scene.render.image_settings
previous = image_settings.file_format, image_settings.color_mode, image_settings.color_depth
image_settings.file_format = "PNG"
image_settings.color_mode = "RGB"
image_settings.color_depth = "8"
hdr.save_render(str(out / "hdr_agx_reference.png"), scene=bpy.context.scene)
image_settings.file_format, image_settings.color_mode, image_settings.color_depth = previous
(out / "expected_vertices.json").write_text(json.dumps(expected))
# Save the generated scene with the other test outputs for inspection.
bpy.ops.wm.save_as_mainfile(filepath=str(out / "smoke_scene.blend"))
settings = bpy.context.scene.mujoco_world
# The export operator writes the same world and remembers the path; the
# browser of a saved file starts in a folder that is not named after the add-on.
assert bpy.ops.mjworld.export_world(filepath=str(out / "operator" / "scene.xml")) == {"FINISHED"}
assert settings.last_export == str(out / "operator" / "scene.xml"), settings.last_export
import mujoco_world  # noqa: E402

assert mujoco_world.default_export_path("") == str(out / "mujoco_export" / "scene.xml")
# Floor plane options: none at all, and a raised plane.
settings.plane = False
no_plane = export_world(bpy.context, str(out / "no_plane" / "scene.xml"))
assert not no_plane["plane"]["enabled"]
settings.plane, settings.plane_height = True, 0.05
raised = export_world(bpy.context, str(out / "raised_plane" / "scene.xml"))
assert raised["plane"]["height_m"] == 0.05, raised["plane"]
settings.plane_height = 0.0
# Exercise the selection-only path too.
bpy.ops.object.select_all(action="DESELECT")
prop.select_set(True)
bpy.context.view_layer.objects.active = prop
settings.selected_only = True
selected = export_world(bpy.context, str(out / "selected" / "scene.xml"))
assert selected["objects"] == 1
settings.selected_only = False
bpy.ops.mjworld.walkie_contacts()
export_world(bpy.context, str(out / "walkie_masks" / "scene.xml"))
harness.complete(out, "smoke")
print("SMOKE_EXPORT_OK", json.dumps({k: report[k] for k in ("objects", "meshes", "textures", "triangles", "warnings")}))
