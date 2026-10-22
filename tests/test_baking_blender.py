"""Run with Blender --background --factory-startup --python this_file.py.

Writes a portable exported fixture under MJWORLD_BAKE_TEST_OUTPUT, or a temporary
directory. A separate MuJoCo Python process can compile the printed scene.xml.
"""

from array import array
import json
import os
from pathlib import Path
import sys
import tempfile
import xml.etree.ElementTree as ET

import bpy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mujoco_world
from mujoco_world.baking import BakeError, bake_static, restore_sources
from mujoco_world.exporter import export_world

mujoco_world.register()
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)
scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.samples = 3
scene.cycles.seed = 0
if bpy.app.version < (5, 0, 0):
    scene.world.use_nodes = True  # Blender 5 always uses the node tree.
scene.world.node_tree.nodes["Background"].inputs["Color"].default_value = (1, 1, 1, 1)
scene.world.node_tree.nodes["Background"].inputs["Strength"].default_value = 6
scene.view_settings.view_transform = "Standard"
scene.view_settings.exposure = -0.3
scene.render.bake.margin = 7
scene.render.bake.use_selected_to_active = True  # Baker must override then restore.


def material(name, rgb, generated=False):
    result = bpy.data.materials.new(name)
    if bpy.app.version < (5, 0, 0):
        result.use_nodes = True
    principled = result.node_tree.nodes.get("Principled BSDF")
    principled.inputs["Base Color"].default_value = (*rgb, 1)
    if generated:
        coordinates = result.node_tree.nodes.new("ShaderNodeTexCoord")
        noise = result.node_tree.nodes.new("ShaderNodeTexNoise")
        ramp = result.node_tree.nodes.new("ShaderNodeValToRGB")
        ramp.color_ramp.elements[0].color = (*rgb, 1)
        ramp.color_ramp.elements[1].color = (*[component * 0.8 for component in rgb], 1)
        result.node_tree.links.new(coordinates.outputs["Generated"], noise.inputs["Vector"])
        result.node_tree.links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
        result.node_tree.links.new(ramp.outputs["Color"], principled.inputs["Base Color"])
    return result


def box(name, location, dimensions, role, mat=None, parent=None):
    bpy.ops.mesh.primitive_cube_add(size=1, location=location)
    obj = bpy.context.object
    obj.name = name
    obj.dimensions = dimensions
    obj.mujoco_world.role = role
    if mat:
        obj.data.materials.append(mat)
    if parent:
        bpy.context.view_layer.update()
        matrix = obj.matrix_world.copy()
        obj.parent = parent
        obj.matrix_world = matrix
    return obj


floor = box("Fixed floor visual", (0, 0, -0.03), (3, 3, 0.06), "VISUAL", material("Floor gray", (0.4, 0.4, 0.4)))
red = box("A red furniture", (-0.6, 0, 0.25), (0.6, 0.6, 0.5), "STATIC", material("Generated red", (0.7, 0.03, 0.02), True))
blue = box("B blue furniture", (0.6, 0, 0.25), (0.6, 0.6, 0.5), "STATIC", material("Blue", (0.02, 0.03, 0.7)))
body = bpy.data.objects.new("Movable body Empty", None)
bpy.context.collection.objects.link(body)
body.mujoco_world.role = "FREE"
if hasattr(body.mujoco_world, "body"):
    body.mujoco_world.body = True
dynamic = box("Movable descendant", (0, 0, 1), (1, 1, 0.25), "STATIC", parent=body)
door_body = bpy.data.objects.new("Door body Empty", None)
bpy.context.collection.objects.link(door_body)
door_body.mujoco_world.role = "HINGE"
if hasattr(door_body.mujoco_world, "body"):
    door_body.mujoco_world.body = True
door = box("Door descendant", (0, -1, 1), (0.6, 0.05, 2), "STATIC", parent=door_body)
helper = box("Collision only", (0, 0, 1.5), (2, 2, 0.1), "COLLIDER")
ignored = box("Excluded", (0, 0, 1.8), (2, 2, 0.1), "EXCLUDE")
sources = [floor, red, blue]
bpy.context.view_layer.update()
source_state = {obj.name: {"mesh": obj.data.as_pointer(), "vertices": [tuple(v.co) for v in obj.data.vertices],
                           "materials": [m.as_pointer() for m in obj.data.materials],
                           "matrix": [list(row) for row in obj.matrix_world],
                           "nodes": [len(m.node_tree.nodes) for m in obj.data.materials]} for obj in sources}
output = Path(os.environ.get("MJWORLD_BAKE_TEST_OUTPUT", tempfile.mkdtemp(prefix="mjworld-bake-test-")))
output.mkdir(parents=True, exist_ok=True)
baseline = export_world(bpy.context, str(output / "baseline" / "scene.xml"))
baseline_xml = ET.parse(output / "baseline" / "scene.xml")
baseline_physics = [geom for geom in baseline_xml.findall(".//geom") if geom.get("group") == "3"]


ORPHAN_MESHES = {mesh.name for mesh in bpy.data.meshes if mesh.users == 0}  # The deleted startup cube.


def assert_no_bake_leftovers():
    """A bake's working image, material copies and clone meshes are private."""
    assert "MuJoCo working static-light atlas" not in bpy.data.images
    assert not [m.name for m in bpy.data.materials if m.name.startswith("MuJoCo bake working material")]
    orphans = {mesh.name for mesh in bpy.data.meshes if mesh.users == 0} - ORPHAN_MESHES
    assert not orphans, f"orphan bake meshes {orphans}"
    assert not [o.name for o in bpy.data.objects if o.get("mjworld_bake_clone")]


try:
    bake_static(bpy.context, str(output), atlas_size=16)
    raise AssertionError("Invalid atlas size was accepted")
except BakeError:
    pass
first = bake_static(bpy.context, str(output), atlas_size=128, samples=2, margin=2)
atlas = bpy.data.objects[first["object"]]
master = atlas.data.materials[0].node_tree.nodes.get("Image Texture").image
pixels = array("f", [0]) * len(master.pixels)
master.pixels.foreach_get(pixels)
red_texels = sum(1 for i in range(0, len(pixels), 4) if pixels[i] > pixels[i + 2] * 3 and pixels[i] > 0.5)
blue_texels = sum(1 for i in range(0, len(pixels), 4) if pixels[i + 2] > pixels[i] * 3 and pixels[i + 2] > 0.5)
assert red_texels > 20 and blue_texels > 20, (red_texels, blue_texels)
assert max(pixels) > 1.5, "HDR linear radiance was clipped"
assert master.packed_file and master.is_float
assert first["source_count"] == 3 and first["excluded_dynamic_count"] == 2
assert atlas.mujoco_world.role == "VISUAL" and atlas.mujoco_world.baked
assert all(obj.hide_get() and obj.hide_render and obj.get("mjworld_bake_source") for obj in sources)
assert all(obj.visible_get() and not obj.hide_render for obj in [dynamic, door, helper, ignored])
assert scene.cycles.samples == 3 and scene.render.bake.margin == 7 and scene.render.bake.use_selected_to_active
assert scene.view_settings.view_transform == "Standard" and abs(scene.view_settings.exposure + 0.3) < 1e-6
assert master["mjworld_display_view_transform"] == "AgX"
# The lightmap profile is pinned, independent of the scene's -0.3 exposure.
assert first["display_profile"] == {"view_transform": "AgX", "look": "None", "exposure": 0.0, "gamma": 1.0,
                                    "display_device": "sRGB"}, first["display_profile"]
assert master["mjworld_display_exposure"] == 0.0 and master["mjworld_display_look"] == "None"
assert first["skipped"] == [] and first["restored"] == [] and first["warnings"] == []
assert atlas["mjworld_bake_passes"] == first["passes"] and list(atlas["mjworld_bake_sources"]) == sorted(o.name for o in sources)
assert "mjworld_bake_clone" not in atlas
assert_no_bake_leftovers()
assert Path(first["master_exr"]).stat().st_size > 100 and Path(first["display_png"]).stat().st_size > 100
for obj in sources:
    state = source_state[obj.name]
    assert obj.data.as_pointer() == state["mesh"]
    assert [tuple(v.co) for v in obj.data.vertices] == state["vertices"]
    assert [m.as_pointer() for m in obj.data.materials] == state["materials"]
    assert [len(m.node_tree.nodes) for m in obj.data.materials] == state["nodes"]
    assert [list(row) for row in obj.matrix_world] == state["matrix"]
    assert len(obj.data.uv_layers) == 1  # Only the original Cube UV map.
# Door and prop repositioning must not leave any permanent lighting change.
dynamic.location.z += 4
door_body.rotation_euler.z = 1.0
second = bake_static(bpy.context, str(output), atlas_size=128, samples=2, margin=2)
second_atlas = bpy.data.objects[second["object"]]
second_master = second_atlas.data.materials[0].node_tree.nodes.get("Image Texture").image
second_pixels = array("f", [0]) * len(second_master.pixels)
second_master.pixels.foreach_get(second_pixels)
max_difference = max(abs(a - b) for a, b in zip(pixels, second_pixels))
assert max_difference < 0.001, f"Moving dynamic bodies changed the static bake by {max_difference}"
assert sum(1 for obj in scene.objects if obj.get("mjworld_baked_atlas", False)) == 1
assert sum(1 for image in bpy.data.images if image.get("mjworld_generated_bake", False)) == 1
assert_no_bake_leftovers()
export =export_world(bpy.context, str(output / "scene.xml"))
assert export["objects"] > 0
xml = ET.parse(output / "scene.xml")
physics = [geom for geom in xml.findall(".//geom") if geom.get("group") == "3"]
assert len(physics) == len(baseline_physics), "Baking removed source collision geometry"
assert len([g for g in physics if g.get("type") == "plane"]) == 1, "Decorative floor gained collision"
# Exporting a packed HDR bake must use its recorded display profile (AgX, no
# look, exposure 0) even when the author's current scene view is different.
texture = xml.find("./asset/texture")
assert texture is not None
assert any("AgX (look None, exposure 0, gamma 1, display sRGB)" in note for note in export["warnings"]), export["warnings"]
reference = output / "agx_reference.png"
image_settings = scene.render.image_settings
previous = (image_settings.file_format, image_settings.color_mode, image_settings.color_depth)
image_settings.file_format, image_settings.color_mode, image_settings.color_depth = "PNG", "RGB", "8"
scene.view_settings.view_transform = "AgX"
scene.view_settings.exposure = 0.0
second_master.save_render(str(reference), scene=scene)
scene.view_settings.view_transform = "Standard"
scene.view_settings.exposure = -0.3
image_settings.file_format, image_settings.color_mode, image_settings.color_depth = previous
reference_image = bpy.data.images.load(str(reference), check_existing=False)
exported_image = bpy.data.images.load(str(output / "scene_assets" / texture.get("file")), check_existing=False)
reference_pixels = array("f", [0]) * len(reference_image.pixels)
exported_pixels = array("f", [0]) * len(exported_image.pixels)
reference_image.pixels.foreach_get(reference_pixels)
exported_image.pixels.foreach_get(exported_pixels)
assert max(abs(a - b) for a, b in zip(reference_pixels, exported_pixels)) < 1e-6, "Export changed the AgX bake appearance"
assert scene.view_settings.view_transform == "Standard"
bpy.ops.wm.save_as_mainfile(filepath=str(output / "baked_fixture.blend"))
restored = restore_sources(bpy.context)
assert restored == 3
assert all(not obj.hide_get() and not obj.hide_render and not obj.get("mjworld_bake_source") for obj in sources)
assert not any(obj.get("mjworld_baked_atlas", False) for obj in scene.objects)
assert_no_bake_leftovers()
summary ={"passed": True, "sources": first["source_count"], "excluded_dynamic": first["excluded_dynamic_count"],
           "hdr_max": max(pixels), "red_texels": red_texels, "blue_texels": blue_texels,
           "dynamic_reposition_max_difference": max_difference, "master_exr": first["master_exr"],
           "display_png": first["display_png"], "scene_xml": str(output / "scene.xml"), "restored_sources": restored}
(output / "bake_test_report.json").write_text(json.dumps(summary, indent=2))
print("STATIC_BAKE_TEST_OK", json.dumps(summary))
