"""Export both existing room variants twice into one shared portable asset folder.

Blender --background --factory-startup --python this_file.py -- ROOM_ROOT OUTPUT
The room root supplies scene.xml/chula_pat14_6.blend and realistic equivalents.
Source projects are opened read-only; no blend file is saved.
"""
import json
import hashlib
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import bpy

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mujoco_world
from mujoco_world.exporter import _export_simple, export_world
from mujoco_world.project import adopt_mjcf

mujoco_world.register()
args = sys.argv[sys.argv.index("--") + 1:]
room, output = (Path(value).resolve() for value in args)
shared = output / "scene_assets"
variants = [(room / "chula_pat14_6.blend", room / "scene.xml", output / "scene.xml"),
            (room / "realistic/realistic.blend", room / "realistic/scene.xml", output / "realistic/scene.xml")]


def check_paths(path):
    root = ET.parse(path).getroot()
    compiler = root.find("compiler")
    for element in root.findall("./asset/*"):
        filename = element.get("file")
        if not filename:
            continue
        directory = compiler.get("meshdir" if element.tag == "mesh" else "texturedir", "")
        resolved = (path.parent / directory / filename).resolve()
        assert not Path(filename).is_absolute() and "\\" not in filename
        assert resolved.parent == shared and resolved.exists(), (filename, resolved)
    assert len(root.findall(".//geom[@type='plane']")) == 1


counts = []
for repetition in range(2):
    for blend, xml, target in variants:
        bpy.ops.wm.open_mainfile(filepath=str(blend))
        adopt_mjcf(bpy.context, str(xml))
        scene = bpy.context.scene
        scene.mujoco_world.asset_directory = str(shared)
        # Exercise the internal temporary ordinary-export path too.
        bpy.ops.mesh.primitive_cube_add(size=0.1, location=(10, 10, 1))
        prop = bpy.context.object
        prop.name, prop.mujoco_world.role = "Shared_export_added_prop", "FREE"
        export_world(bpy.context, str(target))
        assert scene.mujoco_world.asset_directory == str(shared)
        check_paths(target)
    counts.append(len(list(shared.iterdir())))
assert counts[0] == counts[1], counts
# The ordinary exporter uses the same folder through compiler-relative paths.
scene.mujoco_world.selected_only = True
bpy.ops.object.select_all(action="DESELECT")
prop.select_set(True)
generic = output / "generic/scene.xml"
_export_simple(bpy.context, str(generic))
compiler = ET.parse(generic).getroot().find("compiler")
assert compiler.get("meshdir") == compiler.get("texturedir") == "../scene_assets"
check_paths(generic)
# Reordering/changing ordinary-export textures cannot overwrite another world.
image = bpy.data.images.new("Shared generic label", width=2, height=2)
image.pixels[:] = [1, 0, 0, 1] * 4
material = bpy.data.materials.new("Shared generic material")
material.use_nodes = True
texture = material.node_tree.nodes.new("ShaderNodeTexImage")
texture.image = image
material.node_tree.links.new(texture.outputs["Color"], material.node_tree.nodes.get("Principled BSDF").inputs["Base Color"])
prop.data.materials.clear()
prop.data.materials.append(material)
generic_red, generic_blue = output / "generic/red.xml", output / "generic/blue.xml"
_export_simple(bpy.context, str(generic_red))
red_file = ET.parse(generic_red).getroot().find("./asset/texture").get("file")
red_hash = hashlib.sha256((shared / red_file).read_bytes()).hexdigest()
image.pixels[:] = [0, 0, 1, 1] * 4
_export_simple(bpy.context, str(generic_blue))
blue_file = ET.parse(generic_blue).getroot().find("./asset/texture").get("file")
assert red_file != blue_file
assert hashlib.sha256((shared / red_file).read_bytes()).hexdigest() == red_hash
image.pixels[:] = [1, 0, 0, 1] * 4
before_repeat = len(list(shared.iterdir()))
_export_simple(bpy.context, str(generic_red))
assert len(list(shared.iterdir())) == before_repeat
check_paths(generic_red)
check_paths(generic_blue)
print("SHARED_ASSETS_TEST_OK", json.dumps({"output": str(output), "files_after_each_pair": counts,
                                           "scenes": [str(v[2]) for v in variants] + [str(generic), str(generic_red), str(generic_blue)]}))
