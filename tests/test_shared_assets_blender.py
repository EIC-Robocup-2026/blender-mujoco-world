"""Export both variants of an existing room twice into one shared asset folder.

Run:
  blender --background --factory-startup --python-exit-code 1 \
      --python tests/test_shared_assets_blender.py -- OUTPUT_DIR ROOM_ROOT
Then compile the written worlds with MuJoCo:
  python tests/validate_compile.py shared_assets OUTPUT_DIR

ROOM_ROOT is a room project with scene.xml and chula_pat14_6.blend, and
realistic/scene.xml and realistic/realistic.blend; the test expects its
task_apple_1_00 object. Pass a copy of the project: its blend files are
opened but never saved, and every export goes to OUTPUT_DIR.
"""
import json
import hashlib
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import bpy

sys.dont_write_bytecode = True  # Blender ignores PYTHONDONTWRITEBYTECODE.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness  # noqa: E402

output, args = harness.begin("shared_assets", "OUTPUT_DIR ROOM_ROOT", 2, 2)
from mujoco_world.exporter import _export_simple, export_world  # noqa: E402
from mujoco_world.project import adopt_mjcf  # noqa: E402

room = Path(args[0]).resolve()
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
if bpy.app.version < (5, 0, 0):
    material.use_nodes = True  # Blender 5 always uses the node tree.
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


def referenced(path):
    """Hash of every asset file a world references."""
    root = ET.parse(path).getroot()
    compiler = root.find("compiler")
    hashes = {}
    for element in root.findall("./asset/*"):
        if element.get("file"):
            directory = compiler.get("meshdir" if element.tag == "mesh" else "texturedir", "")
            resolved = (path.parent / directory / element.get("file")).resolve()
            hashes[resolved.name] = hashlib.sha256(resolved.read_bytes()).hexdigest()
    return hashes


def export_variant(variant, edit=False):
    blend, xml, target = variant
    bpy.ops.wm.open_mainfile(filepath=str(blend))
    adopt_mjcf(bpy.context, str(xml))
    bpy.context.scene.mujoco_world.asset_directory = str(shared)
    bpy.ops.mesh.primitive_cube_add(size=0.2 if edit else 0.1, location=(10, 10, 1))
    added = bpy.context.object
    added.name, added.mujoco_world.role = "Shared_export_added_prop", "FREE"
    if edit:
        bpy.context.scene.objects["task_apple_1_00"].scale *= 1.5
    bpy.context.view_layer.update()
    export_world(bpy.context, str(target))
    check_paths(target)
    return referenced(target)


def folder_hashes():
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in shared.iterdir()}


def mesh_file(path, geom_name):
    root = ET.parse(path).getroot()
    mesh = root.find(f".//geom[@name='{geom_name}']").get("mesh")
    return Path(root.find(f"./asset/mesh[@name='{mesh}']").get("file")).name


# Editing one variant never changes a file the other variant references: a
# name-based asset whose content differs gets a content-addressed name.
cross_variant = {}
for untouched, edited in ((variants[0], variants[1]), (variants[1], variants[0])):
    before = export_variant(untouched)
    existing = folder_hashes()
    after_edit = export_variant(edited, edit=True)
    edited_apple = mesh_file(edited[2], "task_apple_1_00")
    current = folder_hashes()
    assert all(current[name] == digest for name, digest in existing.items()), \
        sorted(name for name, digest in existing.items() if current.get(name) != digest)
    assert referenced(untouched[2]) == before, sorted(name for name, value in referenced(untouched[2]).items() if before.get(name) != value)
    unchanged = export_variant(edited)
    # The edited apple is written under a content name next to the old file,
    # which the unchanged re-export references again.
    assert edited_apple.startswith("mesh_") and edited_apple not in unchanged, edited_apple
    assert mesh_file(edited[2], "task_apple_1_00") == "task_apple_1_00_authored_mesh.obj"
    assert referenced(untouched[2]) == before
    cross_variant[untouched[2].parent.name] = edited_apple
harness.complete(output, "shared_assets")
print("SHARED_ASSETS_TEST_OK", json.dumps({"output": str(output), "files_after_each_pair": counts,
                                           "cross_variant_content_named": cross_variant,
                                           "scenes": [str(v[2]) for v in variants] + [str(generic), str(generic_red), str(generic_blue)]}))
