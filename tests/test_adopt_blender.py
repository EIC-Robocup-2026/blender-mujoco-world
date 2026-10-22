"""Blender fixture for adopted projects, shared asset folders and export safety.

Run:
  blender --background --factory-startup --python-exit-code 1 \
      --python tests/test_adopt_blender.py -- OUTPUT_DIR
Then validate the written worlds with MuJoCo:
  python tests/validate_adopt.py OUTPUT_DIR

Every world is generated here; nothing outside OUTPUT_DIR is written. Each
section runs even when an earlier one fails, and the run fails at the end
with every failing section listed.
"""
import hashlib
import json
from pathlib import Path
import sys
import traceback
import xml.etree.ElementTree as ET

import bpy

sys.dont_write_bytecode = True  # Blender ignores PYTHONDONTWRITEBYTECODE.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness  # noqa: E402

OUT, _ = harness.begin("adopt")
import mujoco_world  # noqa: E402
from mujoco_world import exporter  # noqa: E402
from mujoco_world.exporter import ExportError, export_world  # noqa: E402
from mujoco_world.project import adopt_mjcf, group_selected  # noqa: E402

CUBE_OBJ = ("v -0.1 -0.1 -0.1\nv 0.1 -0.1 -0.1\nv 0.1 0.1 -0.1\nv -0.1 0.1 -0.1\n"
            "v -0.1 -0.1 0.1\nv 0.1 -0.1 0.1\nv 0.1 0.1 0.1\nv -0.1 0.1 0.1\n"
            "f 1 3 2\nf 1 4 3\nf 5 6 7\nf 5 7 8\nf 1 2 6\nf 1 6 5\n"
            "f 2 3 7\nf 2 7 6\nf 3 4 8\nf 3 8 7\nf 4 1 5\nf 4 5 8\n")

# Mesh physics cases: default (legacy) inertia on a collider, explicit convex
# with a hull limit, exact inertia from a mesh class, and a massless visual.
WORLD = """<mujoco model="adopt_fixture">
  <compiler angle="radian" meshdir="scene_assets" texturedir="scene_assets"/>
  <default>
    <default class="visual">
      <geom contype="0" conaffinity="0" group="1" density="0"/>
    </default>
    <default class="exactmesh">
      <mesh inertia="exact" maxhullvert="24"/>
    </default>
  </default>
  <asset>
    <texture name="crate_tex" type="2d" file="crate_tex.png"/>
    <material name="crate_mat" texture="crate_tex"/>
    <mesh name="bin_solid" file="bin_solid.obj"/>
    <mesh name="bin_hull" file="bin_hull.obj" inertia="convex" maxhullvert="16"/>
    <mesh name="bin_exact" class="exactmesh" file="bin_exact.obj" smoothnormal="true"/>
    <mesh name="bin_skin" file="bin_skin.obj"/>
  </asset>
  <worldbody>
    <light name="sun" pos="0 0 3" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <geom name="Wall" type="box" size="1 0.05 0.5" pos="0 2 0.5"/>
    <body name="Crate" pos="1 0 0.3">
      <freejoint name="Crate_free"/>
      <site name="Crate_site" pos="0 0 0.2"/>
      <geom name="Crate_box" type="box" size="0.2 0.2 0.2" mass="2" material="crate_mat"/>
    </body>
    <body name="Stool" pos="-1 0 0.25">
      <geom name="Stool_seat" type="box" size="0.2 0.2 0.05" pos="0 0 0.2"/>
      <geom name="Stool_leg" type="box" size="0.03 0.03 0.2"/>
    </body>
    <body name="Bench" pos="-1 -1.5 0.25">
      <geom name="Bench_top" type="box" size="0.4 0.15 0.03"/>
      <geom name="Bench_visual" class="visual" type="box" size="0.4 0.15 0.05"/>
    </body>
    <body name="Bin" pos="1 -1.5 0.2">
      <geom name="Bin_solid" type="mesh" mesh="bin_solid" pos="-0.3 0 0"/>
      <geom name="Bin_hull" type="mesh" mesh="bin_hull"/>
      <geom name="Bin_exact" type="mesh" mesh="bin_exact" pos="0.3 0 0"/>
      <geom name="Bin_skin" class="visual" type="mesh" mesh="bin_skin" pos="0 0.3 0"/>
    </body>
    <body name="Door" pos="0 1.5 0.5">
      <joint name="Door_hinge" type="hinge" axis="0 0 1" range="-1.5 0"/>
      <geom name="Door_panel" type="box" size="0.3 0.02 0.4" pos="0.3 0 0" mass="5"/>
    </body>
  </worldbody>
</mujoco>
"""

summary = {"failures": {}, "sections": []}


def section(function):
    """Run one section; record its failure instead of stopping the run."""
    name = function.__name__
    try:
        summary[name] = function()
        summary["sections"].append(name)
        print("SECTION_OK", name)
    except Exception:  # noqa: BLE001 - every failure is reported at the end
        summary["failures"][name] = traceback.format_exc()
        print("SECTION_FAILED", name)
        traceback.print_exc()
    return function


def fresh_scene():
    bpy.ops.wm.read_homefile(use_empty=True)
    scene = bpy.context.scene
    scene.unit_settings.scale_length = 1.0
    return scene


def box(name, pos, half, role="STATIC", parent=None, size=2.0):
    bpy.ops.mesh.primitive_cube_add(size=size, location=pos)
    obj = bpy.context.object
    obj.name, obj.scale = name, half
    obj.mujoco_world.role = role
    if parent:
        bpy.context.view_layer.update()
        world = obj.matrix_world.copy()
        obj.parent = parent
        obj.matrix_world = world
    return obj


def empty(name, pos, role="STATIC", body=False):
    obj = bpy.data.objects.new(name, None)
    bpy.context.scene.collection.objects.link(obj)
    obj.location = pos
    if body:
        obj.mujoco_world.body = True
        obj.mujoco_world.role = role
    return obj


def build_world(directory, world=WORLD):
    """Write the fixture MJCF with its assets and build the matching Blender objects."""
    fresh_scene()
    assets = directory / "scene_assets"
    assets.mkdir(parents=True, exist_ok=True)
    for name in ("bin_solid", "bin_hull", "bin_exact", "bin_skin"):
        (assets / f"{name}.obj").write_text(CUBE_OBJ)
    image = bpy.data.images.new("crate_tex", 4, 4)
    image.pixels[:] = [0.8, 0.3, 0.1, 1] * 16
    image.filepath_raw = str(assets / "crate_tex.png")
    image.file_format = "PNG"
    image.save()
    (directory / "scene.xml").write_text(world)
    bpy.ops.mesh.primitive_plane_add(size=10)
    bpy.context.object.name = "floor"
    box("Wall", (0, 2, 0.5), (1, 0.05, 0.5))
    crate = empty("Crate", (1, 0, 0.3))
    crate_box = box("Crate_box", (1, 0, 0.3), (0.2, 0.2, 0.2), parent=crate)
    material = bpy.data.materials.new("crate textured")
    if bpy.app.version < (5, 0, 0):
        material.use_nodes = True  # Blender 5 always uses the node tree.
    node = material.node_tree.nodes.new("ShaderNodeTexImage")
    node.image = image
    material.node_tree.links.new(node.outputs["Color"], material.node_tree.nodes["Principled BSDF"].inputs["Base Color"])
    crate_box.data.materials.append(material)
    stool = empty("Stool", (-1, 0, 0.25))
    box("Stool_seat", (-1, 0, 0.45), (0.2, 0.2, 0.05), parent=stool)
    box("Stool_leg", (-1, 0, 0.25), (0.03, 0.03, 0.2), parent=stool)
    bench = empty("Bench", (-1, -1.5, 0.25))
    box("Bench_top", (-1, -1.5, 0.25), (0.4, 0.15, 0.03), parent=bench)
    box("Bench_visual", (-1, -1.5, 0.25), (0.4, 0.15, 0.05), parent=bench)
    bin_body = empty("Bin", (1, -1.5, 0.2))
    for name, offset in (("Bin_solid", (-0.3, 0, 0)), ("Bin_hull", (0, 0, 0)), ("Bin_exact", (0.3, 0, 0)), ("Bin_skin", (0, 0.3, 0))):
        box(name, (1 + offset[0], -1.5 + offset[1], 0.2), (1, 1, 1), parent=bin_body, size=0.2)
    door = empty("Door", (0, 1.5, 0.5))
    box("Door_panel", (0.3, 1.5, 0.5), (0.3, 0.02, 0.4), parent=door)
    bpy.context.view_layer.update()
    return directory / "scene.xml"


def adopt(directory):
    source = build_world(directory)
    adopt_mjcf(bpy.context, str(source))
    return source


def xml(path):
    return ET.parse(path).getroot()


def expect_error(fragment, target):
    """The export must fail with ``fragment`` and leave ``target`` unwritten."""
    bpy.context.view_layer.update()
    try:
        export_world(bpy.context, str(target))
    except ExportError as exc:
        assert fragment in str(exc), (fragment, str(exc))
        assert not target.exists(), target
        return str(exc)
    raise AssertionError(f"export should have failed with '{fragment}'")


def snapshot(directory):
    """Bytes of every file below ``directory``."""
    return {str(p.relative_to(directory)): p.read_bytes() for p in sorted(directory.rglob("*")) if p.is_file()}


def referenced(path):
    """Hash of every asset file an exported world references, by file name."""
    root = xml(path)
    compiler = root.find("compiler")
    files = {}
    for element in root.findall("./asset/*"):
        if element.get("file"):
            directory = compiler.get("meshdir" if element.tag == "mesh" else "texturedir", "")
            resolved = (path.parent / directory / element.get("file")).resolve()
            files[element.get("name")] = (resolved.name, hashlib.sha256(resolved.read_bytes()).hexdigest())
    return files


def leftovers(directory):
    return [str(p) for p in directory.rglob("*") if p.name.startswith(".mjworld") or p.name.endswith(".mjworld-tmp")]


def adoption_state():
    """Adoption keys and physics settings of every object and the scene."""
    scene = bpy.context.scene
    state = {"scene": {key: str(scene.get(key)) for key in ("mjworld_mjcf_template", "mjworld_mjcf_source")}}
    for obj in scene.objects:
        cfg = obj.mujoco_world
        state[obj.name] = {"keys": {key: str(obj[key]) for key in obj.keys() if key.startswith("mjworld")},
                           "settings": [cfg.body, cfg.role, cfg.mass, cfg.hinge_min, cfg.hinge_max, cfg.hinge_start, cfg.damping, cfg.hinge_axis]}
    return state


@section
def adopt_two_pass():
    """A rejected adoption leaves the previous adoption untouched; Adopt is undoable."""
    directory = OUT / "adopt_two_pass"
    source = adopt(directory)
    objects = bpy.context.scene.objects
    # Settings the user changed after adopting must survive a failed re-adoption.
    objects["Stool"].mujoco_world.mass = 7.5
    before = adoption_state()
    broken = {"missing_name": WORLD.replace('<geom name="Wall"', '<geom name="Ghost" type="box" size="1 1 1"/>\n    <geom name="Wall"'),
              "bad_range": WORLD.replace('range="-1.5 0"', 'range="-1.5 0 1"'),
              "bad_number": WORLD.replace('range="-1.5 0"', 'range="-1.5 closed"')}
    messages = {}
    for label, text in broken.items():
        path = directory / f"{label}.xml"
        path.write_text(text)
        try:
            adopt_mjcf(bpy.context, str(path))
        except ExportError as exc:
            messages[label] = str(exc)
        else:
            raise AssertionError(f"{label}: adoption should have failed")
        after = adoption_state()
        changed = sorted(name for name in before if before[name] != after.get(name))
        assert not changed, (label, changed)
    # A successful re-adoption without Bench clears all of Bench's adoption keys.
    reduced = directory / "without_bench.xml"
    start = WORLD.index('    <body name="Bench"')
    reduced.write_text(WORLD[:start] + WORLD[WORLD.index('    <body name="Bin"'):])
    adopt_mjcf(bpy.context, str(reduced))
    stale = {obj.name: sorted(key for key in obj.keys() if key.startswith("mjworld_mjcf") or key == "mjworld_settings_baseline")
             for obj in objects if obj.name.startswith("Bench")}
    assert not any(stale.values()), stale
    assert objects["Crate"].get("mjworld_settings_baseline"), "kept bodies still have their baseline"
    # Adoption records which object owns each MJCF name, and can be undone.
    assert objects["Crate"]["mjworld_mjcf_owner"] == "Crate" and objects["Crate_box"]["mjworld_mjcf_owner"] == "Crate_box"
    assert "UNDO" in getattr(mujoco_world.MJWORLD_OT_adopt, "bl_options", set()), "Adopt cannot be undone"
    return {"source": str(source), "errors": messages}


@section
def mesh_inertia():
    """Regenerated meshes keep the source inertia mode, hull limit and normals."""
    directory = OUT / "mesh_inertia"
    adopt(directory / "source")
    target = directory / "scene.xml"
    export_world(bpy.context, str(target))
    root = xml(target)
    meshes = {geom.get("name"): root.find(f"./asset/mesh[@name='{geom.get('mesh')}']").attrib
              for geom in root.iter("geom") if geom.get("type") == "mesh"}
    assert "inertia" not in meshes["Bin_solid"], meshes["Bin_solid"]
    assert (meshes["Bin_hull"].get("inertia"), meshes["Bin_hull"].get("maxhullvert")) == ("convex", "16"), meshes["Bin_hull"]
    assert ([meshes["Bin_exact"].get(key) for key in ("inertia", "maxhullvert", "smoothnormal")]
            == ["exact", "24", "true"]), meshes["Bin_exact"]
    assert meshes["Bin_skin"].get("inertia") == "shell", meshes["Bin_skin"]
    assert len({attrs["name"] for attrs in meshes.values()}) == 4, "different physics never share a mesh"
    return {"source": str(directory / "source" / "scene.xml"), "export": str(target)}


@section
def duplicates():
    """Shift+D copies of adopted objects export under fresh, unique names."""
    directory = OUT / "duplicates"
    adopt(directory / "source")
    scene = bpy.context.scene
    objects = scene.objects
    crate, crate_box, wall = objects["Crate"], objects["Crate_box"], objects["Wall"]
    # Object.copy() is what Shift+D does to ID properties: they are copied.
    crate_copy = crate.copy()
    scene.collection.objects.link(crate_copy)
    box_copy = crate_box.copy()
    box_copy.data = crate_box.data.copy()
    scene.collection.objects.link(box_copy)
    box_copy.parent = crate_copy
    crate_copy.location.x += 1.0
    wall_copy = wall.copy()
    scene.collection.objects.link(wall_copy)
    wall_copy.location.x += 3.0
    bpy.context.view_layer.update()
    assert (crate_copy.name, box_copy.name, wall_copy.name) == ("Crate.001", "Crate_box.001", "Wall.001")
    target = directory / "scene.xml"
    report = export_world(bpy.context, str(target))
    root = xml(target)
    for tag, names in (("body", {"Crate", "Crate.001"}), ("freejoint", {"Crate_free", "Crate.001_free"}),
                       ("site", {"Crate_site", "Crate.001_site"}), ("geom", {"Crate_box", "Crate_box.001", "Wall", "Wall.001"})):
        found = [element.get("name") for element in root.iter(tag)]
        assert names <= set(found) and len(found) == len(set(found)), (tag, found)
    copy_body = root.find(".//body[@name='Crate.001']")
    assert copy_body.find("geom").get("name") == "Crate_box.001" and copy_body.find("geom").get("mass") == "2"
    assert any("copy of adopted body 'Crate'" in warning for warning in report["warnings"]), report["warnings"]
    # Projects adopted before the owner property existed: the object whose name
    # is the MJCF name keeps it.
    for obj in objects:
        if "mjworld_mjcf_owner" in obj:
            del obj["mjworld_mjcf_owner"]
    legacy = directory / "legacy" / "scene.xml"
    export_world(bpy.context, str(legacy))
    assert legacy.read_bytes() == target.read_bytes()
    # With no object holding the adopted name, which one is the original is unknown.
    crate.name = "Crate_original"
    message = expect_error("Several objects carry the adopted MJCF body 'Crate'", directory / "ambiguous" / "scene.xml")
    return {"export": str(target), "legacy": str(legacy), "ambiguous": message}


@section
def inplace_textures():
    """Exporting over the source XML never deletes the source world's own files."""
    directory = OUT / "inplace_textures"
    source = adopt(directory)
    texture = directory / "scene_assets" / "crate_tex.png"
    original = texture.read_bytes()
    export_world(bpy.context, str(source))
    crate_box = bpy.context.scene.objects["Crate_box"]
    textured = crate_box.data.materials[0]
    flat = bpy.data.materials.new("flat green")
    if bpy.app.version < (5, 0, 0):
        flat.use_nodes = True
    flat.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0, 1, 0, 1)
    crate_box.data.materials[0] = flat
    export_world(bpy.context, str(source))
    assert texture.exists() and texture.read_bytes() == original, "source texture removed"
    assert xml(source).find("./asset/texture[@name='crate_tex']") is None, "unused texture left in the XML"
    crate_box.data.materials[0] = textured
    export_world(bpy.context, str(source))
    assert xml(source).find("./asset/texture[@name='crate_tex']") is not None
    assert not leftovers(directory), leftovers(directory)
    return {"export": str(source)}


@section
def unevaluated_objects():
    """Objects Blender does not evaluate are refused, naming them; the eye icon is fine."""
    directory = OUT / "unevaluated"
    adopt(directory / "source")
    scene = bpy.context.scene
    leg = scene.objects["Stool_leg"]
    helpers = bpy.data.collections.new("Hidden helpers")
    scene.collection.children.link(helpers)
    for collection in list(leg.users_collection):
        collection.objects.unlink(leg)
    helpers.objects.link(leg)
    layer = bpy.context.view_layer.layer_collection.children["Hidden helpers"]
    messages = {}
    layer.exclude = True
    messages["excluded_collection"] = expect_error("Stool_leg", directory / "excluded" / "scene.xml")
    layer.exclude = False
    helpers.hide_viewport = True
    messages["collection_monitor"] = expect_error("Stool_leg", directory / "collection_monitor" / "scene.xml")
    helpers.hide_viewport = False
    leg.hide_viewport = True
    messages["object_monitor"] = expect_error("Stool_leg", directory / "object_monitor" / "scene.xml")
    leg.hide_viewport = False
    scene.objects["Stool"].hide_viewport = True
    messages["body_monitor"] = expect_error("Stool", directory / "body_monitor" / "scene.xml")
    scene.objects["Stool"].hide_viewport = False
    leg.hide_set(True)
    target = directory / "scene.xml"
    export_world(bpy.context, str(target))
    assert xml(target).find(".//geom[@name='Stool_leg']") is not None
    # Ordinary export: a body group Empty in an excluded collection.
    fresh_scene()
    hidden = bpy.data.collections.new("Hidden groups")
    bpy.context.scene.collection.children.link(hidden)
    cart = bpy.data.objects.new("Cart", None)
    hidden.objects.link(cart)
    cart.location = (0, 0, 0.2)
    cart.mujoco_world.body = True
    cart.mujoco_world.role = "FREE"
    box("Cart_box", (0, 0, 0.2), (0.2, 0.2, 0.2), parent=cart)
    bpy.context.view_layer.layer_collection.children["Hidden groups"].exclude = True
    messages["ordinary_group"] = expect_error("Cart", directory / "ordinary" / "scene.xml")
    assert all("eye icon" in message for message in messages.values()), messages
    return messages


@section
def weights():
    """Weight applies to adopted bodies made movable and to grouped adopted parts."""
    directory = OUT / "weights"
    adopt(directory / "source")
    scene = bpy.context.scene
    stool = scene.objects["Stool"]
    stool.mujoco_world.role = "FREE"
    stool.mujoco_world.mass = 2.0
    bpy.ops.object.select_all(action="DESELECT")
    for name in ("Bench_top", "Bench_visual"):
        scene.objects[name].select_set(True)
    group = group_selected(bpy.context, "FREE")
    group.mujoco_world.mass = 1.5
    bpy.context.view_layer.update()
    target = directory / "scene.xml"
    export_world(bpy.context, str(target))
    root = xml(target)
    stool_body = root.find(".//body[@name='Stool']")
    assert stool_body.find("freejoint") is not None
    # The Weight follows collider volume: seat 0.016 m3, leg 0.00144 m3.
    stool_masses = {geom.get("name"): float(geom.get("mass")) for geom in stool_body.findall("geom")}
    assert abs(stool_masses["Stool_seat"] - 2.0 * 0.016 / 0.01744) < 1e-7, stool_masses
    assert abs(stool_masses["Stool_leg"] - 2.0 * 0.00144 / 0.01744) < 1e-7, stool_masses
    assert abs(sum(stool_masses.values()) - 2.0) < 1e-7, stool_masses
    group_body = root.find(f".//body[@name='{group.name}']")
    masses = {geom.get("name"): geom.get("mass") for geom in group_body.findall("geom")}
    assert masses == {"Bench_top": "1.5", "Bench_visual": "0"}, masses
    # Untouched adopted bodies keep their source physics exactly.
    assert root.find(".//geom[@name='Crate_box']").get("mass") == "2"
    assert "mass" not in root.find(".//geom[@name='Bin_solid']").attrib
    return {"export": str(target), "source": str(directory / "source" / "scene.xml"), "group": group.name}


@section
def hinge_cursor_shared_mesh():
    """Set hinge at 3D cursor refuses shared mesh data and restores the selection."""
    scene = fresh_scene()
    door = box("Door_leaf", (0, 0, 1), (0.4, 0.02, 1), "HINGE")
    twin = door.copy()
    scene.collection.objects.link(twin)
    twin.location.x += 2
    other = box("Other", (5, 0, 1), (0.1, 0.1, 0.1))
    bpy.context.view_layer.update()
    twin_vertices = [v.co.copy() for v in twin.data.vertices]
    bpy.ops.object.select_all(action="DESELECT")
    door.select_set(True)
    other.select_set(True)
    bpy.context.view_layer.objects.active = door
    scene.cursor.location = (-0.4, 0, 1)
    try:
        result = bpy.ops.mjworld.hinge_at_cursor()
    except RuntimeError as exc:
        result, message = {"CANCELLED"}, str(exc)
    else:
        message = ""
    assert result == {"CANCELLED"}, result
    assert "Make Single User" in message, message
    assert [v.co for v in twin.data.vertices] == twin_vertices
    assert tuple(door.location) == (0, 0, 1)
    door.data = door.data.copy()
    world_corners = [door.matrix_world @ v.co for v in door.data.vertices]
    assert bpy.ops.mjworld.hinge_at_cursor() == {"FINISHED"}
    bpy.context.view_layer.update()
    assert (door.location - scene.cursor.location).length < 1e-6, tuple(door.location)
    assert all((door.matrix_world @ v.co - corner).length < 1e-5 for v, corner in zip(door.data.vertices, world_corners))
    assert [v.co for v in twin.data.vertices] == twin_vertices
    assert {obj.name for obj in bpy.context.selected_objects} == {"Door_leaf", "Other"}
    assert bpy.context.view_layer.objects.active == door
    # A fake user is not another object.
    door.data.use_fake_user = True
    assert bpy.ops.mjworld.hinge_at_cursor() == {"FINISHED"}
    return {"refusal": message}


@section
def failed_export_keeps_output():
    """A failed export leaves the previous XML, report and assets byte-identical."""
    directory = OUT / "failed_export"
    fresh_scene()
    crate = box("Crate", (0, 0, 0.3), (0.2, 0.2, 0.2), "FREE")
    simple = directory / "simple" / "scene.xml"
    export_world(bpy.context, str(simple))
    before = snapshot(simple.parent)
    crate.scale = (0.4, 0.4, 0.4)
    # Fails in the regroup stage, after the parts were built.
    cart = empty("Ghost_cart", (2, 0, 0.2), "FREE", body=True)
    box("Ghost_cart_visual", (2, 0, 0.2), (0.1, 0.1, 0.1), "VISUAL", parent=cart)
    bpy.context.view_layer.update()
    try:
        export_world(bpy.context, str(simple))
    except ExportError as exc:
        assert "collision helper" in str(exc), str(exc)
    else:
        raise AssertionError("regroup should have failed")
    assert snapshot(simple.parent) == before, "regroup failure changed the previous export"
    # Fails while building, after the crate's mesh files were produced.
    bpy.data.objects.remove(bpy.context.scene.objects["Ghost_cart_visual"])
    bpy.data.objects.remove(cart)
    bpy.ops.mesh.primitive_plane_add(size=1, location=(3, 0, 0.5))
    flat = bpy.context.object
    flat.name = "zzz_flat_panel"
    bpy.context.view_layer.update()
    try:
        export_world(bpy.context, str(simple))
    except ExportError as exc:
        assert "zero thickness" in str(exc), str(exc)
    else:
        raise AssertionError("flat collision should have failed")
    assert snapshot(simple.parent) == before, "build failure changed the previous export"
    # Adopted world: the floor check runs after every mesh was regenerated.
    adopt(directory / "adopted_source")
    adopted = directory / "adopted" / "scene.xml"
    export_world(bpy.context, str(adopted))
    adopted_before = snapshot(adopted.parent)
    bpy.context.scene.objects["Bin_solid"].scale = (1.5, 1.5, 1.5)
    bpy.data.objects.remove(bpy.context.scene.objects["floor"])
    bpy.context.view_layer.update()
    try:
        export_world(bpy.context, str(adopted))
    except ExportError as exc:
        assert "floor plane" in str(exc), str(exc)
    else:
        raise AssertionError("missing floor should have failed")
    assert snapshot(adopted.parent) == adopted_before, "adopted failure changed the previous export"
    assert not leftovers(directory), leftovers(directory)
    return {"simple": str(simple), "adopted": str(adopted)}


@section
def shared_folder_safe_names():
    """Exports into a shared folder never change a file another world references."""
    directory = OUT / "shared_folder"
    adopt(directory / "source")
    shared = directory / "shared_assets"
    bpy.context.scene.mujoco_world.asset_directory = str(shared)
    first = directory / "a" / "scene.xml"
    export_world(bpy.context, str(first))
    first_files = referenced(first)
    listing = sorted(p.name for p in shared.iterdir())
    export_world(bpy.context, str(first))
    assert referenced(first) == first_files and sorted(p.name for p in shared.iterdir()) == listing, "repeat export added files"
    assert all(not name.startswith("mesh_") and not name.startswith("texture_") for name, _digest in first_files.values()), first_files
    bpy.context.scene.objects["Bin_solid"].scale = (1.2, 1.2, 1.2)
    bpy.context.view_layer.update()
    second = directory / "b" / "scene.xml"
    export_world(bpy.context, str(second))
    second_files = referenced(second)
    assert referenced(first) == first_files, "the second world changed the first world's files"
    edited = second_files.pop("Bin_solid_authored_mesh")
    assert edited[0].startswith("mesh_") and edited != first_files["Bin_solid_authored_mesh"], edited
    unchanged = {name: value for name, value in first_files.items() if name != "Bin_solid_authored_mesh"}
    assert second_files == unchanged, (second_files, unchanged)
    assert not leftovers(directory), leftovers(directory)
    return {"first": str(first), "second": str(second), "content_named": edited[0]}


def resolved_assets(path):
    """Resolved path of every asset file an exported world references."""
    root = xml(path)
    compiler = root.find("compiler")
    return {element.get("name"): (path.parent / compiler.get("meshdir" if element.tag == "mesh" else "texturedir", "")
                                  / element.get("file")).resolve()
            for element in root.findall("./asset/*") if element.get("file")}


@section
def shared_folder_two_projects():
    """Two adopted projects and an ordinary scene share one folder, like the variants of a room."""
    directory = OUT / "shared_two_projects"
    out = directory / "export"
    shared = out / "scene_assets"
    variants = {"low": (directory / "low", out / "scene.xml"),
                "realistic": (directory / "realistic", out / "realistic" / "scene.xml")}
    blue = [0.1, 0.3, 0.8, 1] * 16

    def export_variant(name):
        source_directory, target = variants[name]
        build_world(source_directory)
        if name == "realistic":
            # Same texture file name as the other project, different content.
            image = bpy.data.images["crate_tex"]
            image.pixels[:] = blue
            image.save()
        adopt_mjcf(bpy.context, str(source_directory / "scene.xml"))
        bpy.context.scene.mujoco_world.asset_directory = str(shared)
        export_world(bpy.context, str(target))
        return target

    listings = []
    for _repetition in range(2):
        for name in variants:
            export_variant(name)
        listings.append(sorted(p.name for p in shared.iterdir()))
    assert listings[0] == listings[1], "repeating both exports changed the shared folder"
    assert not (shared / ".mujoco_world_manifest.json").exists(), "a shared folder must not get a manifest"
    files = {name: resolved_assets(target) for name, (_source, target) in variants.items()}
    for name, assets in files.items():
        assert all(path.parent == shared and path.is_file() for path in assets.values()), (name, assets)
    low_crate, realistic_crate = files["low"]["crate_tex"], files["realistic"]["crate_tex"]
    assert low_crate != realistic_crate, "two different textures share one file"
    for crate, (source_directory, _target) in ((low_crate, variants["low"]), (realistic_crate, variants["realistic"])):
        source = source_directory / "scene_assets" / "crate_tex.png"
        assert crate.read_bytes() == source.read_bytes(), crate
    # An ordinary scene exports into the same folder through relative paths.
    fresh_scene()
    bpy.ops.mesh.primitive_plane_add(size=4)
    bpy.context.object.mujoco_world.role = "VISUAL"
    box("Ordinary_crate", (0, 0, 0.5), (0.1, 0.1, 0.1), "FREE")
    bpy.context.scene.mujoco_world.asset_directory = str(shared)
    ordinary = out / "ordinary" / "scene.xml"
    export_world(bpy.context, str(ordinary))
    compiler = xml(ordinary).find("compiler")
    assert compiler.get("meshdir") == compiler.get("texturedir") == "../scene_assets", compiler.attrib
    assert all(path.parent == shared and path.is_file() for path in resolved_assets(ordinary).values())
    # The adopted worlds still reference their own unchanged files.
    assert {name: resolved_assets(target) for name, (_source, target) in variants.items()} == files
    assert low_crate.read_bytes() == (variants["low"][0] / "scene_assets" / "crate_tex.png").read_bytes()
    assert not leftovers(directory), leftovers(directory)
    return {"low": str(variants["low"][1]), "realistic": str(variants["realistic"][1]), "ordinary": str(ordinary),
            "low_crate": low_crate.name, "realistic_crate": realistic_crate.name}


@section
def door_masks():
    """Doors have their own masks; blank ones follow the fixed masks; the Walkie preset uses 32/121."""
    directory = OUT / "door_masks"
    scene = fresh_scene()
    settings = scene.mujoco_world
    settings.object_contype, settings.object_conaffinity = "64", "56"
    settings.movable_contype, settings.movable_conaffinity = "32", "121"
    box("Wall", (0, 2, 0.5), (1, 0.05, 0.5))
    box("Door", (0, 1, 1), (0.4, 0.02, 1), "HINGE")
    cart = empty("Cart", (2, 0, 0.2), "FREE", body=True)
    cart.mujoco_world.mass = 3
    box("Cart_base", (2, 0, 0.1), (0.3, 0.2, 0.05), parent=cart)
    shelf = empty("Cart_shelf", (2, 0, 0.4), "STATIC", body=True)
    shelf.parent = cart
    shelf.matrix_parent_inverse = cart.matrix_world.inverted()
    bpy.context.view_layer.update()
    box("Shelf_board", (2, 0, 0.4), (0.3, 0.2, 0.02), parent=shelf)
    bpy.context.view_layer.update()

    def masks(path):
        root = xml(path)
        return {geom.get("name"): (geom.get("contype"), geom.get("conaffinity")) for geom in root.iter("geom") if geom.get("contype") != "0"}

    blank = directory / "blank" / "scene.xml"
    export_world(bpy.context, str(blank))
    found = masks(blank)
    door = next(name for name in found if name.startswith("Door_"))
    assert found[door] == ("64", "56"), found
    # A static group inside a free body moves with it, as world_loader sees it.
    shelf_collider = next(name for name in found if name.startswith("Shelf_board_"))
    assert found[shelf_collider] == ("32", "121"), found
    bpy.ops.mjworld.walkie_contacts()
    assert (settings.hinge_contype, settings.hinge_conaffinity) == ("32", "121")
    walkie = directory / "walkie" / "scene.xml"
    export_world(bpy.context, str(walkie))
    found = masks(walkie)
    assert found[door] == ("32", "121") and found["floor"] == ("1", "32"), found
    assert next(value for name, value in found.items() if name.startswith("Wall_")) == ("64", "56"), found
    return {"blank": str(blank), "walkie": str(walkie)}


@section
def per_xml_in_shared_folder():
    """A per-XML export whose folder another world uses as its shared folder replaces none of that world's files."""
    directory = OUT / "per_xml_shared"
    adopt(directory / "source")
    scene = bpy.context.scene
    out = directory / "export"
    shared = out / "scene_assets"
    scene.mujoco_world.asset_directory = str(shared)
    first = out / "shared" / "scene.xml"
    export_world(bpy.context, str(first))
    first_files = referenced(first)
    # The per-XML folder of out/scene.xml is out/scene_assets: the same folder.
    scene.mujoco_world.asset_directory = ""
    scene.objects["Bin_solid"].scale = (1.2, 1.2, 1.2)
    bpy.context.view_layer.update()
    second = out / "scene.xml"
    export_world(bpy.context, str(second))
    assert referenced(first) == first_files, "the per-XML export changed the shared world's files"
    second_files = referenced(second)
    edited = second_files.pop("Bin_solid_authored_mesh")
    assert edited[0].startswith("mesh_") and edited != first_files["Bin_solid_authored_mesh"], edited
    unchanged = {name: value for name, value in first_files.items() if name != "Bin_solid_authored_mesh"}
    assert second_files == unchanged, (second_files, unchanged)
    manifest = shared / ".mujoco_world_manifest.json"
    # Identical files are referenced as they are; only the file this world wrote is its own.
    assert json.loads(manifest.read_text())["files"] == [edited[0]], manifest.read_text()
    # The next per-XML export removes its own previous file, still none of the shared world's.
    scene.objects["Bin_solid"].scale = (1.4, 1.4, 1.4)
    bpy.context.view_layer.update()
    export_world(bpy.context, str(second))
    again = referenced(second)["Bin_solid_authored_mesh"]
    assert again[0].startswith("mesh_") and again != edited, again
    assert not (shared / edited[0]).exists(), "the per-XML world's own stale file was kept"
    assert referenced(first) == first_files, "the second per-XML export changed the shared world's files"
    assert json.loads(manifest.read_text())["files"] == [again[0]], manifest.read_text()
    assert not leftovers(directory), leftovers(directory)
    return {"shared": str(first), "per_xml": str(second), "content_named": again[0]}


@section
def failed_xml_install():
    """When the new assets or XML cannot be installed, the old XML, its report and every file it names stay."""
    directory = OUT / "failed_install"
    original = exporter._install
    real_replace = exporter.os.replace

    def broken(temporary, path):
        if Path(path).suffix == ".xml":
            raise OSError("simulated XML install failure")
        original(temporary, path)

    def attempt(target):
        bpy.context.view_layer.update()
        exporter._install = broken
        try:
            export_world(bpy.context, str(target))
        except OSError as exc:
            assert "simulated XML install failure" in str(exc), str(exc)
        else:
            raise AssertionError("the simulated install failure did not stop the export")
        finally:
            exporter._install = original

    def failing_move(source, target, *args, **kwargs):
        # Only moves out of an AssetWriter's staging folder fail.
        if ".mjworld-stage-" in str(source):
            raise OSError("simulated asset install failure")
        return real_replace(source, target, *args, **kwargs)

    def asset_attempt(target):
        """Fail the first staged asset move. Assets go in before the XML, so the XML must not change."""
        bpy.context.view_layer.update()
        exporter.os.replace = failing_move
        try:
            export_world(bpy.context, str(target))
        except OSError as exc:
            assert "simulated asset install failure" in str(exc), str(exc)
        else:
            raise AssertionError("the export moved no staged asset, so the simulated failure never happened")
        finally:
            exporter.os.replace = real_replace

    def check(target, old_xml, old_report):
        assert target.read_bytes() == old_xml, "the old XML was replaced"
        assert (target.parent / "export_report.json").read_bytes() == old_report, "the old report was replaced"
        missing = [str(path) for path in resolved_assets(target).values() if not path.is_file()]
        assert not missing, f"files the old XML names were removed: {missing}"
        assert not leftovers(directory), leftovers(directory)

    def folder_matches(target):
        """The asset folder holds exactly the files the XML names, plus the manifest."""
        folder = target.parent / "scene_assets"
        files = sorted(p.name for p in folder.iterdir() if p.name != ".mujoco_world_manifest.json")
        named = sorted({path.name for path in resolved_assets(target).values()})
        assert files == named, (files, named)
        assert json.loads((folder / ".mujoco_world_manifest.json").read_text())["files"] == named
        return named

    results = {}
    # Ordinary export: a renamed object writes new files, a deleted one leaves stale files.
    fresh_scene()
    box("Crate", (0, 0, 0.3), (0.2, 0.2, 0.2), "FREE")
    box("Stool", (1, 0, 0.3), (0.2, 0.2, 0.2), "FREE")
    simple = directory / "simple" / "scene.xml"
    export_world(bpy.context, str(simple))
    old = simple.read_bytes(), (simple.parent / "export_report.json").read_bytes()
    bpy.context.scene.objects["Crate"].name = "Crate_renamed"
    bpy.data.objects.remove(bpy.context.scene.objects["Stool"])
    asset_attempt(simple)
    check(simple, *old)
    attempt(simple)
    check(simple, *old)
    export_world(bpy.context, str(simple))
    results["simple"] = folder_matches(simple)
    # Adopted export: a regenerated mesh and a deleted visual.
    adopt(directory / "adopted_source")
    adopted = directory / "adopted" / "scene.xml"
    export_world(bpy.context, str(adopted))
    old = adopted.read_bytes(), (adopted.parent / "export_report.json").read_bytes()
    bpy.context.scene.objects["Bin_solid"].scale = (1.5, 1.5, 1.5)
    bpy.data.objects.remove(bpy.context.scene.objects["Bin_skin"])
    asset_attempt(adopted)
    check(adopted, *old)
    attempt(adopted)
    check(adopted, *old)
    export_world(bpy.context, str(adopted))
    results["adopted"] = folder_matches(adopted)
    return results


PART_NAMES = """<mujoco model="part_names">
  <compiler angle="radian"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="Lamp_000" pos="1 0 0.3">
      <freejoint name="Lamp_000_free"/>
      <geom name="Lamp_000_box" type="box" size="0.1 0.1 0.1" mass="1"/>
    </body>
    <body name="Shelf" pos="-1 0 0.5">
      <geom name="Shelf_board" type="box" size="0.3 0.2 0.02"/>
      <geom name="Book_000_collision" type="box" size="0.05 0.05 0.05" pos="0.2 0 0.07"/>
    </body>
    <body name="Cabinet" pos="0 2 0.5">
      <joint name="Drawer_hinge" type="hinge" axis="0 0 1" range="-1 0"/>
      <geom name="Cabinet_box" type="box" size="0.2 0.2 0.2" mass="1"/>
    </body>
    <body name="Tub" pos="0 -2 0.3">
      <freejoint name="Box_free"/>
      <geom name="Tub_box" type="box" size="0.1 0.1 0.1" mass="1"/>
    </body>
  </worldbody>
</mujoco>
"""


@section
def new_part_names():
    """New objects whose MJCF names the adopted world already uses are refused before anything is written."""
    directory = OUT / "part_names"
    source = directory / "source" / "scene.xml"
    source.parent.mkdir(parents=True)
    source.write_text(PART_NAMES)
    fresh_scene()
    bpy.ops.mesh.primitive_plane_add(size=10)
    bpy.context.object.name = "floor"
    lamp_body = empty("Lamp_000", (1, 0, 0.3))
    box("Lamp_000_box", (1, 0, 0.3), (0.1, 0.1, 0.1), parent=lamp_body)
    shelf = empty("Shelf", (-1, 0, 0.5))
    box("Shelf_board", (-1, 0, 0.5), (0.3, 0.2, 0.02), parent=shelf)
    box("Book_000_collision", (-0.8, 0, 0.57), (0.05, 0.05, 0.05), parent=shelf)
    cabinet = empty("Cabinet", (0, 2, 0.5))
    box("Cabinet_box", (0, 2, 0.5), (0.2, 0.2, 0.2), parent=cabinet)
    tub = empty("Tub", (0, -2, 0.3))
    box("Tub_box", (0, -2, 0.3), (0.1, 0.1, 0.1), parent=tub)
    bpy.context.view_layer.update()
    adopt_mjcf(bpy.context, str(source))
    messages = {}
    # A new top-level movable object: its body and free joint names are taken.
    lamp = box("Lamp", (2, 0, 0.3), (0.1, 0.1, 0.1), "FREE")
    messages["body"] = expect_error("'Lamp' (body 'Lamp_000')", directory / "body" / "scene.xml")
    assert "'Lamp' (freejoint 'Lamp_000_free')" in messages["body"], messages["body"]
    lamp.name = "Lantern"
    # A new part grouped into an adopted body: its geoms join that body.
    book = box("Book", (-0.6, 0, 0.6), (0.05, 0.05, 0.05), parent=shelf)
    messages["grouped"] = expect_error("'Book' (geom 'Book_000_collision')", directory / "grouped" / "scene.xml")
    assert "Lantern" not in messages["grouped"], messages["grouped"]
    book.name = "Novel"
    # A new body group's body is named after its Empty, and its joint after the body.
    drawer = empty("Drawer", (-2, 2, 0.5), "HINGE", body=True)
    box("Drawer_front", (-2, 2, 0.5), (0.2, 0.02, 0.2), parent=drawer)
    messages["hinge_group"] = expect_error("'Drawer' (joint 'Drawer_hinge')", directory / "hinge_group" / "scene.xml")
    drawer.name = "Hatch"
    crate = empty("Box", (2, -2, 0.3), "FREE", body=True)
    box("Box_part", (2, -2, 0.3), (0.1, 0.1, 0.1), parent=crate)
    messages["free_group"] = expect_error("'Box' (freejoint 'Box_free')", directory / "free_group" / "scene.xml")
    crate.name = "Tote"
    # A renamed adopted body keeps its MJCF name, so a new group may not take it.
    shelf.name = "Shelf_old"
    cupboard = empty("Shelf", (-2, -2, 0.5), body=True)
    box("Cupboard_board", (-2, -2, 0.5), (0.3, 0.2, 0.02), parent=cupboard)
    messages["renamed_body"] = expect_error("'Shelf' (body 'Shelf')", directory / "renamed_body" / "scene.xml")
    cupboard.name = "Cupboard"
    # An adopted body made movable gets a free joint named after its object.
    shelf.name = "Box"
    shelf.mujoco_world.role = "FREE"
    messages["new_role"] = expect_error("'Box' (freejoint 'Box_free')", directory / "new_role" / "scene.xml")
    shelf.mujoco_world.role = "STATIC"
    shelf.name = "Shelf"
    target = directory / "scene.xml"
    export_world(bpy.context, str(target))
    root = xml(target)
    # One namespace per kind; a joint and a free joint share one.
    for tags in (("body",), ("geom",), ("joint", "freejoint"), ("site",)):
        names = [element.get("name") for tag in tags for element in root.iter(tag) if element.get("name")]
        assert len(names) == len(set(names)), (tags, names)
    shelf_geoms = [geom.get("name") for geom in root.find(".//body[@name='Shelf']").findall("geom")]
    assert any(name.startswith("Novel_") for name in shelf_geoms), shelf_geoms
    # A new part's body name carries its index among the new parts.
    assert any(body.get("name").startswith("Lantern_") for body in root.find("worldbody").findall("body"))
    assert root.find("./worldbody/body[@name='Hatch']/joint[@name='Hatch_hinge']") is not None
    assert root.find("./worldbody/body[@name='Tote']/freejoint[@name='Tote_free']") is not None
    assert root.find("./worldbody/body[@name='Cupboard']") is not None
    assert root.find("./worldbody/body[@name='Shelf']/freejoint") is None
    return {"messages": messages, "export": str(target)}


(OUT / "adopt_summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
if summary["failures"]:
    raise AssertionError("Failed sections: " + ", ".join(summary["failures"]))
harness.complete(OUT, "adopt", sections=summary["sections"])
print("ADOPT_FIXTURE_OK", json.dumps({"output": str(OUT), "sections": summary["sections"]}))
