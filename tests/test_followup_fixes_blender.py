"""Blender fixture for the adoption, weight, asset and registration follow-up fixes.

Run:
  blender --background --factory-startup --python-exit-code 1 \
      --python tests/test_followup_fixes_blender.py -- OUTPUT_DIR
Then validate the written worlds with MuJoCo:
  python tests/validate_followup_fixes.py OUTPUT_DIR

Every scene is generated here; nothing outside OUTPUT_DIR is written. Each
section runs even when an earlier one fails, and the run fails at the end
with every failing section listed.
"""
import importlib
import json
from pathlib import Path
import shutil
import struct
import sys
import traceback
import xml.etree.ElementTree as ET
import zlib

import bmesh
import bpy
from mathutils import Matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mujoco_world  # noqa: E402
from mujoco_world import exporter  # noqa: E402
from mujoco_world.baking import bake_static  # noqa: E402
from mujoco_world.exporter import ExportError, export_world  # noqa: E402
from mujoco_world.project import adopt_mjcf, group_selected  # noqa: E402

assert Path(mujoco_world.__file__).resolve().parent == ROOT / "mujoco_world", mujoco_world.__file__
mujoco_world.register()
args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
if not args:
    raise SystemExit("Pass an output directory after --")
OUT = Path(args[0]).resolve()
shutil.rmtree(OUT, ignore_errors=True)
OUT.mkdir(parents=True)
summary = {"sections": [], "failures": {}, "worlds": {}, "blender": bpy.app.version_string}
VERSION = ".".join(str(part) for part in mujoco_world.bl_info["version"])
CUBE_OBJ = ("v -0.1 -0.1 -0.1\nv 0.1 -0.1 -0.1\nv 0.1 0.1 -0.1\nv -0.1 0.1 -0.1\n"
            "v -0.1 -0.1 0.1\nv 0.1 -0.1 0.1\nv 0.1 0.1 0.1\nv -0.1 0.1 0.1\n"
            "f 1 3 2\nf 1 4 3\nf 5 6 7\nf 5 7 8\nf 1 2 6\nf 1 6 5\n"
            "f 2 3 7\nf 2 7 6\nf 3 4 8\nf 3 8 7\nf 4 1 5\nf 4 5 8\n")


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


def keep_world(obj, parent):
    """Set ``obj.parent`` without moving the object."""
    bpy.context.view_layer.update()
    world = obj.matrix_world.copy()
    obj.parent = parent
    obj.matrix_world = world


def box(name, pos, half, parent=None, role="STATIC", size=2.0, origin_shift=None):
    """A cube scaled to ``half``; ``origin_shift`` moves the mesh off the origin (local units)."""
    bpy.ops.mesh.primitive_cube_add(size=size, location=pos)
    obj = bpy.context.object
    obj.name, obj.scale = name, half
    obj.mujoco_world.role = role
    if origin_shift:
        obj.data.transform(Matrix.Translation(origin_shift))
    if parent:
        keep_world(obj, parent)
    return obj


def empty(name, pos, parent=None):
    obj = bpy.data.objects.new(name, None)
    bpy.context.scene.collection.objects.link(obj)
    obj.location = pos
    if parent:
        keep_world(obj, parent)
    return obj


def floor():
    bpy.ops.mesh.primitive_plane_add(size=10)
    bpy.context.object.name = "floor"
    return bpy.context.object


def png(path, rgb=(200, 120, 60), size=4, gray=False):
    """Write a small solid PNG without going through Blender."""
    pixel = bytes(rgb[:1]) if gray else bytes(rgb)
    raw = b"".join(b"\x00" + pixel * size for _ in range(size))
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    header = struct.pack(">IIBBBBB", size, size, 8, 0 if gray else 2, 0, 0, 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def write_world(directory, text):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "scene.xml"
    path.write_text(text, encoding="utf-8")
    return path


def export(target, name=None):
    """Export the current scene; record the world for the validator when ``name`` is given."""
    bpy.context.view_layer.update()
    report = export_world(bpy.context, str(target))
    if name:
        summary["worlds"][name] = str(target)
    return report


def expect_error(fragment, call):
    try:
        call()
    except ExportError as exc:
        assert fragment in str(exc), (fragment, str(exc))
        return str(exc)
    raise AssertionError(f"expected an error containing '{fragment}'")


def xml(path):
    return ET.parse(path).getroot()


def geom(root, name):
    return next(element for element in root.iter("geom") if element.get("name") == name)


def floats(text):
    return [float(v) for v in text.split()]


def close(a, b, tolerance=1e-6):
    return len(a) == len(b) and all(abs(x - y) <= tolerance for x, y in zip(a, b))


@section
def capsule_stretch():
    """A stretched capsule keeps its caps: the overall length scales, not just the cylinder part."""
    directory = OUT / "capsule"
    source = write_world(directory / "source", """<mujoco model="capsule">
  <compiler angle="radian"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <geom name="Rod" type="capsule" size="0.05 0.2" pos="0 0 0.5"/>
    <geom name="Peg" type="capsule" size="0.05 0.2" pos="1 0 0.5"/>
  </worldbody>
</mujoco>
""")
    fresh_scene()
    floor()
    for name, x in (("Rod", 0), ("Peg", 1)):
        bpy.ops.mesh.primitive_cylinder_add(radius=0.05, depth=0.5, location=(x, 0, 0.5))
        bpy.context.object.name = name
    adopt_mjcf(bpy.context, str(source))
    scene = bpy.context.scene
    scene.objects["Rod"].scale.z = 2.0
    report = export(directory / "scene.xml", "capsule")
    size = floats(geom(xml(directory / "scene.xml"), "Rod").get("size"))
    # The visual is 1.0 m long; the old code wrote 0.05 0.4 (0.9 m).
    assert close(size, [0.05, 0.45]), size
    assert geom(xml(directory / "scene.xml"), "Peg").get("size") == "0.05 0.2"
    scene.objects["Peg"].scale.z = 0.1
    message = expect_error("shorter than it is wide", lambda: export(directory / "short" / "scene.xml"))
    return {"rod_size": size, "error": message, "addon_version": report["addon_version"]}


@section
def rejected_structures():
    """Adoption refuses <frame>, <replicate> and nested <include> and leaves the scene unadopted."""
    directory = OUT / "rejected"
    worlds = {
        "frame": """<mujoco><worldbody><geom name="floor" type="plane" size="5 5 0.1"/>
  <frame pos="1 0 0"><geom name="g2" type="box" size="0.1 0.1 0.1"/></frame></worldbody></mujoco>""",
        "replicate": """<mujoco><worldbody><geom name="floor" type="plane" size="5 5 0.1"/>
  <replicate count="2" offset="0.3 0 0"><geom name="r" type="box" size="0.1 0.1 0.1"/></replicate></worldbody></mujoco>""",
        "include": """<mujoco><worldbody><geom name="floor" type="plane" size="5 5 0.1"/>
  <include file="part.xml"/></worldbody></mujoco>""",
    }
    fragments = {"frame": "<frame>", "replicate": "<replicate>", "include": "Expand MJCF includes"}
    messages = {}
    for name, text in worlds.items():
        fresh_scene()
        floor()
        source = write_world(directory / name, text)
        (directory / name / "part.xml").write_text('<mujoco><geom name="shelf" type="box" size="0.1 0.1 0.1"/></mujoco>')
        messages[name] = expect_error(fragments[name], lambda: adopt_mjcf(bpy.context, str(source)))
        assert "mjworld_mjcf_template" not in bpy.context.scene, name
    return messages


@section
def joint_types():
    """Untyped joints are hinges; slide joints, typed directly or by class, keep their settings."""
    directory = OUT / "joints"
    source = write_world(directory / "source", """<mujoco model="joints">
  <default>
    <default class="slider"><joint type="slide" axis="1 0 0"/></default>
  </default>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="Door" pos="0 1 0.5">
      <joint name="Door_joint" axis="0 0 1" range="-90 0" damping="2"/>
      <geom name="Door_panel" type="box" size="0.3 0.02 0.4" pos="0.3 0 0" mass="5"/>
    </body>
    <body name="Drawer" pos="2 0 0.5">
      <joint name="Drawer_slide" type="slide" axis="1 0 0" range="0 0.4"/>
      <geom name="Drawer_box" type="box" size="0.2 0.2 0.1" mass="1"/>
    </body>
    <body name="Tray" pos="-2 0 0.5">
      <joint name="Tray_slide" class="slider" range="0 0.3"/>
      <geom name="Tray_box" type="box" size="0.2 0.2 0.02" mass="1"/>
    </body>
  </worldbody>
</mujoco>
""")
    fresh_scene()
    floor()
    for body, part, pos, offset, half in (("Door", "Door_panel", (0, 1, 0.5), (0.3, 0, 0), (0.3, 0.02, 0.4)),
                                          ("Drawer", "Drawer_box", (2, 0, 0.5), (0, 0, 0), (0.2, 0.2, 0.1)),
                                          ("Tray", "Tray_box", (-2, 0, 0.5), (0, 0, 0), (0.2, 0.2, 0.02))):
        owner = empty(body, pos)
        box(part, tuple(p + o for p, o in zip(pos, offset)), half, parent=owner)
    adopt_mjcf(bpy.context, str(source))
    scene = bpy.context.scene
    door, drawer = scene.objects["Door"].mujoco_world, scene.objects["Drawer"].mujoco_world
    # The untyped joint was skipped before: its limits came out as the add-on default.
    assert door.role == "HINGE" and (door.hinge_min, door.hinge_max) == (-90, 0) and abs(door.damping - 2) < 1e-6, (
        door.hinge_min, door.hinge_max, door.damping)
    assert drawer.role == "HINGE" and scene.objects["Tray"].mujoco_world.role == "HINGE"
    door.hinge_min, door.damping = -45, 0.5
    drawer.hinge_min = -30
    report = export(directory / "scene.xml", "joints")
    root = xml(directory / "scene.xml")
    joints = {joint.get("name"): joint.attrib for joint in root.iter("joint")}
    assert joints["Door_joint"]["range"] == "-45 0" and joints["Door_joint"]["damping"] == "0.5", joints
    assert "type" not in joints["Door_joint"], joints
    assert joints["Drawer_slide"] == {"name": "Drawer_slide", "type": "slide", "axis": "1 0 0", "range": "0 0.4"}, joints
    assert joints["Tray_slide"] == {"name": "Tray_slide", "class": "slider", "range": "0 0.3"}, joints
    notes = [w for w in report["warnings"] if "door settings were not applied" in w]
    assert len(notes) == 1 and notes[0].startswith("Drawer:"), report["warnings"]
    return {"joints": joints, "warning": notes[0]}


@section
def class_types():
    """Type and size inherited from a <default> class: meshes regenerate, boxes scale, a class plane counts."""
    directory = OUT / "classes"
    source = write_world(directory / "source", """<mujoco model="classes">
  <compiler angle="radian" meshdir="meshes"/>
  <default>
    <default class="vis"><geom type="mesh" contype="0" conaffinity="0" group="1"/></default>
    <default class="slab"><geom type="box" size="0.2 0.1 0.05"/></default>
    <default class="ground"><geom type="plane" size="5 5 0.1"/></default>
  </default>
  <asset><mesh name="m" file="m.obj"/></asset>
  <worldbody>
    <geom name="floor" class="ground"/>
    <body name="Shelf" pos="0 0 0.5">
      <geom name="Shelf_slab" class="slab"/>
      <geom name="Shelf_shell" class="vis" mesh="m" pos="0 0 0.2"/>
    </body>
  </worldbody>
</mujoco>
""")
    (directory / "source" / "meshes").mkdir(parents=True, exist_ok=True)
    (directory / "source" / "meshes" / "m.obj").write_text(CUBE_OBJ)
    fresh_scene()
    floor()
    shelf = empty("Shelf", (0, 0, 0.5))
    box("Shelf_slab", (0, 0, 0.5), (0.2, 0.1, 0.05), parent=shelf)
    box("Shelf_shell", (0, 0, 0.7), (1, 1, 1), parent=shelf, size=0.2)
    adopt_mjcf(bpy.context, str(source))
    scene = bpy.context.scene
    assert scene.objects["Shelf_shell"].mujoco_world.role == "VISUAL", scene.objects["Shelf_shell"].mujoco_world.role
    scene.objects["Shelf_slab"].scale.x *= 2
    export(directory / "scene.xml", "classes")
    root = xml(directory / "scene.xml")
    slab, shell = geom(root, "Shelf_slab"), geom(root, "Shelf_shell")
    assert close(floats(slab.get("size")), [0.4, 0.1, 0.05]) and slab.get("class") == "slab", slab.attrib
    assert shell.get("mesh") == "Shelf_shell_authored_mesh" and "type" not in shell.attrib, shell.attrib
    assert [m.get("name") for m in root.iter("mesh")] == ["Shelf_shell_authored_mesh"]
    return {"slab": slab.attrib, "shell": shell.attrib}


@section
def moved_classes():
    """Geoms and bodies moved out of a childclass body keep its class; moves are reported."""
    directory = OUT / "moved"
    source = write_world(directory / "source", """<mujoco model="moved">
  <compiler angle="radian"/>
  <default>
    <default class="grippy"><geom friction="0.9 0.01 0.001" condim="4"/></default>
  </default>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="Shelf" pos="0 0 0.5" childclass="grippy">
      <geom name="Shelf_box" type="box" size="0.2 0.2 0.05"/>
      <geom name="Shelf_lip" type="box" size="0.2 0.02 0.05" pos="0 0.2 0.1"/>
      <body name="Shelf_drawer" pos="0 0 -0.2">
        <geom name="Shelf_drawer_box" type="box" size="0.15 0.15 0.05"/>
      </body>
    </body>
    <body name="Crate" pos="2 0 0.3">
      <freejoint name="Crate_free"/>
      <inertial pos="0 0 0" mass="2" diaginertia="0.01 0.01 0.01"/>
      <geom name="Crate_box" type="box" size="0.2 0.2 0.2"/>
    </body>
  </worldbody>
</mujoco>
""")
    fresh_scene()
    floor()
    shelf = empty("Shelf", (0, 0, 0.5))
    shelf_box = box("Shelf_box", (0, 0, 0.5), (0.2, 0.2, 0.05), parent=shelf)
    box("Shelf_lip", (0, 0.2, 0.6), (0.2, 0.02, 0.05))  # never parented under Shelf
    drawer = empty("Shelf_drawer", (0, 0, 0.3), parent=shelf)
    box("Shelf_drawer_box", (0, 0, 0.3), (0.15, 0.15, 0.05), parent=drawer)
    crate = empty("Crate", (2, 0, 0.3))
    crate_box = box("Crate_box", (2, 0, 0.3), (0.2, 0.2, 0.2), parent=crate)
    result = adopt_mjcf(bpy.context, str(source))
    assert result["warnings"] == ["Shelf_lip: MJCF geom 'Shelf_lip' belongs to body 'Shelf', but its Blender object is under "
                                  "the world body; the export follows Blender."], result["warnings"]
    for obj in (shelf_box, drawer, crate_box):
        keep_world(obj, None)
    report = export(directory / "scene.xml", "moved")
    root = xml(directory / "scene.xml")
    world = root.find("worldbody")
    assert geom(root, "Shelf_box").get("class") == "grippy" and geom(root, "Shelf_lip").get("class") == "grippy"
    assert geom(root, "Shelf_box") in list(world)
    moved_body = next(body for body in root.iter("body") if body.get("name") == "Shelf_drawer")
    assert moved_body in list(world) and moved_body.get("childclass") == "grippy", moved_body.attrib
    assert "class" not in geom(root, "Shelf_drawer_box").attrib
    shelf_body = next(body for body in root.iter("body") if body.get("name") == "Shelf")
    assert shelf_body.get("childclass") == "grippy" and not shelf_body.findall("geom")
    moved_notes = [w for w in report["warnings"] if "in the source MJCF, but is exported in" in w]
    assert any(w.startswith("Shelf_box: adopted geom 'Shelf_box' belongs to body 'Shelf'") for w in moved_notes), moved_notes
    assert any(w.startswith("Crate_box:") and "body 'Crate'" in w for w in moved_notes), moved_notes
    assert any(w.startswith("Crate: this moving body has no colliding geoms left") for w in report["warnings"]), report["warnings"]
    return {"warnings": report["warnings"]}


@section
def name_maps():
    """body_map and geom_map separate a body and a geom sharing a name; two elements on one object fail."""
    directory = OUT / "names"
    source = write_world(directory / "source", """<mujoco model="names">
  <compiler angle="radian"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="table" pos="0 0 0.4">
      <geom name="table" type="box" size="0.5 0.3 0.02"/>
    </body>
    <geom name="g1" type="box" size="0.1 0.1 0.1" pos="2 0 0.1"/>
    <geom name="g2" type="box" size="0.1 0.1 0.1" pos="3 0 0.1"/>
  </worldbody>
</mujoco>
""")
    fresh_scene()
    floor()
    table = empty("Table empty", (0, 0, 0.4))
    box("Table top", (0, 0, 0.4), (0.5, 0.3, 0.02), parent=table)
    box("Thing", (2, 0, 0.1), (0.1, 0.1, 0.1))
    box("Thing 2", (3, 0, 0.1), (0.1, 0.1, 0.1))
    maps = {"body_map": {"table": "Table empty"}, "geom_map": {"table": "Table top"}}
    message = expect_error("geom 'g1' and geom 'g2' both match 'Thing'", lambda: adopt_mjcf(
        bpy.context, str(source), object_map={"g1": "Thing", "g2": "Thing"}, **maps))
    assert "mjworld_mjcf_template" not in bpy.context.scene
    adopt_mjcf(bpy.context, str(source), object_map={"g1": "Thing", "g2": "Thing 2"}, **maps)
    export(directory / "scene.xml", "names")
    root = xml(directory / "scene.xml")
    body = next(b for b in root.iter("body") if b.get("name") == "table")
    assert [g.get("name") for g in body.findall("geom")] == ["table"]
    assert close(floats(geom(root, "g2").get("pos")), [3, 0, 0.1])
    return {"error": message}


@section
def mesh_names():
    """Thai and colliding geom names get distinct mesh assets."""
    directory = OUT / "mesh_names"
    names = ["โต๊ะ", "ชั้นวาง", "a b", "a_b"]
    geoms = "\n".join(f'    <geom name="{name}" type="mesh" mesh="m" pos="{i} 0 0.1"/>' for i, name in enumerate(names))
    source = write_world(directory / "source", f"""<mujoco model="mesh_names">
  <compiler angle="radian" meshdir="meshes"/>
  <asset><mesh name="m" file="m.obj"/></asset>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
{geoms}
  </worldbody>
</mujoco>
""")
    (directory / "source" / "meshes").mkdir(parents=True, exist_ok=True)
    (directory / "source" / "meshes" / "m.obj").write_text(CUBE_OBJ)
    fresh_scene()
    floor()
    for i, name in enumerate(names):
        # Different sizes, so every object needs its own mesh asset.
        box(name, (i, 0, 0.1), (1, 1, 1), size=0.1 + 0.02 * i)
    adopt_mjcf(bpy.context, str(source))
    export(directory / "scene.xml", "mesh_names")
    root = xml(directory / "scene.xml")
    meshes = {g.get("name"): g.get("mesh") for g in root.iter("geom") if g.get("mesh")}
    assert len(set(meshes.values())) == 4, meshes
    assert meshes["a b"] == "a_b_authored_mesh" and meshes["a_b"].startswith("a_b_") and meshes["a_b"] != "a_b_authored_mesh", meshes
    assert all(meshes[name].startswith("object_") and meshes[name] != "object_authored_mesh" for name in names[:2]), meshes
    files = [m.get("file") for m in root.iter("mesh")]
    assert len(set(files)) == 4 and all((directory / f).exists() for f in files), files
    return {"meshes": meshes}


@section
def origin_offset():
    """A primitive whose Blender origin is not at the geom centre exports where the source put it."""
    directory = OUT / "offsets"
    source = write_world(directory / "source", """<mujoco model="offsets">
  <compiler angle="radian"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <geom name="Table_top" type="box" size="0.5 0.3 0.02" pos="0 0 0.7"/>
    <body name="Cart" pos="2 0 0.2">
      <freejoint name="Cart_free"/>
      <geom name="Cart_box" type="box" size="0.2 0.2 0.1" pos="0 0 0.1" mass="3"/>
    </body>
  </worldbody>
</mujoco>
""")
    fresh_scene()
    floor()
    # Origins at the bottom face, as furniture is often modelled.
    table = box("Table_top", (0, 0, 0.68), (0.5, 0.3, 0.02), origin_shift=(0, 0, 1))
    cart = empty("Cart", (2, 0, 0.2))
    box("Cart_box", (2, 0, 0.2), (0.2, 0.2, 0.1), parent=cart, origin_shift=(0, 0, 1))
    adopt_mjcf(bpy.context, str(source))
    assert close(list(table["mjworld_mjcf_geom_offset"]), [0, 0, 1], 1e-5), list(table["mjworld_mjcf_geom_offset"])
    export(directory / "untouched" / "scene.xml", "offsets_untouched")
    root = xml(directory / "untouched" / "scene.xml")
    # Untouched geoms keep their source text; the old export wrote 0 0 0.680000007.
    assert geom(root, "Table_top").get("pos") == "0 0 0.7", geom(root, "Table_top").attrib
    assert geom(root, "Cart_box").get("pos") == "0 0 0.1", geom(root, "Cart_box").attrib
    table.location.x += 0.25
    table.scale.z *= 2
    cart.location.y += 0.5
    export(directory / "edited" / "scene.xml", "offsets_edited")
    root = xml(directory / "edited" / "scene.xml")
    top = geom(root, "Table_top")
    # The bottom stays at 0.68, so the doubled top is centred at 0.72.
    assert close(floats(top.get("pos")), [0.25, 0, 0.72]) and close(floats(top.get("size")), [0.5, 0.3, 0.04]), top.attrib
    assert close(floats(geom(root, "Cart_box").get("pos")), [0, 0, 0.1]), geom(root, "Cart_box").attrib
    cart_body = next(b for b in root.iter("body") if b.get("name") == "Cart")
    assert close(floats(cart_body.get("pos")), [2, 0.5, 0.2]), cart_body.attrib
    return {"table": top.attrib}


def instancing_tree():
    """A Geometry Nodes group that keeps the mesh and adds cube instances on its points."""
    tree = bpy.data.node_groups.new("Scatter cubes", "GeometryNodeTree")
    tree.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    tree.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    nodes, links = tree.nodes, tree.links
    group_in, group_out = nodes.new("NodeGroupInput"), nodes.new("NodeGroupOutput")
    points, cube, join = nodes.new("GeometryNodeInstanceOnPoints"), nodes.new("GeometryNodeMeshCube"), nodes.new("GeometryNodeJoinGeometry")
    cube.inputs["Size"].default_value = (0.05, 0.05, 0.05)
    links.new(group_in.outputs[0], points.inputs["Points"])
    links.new(cube.outputs["Mesh"], points.inputs["Instance"])
    links.new(group_in.outputs[0], join.inputs[0])
    links.new(points.outputs["Instances"], join.inputs[0])
    links.new(join.outputs[0], group_out.inputs[0])
    return tree


def unexportable(prefix):
    """Add a text object, a curve and a Geometry Nodes instancer; return their names."""
    scene = bpy.context.scene
    text = bpy.data.objects.new(prefix + "Label", bpy.data.curves.new(prefix + "Label", "FONT"))
    curve_data = bpy.data.curves.new(prefix + "Rail", "CURVE")
    curve_data.splines.new("POLY").points.add(1)
    curve = bpy.data.objects.new(prefix + "Rail", curve_data)
    for obj in (text, curve):
        scene.collection.objects.link(obj)
    host = box(prefix + "Scatter", (3, 3, 0.1), (0.1, 0.1, 0.1))
    host.modifiers.new("Scatter", "NODES").node_group = instancing_tree()
    return text.name, curve.name, host.name


@section
def dropped_warnings():
    """Text, curves, instancers and Geometry Nodes instances are reported in both export paths."""
    directory = OUT / "dropped"
    fresh_scene()
    box("Crate", (0, 0, 0.2), (0.2, 0.2, 0.2))
    label, rail, scatter = unexportable("")
    hidden = bpy.data.collections.new("Instanced")
    hidden.objects.link(bpy.data.objects.new("Inside", bpy.data.meshes.new("Inside")))
    copies = bpy.data.objects.new("Copies", None)
    copies.instance_type, copies.instance_collection = "COLLECTION", hidden
    bpy.context.scene.collection.objects.link(copies)
    report = export(directory / "simple" / "scene.xml")
    expected = [f"{label}: convert this font to mesh", f"{rail}: convert this curve to mesh",
                f"{scatter}: its Geometry Nodes instances are not exported", "Copies: instances are not exported"]
    for fragment in expected:
        assert any(w.startswith(fragment) for w in report["warnings"]), (fragment, report["warnings"])
    assert report["addon_version"] == VERSION, report["addon_version"]
    source = write_world(directory / "adopted_source", """<mujoco model="dropped">
  <compiler angle="radian"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <geom name="Crate" type="box" size="0.2 0.2 0.2" pos="0 0 0.2"/>
  </worldbody>
</mujoco>
""")
    fresh_scene()
    floor()
    box("Crate", (0, 0, 0.2), (0.2, 0.2, 0.2))
    adopt_mjcf(bpy.context, str(source))
    label, rail, scatter = unexportable("Adopted ")
    adopted = export(directory / "adopted" / "scene.xml", "dropped_adopted")
    for fragment in (f"{label}: convert this font", f"{rail}: convert this curve", f"{scatter}: its Geometry Nodes instances"):
        assert sum(w.startswith(fragment) for w in adopted["warnings"]) == 1, (fragment, adopted["warnings"])
    assert adopted["addon_version"] == VERSION, adopted["addon_version"]
    return {"simple": report["warnings"], "adopted": adopted["warnings"]}


def closed_volume(obj):
    """Volume of a closed convex mesh object in world space (bmesh, no hull)."""
    mesh = bmesh.new()
    mesh.from_mesh(obj.data)
    mesh.transform(obj.matrix_world)
    try:
        return abs(mesh.calc_volume())
    finally:
        mesh.free()


@section
def volume_mass():
    """A grouped body's Weight is split by collider volume, so its centre of mass follows the bulk."""
    directory = OUT / "volume"
    fresh_scene()
    slab = box("Slab", (0, 0, 0.1), (0.25, 0.25, 0.05))
    knob = box("Knob", (0, 0, 1.0), (0.005, 0.005, 0.005))
    bpy.ops.mesh.primitive_ico_sphere_add(radius=0.1, subdivisions=2, location=(0, 0, 0.5))
    ball = bpy.context.object
    ball.name = "Ball"
    ball.mujoco_world.collision = "CONVEX"
    bpy.ops.object.select_all(action="DESELECT")
    for obj in (slab, knob, ball):
        obj.select_set(True)
    group = group_selected(bpy.context, "FREE")
    group.mujoco_world.mass = 3.0
    export(directory / "scene.xml", "volume")
    root = xml(directory / "scene.xml")
    body = next(b for b in root.iter("body") if b.get("name") == group.name)
    masses = {g.get("name"): float(g.get("mass")) for g in body.findall("geom") if g.get("contype") != "0"}
    volumes = {"Slab": 0.025, "Knob": 1e-6, "Ball": closed_volume(ball)}
    total = sum(volumes.values())
    by_part = {next(part for part in volumes if name.startswith(part)): mass for name, mass in masses.items()}
    for part, volume in volumes.items():
        assert abs(by_part[part] - 3.0 * volume / total) < 1e-6 * 3.0, (part, by_part, volumes)
    assert abs(sum(masses.values()) - 3.0) < 1e-6, masses
    centre = sum(3.0 * volumes[p] / total * z for p, z in (("Slab", 0.1), ("Knob", 1.0), ("Ball", 0.5))) / 3.0
    return {"body": group.name, "masses": by_part, "volumes": volumes, "expected_com_z": centre}


@section
def moved_blend_source():
    """A blend saved elsewhere still finds the adopted source through the stored absolute path."""
    directory = OUT / "moved_blend"
    source_dir = directory / "source"
    source = write_world(source_dir, """<mujoco model="moved_blend">
  <compiler angle="radian" texturedir="tex"/>
  <asset><texture name="wood" type="2d" file="wood.png"/><material name="wood_mat" texture="wood"/></asset>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <geom name="Crate" type="box" size="0.2 0.2 0.2" pos="0 0 0.2" material="wood_mat"/>
  </worldbody>
</mujoco>
""")
    png(source_dir / "tex" / "wood.png")
    fresh_scene()
    floor()
    box("Crate", (0, 0, 0.2), (0.2, 0.2, 0.2))
    (directory / "a").mkdir(parents=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(directory / "a" / "project.blend"))
    adopt_mjcf(bpy.context, str(source))
    assert bpy.context.scene["mjworld_mjcf_source"].startswith("//"), bpy.context.scene["mjworld_mjcf_source"]
    (directory / "b" / "deeper").mkdir(parents=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(directory / "b" / "deeper" / "project.blend"))
    report = export(directory / "b" / "deeper" / "out" / "scene.xml", "moved_blend")
    texture = next(t for t in xml(directory / "b" / "deeper" / "out" / "scene.xml").iter("texture"))
    assert texture.get("file") == "scene_assets/wood.png", texture.attrib
    source_dir.rename(directory / "source_moved")
    message = expect_error("adopt the world again", lambda: export(directory / "b" / "deeper" / "broken" / "scene.xml"))
    assert str(source_dir / "scene.xml") in message, message
    return {"warnings": report["warnings"], "error": message}


@section
def manifest_cleanup():
    """A per-XML asset folder drops the files of its previous export, and nothing else."""
    directory = OUT / "manifest"
    fresh_scene()
    for i in range(3):
        box(f"Crate_{i}", (i, 0, 0.2), (0.2, 0.2, 0.2), role="FREE")
    target = directory / "scene.xml"
    export(target)
    assets = directory / "scene_assets"
    foreign = assets / "notes.txt"
    foreign.write_text("kept")
    outside = directory / "outside.txt"
    outside.write_text("kept")
    manifest = assets / ".mujoco_world_manifest.json"
    recorded = json.loads(manifest.read_text())["files"]
    first = sorted(p.name for p in assets.iterdir() if p.name != manifest.name and p != foreign)
    assert recorded == first, (recorded, first)
    # Only plain file names in the manifest are trusted.
    manifest.write_text(json.dumps({"files": recorded + ["../outside.txt", ".hidden"]}))
    bpy.data.objects.remove(bpy.context.scene.objects["Crate_0"])
    bpy.context.scene.objects["Crate_2"].name = "Crate_two"
    export(target, "manifest")
    root = xml(target)
    referenced = sorted(Path(e.get("file")).name for e in root.iter() if e.get("file"))
    left = sorted(p.name for p in assets.iterdir() if p.name not in {manifest.name, foreign.name})
    assert left == referenced, (left, referenced)
    assert json.loads(manifest.read_text())["files"] == referenced
    assert foreign.exists() and outside.exists()
    return {"first": first, "second": referenced}


@section
def asset_dirs():
    """assetdir, texturedir precedence, strippath, cube texture faces and height fields resolve like MuJoCo."""
    directory = OUT / "asset_dirs"
    sky = {side: f"sky_{side[4]}.png" for side in ("fileright", "fileleft", "fileup", "filedown", "filefront", "fileback")}
    worlds = {
        "assetdir": ("""<compiler angle="radian" assetdir="assets"/>
  <asset>
    <texture name="panel" type="2d" file="panel.png"/>
    <texture name="sky" type="skybox" """ + " ".join(f'{k}="{v}"' for k, v in sky.items()) + """/>
    <hfield name="terrain" file="terrain.png" size="1 1 0.1 0.05"/>
    <material name="panel_mat" texture="panel"/>
  </asset>""", ["assets/panel.png", "assets/terrain.png"] + [f"assets/{v}" for v in sky.values()]),
        "strippath": ("""<compiler angle="radian" assetdir="assets" texturedir="textures" strippath="true"/>
  <asset>
    <texture name="panel" type="2d" file="old/location/panel.png"/>
    <material name="panel_mat" texture="panel"/>
  </asset>""", ["textures/panel.png"]),
    }
    results = {}
    for name, (header, files) in worlds.items():
        source = write_world(directory / name / "source", f"""<mujoco model="{name}">
  {header}
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <geom name="Panel" type="box" size="0.2 0.2 0.2" pos="0 0 0.2" material="panel_mat"/>
  </worldbody>
</mujoco>
""")
        for file in files:
            png(directory / name / "source" / file, gray="terrain" in file)
        summary["worlds"][name + "_source"] = str(source)
        fresh_scene()
        floor()
        box("Panel", (0, 0, 0.2), (0.2, 0.2, 0.2))
        adopt_mjcf(bpy.context, str(source))
        target = directory / name / "export" / "scene.xml"
        export(target, name)
        root = xml(target)
        compiler = root.find("compiler").attrib
        assert not {"assetdir", "meshdir", "texturedir", "strippath"} & set(compiler), compiler
        written = {f"{e.tag}:{k}": v for e in root.find("asset") for k, v in e.attrib.items() if k.startswith("file")}
        assert all(v.startswith("scene_assets/") and (target.parent / v).exists() for v in written.values()), written
        results[name] = written
    assert results["assetdir"]["texture:fileright"] == "scene_assets/sky_right.png", results["assetdir"]
    assert results["assetdir"]["hfield:file"] == "scene_assets/terrain.png", results["assetdir"]
    return results


@section
def rebake_names():
    """Rebaking keeps the atlas, image, material and collection names instead of alternating with '.001'."""
    scene = fresh_scene()
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 2
    world = bpy.data.worlds.new("Fixture world")
    scene.world = world
    if world.node_tree is None:
        world.use_nodes = True  # Blender 4.x; Blender 5 worlds always have a tree.
    box("Wall", (0, 0, 0.5), (1, 0.05, 0.5))
    rounds = []
    for _ in range(2):
        result = bake_static(bpy.context, str(OUT / "rebake"), atlas_size=32, samples=1, margin=1)
        names = {"object": result["object"],
                 "images": sorted(i.name for i in bpy.data.images if i.get("mjworld_generated_bake")),
                 "materials": sorted(m.name for m in bpy.data.materials if m.get("mjworld_generated_bake")),
                 "collections": sorted(c.name for c in bpy.data.collections if c.get("mjworld_generated_bake"))}
        rounds.append(names)
    expected = {"object": "MuJoCo baked static visual", "images": ["static_lighting_linear.exr"],
                "materials": ["MuJoCo baked static lighting"], "collections": ["MuJoCo baked static visual"]}
    assert rounds == [expected, expected], rounds
    return rounds


@section
def registration():
    """register() and unregister() are re-entrant, and reloading the package reloads its submodules."""
    menu = bpy.types.TOPBAR_MT_file_export
    def entries():
        return getattr(menu.draw, "_draw_funcs", []).count(mujoco_world.export_menu)
    mujoco_world.register()
    assert entries() == 1, entries()
    mujoco_world.unregister()
    mujoco_world.unregister()
    assert not hasattr(bpy.types.Scene, "mujoco_world") and not hasattr(bpy.types.Object, "mujoco_world")
    assert entries() == 0
    # Reload Scripts re-runs __init__.py in its old namespace; the submodule
    # must be executed again too, which restores the function removed here.
    exporter.addon_version = None
    importlib.reload(mujoco_world)
    assert callable(sys.modules["mujoco_world.exporter"].addon_version), "exporter was not reloaded"
    mujoco_world.register()
    assert hasattr(bpy.types.Scene, "mujoco_world") and entries() == 1
    assert mujoco_world.bl_info["author"] == "Walkie contributors"
    return {"version": VERSION}


(OUT / "followup.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
if summary["failures"]:
    raise AssertionError("Failed sections: " + ", ".join(summary["failures"]))
print("FOLLOWUP_FIXES_OK", json.dumps({"output": str(OUT), "sections": summary["sections"]}))
