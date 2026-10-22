"""Blender fixture for the static bake, texture export and operator fixes.

Run:
  blender --background --factory-startup --python-exit-code 1 \
      --python tests/test_review_fixes_blender.py -- OUTPUT_DIR
Then validate the written worlds with MuJoCo:
  python tests/validate_review_fixes.py OUTPUT_DIR

Every scene is generated here; nothing outside OUTPUT_DIR is written. Each
section runs even when an earlier one fails, and the run fails at the end
with every failing section listed.
"""
from array import array
import json
import os
from pathlib import Path
import struct
import sys
import traceback
import warnings
import xml.etree.ElementTree as ET
import zlib

import addon_utils
import bpy

sys.dont_write_bytecode = True  # Blender ignores PYTHONDONTWRITEBYTECODE.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness  # noqa: E402

OUT, _ = harness.begin("review_fixes")
import mujoco_world  # noqa: E402
from mujoco_world import baking  # noqa: E402
from mujoco_world.baking import BakeError, bake_covered, bake_leftovers, bake_static, restore_sources  # noqa: E402
from mujoco_world.exporter import ExportError, export_world, srgb  # noqa: E402
from mujoco_world.project import adopt_mjcf  # noqa: E402

summary = {"sections": [], "failures": {}, "worlds": {}, "blender": bpy.app.version_string}
PINNED = {"view_transform": "AgX", "look": "None", "exposure": 0.0, "gamma": 1.0, "display_device": "sRGB"}


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


def bake_scene(strength=3.0):
    """An empty scene lit by a uniform white world, rendered by Cycles."""
    scene = fresh_scene()
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 2
    scene.cycles.seed = 0
    world = bpy.data.worlds.new("Fixture world")
    scene.world = world
    if world.node_tree is None:
        world.use_nodes = True  # Blender 4.x; Blender 5 worlds always have a tree.
    background = world.node_tree.nodes["Background"]
    background.inputs["Color"].default_value = (1, 1, 1, 1)
    background.inputs["Strength"].default_value = strength
    return scene


def principled(material):
    return next(node for node in material.node_tree.nodes if node.type == "BSDF_PRINCIPLED")


def material(name, rgb=(0.8, 0.8, 0.8), metallic=0.0, emission=None, strength=0.0):
    result = bpy.data.materials.new(name)
    if bpy.app.version < (5, 0, 0):
        result.use_nodes = True  # Blender 5 always uses the node tree.
    shader = principled(result)
    shader.inputs["Base Color"].default_value = (*rgb, 1)
    shader.inputs["Metallic"].default_value = metallic
    if emission:
        shader.inputs["Emission Color"].default_value = (*emission, 1)
        shader.inputs["Emission Strength"].default_value = strength
    return result


def image_material(name, image, *, via=None):
    """Principled material whose Base Color comes from ``image``."""
    result = material(name)
    tree = result.node_tree
    node = tree.nodes.new("ShaderNodeTexImage")
    node.image = image
    target = principled(result).inputs["Base Color"]
    if via == "reroute":
        reroute = tree.nodes.new("NodeReroute")
        tree.links.new(node.outputs["Color"], reroute.inputs[0])
        tree.links.new(reroute.outputs[0], target)
    else:
        link = tree.links.new(node.outputs["Color"], target)
        if via == "muted_link":
            link.is_muted = True
        elif via == "muted_node":
            node.mute = True
    return result


def box(name, pos, half, role="STATIC", mat=None, parent=None):
    bpy.ops.mesh.primitive_cube_add(size=2, location=pos)
    obj = bpy.context.object
    obj.name, obj.scale = name, half
    obj.mujoco_world.role = role
    if mat:
        obj.data.materials.append(mat)
    if parent:
        bpy.context.view_layer.update()
        world = obj.matrix_world.copy()
        obj.parent = parent
        obj.matrix_world = world
    return obj


def wire(name):
    """A visible mesh with edges but no faces, such as a guide line."""
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata([(0, 0, 0), (1, 0, 0), (1, 1, 0)], [(0, 1), (1, 2)], [])
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def texels(image):
    pixels = array("f", [0]) * len(image.pixels)
    image.pixels.foreach_get(pixels)
    return [tuple(pixels[i:i + 4]) for i in range(0, len(pixels), 4)]


def atlas_image(result):
    atlas = bpy.data.objects[result["object"]]
    return atlas.data.materials[0].node_tree.nodes.get("Image Texture").image


def xml(path):
    return ET.parse(path).getroot()


def record(name, path):
    summary["worlds"][name] = str(path)
    return path


def visual_geoms(path, prefix):
    return [geom for body in xml(path).iter("body") if body.get("name", "").startswith(prefix)
            for geom in body.findall("geom") if geom.get("group") == "1"]


def texture_file(path, element):
    compiler = xml(path).find("compiler")
    return path.parent / compiler.get("texturedir", "") / element.get("file")


def bytes_rgb(path):
    """8-bit RGB of every pixel of a PNG, read through Blender."""
    image = bpy.data.images.load(str(path), check_existing=False)
    try:
        assert not image.is_float, path
        return [tuple(round(c * 255) for c in texel[:3]) for texel in texels(image)]
    finally:
        bpy.data.images.remove(image)


def data_counts():
    """Names of every image, material, mesh, object and collection."""
    return {kind: sorted(item.name for item in getattr(bpy.data, kind))
            for kind in ("images", "materials", "meshes", "objects", "collections")}


def write_png16(path, width, height, value):
    """A 16-bit RGB PNG without colour chunks; Blender loads it as a float image."""
    rows = b"".join(b"\x00" + b"".join(struct.pack(">HHH", *value(x, y)) for x in range(width)) for y in range(height))

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 16, 2, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))


# ---------------------------------------------------------------- baking


@section
def bake_metal_emission():
    """Metal bakes its base colour as diffuse, and emission reaches the atlas."""
    bake_scene()
    box("Floor", (0, 0, -0.05), (1.5, 1.5, 0.05), "VISUAL", material("grey", (0.3, 0.3, 0.3)))
    metal = box("Metal", (-0.5, 0, 0.3), (0.25, 0.25, 0.25), mat=material("red metal", (0.9, 0.05, 0.03), metallic=1.0))
    box("Panel", (0.5, 0, 0.3), (0.25, 0.25, 0.25),
        mat=material("green panel", (0.02, 0.02, 0.02), emission=(0.05, 1, 0.05), strength=8))
    result = bake_static(bpy.context, str(OUT / "metal_bake"), atlas_size=64, samples=2, margin=1)
    pixels = texels(atlas_image(result))
    red = sum(1 for r, g, b, _a in pixels if r > 0.3 and r > 3 * g and r > 3 * b)
    green = sum(1 for r, g, b, _a in pixels if g > 2 and g > 3 * r)
    assert red > 30, f"metal baked black: {red} red texels"
    assert green > 30, f"emission missing: {green} bright green texels"
    # The metal's own material is untouched; only the private copy changed.
    assert principled(metal.data.materials[0]).inputs["Metallic"].default_value == 1.0
    assert "metallic" in result["passes"] and "emission" in result["passes"]
    return {"red_texels": red, "green_texels": green, "passes": result["passes"]}


@section
def material_copy_private():
    """The bake copy zeroes Metallic (and its link) but keeps Transmission."""
    fresh_scene()
    source = material("glassy metal", metallic=0.7)
    shader = principled(source)
    shader.inputs["Transmission Weight"].default_value = 0.6
    value = source.node_tree.nodes.new("ShaderNodeValue")
    source.node_tree.links.new(value.outputs[0], shader.inputs["Metallic"])
    image = bpy.data.images.new("copy target", 4, 4)
    copy = baking._material_copy(source, image, {})
    copied = principled(copy)
    assert copied.inputs["Metallic"].default_value == 0.0 and not copied.inputs["Metallic"].is_linked
    assert abs(copied.inputs["Transmission Weight"].default_value - 0.6) < 1e-6
    assert shader.inputs["Metallic"].is_linked and abs(shader.inputs["Metallic"].default_value - 0.7) < 1e-6
    assert copy.node_tree.nodes.active.image == image
    plain = bpy.data.materials.new("never used nodes")
    plain_copy = baking._material_copy(plain, image, {})
    assert plain_copy.node_tree is not None
    return {"node_tree_before": bpy.app.version >= (5, 0, 0) or plain.node_tree is not None}


def set_look(view, names=("AgX - Medium High Contrast", "AgX - Punchy", "Medium High Contrast", "Punchy")):
    for name in names:
        try:
            view.look = name
            return name
        except TypeError:
            continue
    return None


@section
def bake_colour_management():
    """Video output, Output colour Override, a look and exposure do not reach the lightmap."""
    scene = bake_scene()
    box("Floor", (0, 0, -0.05), (1.5, 1.5, 0.05), "VISUAL", material("grey", (0.5, 0.5, 0.5)))
    box("Lamp", (0, 0, 0.3), (0.25, 0.25, 0.25), mat=material("lamp", (0.1, 0.1, 0.1), emission=(1, 0.8, 0.5), strength=6))
    settings = scene.render.image_settings
    if hasattr(settings, "media_type"):
        settings.media_type = "VIDEO"  # Blender 5: file_format is now limited to FFMPEG.
    else:
        try:
            settings.file_format = "FFMPEG"
        except TypeError:
            pass
    settings.color_management = "OVERRIDE"
    settings.view_settings.view_transform = "Standard"
    scene.view_settings.view_transform = "AgX"
    look = set_look(scene.view_settings)
    scene.view_settings.exposure = 1.5
    scene.view_settings.gamma = 1.2

    def state():
        return (getattr(settings, "media_type", None), settings.file_format, settings.color_management,
                settings.view_settings.view_transform, scene.view_settings.view_transform, scene.view_settings.look,
                round(scene.view_settings.exposure, 6), round(scene.view_settings.gamma, 6))

    before = state()
    result = bake_static(bpy.context, str(OUT / "colour_bake"), atlas_size=64, samples=2, margin=1)
    assert state() == before, (state(), before)
    assert result["display_profile"] == PINNED, result["display_profile"]
    master = atlas_image(result)
    reference_scene = bpy.data.scenes.new("Reference AgX")
    reference_scene.view_settings.view_transform = "AgX"
    reference_scene.view_settings.look = "None"
    reference_scene.view_settings.exposure, reference_scene.view_settings.gamma = 0.0, 1.0
    reference_settings = reference_scene.render.image_settings
    reference_settings.file_format, reference_settings.color_mode, reference_settings.color_depth = "PNG", "RGB", "8"
    reference = OUT / "colour_bake" / "agx_reference.png"
    master.save_render(str(reference), scene=reference_scene)
    bpy.data.scenes.remove(reference_scene)
    expected = bytes_rgb(reference)
    lightmap = bytes_rgb(Path(result["display_png"]))
    # The lightmap PNG comes from the float working image, the reference from
    # the half-float EXR master, so allow one quantisation step.
    worst = max(abs(a - b) for x, y in zip(expected, lightmap) for a, b in zip(x, y))
    assert worst <= 1, f"lightmap differs from the pinned AgX reference by {worst}/255"
    assert max(max(texel) for texel in expected) > 100, "reference is black"
    target = record("colour_export", OUT / "colour_export" / "scene.xml")
    report = export_world(bpy.context, str(target))
    assert state() == before
    note = next(warning for warning in report["warnings"] if "HDR bake" in warning)
    assert "AgX (look None, exposure 0, gamma 1, display sRGB)" in note, note
    texture = xml(target).find("./asset/texture")
    exported = bytes_rgb(texture_file(target, texture))
    assert exported == expected, "export changed the recorded AgX appearance"
    return {"look": look, "lightmap_max_difference": worst, "note": note}


@section
def bake_stale_sources():
    """A source that is no longer baked comes back; an uncovered mark keeps its visual."""
    scene = bake_scene()
    box("Floor", (0, 0, -0.05), (1.5, 1.5, 0.05), "VISUAL", material("grey", (0.3, 0.3, 0.3)))
    box("A fixed", (-0.5, 0, 0.3), (0.2, 0.2, 0.2), mat=material("red", (0.8, 0.1, 0.1)))
    b = box("B fixed", (0.5, 0, 0.3), (0.2, 0.2, 0.2), mat=material("blue", (0.1, 0.1, 0.8)))
    first = bake_static(bpy.context, str(OUT / "stale_bake"), atlas_size=64, samples=2, margin=1)
    assert first["source_count"] == 3 and first["restored"] == []
    assert bake_covered(scene) == {"Floor", "A fixed", "B fixed"}
    b.mujoco_world.role = "FREE"
    second = bake_static(bpy.context, str(OUT / "stale_bake"), atlas_size=64, samples=2, margin=1)
    assert second["source_count"] == 2 and second["restored"] == ["B fixed"], second
    assert b.visible_get() and not b.hide_render
    assert not any(key.startswith("mjworld_bake") for key in b.keys()), list(b.keys())
    moved = record("stale_moved", OUT / "stale_moved" / "scene.xml")
    report = export_world(bpy.context, str(moved))
    assert len(visual_geoms(moved, "B_fixed")) == 1 and not visual_geoms(moved, "A_fixed")
    assert not [w for w in report["warnings"] if "marked as a bake source" in w], report["warnings"]
    # Deleting the atlas by hand leaves marked sources that nothing covers.
    atlas = bpy.data.objects[second["object"]]
    bpy.data.objects.remove(atlas)
    assert bake_covered(scene) == set()
    assert bpy.ops.mjworld.restore_sources.poll(), "restore must stay available for hidden sources"
    deleted = record("stale_deleted_atlas", OUT / "stale_deleted_atlas" / "scene.xml")
    report = export_world(bpy.context, str(deleted))
    assert len(visual_geoms(deleted, "A_fixed")) == 1
    assert any(w.startswith("A fixed: marked as a bake source, but no baked atlas contains it") for w in report["warnings"])
    # A legacy atlas without a source list (the Chula realistic room) covers
    # every marked source, as before.
    third = bake_static(bpy.context, str(OUT / "stale_bake"), atlas_size=64, samples=2, margin=1)
    legacy = bpy.data.objects[third["object"]]
    del legacy["mjworld_bake_sources"]
    assert bake_covered(scene) == {"Floor", "A fixed"}
    legacy_path = record("stale_legacy", OUT / "stale_legacy" / "scene.xml")
    report = export_world(bpy.context, str(legacy_path))
    assert not visual_geoms(legacy_path, "A_fixed") and not [w for w in report["warnings"] if "marked as a bake source" in w]
    # An atlas set to Leave out is not exported, so it covers nothing.
    legacy.mujoco_world.role = "EXCLUDE"
    assert bake_covered(scene) == set()
    return {"restored": second["restored"]}


@section
def bake_failure_cleanup():
    """A clone-loop failure surfaces as BakeError and leaves no bake data behind."""
    bake_scene()
    a = box("A cube", (-0.5, 0, 0.3), (0.2, 0.2, 0.2), mat=material("red", (0.8, 0.1, 0.1)))
    b = box("B many UV maps", (0.5, 0, 0.3), (0.2, 0.2, 0.2), mat=material("blue", (0.1, 0.1, 0.8)))
    for index in range(7):
        b.data.uv_layers.new(name=f"extra_{index}")
    assert len(b.data.uv_layers) == 8
    before = data_counts()
    try:
        bake_static(bpy.context, str(OUT / "failed_bake"), atlas_size=64, samples=2, margin=1)
    except BakeError as exc:
        message = str(exc)
    else:
        raise AssertionError("a ninth UV map was accepted")
    assert "8 UV maps" in message and "B many UV maps" in message, message
    after = data_counts()
    assert after == before, {kind: sorted(set(after[kind]) ^ set(before[kind])) for kind in after}
    assert all(obj.visible_get() and not obj.hide_render and not obj.get("mjworld_bake_source") for obj in (a, b))
    # A stray piece from an older failure is neither baked nor exported, and
    # 'Edit original surfaces' removes it.
    b.data.uv_layers.remove(b.data.uv_layers["extra_0"])
    b.data.update()  # uv_layers.remove does not tag the depsgraph.
    stray =box("MuJoCo bake piece stray", (0, 1, 0.3), (0.2, 0.2, 0.2))
    stray["mjworld_bake_clone"] = True
    result = bake_static(bpy.context, str(OUT / "failed_bake"), atlas_size=64, samples=2, margin=1)
    assert result["source_count"] == 2, result
    target = record("stray_clone", OUT / "stray_clone" / "scene.xml")
    export_world(bpy.context, str(target))
    assert not [body for body in xml(target).iter("body") if body.get("name", "").startswith("MuJoCo_bake_piece")]
    assert restore_sources(bpy.context) == 2
    assert "MuJoCo bake piece stray" not in bpy.data.objects
    return {"error": message}


@section
def bake_restore_edges():
    """'Edit original surfaces': nothing baked, a full round trip, a deleted source, and user data."""
    scene = bake_scene()
    box("Floor", (0, 0, -0.05), (1.5, 1.5, 0.05), "VISUAL", material("grey", (0.3, 0.3, 0.3)))
    kept = box("Kept fixed", (-0.5, 0, 0.3), (0.2, 0.2, 0.2), mat=material("red", (0.8, 0.1, 0.1)))
    box("Deleted fixed", (0.5, 0, 0.3), (0.2, 0.2, 0.2), mat=material("blue", (0.1, 0.1, 0.8)))
    before = data_counts()
    # Nothing baked: the operator is unavailable and the function changes nothing.
    assert not bpy.ops.mjworld.restore_sources.poll()
    assert restore_sources(bpy.context) == 0 and data_counts() == before
    # A bake and a restore leave exactly the data that was there before; the
    # lightmap files stay on disk.
    first = bake_static(bpy.context, str(OUT / "restore_edges"), atlas_size=64, samples=2, margin=1)
    assert restore_sources(bpy.context) == 3
    after = data_counts()
    assert after == before, {kind: sorted(set(after[kind]) ^ set(before[kind])) for kind in after}
    assert Path(first["master_exr"]).is_file() and Path(first["display_png"]).is_file()
    # A source deleted after baking, and a user object placed in the bake's
    # collection that uses the baked material: the rest is restored, the
    # user's object, collection and material survive.
    second = bake_static(bpy.context, str(OUT / "restore_edges"), atlas_size=64, samples=2, margin=1)
    atlas = bpy.data.objects[second["object"]]
    baked_material = atlas.data.materials[0]
    material_name, collection_name = baked_material.name, atlas.users_collection[0].name
    bpy.ops.mesh.primitive_cube_add(size=0.2, location=(0, 1, 0.3))
    user = bpy.context.object
    user.name = "User prop"
    for owner in list(user.users_collection):
        owner.objects.unlink(user)
    bpy.data.collections[collection_name].objects.link(user)
    user.data.materials.append(baked_material)
    bpy.data.objects.remove(bpy.data.objects["Deleted fixed"])
    assert restore_sources(bpy.context) == 2
    assert kept.visible_get() and not kept.hide_render
    assert not any(key.startswith("mjworld_bake") for key in kept.keys()), list(kept.keys())
    assert second["object"] not in bpy.data.objects and not bake_leftovers(scene)
    assert "User prop" in bpy.data.collections[collection_name].objects
    assert material_name in bpy.data.materials and user.data.materials[0].name == material_name
    return {"first_sources": first["source_count"], "second_sources": second["source_count"]}


@section
def bake_faceless():
    """A faceless mesh is skipped with a note in the bake and in the export."""
    bake_scene()
    box("A cube", (0, 0, 0.3), (0.2, 0.2, 0.2), mat=material("red", (0.8, 0.1, 0.1)))
    guide = wire("Wire guide")
    result = bake_static(bpy.context, str(OUT / "faceless_bake"), atlas_size=64, samples=2, margin=1)
    assert result["skipped"] == ["Wire guide"] and result["source_count"] == 1, result
    assert not guide.get("mjworld_bake_source") and guide.visible_get()
    target = record("faceless", OUT / "faceless" / "scene.xml")
    report = export_world(bpy.context, str(target))
    assert any(w.startswith("Wire guide: mesh has no faces, so it was left out") for w in report["warnings"]), report["warnings"]
    bake_scene()
    wire("Only guide")
    before = data_counts()
    try:
        bake_static(bpy.context, str(OUT / "faceless_bake"), atlas_size=64, samples=2, margin=1)
    except BakeError as exc:
        assert "with faces" in str(exc), str(exc)
    else:
        raise AssertionError("a bake without faces succeeded")
    assert data_counts() == before
    try:
        export_world(bpy.context, str(OUT / "faceless_only" / "scene.xml"))
    except ExportError as exc:
        assert "Nothing to export" in str(exc) and "Only guide" in str(exc), str(exc)
    else:
        raise AssertionError("an export without faces succeeded")
    return {"skipped": result["skipped"]}


@section
def bake_clone_shading():
    """Object colour and Attribute (Object) properties reach the bake clones."""
    bake_scene()
    tinted = material("object colour")
    info = tinted.node_tree.nodes.new("ShaderNodeObjectInfo")
    tinted.node_tree.links.new(info.outputs["Color"], principled(tinted).inputs["Base Color"])
    blue = box("A object colour", (-0.5, 0, 0.3), (0.2, 0.2, 0.2), mat=tinted)
    blue.color = (0.05, 0.1, 0.9, 1)
    attribute_material = material("custom property colour")
    attribute = attribute_material.node_tree.nodes.new("ShaderNodeAttribute")
    attribute.attribute_type, attribute.attribute_name = "OBJECT", "tint"
    attribute_material.node_tree.links.new(attribute.outputs["Color"], principled(attribute_material).inputs["Base Color"])
    red = box("B custom property", (0.5, 0, 0.3), (0.2, 0.2, 0.2), mat=attribute_material)
    red["tint"] = [0.9, 0.08, 0.05]
    random_material = material("random colour")
    random_info = random_material.node_tree.nodes.new("ShaderNodeObjectInfo")
    random_material.node_tree.links.new(random_info.outputs["Random"], principled(random_material).inputs["Roughness"])
    box("C random", (0, 0.6, 0.3), (0.2, 0.2, 0.2), mat=random_material)
    result = bake_static(bpy.context, str(OUT / "shading_bake"), atlas_size=64, samples=2, margin=1)
    pixels = texels(atlas_image(result))
    blue_texels = sum(1 for r, g, b, _a in pixels if b > 0.3 and b > 3 * r)
    red_texels = sum(1 for r, g, b, _a in pixels if r > 0.3 and r > 3 * g and r > 3 * b)
    assert blue_texels > 20, f"object colour lost: {blue_texels} blue texels"
    assert red_texels > 20, f"custom property lost: {red_texels} red texels"
    assert any("C random" in w and "Object Info Random" in w for w in result["warnings"]), result["warnings"]
    return {"blue_texels": blue_texels, "red_texels": red_texels}


@section
def bake_prechecks():
    """Cheap failures stop the bake before the Cycles pass."""
    scene = bake_scene()
    box("A cube", (0, 0, 0.3), (0.2, 0.2, 0.2))
    assert not bpy.data.filepath
    cwd = OUT / "cwd"
    cwd.mkdir()
    previous = os.getcwd()
    os.chdir(cwd)
    try:
        try:
            bake_static(bpy.context, "//baked_lighting", atlas_size=64, samples=2)
        except BakeError as exc:
            relative = str(exc)
        else:
            raise AssertionError("a '//' bake folder was accepted in an unsaved file")
        assert "not saved" in relative and not (cwd / "baked_lighting").exists(), relative
        # The operator passes the raw setting, so it refuses too.
        try:
            bpy.ops.mjworld.bake_lighting()
        except RuntimeError as exc:
            assert "not saved" in str(exc), str(exc)
        else:
            raise AssertionError("the bake operator accepted '//' in an unsaved file")
        scene.mujoco_world.asset_directory = "//shared_assets"
        try:
            export_world(bpy.context, str(OUT / "unsaved_assets" / "scene.xml"))
        except ExportError as exc:
            assert "not saved" in str(exc), str(exc)
        else:
            raise AssertionError("a '//' asset folder was accepted in an unsaved file")
        scene.mujoco_world.asset_directory = ""
        try:
            export_world(bpy.context, "//scene.xml")
        except ExportError as exc:
            assert "not saved" in str(exc), str(exc)
        else:
            raise AssertionError("a '//' export path was accepted in an unsaved file")
        assert not list(cwd.iterdir()), list(cwd.iterdir())
    finally:
        os.chdir(previous)
    locked = OUT / "locked"
    locked.mkdir()
    locked.chmod(0o500)
    try:
        if not os.access(locked, os.W_OK):  # Root can write anyway.
            try:
                bake_static(bpy.context, str(locked), atlas_size=64, samples=2)
            except BakeError as exc:
                assert "Cannot write" in str(exc), str(exc)
            else:
                raise AssertionError("an unwritable bake folder was accepted")
    finally:
        locked.chmod(0o700)
    addon_utils.disable("cycles")
    try:
        assert not hasattr(scene, "cycles")
        try:
            bake_static(bpy.context, str(OUT / "no_cycles"), atlas_size=64, samples=2)
        except BakeError as exc:
            assert "Cycles" in str(exc), str(exc)
        else:
            raise AssertionError("a bake without Cycles started")
    finally:
        addon_utils.enable("cycles")
    assert hasattr(bpy.context.scene, "cycles")
    return {"relative": relative}


# ---------------------------------------------------------------- textures


@section
def render_uv():
    """Textures use the render-active UV map, as Blender renders them."""
    fresh_scene()
    image = bpy.data.images.new("uv label", 4, 4)
    image.pixels[:] = [0.8, 0.2, 0.1, 1] * 16
    obj = box("Label box", (0, 0, 0.3), (0.2, 0.2, 0.2), mat=image_material("label", image))
    render = obj.data.uv_layers["UVMap"]
    lightmap = obj.data.uv_layers.new(name="Lightmap")
    for loop in lightmap.data:
        loop.uv = (0.123, 0.456)
    obj.data.uv_layers.active = lightmap
    render.active_render = True
    target = record("render_uv", OUT / "render_uv" / "scene.xml")
    export_world(bpy.context, str(target))
    root = xml(target)
    mesh = next(m for m in root.iter("mesh") if m.get("name").endswith("_visual_0"))
    lines = (target.parent / root.find("compiler").get("meshdir") / mesh.get("file")).read_text().splitlines()
    uvs = {line for line in lines if line.startswith("vt ")}
    assert len(uvs) > 1, uvs
    return {"distinct_uvs": len(uvs)}


@section
def texture_images():
    """16-bit PNG, float albedo, UDIM, missing, reroute and muted cases."""
    fresh_scene()
    folder = OUT / "texture_sources"
    folder.mkdir()
    png16 = folder / "tex16.png"
    write_png16(png16, 4, 4, lambda x, y: (65535 * x // 3, 30000, 65535 * y // 3))
    deep = bpy.data.images.load(str(png16))
    assert deep.is_float, "Blender should load a 16-bit PNG as float"
    box("A deep png", (-1, 0, 0.3), (0.2, 0.2, 0.2), mat=image_material("deep", deep))
    linear = bpy.data.images.new("float albedo", 2, 2, float_buffer=True)
    linear.pixels[:] = [0.5, 0.25, 0.0, 1, 1.5, 0.5, 0.25, 1] * 2
    box("B float albedo", (-0.5, 0, 0.3), (0.2, 0.2, 0.2), mat=image_material("float", linear))
    tiled = bpy.data.images.new("udim tiles", 4, 4, tiled=True)
    box("C tiled", (0, 0, 0.3), (0.2, 0.2, 0.2), mat=image_material("tiled", tiled))
    missing = bpy.data.images.new("missing file", 4, 4)
    missing.source = "FILE"
    missing.filepath = str(folder / "does_not_exist.png")
    box("D missing", (0.5, 0, 0.3), (0.2, 0.2, 0.2), mat=image_material("missing", missing))
    plain = bpy.data.images.new("plain label", 2, 2)
    plain.pixels[:] = [0.2, 0.6, 0.9, 1] * 4
    box("E reroute", (1, 0, 0.3), (0.2, 0.2, 0.2), mat=image_material("reroute", plain, via="reroute"))
    box("F muted link", (1.5, 0, 0.3), (0.2, 0.2, 0.2), mat=image_material("muted link", plain, via="muted_link"))
    box("G muted node", (2, 0, 0.3), (0.2, 0.2, 0.2), mat=image_material("muted node", plain, via="muted_node"))
    target = record("texture_images", OUT / "texture_images" / "scene.xml")
    report = export_world(bpy.context, str(target))
    notes = report["warnings"]
    root = xml(target)
    materials = {m.get("name"): m for m in root.iter("material")}
    textures = {t.get("name"): t for t in root.iter("texture")}

    def texture_of(prefix):
        geom = visual_geoms(target, prefix)
        assert len(geom) == 1, (prefix, geom)
        name = materials[geom[0].get("material")].get("texture")
        return textures[name] if name else None

    deep_texture = texture_of("A_deep_png")
    assert texture_file(target, deep_texture).read_bytes() == png16.read_bytes(), "16-bit PNG was not copied unchanged"
    float_texture = texture_of("B_float_albedo")
    encoded = bytes_rgb(texture_file(target, float_texture))
    expected = [round(srgb(c) * 255) for c in (0.5, 0.25, 0.0)]
    assert all(abs(a - b) <= 1 for a, b in zip(encoded[0], expected)), (encoded[0], expected)
    assert encoded[1][0] == 255, encoded[1]
    assert any("floating-point image 'float albedo' was written as an 8-bit sRGB PNG" in n for n in notes), notes
    assert any("float albedo" in n and "clipped" in n for n in notes), notes
    assert texture_of("C_tiled") is None and texture_of("D_missing") is None
    assert any(n.startswith("C tiled / tiled: image 'udim tiles' is a UDIM (tiled) image") for n in notes), notes
    assert any(n.startswith("D missing / missing: image 'missing file' has no pixels") for n in notes), notes
    assert texture_of("E_reroute") is not None
    assert texture_of("F_muted_link") is None
    assert texture_of("G_muted_node") is None
    assert any(n.startswith("G muted node / muted node: Colour node") and "muted" in n for n in notes), notes
    assert not any(n.startswith("E reroute") for n in notes), notes
    return {"notes": notes}


@section
def byte_colourspace():
    """Non-Color byte albedo keeps Blender's brightness in both texture modes."""
    scene = fresh_scene()
    data = bpy.data.images.new("data albedo", 2, 2)
    data.colorspace_settings.name = "Non-Color"  # Before the pixels: changing it reloads the buffer.
    data.pixels[:] = [128 / 255, 64 / 255, 32 / 255, 1] * 4
    box("A data albedo", (0, 0, 0.3), (0.2, 0.2, 0.2), mat=image_material("data", data))
    srgb_image = bpy.data.images.new("srgb albedo", 2, 2)
    srgb_image.pixels[:] = [128 / 255, 64 / 255, 32 / 255, 1] * 4
    box("B srgb albedo", (0.6, 0, 0.3), (0.2, 0.2, 0.2), mat=image_material("srgb", srgb_image))
    results = {}
    for mode in ("DISPLAY", "SRGB"):
        scene.mujoco_world.texture_mode = mode
        target = record(f"byte_{mode.lower()}", OUT / f"byte_{mode.lower()}" / "scene.xml")
        report = export_world(bpy.context, str(target))
        root = xml(target)
        materials = {m.get("name"): m for m in root.iter("material")}
        textures = {t.get("name"): t for t in root.iter("texture")}
        found = {}
        for prefix in ("A_data_albedo", "B_srgb_albedo"):
            texture = textures[materials[visual_geoms(target, prefix)[0].get("material")].get("texture")]
            found[prefix] = (texture.get("colorspace"), bytes_rgb(texture_file(target, texture))[0])
        assert any("is tagged Non-Color" in n for n in report["warnings"]), report["warnings"]
        results[mode] = found
    expected = tuple(round(srgb(c / 255) * 255) for c in (128, 64, 32))
    assert results["DISPLAY"]["A_data_albedo"][0] == "linear"
    assert all(abs(a - b) <= 1 for a, b in zip(results["DISPLAY"]["A_data_albedo"][1], expected)), results
    assert results["DISPLAY"]["B_srgb_albedo"] == ("linear", (128, 64, 32)), results
    assert results["SRGB"]["A_data_albedo"] == ("linear", (128, 64, 32)), results
    assert results["SRGB"]["B_srgb_albedo"] == ("sRGB", (128, 64, 32)), results
    return {mode: {key: list(value) for key, value in found.items()} for mode, found in results.items()}


ADOPTED = """<mujoco model="adopted_colourspace">
  <compiler angle="radian" meshdir="scene_assets" texturedir="scene_assets"/>
  <asset>
    <texture name="crate_tex" type="2d" file="crate_tex.png"/>
    <material name="crate_mat" texture="crate_tex"/>
  </asset>
  <worldbody>
    <light name="sun" pos="0 0 3" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <geom name="Wall" type="box" size="1 0.05 0.5" pos="0 2 0.5"/>
    <body name="Crate" pos="1 0 0.3">
      <freejoint name="Crate_free"/>
      <geom name="Crate_box" type="box" size="0.2 0.2 0.2" mass="2" material="crate_mat"/>
    </body>
  </worldbody>
</mujoco>
"""


@section
def adopted_colourspace():
    """Adopted export pins retained and atlas texture colourspace by texture mode."""
    scene = bake_scene()
    directory = OUT / "adopted_source"
    assets = directory / "scene_assets"
    assets.mkdir(parents=True)
    image = bpy.data.images.new("crate_tex", 4, 4)
    image.pixels[:] = [0.8, 0.3, 0.1, 1] * 16
    image.filepath_raw = str(assets / "crate_tex.png")
    image.file_format = "PNG"
    image.save()
    (directory / "scene.xml").write_text(ADOPTED)
    bpy.ops.mesh.primitive_plane_add(size=10)
    bpy.context.object.name = "floor"
    box("Wall", (0, 2, 0.5), (1, 0.05, 0.5), mat=material("wall", (0.7, 0.7, 0.6)))
    crate = bpy.data.objects.new("Crate", None)
    scene.collection.objects.link(crate)
    crate.location = (1, 0, 0.3)
    box("Crate_box", (1, 0, 0.3), (0.2, 0.2, 0.2), parent=crate, mat=image_material("crate textured", image))
    bpy.context.view_layer.update()
    adopt_mjcf(bpy.context, str(directory / "scene.xml"))
    wire("Guide edges")
    baked = bake_static(bpy.context, str(OUT / "adopted_bake"), atlas_size=64, samples=2, margin=1)
    found = {}
    for mode in ("DISPLAY", "SRGB"):
        scene.mujoco_world.texture_mode = mode
        target = record(f"adopted_{mode.lower()}", OUT / f"adopted_{mode.lower()}" / "scene.xml")
        report = export_world(bpy.context, str(target))
        textures = {t.get("name"): t.get("colorspace") for t in xml(target).iter("texture")}
        assert any(w.startswith("Guide edges: mesh has no faces") for w in report["warnings"]), report["warnings"]
        found[mode] = textures
    expected = {"DISPLAY": "linear", "SRGB": "sRGB"}
    for mode, textures in found.items():
        assert textures.get("crate_tex") == expected[mode], (mode, textures)
        atlas = [name for name in textures if name != "crate_tex"]
        assert atlas and all(textures[name] == expected[mode] for name in atlas), (mode, textures)
    return {"textures": found, "baked": baked["source_count"]}


# ---------------------------------------------------------------- operators


@section
def operators():
    """Operator options, polls, reports and linked-data refusals."""
    scene = fresh_scene()
    crate = bpy.data.objects.new("Crate Empty", None)
    scene.collection.objects.link(crate)
    crate.mujoco_world.role = "FREE"
    assert crate.mujoco_world.body, "the role dropdown did not make the Empty a body"
    label = bpy.data.objects.new("Label Empty", None)
    scene.collection.objects.link(label)
    label.mujoco_world.role = "VISUAL"
    assert not label.mujoco_world.body
    assert {"REGISTER", "UNDO"} <= mujoco_world.MJWORLD_OT_bake.bl_options
    assert "invoke" in mujoco_world.MJWORLD_OT_restore.__dict__
    settings_rna = mujoco_world.MJWORLD_ObjectSettings.bl_rna.properties
    assert all(settings_rna[name].is_overridable for name in ("role", "body", "mass", "collision", "hinge_min"))
    assert bpy.types.Object.bl_rna.properties["mujoco_world"].is_overridable
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        scene.mujoco_world.asset_directory = "//scene_assets"
        scene.mujoco_world.bake_directory = "//bake"
        scene.mujoco_world.last_export = "//mujoco_export/scene.xml"
    relative_warnings = [str(w.message) for w in caught if "blend relative" in str(w.message)]
    if bpy.app.version >= (4, 5, 0):
        assert not relative_warnings, relative_warnings
    scene.mujoco_world.asset_directory = scene.mujoco_world.bake_directory = scene.mujoco_world.last_export = ""
    # Export browser default.
    assert mujoco_world.default_export_path("") == ""
    assert mujoco_world.default_export_path("//mujoco_export/scene.xml") == ""
    assert mujoco_world.default_export_path("/tmp/world/scene.xml") == "/tmp/world/scene.xml"
    saved = OUT / "operators" / "saved.blend"
    saved.parent.mkdir()
    bpy.ops.wm.save_as_mainfile(filepath=str(saved))
    assert mujoco_world.default_export_path("") == str(saved.parent / "mujoco_export" / "scene.xml")
    assert mujoco_world.default_export_path("//other/scene.xml") == str(saved.parent / "other" / "scene.xml")
    # Restore: poll and report.
    scene = fresh_scene()
    assert not bpy.ops.mjworld.restore_sources.poll()
    source = box("Hidden source", (0, 0, 0.3), (0.2, 0.2, 0.2))
    source["mjworld_bake_source"] = True
    source.hide_render = True
    assert bpy.ops.mjworld.restore_sources.poll()
    bpy.context.view_layer.objects.active = source
    bpy.ops.object.mode_set(mode="EDIT")
    assert not bpy.ops.mjworld.restore_sources.poll(), "restore must not run in Edit Mode"
    bpy.ops.object.mode_set(mode="OBJECT")
    assert bpy.ops.mjworld.restore_sources() == {"FINISHED"}
    assert not source.hide_render and "mjworld_bake_source" not in source
    # Unexpected errors are reported briefly instead of escaping as a traceback.
    original = baking.bake_static

    def broken(*_args, **_kwargs):
        raise TypeError("simulated failure")

    baking.bake_static = broken
    try:
        bpy.ops.mjworld.bake_lighting()
    except RuntimeError as exc:
        failure = str(exc)
    else:
        raise AssertionError("the simulated failure was not reported")
    finally:
        baking.bake_static = original
    assert "TypeError: simulated failure (details in the system console)" in failure, failure
    # Linked objects are refused; a library override is editable.
    library = OUT / "operators" / "library.blend"
    chair = box("Linked chair", (0, 0, 0.3), (0.2, 0.2, 0.2))
    bpy.data.libraries.write(str(library), {chair, chair.data})
    scene = fresh_scene()
    with bpy.data.libraries.load(str(library), link=True) as (_source, target):
        target.objects = ["Linked chair"]
    linked = target.objects[0]
    scene.collection.objects.link(linked)
    local = box("Local stool", (1, 0, 0.3), (0.2, 0.2, 0.2))
    assert linked.library is not None and not linked.is_editable
    bpy.ops.object.select_all(action="DESELECT")
    linked.select_set(True)
    bpy.context.view_layer.objects.active = linked
    refusals = {}
    for name, call in (("set_role", lambda: bpy.ops.mjworld.set_role(role="FREE")),
                       ("group_parts", lambda: bpy.ops.mjworld.group_parts(role="FREE")),
                       ("hinge_at_cursor", lambda: bpy.ops.mjworld.hinge_at_cursor())):
        try:
            call()
        except RuntimeError as exc:
            refusals[name] = str(exc)
        else:
            raise AssertionError(f"{name} edited a linked object")
        assert "linked" in refusals[name], refusals[name]
    assert linked.mujoco_world.role == "STATIC" and linked.parent is None
    local.select_set(True)
    assert bpy.ops.mjworld.set_role(role="FREE") == {"FINISHED"}
    assert local.mujoco_world.role == "FREE" and linked.mujoco_world.role == "STATIC"
    override = linked.override_create(remap_local_usages=True)
    assert override.library is None and override.override_library is not None
    assert not override.mujoco_world.is_property_readonly("role"), "role is read-only on a library override"
    return {"failure": failure, "refusals": refusals}


(OUT / "review_fixes.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
if summary["failures"]:
    raise AssertionError("Failed sections: " + ", ".join(summary["failures"]))
harness.complete(OUT, "review_fixes", sections=summary["sections"])
print("REVIEW_FIXES_OK", json.dumps({"output": str(OUT), "sections": summary["sections"]}))
