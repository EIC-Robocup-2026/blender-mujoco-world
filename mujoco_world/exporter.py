"""MJCF writer. Visual shape, collision approximation and mass are separate."""

import json
import hashlib
import math
import os
from array import array
from pathlib import Path
import re
import shutil
import struct
import tempfile
import xml.etree.ElementTree as ET

import bpy
from mathutils import Matrix, Vector


class ExportError(ValueError):
    pass


def numbers(values):
    return " ".join(f"{float(v):.9g}" for v in values)


def _float32(value):
    return struct.unpack("f", struct.pack("f", value))[0]


def property_value(value):
    """Return a Blender float property as the shortest decimal it stores.

    Blender stores FloatProperty values as float32, so an entered 0.3 reads
    back as 0.30000001192092896. The result is the shortest decimal that maps
    to the same float32, so 0.3 is written as 0.3 and no stored value changes.
    Geometry keeps numbers() precision; this is only for property values.
    """
    value = float(value)
    try:
        stored = _float32(value)
    except OverflowError:
        return value
    for digits in range(1, 10):
        candidate = float(f"{value:.{digits}g}")
        if _float32(candidate) == stored:
            return candidate
    return value


def property_numbers(values):
    return numbers(property_value(v) for v in values)


def xml_path(filepath):
    """Resolve an export target; an existing .xml suffix keeps its case."""
    path = Path(bpy.path.abspath(str(filepath))).expanduser().resolve()
    return path if path.suffix.lower() == ".xml" else path.with_suffix(".xml")


def check_door(obj, closed_at_zero=True):
    """Validate door limits before any file is written.

    An authored door is modelled in its closed pose, so its limits must contain
    0. A joint added to an adopted body keeps its Blender pose at the starting
    angle instead, so only ordered limits are required there.
    """
    cfg = obj.mujoco_world
    low, high, start = (property_value(v) for v in (cfg.hinge_min, cfg.hinge_max, cfg.hinge_start))
    if closed_at_zero and (low >= high or not low <= 0 <= high):
        raise ExportError(f"{obj.name}: door limits must be ordered and contain 0 (the modelled closed pose).")
    if low >= high:
        raise ExportError(f"{obj.name}: the door's open limit must be below its closed limit.")
    if not low <= start <= high:
        raise ExportError(f"{obj.name}: starting angle must be inside the door limits.")


def door_radians(cfg):
    """Door limits and starting angle in radians, the compiler angle unit of every export."""
    return [math.radians(property_value(v)) for v in (cfg.hinge_min, cfg.hinge_max, cfg.hinge_start)]


def _flat_half_thickness(geometry, triangles):
    """Half thickness across the best-fit plane of the triangle vertices.

    The local bounding box misses a flat mesh tilted in edit mode; the smallest
    principal axis of the centred vertices does not.
    """
    import numpy as np
    ids = sorted({v for tri in triangles for v, _loop in tri})
    if len(ids) < 4:
        return 0.0
    points = np.array([tuple(geometry["vertices"][v]) for v in ids], dtype=float)
    centred = points - points.mean(axis=0)
    normal = np.linalg.svd(centred, full_matrices=False)[2][-1]
    distances = centred @ normal
    return float(distances.max() - distances.min()) / 2


def slug(value):
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_.")
    return value or "object"


def srgb(value):
    value = max(0.0, min(1.0, value))
    return 12.92 * value if value <= 0.0031308 else 1.055 * value ** (1 / 2.4) - 0.055


def _socket_source(socket):
    if socket and socket.is_linked:
        return socket.links[0].from_node
    return None


def inspect_material(material):
    """Return only the small material subset that can be represented faithfully."""
    if material is None:
        return {"color": [0.7, 0.7, 0.7, 1], "image": None, "emission": False, "warnings": []}
    result = {"color": list(material.diffuse_color), "image": None, "emission": False, "warnings": []}
    if not material.use_nodes or not material.node_tree:
        return result
    outputs = [n for n in material.node_tree.nodes if n.type == "OUTPUT_MATERIAL" and n.is_active_output]
    shader = _socket_source(outputs[0].inputs.get("Surface")) if outputs else None
    if shader is None:
        result["warnings"].append("No active surface shader; using material colour.")
        return result
    if shader.type == "BSDF_PRINCIPLED":
        color_socket = shader.inputs.get("Base Color")
        result["color"] = list(color_socket.default_value)
    elif shader.type == "EMISSION":
        color_socket = shader.inputs.get("Color")
        result["color"] = list(color_socket.default_value)
        result["emission"] = True
    else:
        result["warnings"].append("Shader is not Principled or Emission; bake its appearance to a UV image first.")
        return result
    source = _socket_source(color_socket)
    if source:
        if source.type == "TEX_IMAGE" and source.image and source.projection == "FLAT":
            vector = _socket_source(source.inputs.get("Vector"))
            if vector and vector.type not in {"UVMAP", "TEX_COORD"}:
                result["warnings"].append("Image uses a Mapping/procedural vector; bake to the active UV map first.")
            elif vector and vector.type == "TEX_COORD" and source.inputs["Vector"].links[0].from_socket.name != "UV":
                result["warnings"].append("Image does not use UV coordinates; bake to the active UV map first.")
            else:
                result["image"] = source.image
                result["uv_map"] = vector.uv_map if vector and vector.type == "UVMAP" else None
        else:
            result["warnings"].append("Procedural/combined colour cannot be translated; bake it to a UV image first.")
    if shader.type == "BSDF_PRINCIPLED":
        for field in ("Normal", "Metallic", "Roughness", "Alpha"):
            sock = shader.inputs.get(field)
            if sock and sock.is_linked:
                result["warnings"].append(f"Linked {field} is not exported; this exporter carries colour/lighting only.")
        if shader.inputs.get("Alpha"):
            result["color"][3] = float(shader.inputs["Alpha"].default_value)
    return result


def _write_image(image, path, *, baked=False, scene=None):
    if image.is_float:
        if not baked:
            raise ExportError(f"{image.name}: floating-point albedo needs a saved display PNG. Save it as PNG before export, or enable 'Image already contains lighting' if it is a light bake.")
        if scene is None:
            raise ExportError(f"{image.name}: scene colour management is needed to convert the HDR bake.")
        settings = scene.render.image_settings
        previous = (settings.file_format, settings.color_mode, settings.color_depth)
        view = scene.view_settings
        previous_view = (view.view_transform, view.look, view.exposure, view.gamma)
        try:
            settings.file_format = "PNG"
            settings.color_mode = "RGB"
            settings.color_depth = "8"
            if image.get("mjworld_display_view_transform"):
                view.view_transform = image["mjworld_display_view_transform"]
                view.look = image.get("mjworld_display_look", "None")
                view.exposure = image.get("mjworld_display_exposure", 0.0)
                view.gamma = image.get("mjworld_display_gamma", 1.0)
            # save_render applies the scene's view transform/exposure to an HDR
            # bake. Plain save() would silently clip radiance above one.
            image.save_render(str(path), scene=scene)
        finally:
            settings.file_format, settings.color_mode, settings.color_depth = previous
            view.view_transform, view.look, view.exposure, view.gamma = previous_view
        return
    # A datablock copy keeps packed source files and original file paths untouched.
    copy = image.copy()
    try:
        # Blender's Image.copy() copies settings but can leave a generated/dirty
        # image buffer blank. Transfer pixels explicitly, including unsaved bakes.
        buffer = array("f", [0.0]) * len(image.pixels)
        image.pixels.foreach_get(buffer)
        copy.pixels.foreach_set(buffer)
        copy.update()
        copy.filepath_raw = str(path)
        copy.file_format = "PNG"
        copy.save()
    finally:
        bpy.data.images.remove(copy)


def _extract_geometry(obj, depsgraph, unit):
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph)
    if mesh is None:
        raise ExportError(f"{obj.name}: could not evaluate mesh")
    try:
        mesh.calc_loop_triangles()
        world = evaluated.matrix_world.copy()
        origin = world.translation.copy()
        # Bake arbitrary scale, reflection and parent-induced shear into vertices.
        # Only a proper unit rotation remains on the MuJoCo body.
        quaternion = world.to_quaternion().normalized()
        rotation = quaternion.to_matrix()
        inverse = rotation.transposed()
        vertices = [inverse @ (world.to_3x3() @ v.co) * unit for v in mesh.vertices]
        if not vertices or not mesh.loop_triangles:
            raise ExportError(f"{obj.name}: mesh has no triangles; leave it out or convert it to a mesh")
        flip = world.to_3x3().determinant() < 0
        triangles = []
        for tri in mesh.loop_triangles:
            loops = list(tri.loops)
            if flip:
                loops.reverse()
            triangles.append((tri.material_index, [(mesh.loops[i].vertex_index, i) for i in loops]))
        uv_layers = {uv.name: [tuple(v.uv) for v in uv.data] for uv in mesh.uv_layers}
        active_uv = mesh.uv_layers.active.name if mesh.uv_layers.active else None
        lo = Vector(tuple(min(v[i] for v in vertices) for i in range(3)))
        hi = Vector(tuple(max(v[i] for v in vertices) for i in range(3)))
        return {"position": origin * unit, "quaternion": quaternion, "vertices": vertices,
                "triangles": triangles, "uv_layers": uv_layers, "active_uv": active_uv,
                "center": (lo + hi) / 2, "half": (hi - lo) / 2}
    finally:
        evaluated.to_mesh_clear()


def _write_obj(path, geometry, triangles, uv_name=None):
    """Write a per-material mesh; per-loop UVs preserve seams."""
    vertex_ids = sorted({v for tri in triangles for v, _loop in tri})
    remap = {v: i + 1 for i, v in enumerate(vertex_ids)}
    uv = geometry["uv_layers"].get(uv_name)
    with path.open("w", encoding="utf-8") as out:
        out.write("# MuJoCo World export; metres; Blender Z up\n")
        for v in vertex_ids:
            out.write("v " + numbers(geometry["vertices"][v]) + "\n")
        if len(vertex_ids) == 3:
            # MuJoCo requires four mesh vertices even for a visual triangle.
            # An unused point inside that triangle satisfies the file format
            # without adding a face, changing the surface or thickening a floor.
            midpoint = sum((geometry["vertices"][v] for v in vertex_ids), Vector()) / 3
            out.write("v " + numbers(midpoint) + "\n")
        if uv:
            for tri in triangles:
                for _vertex, loop in tri:
                    out.write("vt " + numbers(uv[loop]) + "\n")
        tex_id = 1
        for tri in triangles:
            if uv:
                out.write("f " + " ".join(f"{remap[v]}/{tex_id+i}" for i, (v, _loop) in enumerate(tri)) + "\n")
                tex_id += 3
            else:
                out.write("f " + " ".join(str(remap[v]) for v, _loop in tri) + "\n")


def _mask(value, label):
    try:
        result = int(value, 0)
    except ValueError as exc:
        raise ExportError(f"{label} must be a nonnegative integer (for example 1 or 0x20).") from exc
    if not 0 <= result <= 2147483647:
        raise ExportError(f"{label} must be between 0 and 2147483647.")
    return str(result)


def _asset_paths(scene, filepath):
    """Resolve a blend-relative shared folder and its portable XML prefix."""
    configured = scene.mujoco_world.asset_directory.strip()
    directory = (Path(bpy.path.abspath(configured)).expanduser().resolve() if configured
                 else filepath.parent / (filepath.stem + "_assets"))
    try:
        prefix = Path(os.path.relpath(directory, filepath.parent)).as_posix()
    except ValueError as exc:
        raise ExportError("The XML and assets must be on the same filesystem volume for portable relative paths.") from exc
    directory.mkdir(parents=True, exist_ok=True)
    return directory, prefix, bool(configured)


def _content_name(path, data):
    """Content-addressed file name for ``path`` in the same folder."""
    label = "texture" if path.suffix.lower() == ".png" else "mesh"
    return label + "_" + hashlib.sha256(data).hexdigest()[:24] + path.suffix.lower()


class AssetWriter:
    """Writes the mesh and texture files of one export.

    A shared folder holds files that other exported worlds reference, so a
    shared export never changes an existing file. Index-named files (the
    ordinary exporter) always get content-addressed names. Name-based files
    (adopted worlds) keep their name when that file is absent or byte-identical;
    different content gets a content-addressed name instead.

    A per-XML folder belongs to one world. Its files are staged and replace the
    previous export's files only in commit(), so a failed export leaves the
    previous output intact. Only files this export writes are ever removed,
    never a protected source file.
    """

    def __init__(self, directory, shared):
        self.directory = Path(directory)
        self.shared = shared
        self.written = set()
        self._staged = {}
        self._remove = set()
        self._protected = set()
        self._stage = None

    def protect(self, path):
        """Never remove ``path`` (an adopted world's own source asset)."""
        self._protected.add(Path(path).resolve())

    def _staging(self):
        if self._stage is None:
            self._stage = tempfile.TemporaryDirectory(prefix=".mjworld-stage-", dir=self.directory)
        return Path(self._stage.name)

    def _place(self, target, source, *, content_named, move):
        """Put ``source`` into the shared folder without changing existing files."""
        data = source.read_bytes()
        if not content_named and (not target.exists() or target.read_bytes() == data):
            final = target
        else:
            final = target.parent / _content_name(target, data)
        if not final.exists():
            if move:
                source.replace(final)
            else:
                shutil.copyfile(source, final)
        self.written.add(final.resolve())
        return final.name

    def write(self, filename, writer, *, content_named=False):
        """Write one asset with ``writer(path)``; return the file name to reference."""
        target = self.directory / filename
        if not self.shared:
            staged = self._staging() / filename
            writer(staged)
            self._staged[target.resolve()] = staged
            self.written.add(target.resolve())
            return filename
        with tempfile.TemporaryDirectory(prefix=".mjworld-", dir=self.directory) as temporary:
            working = Path(temporary) / filename
            writer(working)
            return self._place(target, working, content_named=content_named, move=True)

    def copy(self, source, filename):
        """Copy an existing asset file; return the file name to reference."""
        target = self.directory / filename
        if Path(source).resolve() == target.resolve():
            return filename
        if not self.shared:
            return self.write(filename, lambda path: shutil.copyfile(source, path))
        return self._place(target, Path(source), content_named=False, move=False)

    def discard(self, path):
        """Drop an asset that ended up unreferenced, if this export wrote it."""
        path = Path(path).resolve()
        if self.shared or path not in self.written or path in self._protected:
            return
        self._staged.pop(path, None)
        self.written.discard(path)
        self._remove.add(path)

    def commit(self):
        """Move staged files into place and remove discarded ones."""
        for target, staged in self._staged.items():
            os.replace(staged, target)
        self._staged.clear()
        for path in self._remove - self._protected:
            path.unlink(missing_ok=True)
        self._remove.clear()

    def close(self):
        if self._stage is not None:
            self._stage.cleanup()
            self._stage = None


def _replace_file(path, write):
    """Write ``path`` through a sibling temporary file, so a failure keeps the old file."""
    path = Path(path)
    temporary = path.with_name("." + path.name + ".mjworld-tmp")
    try:
        write(temporary)
        if path.exists():
            shutil.copymode(path, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_xml(root, path):
    ET.indent(root, space="  ")
    _replace_file(path, lambda target: ET.ElementTree(root).write(target, encoding="utf-8", xml_declaration=True))


def write_text(path, text):
    _replace_file(path, lambda target: target.write_text(text, encoding="utf-8"))


def mask_value(settings, name):
    """Normalized contact mask from a scene setting (decimal, hex or binary input)."""
    return _mask(getattr(settings, name), name.replace("_", " "))


def role_masks(settings, role):
    """Contact type and mask for a collision geom of a body that moves as ``role``.

    ``role`` is FREE or HINGE for geoms that move with such a body, else None.
    A blank door mask follows the fixed mask, which doors used before they had
    their own settings.
    """
    def pick(kind):
        if role == "FREE":
            return mask_value(settings, "movable_" + kind)
        if role == "HINGE" and getattr(settings, "hinge_" + kind).strip():
            return mask_value(settings, "hinge_" + kind)
        return mask_value(settings, "object_" + kind)
    return pick("contype"), pick("conaffinity")


def _export_simple(context, filepath, *, regroup=False):
    """Write an ordinary (not adopted) world; works in UI and Blender background mode.

    With ``regroup``, body group Empties become compound bodies first. Files
    are written only once the whole world is built, so a failed export keeps
    the previous XML, report and per-XML assets.
    """
    from .project import _included, body_owner, check_evaluated, is_body, regroup_export, validate_parts
    scene = context.scene
    settings = scene.mujoco_world
    filepath = xml_path(filepath)
    if context.mode != "OBJECT":
        raise ExportError("Switch to Object Mode before exporting.")
    unit = float(scene.unit_settings.scale_length)
    if unit <= 0:
        raise ExportError("Scene unit scale must be positive.")
    candidates = list(context.selected_objects) if settings.selected_only else list(scene.objects)
    if settings.selected_only:
        for obj in list(candidates):
            if is_body(obj):
                candidates.extend(obj.children_recursive)
    # 'Leave out' on a body group Empty also leaves out every part it owns.
    objects = sorted(set(o for o in candidates if o.type == "MESH" and
                         ((o.visible_get(view_layer=context.view_layer) and not o.hide_render) or o.get("mjworld_bake_source"))
                         and _included(o)), key=lambda o: o.name)
    if not objects:
        raise ExportError("No visible mesh objects to export. Select a mesh, or turn off 'Export selected objects only'.")
    plane_mask = (mask_value(settings, "plane_contype"), mask_value(settings, "plane_conaffinity"))
    masks = {role: role_masks(settings, role) for role in (None, "FREE", "HINGE")}
    labels = [f"{slug(obj.name)}_{index:03d}" for index, obj in enumerate(objects)]
    # Validate before anything is written, so a rejected export leaves no XML.
    validate_parts(scene, objects, labels)
    for obj in objects:
        if obj.mujoco_world.role == "HINGE":
            check_door(obj)
    owners = {owner for obj in objects for owner in [body_owner(obj)] if owner}
    check_evaluated(context, objects + sorted(owners, key=lambda o: o.name))
    filepath.parent.mkdir(parents=True, exist_ok=True)
    assets_path, asset_prefix, shared_assets = _asset_paths(scene, filepath)
    assets = AssetWriter(assets_path, shared_assets)
    try:
        root, report = _build_simple(context, filepath, objects, labels, candidates, unit, plane_mask, masks, assets, asset_prefix)
        if regroup:
            report = regroup_export(context, root, report)
        assets.commit()
        write_xml(root, filepath)
        write_text(filepath.parent / "export_report.json", json.dumps(report, indent=2) + "\n")
        write_text(filepath.parent / "OPEN_WORLD.txt",
                   f"MuJoCo 3.15+\n\nOpen with:\n  python -m mujoco.viewer --mjcf={filepath.name}\n\n"
                   f"Keep the XML and asset folder ({asset_prefix}) together. All asset paths are relative.\n"
                   "In the viewer, double-click a movable object/door, then Ctrl+right-drag to apply force.\n"
                   "Visual meshes do not collide. Collision shapes and the analytic floor are in group 3.\n"
                   "For an omniwheel robot, preserve its explicit wheel-capsule/floor contact pairs when merging.\n"
                   "This isolated world does not contain a robot or ROS controller configuration.\n")
    finally:
        assets.close()
    return report


def _build_simple(context, filepath, objects, labels, candidates, unit, plane_mask, masks, assets, asset_prefix):
    """Build the ordinary world in memory; asset files go through ``assets``."""
    from .project import _included
    scene = context.scene
    settings = scene.mujoco_world
    warnings = []
    for candidate in candidates:
        if candidate.visible_get(view_layer=context.view_layer) and not candidate.hide_render and _included(candidate):
            if candidate.type in {"CURVE", "SURFACE", "FONT", "META"}:
                warnings.append(f"{candidate.name}: convert this {candidate.type.lower()} to mesh before exporting; it was left out.")
            if candidate.is_instancer:
                warnings.append(f"{candidate.name}: instances are not exported. Realize instances / Make Instances Real before exporting.")
    root = ET.Element("mujoco", model=slug(filepath.stem))
    # Radians match robot models such as Walkie, which merge this world and
    # require one angle convention. Inertia settings stay at MuJoCo defaults:
    # compiler settings are global, so values written here would also change a
    # merged robot's inertials. Visual geoms carry mass 0 instead.
    ET.SubElement(root, "compiler", angle="radian", meshdir=asset_prefix, texturedir=asset_prefix, autolimits="true")
    ET.SubElement(root, "option", timestep="0.002", gravity="0 0 -9.81", integrator="implicitfast", cone="elliptic")
    ET.SubElement(root, "size", memory="8M")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="1280", offheight="960")
    ET.SubElement(visual, "headlight", diffuse="0.4 0.4 0.4", ambient="0.3 0.3 0.3", specular="0.05 0.05 0.05")
    ET.SubElement(visual, "rgba", haze="0.65 0.68 0.7 1")
    asset = ET.SubElement(root, "asset")
    world = ET.SubElement(root, "worldbody")
    if settings.plane:
        ET.SubElement(world, "geom", name="floor", type="plane", pos=numbers([0, 0, property_value(settings.plane_height)]),
                      size="0 0 0.05", rgba="0 0 0 0", group="3", contype=plane_mask[0],
                      conaffinity=plane_mask[1], friction="0.8 0.005 0.0001", mass="0")
    depsgraph = context.evaluated_depsgraph_get()
    report_objects = []
    exported_images = {}
    material_cache = {}
    all_world_vertices = []
    for obj, label in zip(objects, labels):
        cfg = obj.mujoco_world
        geometry = _extract_geometry(obj, depsgraph, unit)
        center, half = geometry["center"], geometry["half"]
        rotation = geometry["quaternion"].to_matrix()
        all_world_vertices.extend(geometry["position"] + rotation @ v for v in geometry["vertices"])
        if obj.parent:
            warnings.append(f"{obj.name}: parent transforms were baked in. Exported as an independent body; hierarchy/joints are not inherited.")
        body = ET.SubElement(world, "body", name=label, pos=numbers(geometry["position"]), quat=numbers(geometry["quaternion"]))
        if cfg.role == "FREE":
            ET.SubElement(body, "freejoint", name=label + "_free")
        elif cfg.role == "HINGE":
            low, high, start = door_radians(cfg)
            axis = {"X": [1, 0, 0], "Y": [0, 1, 0], "Z": [0, 0, 1]}[cfg.hinge_axis]
            from mathutils import Quaternion
            body.set("quat", numbers(geometry["quaternion"] @ Quaternion(Vector(axis), start)))
            ET.SubElement(body, "joint", name=label + "_hinge", type="hinge", axis=numbers(axis), pos="0 0 0",
                          range=numbers([low, high]), ref=numbers([start]), damping=property_numbers([cfg.damping]))
        by_material = {}
        if cfg.role != "COLLIDER" and not obj.get("mjworld_bake_source"):
            for material_id, tri in geometry["triangles"]:
                by_material.setdefault(material_id, []).append(tri)
        total_triangles = 0
        for material_id, triangles in sorted(by_material.items()):
            mat = obj.material_slots[material_id].material if material_id < len(obj.material_slots) else None
            info = inspect_material(mat)
            for note in info["warnings"]:
                warnings.append(f"{obj.name} / {mat.name if mat else 'no material'}: {note}")
            uv_name = info.get("uv_map") or geometry["active_uv"]
            image = info["image"]
            if image and uv_name not in geometry["uv_layers"]:
                warnings.append(f"{obj.name}: image has no matching UV map; export used the flat material colour. UV unwrap and bake first.")
                image = None
            baked = bool(cfg.baked or info["emission"])
            key = (mat.as_pointer() if mat else 0, baked, image.as_pointer() if image else 0)
            if key not in material_cache:
                material_name = f"material_{len(material_cache):03d}"
                rgba = info["color"][:]
                if settings.texture_mode == "DISPLAY":
                    rgba[:3] = [srgb(c) for c in rgba[:3]]
                material_args = {"name": material_name, "rgba": numbers(rgba), "specular": "0", "shininess": "0.2",
                                 "emission": property_numbers([settings.baked_emission]) if baked else "0"}
                if image:
                    image_key = (image.as_pointer(), baked)
                    if image_key not in exported_images:
                        image_name = f"texture_{len(exported_images):03d}"
                        image_file = assets.write(image_name + ".png", lambda target: _write_image(image, target, baked=baked, scene=scene),
                                                  content_named=True)
                        if image.is_float:
                            warnings.append(f"{obj.name}: HDR bake '{image.name}' converted to display PNG using {scene.view_settings.view_transform}, exposure {scene.view_settings.exposure:g}. Keep that view transform consistent when comparing Blender and MuJoCo.")
                        ET.SubElement(asset, "texture", name=image_name, type="2d", file=image_file,
                                      colorspace="linear" if settings.texture_mode == "DISPLAY" else "sRGB")
                        exported_images[image_key] = image_name
                    material_args["texture"] = exported_images[image_key]
                    material_args["rgba"] = "1 1 1 " + f"{rgba[3]:.9g}"
                    material_args["texuniform"] = "false"
                ET.SubElement(asset, "material", **material_args)
                material_cache[key] = material_name
            mesh_name = f"{label}_visual_{material_id}"
            mesh_file = assets.write(mesh_name + ".obj", lambda target: _write_obj(target, geometry, triangles, uv_name if image else None),
                                     content_named=True)
            ET.SubElement(asset, "mesh", name=mesh_name, file=mesh_file, inertia="shell")
            ET.SubElement(body, "geom", name=mesh_name, type="mesh", mesh=mesh_name,
                          material=material_cache[key], contype="0", conaffinity="0", group="1", mass="0")
            total_triangles += len(triangles)
        collision_type = None
        if cfg.role != "VISUAL":
            contype, conaffinity = masks[cfg.role if cfg.role in {"FREE", "HINGE"} else None]
            collision_args = {"name": label + "_collision", "pos": numbers(center), "group": "3", "rgba": "0 0 0 0",
                              "contype": contype, "conaffinity": conaffinity,
                              "friction": property_numbers(cfg.friction),
                              "mass": property_numbers([cfg.mass]) if cfg.role in {"FREE", "HINGE"} else "0"}
            flat_error = f"{obj.name}: collision mesh has zero thickness. Choose Visual floor / Visual only, or give it thickness."
            if min(half) < 0.00001:
                raise ExportError(flat_error)
            if cfg.collision in {"BOX", "CONVEX"} and _flat_half_thickness(geometry, [tri for _m, tri in geometry["triangles"]]) < 0.00001:
                # A flat mesh tilted in its local axes passes the box test above.
                if cfg.collision == "CONVEX":
                    raise ExportError(flat_error)
                warnings.append(f"{obj.name}: this flat mesh is tilted in its local axes, so its collision box is a solid block around it. Give it thickness or use separate collision helpers.")
            if cfg.collision == "BOX":
                collision_args.update(type="box", size=numbers(half))
            elif cfg.collision == "SPHERE":
                radius = max((v - center).length for v in geometry["vertices"])
                collision_args.update(type="sphere", size=numbers([radius]))
            elif cfg.collision == "CYLINDER":
                radius = max(math.hypot(v.x - center.x, v.y - center.y) for v in geometry["vertices"])
                collision_args.update(type="cylinder", size=numbers([radius, half.z]))
            else:
                collision_name = label + "_hull"
                collision_file = assets.write(collision_name + ".obj",
                                              lambda target: _write_obj(target, geometry, [tri for _m, tri in geometry["triangles"]]),
                                              content_named=True)
                ET.SubElement(asset, "mesh", name=collision_name, file=collision_file, inertia="convex")
                collision_args.update(type="mesh", mesh=collision_name, pos="0 0 0")
                warnings.append(f"{obj.name}: convex collision fills cavities. For shelves, tables or doorways use separate box pieces.")
            collision_type = collision_args["type"]
            ET.SubElement(body, "geom", **collision_args)
        report_objects.append({"source": obj.name, "body": label, "role": cfg.role, "collision": collision_type,
                               "triangles": total_triangles, "position_m": list(geometry["position"]),
                               "quaternion_wxyz": list(geometry["quaternion"]),
                               "bounds_center_local_m": list(center), "bounds_half_local_m": list(half),
                               "mass_kg": property_value(cfg.mass) if cfg.role in {"FREE", "HINGE"} else 0})
    midpoint = [0.5 * (min(v[i] for v in all_world_vertices) + max(v[i] for v in all_world_vertices)) for i in range(3)]
    extent = max(1.0, *(max(v[i] for v in all_world_vertices) - min(v[i] for v in all_world_vertices) for i in range(3)))
    ET.SubElement(root, "statistic", center=numbers(midpoint), extent=f"{extent:.9g}")
    warnings = list(dict.fromkeys(warnings))
    report = {"addon_version": "0.1.0", "blender_version": bpy.app.version_string, "file": filepath.name,
              "objects": len(objects), "meshes": sum(1 for x in asset if x.tag == "mesh"), "textures": len(exported_images),
              "triangles": sum(x["triangles"] for x in report_objects), "metres_per_blender_unit": unit,
              "plane": {"enabled": settings.plane, "height_m": property_value(settings.plane_height),
                        "name": "floor", "only_visual_floor_meshes": all(x["role"] == "VISUAL" for x in report_objects if "floor" in x["source"].lower())},
              "texture_input": "linear/display values" if settings.texture_mode == "DISPLAY" else "sRGB decoded",
              "object_details": report_objects, "warnings": warnings}
    return root, report


def export_world(context, filepath):
    """Export an authored hierarchy, adopted MJCF project or ordinary mesh scene."""
    from .project import export_adopted, is_body
    # Resolve once: every stage must read and write the same file name.
    filepath = xml_path(filepath)
    if context.scene.get("mjworld_mjcf_template"):
        return export_adopted(context, filepath)
    return _export_simple(context, filepath, regroup=any(is_body(obj) for obj in context.scene.objects))
