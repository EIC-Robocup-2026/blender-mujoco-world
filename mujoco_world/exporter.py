"""MJCF writer. Visual shape, collision approximation and mass are separate."""

import json
import hashlib
import math
import os
from array import array
from pathlib import Path
import re
import secrets
import shutil
import struct
import tempfile
import xml.etree.ElementTree as ET

import bpy
from mathutils import Vector


class ExportError(ValueError):
    pass


def addon_version():
    """The add-on version from bl_info, as written into every export report."""
    from . import bl_info
    return ".".join(str(part) for part in bl_info["version"])


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


def blend_path(value, what):
    """Resolve a path that may be relative to the .blend file ('//').

    An unsaved .blend has no folder, and Blender would resolve '//' against
    its working directory instead, so such a path is refused.
    """
    value = str(value)
    if value.startswith("//") and not bpy.data.filepath:
        raise ExportError(f"{what} '{value}' is relative to the .blend file, which is not saved yet. "
                          "Save the .blend first, or choose an absolute folder.")
    return Path(bpy.path.abspath(value)).expanduser().resolve()


def xml_path(filepath):
    """Resolve an export target; an existing .xml suffix keeps its case."""
    path = blend_path(filepath, "The export path")
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


def material_uses_nodes(material):
    """Whether ``material`` renders through its node tree.

    Blender 5.0 deprecated Material.use_nodes (removal is planned for 6.0):
    there every material with a node tree uses it, and reading the property
    only emits a deprecation warning.
    """
    if material is None or material.node_tree is None:
        return False
    return True if bpy.app.version >= (5, 0, 0) else bool(material.use_nodes)


def _live_link(socket):
    """The link that drives ``socket`` as Blender renders it, followed through reroutes.

    Muted and invalid links render as the socket's default value, so they
    count as no link.
    """
    seen = set()
    while socket is not None and socket.is_linked:
        link = next((link for link in socket.links if link.is_valid and not link.is_muted), None)
        if link is None:
            return None
        node = link.from_node
        if node.type != "REROUTE" or node.as_pointer() in seen:
            return link
        seen.add(node.as_pointer())
        socket = node.inputs[0] if node.inputs else None
    return None


def _socket_source(socket):
    link = _live_link(socket)
    return link.from_node if link else None


def inspect_material(material):
    """Return only the small material subset that can be represented faithfully."""
    if material is None:
        return {"color": [0.7, 0.7, 0.7, 1], "image": None, "emission": False, "warnings": []}
    result = {"color": list(material.diffuse_color), "image": None, "emission": False, "warnings": []}
    if not material_uses_nodes(material):
        return result
    outputs = [n for n in material.node_tree.nodes if n.type == "OUTPUT_MATERIAL" and n.is_active_output]
    shader = _socket_source(outputs[0].inputs.get("Surface")) if outputs else None
    if shader is None:
        result["warnings"].append("No active surface shader; using material colour.")
        return result
    if shader.mute:
        result["warnings"].append(f"Shader node '{shader.name}' is muted; using material colour.")
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
        if source.mute:
            result["warnings"].append(f"Colour node '{source.name}' is muted; bake its appearance to a UV image first.")
        elif source.type == "TEX_IMAGE" and source.image and source.projection == "FLAT":
            vector_link = _live_link(source.inputs.get("Vector"))
            vector = vector_link.from_node if vector_link else None
            if vector and (vector.mute or vector.type not in {"UVMAP", "TEX_COORD"}):
                result["warnings"].append("Image uses a Mapping/procedural vector; bake to the active UV map first.")
            elif vector and vector.type == "TEX_COORD" and vector_link.from_socket.name != "UV":
                result["warnings"].append("Image does not use UV coordinates; bake to the active UV map first.")
            else:
                result["image"] = source.image
                result["uv_map"] = vector.uv_map if vector and vector.type == "UVMAP" else None
        else:
            result["warnings"].append("Procedural/combined colour cannot be translated; bake it to a UV image first.")
    if shader.type == "BSDF_PRINCIPLED":
        for field in ("Normal", "Metallic", "Roughness", "Alpha"):
            sock = shader.inputs.get(field)
            if sock and _live_link(sock):
                result["warnings"].append(f"Linked {field} is not exported; this exporter carries colour/lighting only.")
        if shader.inputs.get("Alpha"):
            result["color"][3] = float(shader.inputs["Alpha"].default_value)
    return result


STANDARD_PROFILE = {"view_transform": "Standard", "look": "None", "exposure": 0.0, "gamma": 1.0}


def profile_text(profile):
    """Readable colour profile for report notes."""
    return (f"{profile['view_transform']} (look {profile['look']}, exposure {profile['exposure']:g}, "
            f"gamma {profile['gamma']:g}, display {profile.get('display_device', 'sRGB')})")


def _apply_profile(scene, profile):
    """Pin ``scene``'s display and view to ``profile``; record the display used."""
    display = scene.display_settings
    try:
        display.display_device = "sRGB"
    except TypeError:
        pass  # A custom OCIO configuration without an sRGB display keeps its default.
    view = scene.view_settings
    for name in ("view_transform", "look"):
        try:
            setattr(view, name, profile[name])
        except TypeError as exc:
            raise ExportError(f"The colour {name.replace('_', ' ')} '{profile[name]}' is not available in "
                              "this Blender's colour management configuration.") from exc
    view.exposure, view.gamma = profile["exposure"], profile["gamma"]
    view.use_curve_mapping = False
    if hasattr(view, "use_white_balance"):
        view.use_white_balance = False
    profile["display_device"] = display.display_device


def check_profile(profile):
    """Raise ExportError when ``profile`` cannot be applied in this Blender."""
    scene = bpy.data.scenes.new(".mjworld colour check")
    try:
        _apply_profile(scene, dict(STANDARD_PROFILE, **profile))
    finally:
        bpy.data.scenes.remove(scene)


def save_image(image, path, *, file_format="PNG", color_depth="8", profile=None):
    """Save ``image`` through a temporary scene; return the colour profile applied.

    The temporary scene keeps the file independent of the user's output
    settings: a Blender 5 video or multilayer output (which rejects PNG and
    OpenEXR), Output colour management set to Override, curves and white
    balance. The view uses ``profile`` (view transform, look, exposure and
    gamma; Standard, None, 0 and 1 by default) on the sRGB display. OpenEXR
    output stays scene-linear; the view does not apply to it.
    """
    applied = dict(STANDARD_PROFILE, **(profile or {}))
    scene = bpy.data.scenes.new(".mjworld colour profile")
    try:
        settings = scene.render.image_settings
        if hasattr(settings, "media_type"):
            settings.media_type = "IMAGE"
        settings.file_format = file_format
        settings.color_mode = "RGB"
        settings.color_depth = color_depth
        if file_format == "OPEN_EXR":
            settings.exr_codec = "ZIP"
        settings.color_management = "FOLLOW_SCENE"
        _apply_profile(scene, applied)
        image.save_render(str(path), scene=scene)
    finally:
        bpy.data.scenes.remove(scene)
    return applied


def display_profile(image, scene):
    """Colour profile that converts the HDR ``image`` to its display PNG.

    A bake records the profile of its own lightmap PNG on the image. Any other
    HDR image follows the scene's view transform, look, exposure and gamma.
    Recorded fields that are missing default to look None, exposure 0, gamma 1.
    """
    if image.get("mjworld_display_view_transform"):
        return {"view_transform": str(image["mjworld_display_view_transform"]),
                "look": str(image.get("mjworld_display_look", "None")),
                "exposure": float(image.get("mjworld_display_exposure", 0.0)),
                "gamma": float(image.get("mjworld_display_gamma", 1.0))}
    view = scene.view_settings
    return {"view_transform": view.view_transform, "look": view.look,
            "exposure": float(view.exposure), "gamma": float(view.gamma)}


def image_problem(image):
    """Why ``image`` cannot be exported as one flat texture, or None."""
    if image.source == "TILED":
        return "is a UDIM (tiled) image, which MuJoCo cannot use; bake it to one flat UV image"
    if image.source not in {"FILE", "GENERATED"}:
        return f"is a {image.source.lower()} image; use a single still image"
    width, height = image.size
    if width and height:
        return None
    path = image.filepath_raw or image.filepath
    if not path:
        return "has no pixels"
    if path.startswith("//") and not bpy.data.filepath:
        return f"has the path '{path}', relative to a .blend file that is not saved yet; save the .blend or use an absolute path"
    return f"has no pixels: '{bpy.path.abspath(path, library=image.library)}' is missing or unreadable"


def texture_colorspace(settings, linear):
    """MuJoCo colorspace of a written PNG; ``linear`` when it holds linear values."""
    return "linear" if linear or settings.texture_mode == "DISPLAY" else "sRGB"


def _linear_colorspace(image):
    """Whether the image's values are linear: Non-Color data or a linear colour space."""
    settings = image.colorspace_settings
    return bool(settings.is_data) or settings.name.startswith("Linear") or settings.name in {"ACEScg", "ACES2065-1"}


def _unmodified_png(image):
    """The PNG file bytes of an image that Blender has not changed, else None."""
    if image.source != "FILE" or image.is_dirty or image.file_format != "PNG":
        return None
    if image.packed_file:
        data = bytes(image.packed_file.data)
    else:
        path = Path(bpy.path.abspath(image.filepath_raw or image.filepath, library=image.library))
        if not path.is_file():
            return None
        data = path.read_bytes()
    return data if data.startswith(b"\x89PNG\r\n\x1a\n") else None


def _rgba_pixels(image):
    """The image buffer as an (N, 4) float32 numpy array."""
    import numpy as np
    channels = image.channels
    values = np.empty(len(image.pixels), dtype=np.float32)
    image.pixels.foreach_get(values)
    values = values.reshape(-1, channels)
    rgba = np.ones((values.shape[0], 4), dtype=np.float32)
    rgba[:, :3] = values[:, :3] if channels >= 3 else values[:, :1]
    if channels in {2, 4}:
        rgba[:, 3] = values[:, -1]
    return rgba


def _encode_srgb(rgba):
    """Encode linear RGB with the sRGB transfer function srgb() uses; alpha is kept."""
    import numpy as np
    result = rgba.copy()
    rgb = np.clip(rgba[:, :3], 0.0, 1.0)
    result[:, :3] = np.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * np.power(rgb, 1 / 2.4) - 0.055)
    return result


def _save_png_pixels(image, path, rgba):
    """Save display-encoded ``rgba`` values with the size of ``image`` as an 8-bit PNG."""
    width, height = image.size
    target = bpy.data.images.new(".mjworld texture", width, height, alpha=True, float_buffer=False)
    try:
        # Set the colour space before the pixels: changing it reloads the buffer.
        target.colorspace_settings.name = "sRGB"
        target.pixels.foreach_set(rgba.ravel())
        target.filepath_raw = str(path)
        target.file_format = "PNG"
        target.save()
    finally:
        bpy.data.images.remove(target)


def _copy_png(image, path):
    """Save the byte buffer of ``image`` unchanged as PNG."""
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


def _encode_image(image, path, baked, scene):
    if image.is_float:
        # 16-bit PNGs load as float buffers. An unchanged sRGB one is copied,
        # also when it holds lighting: its values are display values already,
        # and MuJoCo reads 16-bit PNGs as 8-bit. A lightmap this add-on baked
        # records its display profile and is always converted with it.
        data = None
        if not _linear_colorspace(image) and not (baked and image.get("mjworld_display_view_transform")):
            data = _unmodified_png(image)
        if data is not None:
            path.write_bytes(data)
            return {"profile": None, "linear": False, "hdr": False, "notes": []}
        if baked:
            # save_render applies the display profile to the HDR bake; plain
            # save() would silently clip radiance above one.
            return {"profile": save_image(image, path, profile=display_profile(image, scene)), "linear": False,
                    "hdr": True, "notes": []}
        rgba = _rgba_pixels(image)
        clipped = bool((rgba[:, :3] > 1.0).any())
        _save_png_pixels(image, path, _encode_srgb(rgba))
        note = f"floating-point image '{image.name}' was written as an 8-bit sRGB PNG"
        if clipped:
            note += ("; values above 1 were clipped. If it is a light bake, enable "
                     "'Image already contains lighting' to convert it with a view transform")
        return {"profile": dict(STANDARD_PROFILE, display_device="sRGB"), "linear": False, "hdr": False, "notes": [note]}
    if not _linear_colorspace(image):
        _copy_png(image, path)
        return {"profile": None, "linear": False, "hdr": False, "notes": []}
    notes = [f"image '{image.name}' is tagged {image.colorspace_settings.name}, so its values are linear; "
             "use the sRGB colour space for colour textures"]
    if scene.mujoco_world.texture_mode == "DISPLAY":
        # Display PNG values: encode what Blender shows as linear colour.
        _save_png_pixels(image, path, _encode_srgb(_rgba_pixels(image)))
        return {"profile": None, "linear": False, "hdr": False, "notes": notes}
    _copy_png(image, path)
    return {"profile": None, "linear": True, "hdr": False, "notes": notes}


def _write_image(image, path, *, baked=False, scene, where=None):
    """Write ``image`` to the PNG ``path`` and describe the result.

    Returns ``{"profile", "linear", "hdr", "notes"}``: the colour profile a
    float image was converted with (None otherwise), whether the PNG holds
    linear values (a linear byte image in sRGB texture mode; tag its texture
    with texture_colorspace()), whether an HDR light bake was converted to
    display values with the profile (worth a warning: the view transform must
    stay consistent), and notes for the report. Errors name the original
    image, after ``where`` (object and material).
    """
    try:
        return _encode_image(image, Path(path), baked, scene)
    except (RuntimeError, ExportError) as exc:
        label = f"{where}: image '{image.name}'" if where else f"Image '{image.name}'"
        raise ExportError(f"{label} could not be exported: {exc}") from exc


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
        # Blender renders an image without an explicit UV map with the
        # render-active map, not the one active for editing.
        render = next((uv for uv in mesh.uv_layers if uv.active_render), None) or mesh.uv_layers.active
        render_uv = render.name if render else None
        lo = Vector(tuple(min(v[i] for v in vertices) for i in range(3)))
        hi = Vector(tuple(max(v[i] for v in vertices) for i in range(3)))
        return {"position": origin * unit, "quaternion": quaternion, "vertices": vertices,
                "triangles": triangles, "uv_layers": uv_layers, "render_uv": render_uv,
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
    directory = (blend_path(configured, "The shared asset folder") if configured
                 else filepath.parent / (filepath.stem + "_assets"))
    try:
        prefix = Path(os.path.relpath(directory, filepath.parent)).as_posix()
    except ValueError as exc:
        raise ExportError("The XML and assets must be on the same filesystem volume for portable relative paths.") from exc
    directory.mkdir(parents=True, exist_ok=True)
    return directory, prefix, bool(configured)


def _content_name(path, data):
    """Content-addressed file name for ``path`` in the same folder."""
    return _digest_name(path, hashlib.sha256(data).hexdigest())


def _digest_name(path, digest):
    """Content-addressed file name for ``path`` whose content has the SHA-256 hex ``digest``."""
    label = "texture" if path.suffix.lower() == ".png" else "mesh"
    return label + "_" + digest[:24] + path.suffix.lower()


def file_digest(path):
    """SHA-256 hex digest of the file at ``path``."""
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _holds(path, data):
    """Whether ``path`` is a readable file whose bytes are ``data``."""
    try:
        return path.is_file() and path.read_bytes() == data
    except OSError:
        return False


class AssetWriter:
    """Writes the mesh and texture files of one export.

    Files are addressed by name inside the resolved folder. A file name is
    never resolved, so a symbolic link in the folder is replaced or left alone,
    never written through or removed together with its target.

    A shared folder holds files that other exported worlds reference, so a
    shared export never changes an existing file. Index-named files (the
    ordinary exporter) always get content-addressed names. Name-based files
    (adopted worlds) keep their name when that file is absent or byte-identical;
    different content gets a content-addressed name instead.

    A per-XML folder belongs to one world. Its files are staged and replace the
    previous export's files only in commit(), so a failed export leaves the
    previous output intact. The folder's manifest (MANIFEST) records the name
    and SHA-256 digest of every file the last export wrote; finish(), called
    once the new XML is in place, removes those the new export no longer
    writes, so renamed or deleted objects leave no stale files. A stale file is
    removed only when it is a regular file directly in the folder whose content
    still matches its recorded digest: files the manifest does not list, files
    edited since, symbolic links (and whatever they point to) and protected
    source files are kept. A shared folder has no manifest: other worlds may
    still use any of its files.

    A per-XML export replaces only the files its folder's manifest lists (and
    creates absent ones). Any other existing file is not this world's: another
    world may use the folder as its shared asset folder, an older add-on
    version wrote it, or it is an adopted world's own source file. It is
    referenced as it is when its content is identical, and different content
    gets a content-addressed name beside it instead.

    Within one export a name holds one content. A second file written under a
    name the export already used for other content (a texture and a height
    field of the same name, say) gets a content-addressed name; identical
    content shares the name.
    """

    MANIFEST = ".mujoco_world_manifest.json"

    def __init__(self, directory, shared):
        self.directory = Path(directory).resolve()
        self.shared = shared
        self.written = set()
        self._staged = {}
        self._used = {}
        self._protected = set()
        self._stage = None
        self._count = 0
        self._previous = {} if shared else self._read_manifest()

    def _entry(self, path):
        """``path`` with its folder resolved but its file name kept, so a link is not followed."""
        path = Path(path)
        return path.parent.resolve() / path.name

    def _read_manifest(self):
        """Name and recorded digest of each file the previous export into this folder wrote.

        A name without a digest maps to None; such a file is never removed.
        """
        try:
            manifest = json.loads((self.directory / self.MANIFEST).read_text(encoding="utf-8"))
            names, digests = manifest.get("files", []), manifest.get("sha256", {})
        except (OSError, ValueError, AttributeError):
            return {}
        if not isinstance(names, list):
            return {}
        if not isinstance(digests, dict):
            digests = {}
        # Only plain file names directly inside the folder; anything else is ignored.
        return {name: digests[name] if isinstance(digests.get(name), str) else None for name in names
                if isinstance(name, str) and name and name == Path(name).name and not name.startswith(".")
                and "\\" not in name}

    def protect(self, path):
        """Never remove ``path`` (an adopted world's own source asset), nor the file it links to."""
        self._protected.update({self._entry(path), Path(path).resolve()})

    def _staging(self):
        if self._stage is None:
            self._stage = tempfile.TemporaryDirectory(prefix=".mjworld-stage-", dir=self.directory)
        return Path(self._stage.name)

    def _place(self, target, source, *, content_named, move):
        """Put ``source`` into the shared folder without changing existing files."""
        data = source.read_bytes()
        # A dangling link counts as an existing file, so no file is created through it.
        if not content_named and (not os.path.lexists(target) or _holds(target, data)):
            final = target
        else:
            final = target.parent / _content_name(target, data)
        if not os.path.lexists(final):
            if move:
                source.replace(final)
            else:
                shutil.copyfile(source, final)
        elif not _holds(final, data):
            raise ExportError(f"The shared asset folder already has '{final.name}' with other content, and an export never "
                              "changes a file there. Remove or rename that file, then export again.")
        self.written.add(final)
        return final.name

    def write(self, filename, writer, *, content_named=False):
        """Write one asset with ``writer(path)``; return the file name to reference."""
        target = self.directory / filename
        if not self.shared:
            # One folder per file, so two files of one name never overwrite each other here.
            self._count += 1
            staged = self._staging() / str(self._count) / filename
            staged.parent.mkdir()
            writer(staged)
            digest = file_digest(staged)
            if self._settle(target, staged, digest):
                return filename
            final = target.parent / _digest_name(target, digest)
            if self._settle(final, staged, digest):
                return final.name
            raise ExportError(f"The asset folder {self.directory} already has '{final.name}' with other content, which this "
                              "world's last export did not write, so the export does not replace it. Remove or rename that "
                              "file, then export again.")
        with tempfile.TemporaryDirectory(prefix=".mjworld-", dir=self.directory) as temporary:
            working = Path(temporary) / filename
            writer(working)
            return self._place(target, working, content_named=content_named, move=True)

    def _settle(self, target, staged, digest):
        """Use the per-XML name ``target`` for the staged file with SHA-256 ``digest``; return whether it can.

        The name can hold the file when this export has not used it for other
        content and it is free: absent, or listed in the previous manifest and
        not an adopted world's source file. An existing file that is not this
        world's is referenced as it is when its content is identical.
        """
        used = self._used.get(target)
        if used is not None:
            return used == digest
        if not os.path.lexists(target) or (target.name in self._previous and target not in self._protected):
            self._staged[target] = staged
            self.written.add(target)
        elif not self._identical(target, digest):
            return False
        self._used[target] = digest
        return True

    @staticmethod
    def _identical(path, digest):
        """Whether ``path`` is a regular file (not a link) whose content has SHA-256 ``digest``."""
        try:
            return not path.is_symlink() and path.is_file() and file_digest(path) == digest
        except OSError:
            return False

    def copy(self, source, filename):
        """Copy an existing asset file; return the file name to reference."""
        target = self.directory / filename
        if Path(source).resolve() == target.resolve():
            # The source already is the asset file (an export into the adopted world's own folder).
            if self.shared:
                return filename
            digest = file_digest(source)
            if self._used.setdefault(target, digest) == digest:
                return filename
        if not self.shared:
            return self.write(filename, lambda path: shutil.copyfile(source, path))
        return self._place(target, Path(source), content_named=False, move=False)

    def discard(self, path):
        """Drop an asset that ended up unreferenced, if this export wrote it.

        The previous export's file of that name then counts as stale.
        """
        path = self._entry(path)
        if self.shared or path not in self.written or path in self._protected:
            return
        self._staged.pop(path, None)
        self._used.pop(path, None)
        self.written.discard(path)

    def _stale(self, name, digest):
        """The previous export's file ``name`` if it may be removed, else None."""
        path = self.directory / name
        if digest is None or path.parent != self.directory or path in self._protected:
            return None
        try:
            # A link is never removed, and neither is the file it points to.
            if path.is_symlink() or not path.is_file() or path.resolve() in self._protected:
                return None
            # A file edited since the export wrote it is the user's now.
            return path if file_digest(path) == digest else None
        except OSError:
            return None

    def _record(self, digests):
        """Write the manifest for ``digests`` (file name to SHA-256, or None when unknown)."""
        record = {"files": sorted(digests), "sha256": {name: digests[name] for name in sorted(digests) if digests[name]}}
        _replace_file(self.directory / self.MANIFEST,
                      lambda target: target.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8"))

    def commit(self):
        """Move the staged files into place, before the XML that references them.

        A per-XML folder's manifest first lists the new files beside the
        previous ones, so a file placed here stays this world's even when the
        XML is not replaced afterwards. Nothing is removed until finish().
        """
        if self.shared:
            # A folder that became shared no longer belongs to one world.
            (self.directory / self.MANIFEST).unlink(missing_ok=True)
            return
        if self._staged:
            self._record({**self._previous, **{target.name: self._used[target] for target in self._staged}})
        for target, staged in self._staged.items():
            # Replacing the name replaces a link there, never the file it points to.
            os.replace(staged, target)
        self._staged.clear()

    def finish(self):
        """Once the new XML is in place, remove the stale files of a per-XML folder and record its manifest."""
        if self.shared:
            return
        files = {path.name: self._used[path] for path in self.written if path.parent == self.directory}
        for name, digest in self._previous.items():
            if name not in files:
                path = self._stale(name, digest)
                if path is not None:
                    path.unlink(missing_ok=True)
        self._record(files)
        self._previous = files

    def close(self):
        if self._stage is not None:
            self._stage.cleanup()
            self._stage = None


def _new_sibling(path):
    """Create an empty file beside ``path`` under a new random name; return its path.

    O_EXCL never opens an existing name, a link included, so a link planted
    under a predictable name cannot redirect the write. Unlike mkstemp, the
    file gets the usual permissions (0o666 less the umask).
    """
    for _attempt in range(100):
        temporary = path.with_name(f".{path.name}.{secrets.token_hex(6)}.mjworld-tmp")
        try:
            os.close(os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666))
        except FileExistsError:
            continue
        return temporary
    raise ExportError(f"Could not create a temporary file beside {path}.")


def _replace_file(path, write):
    """Write ``path`` through a new sibling temporary file, so a failure keeps the old file.

    Replacing ``path`` replaces a link there, not the file it points to.
    """
    path = Path(path)
    temporary = _new_sibling(path)
    try:
        write(temporary)
        if path.is_file() and not path.is_symlink():
            shutil.copymode(path, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def xml_bytes(root):
    """The file content of the MJCF ``root``: indented two spaces, UTF-8 with an XML declaration."""
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def _install(temporary, path):
    """Move the finished output file ``temporary`` to ``path``; replacing a link there, not its target."""
    os.replace(temporary, path)


def publish(assets, outputs):
    """Install an export: its asset files, then its output files, then the asset cleanup.

    ``outputs`` lists ``(path, data)`` pairs, the XML first, then the report and
    other text files. Every output is written to a temporary sibling before
    anything is replaced, so a failure while writing changes nothing. Then
    ``assets.commit()`` moves the staged assets into place, the outputs replace
    their files in order, and only then does ``assets.finish()`` remove the
    files the previous export used: until the new XML is in place, every file
    the old XML references still exists.
    """
    temporaries = []
    try:
        for path, data in outputs:
            path = Path(path)
            temporary = _new_sibling(path)
            temporaries.append((temporary, path))
            temporary.write_bytes(data)
            if path.is_file() and not path.is_symlink():
                shutil.copymode(path, temporary)
        assets.commit()
        for temporary, path in temporaries:
            _install(temporary, path)
        assets.finish()
    finally:
        for temporary, _path in temporaries:
            temporary.unlink(missing_ok=True)


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


def dropped_object_notes(context, candidates):
    """Notes for visible objects whose geometry the exporters cannot translate.

    Curves, surfaces, text and metaballs need converting to mesh first.
    Instancers (collection, vertex or face instancing, particles) and Geometry
    Nodes instances are not realized: only a host mesh's own geometry exports.
    The exporters call this for both ordinary and adopted worlds; it only adds
    report notes.
    """
    from .project import _included
    shown = [obj for obj in candidates if obj.visible_get(view_layer=context.view_layer) and not obj.hide_render and _included(obj)]
    notes = []
    for obj in shown:
        if obj.type in {"CURVE", "SURFACE", "FONT", "META"}:
            notes.append(f"{obj.name}: convert this {obj.type.lower()} to mesh before exporting; it was left out.")
        if obj.is_instancer:
            notes.append(f"{obj.name}: instances are not exported. Realize instances / Make Instances Real before exporting.")
    hosts = {obj for obj in shown if not obj.is_instancer}
    reported = set()
    for instance in context.evaluated_depsgraph_get().object_instances:
        if not instance.is_instance or instance.parent is None:
            continue
        host = instance.parent.original
        if host in hosts and host not in reported:
            reported.add(host)
            notes.append(f"{host.name}: its Geometry Nodes instances are not exported, only its own mesh. "
                         "Add a Realize Instances node before exporting.")
    return notes


def skip_faceless(context, objects):
    """Leave out meshes without faces that carry no physics of their own.

    A faceless Visual or Fixed mesh (a wire guide, or modifiers that output no
    faces) is dropped with a note. Movable, door and collision-helper meshes,
    and Fixed parts of a moving body, still fail in _extract_geometry, so a
    collider never disappears quietly. Returns the kept objects and the notes.
    """
    from .project import body_owner, moving_role
    depsgraph = context.evaluated_depsgraph_get()
    kept, notes = [], []
    for obj in objects:
        data = obj.evaluated_get(depsgraph).data
        if (data is None or len(data.polygons) == 0) and obj.mujoco_world.role in {"VISUAL", "STATIC"} \
                and moving_role(body_owner(obj)) is None:
            notes.append(f"{obj.name}: mesh has no faces, so it was left out. Give it faces, or set it to Leave out.")
            continue
        kept.append(obj)
    return kept, notes


def uncovered_bake_source(obj):
    """Warning for a bake source whose visual is exported because no atlas covers it."""
    return (f"{obj.name}: marked as a bake source, but no baked atlas contains it (it was copied after baking, or the "
            "atlas was baked without it), so its own visual was exported. Rebake, or use 'Edit original surfaces' to "
            "clear the mark.")


def _export_simple(context, filepath, *, regroup=False, candidates=None):
    """Write an ordinary (not adopted) world; works in UI and Blender background mode.

    With ``regroup``, body group Empties become compound bodies first. Files
    are written only once the whole world is built, so a failed export keeps
    the previous XML, report and per-XML assets. ``candidates`` lists the
    objects to export, like a selection, regardless of selection and
    visibility (hidden bake sources keep their colliders); by default the
    selection or the whole scene is used, as 'Export selected objects only'
    says.
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
    chosen = candidates is not None or settings.selected_only
    if candidates is None:
        candidates = list(context.selected_objects) if settings.selected_only else list(scene.objects)
    candidates = list(candidates)
    if chosen:
        for obj in list(candidates):
            if is_body(obj):
                candidates.extend(obj.children_recursive)
    # 'Leave out' on a body group Empty also leaves out every part it owns.
    # Leftover pieces of a failed bake are never exported.
    objects = sorted(set(o for o in candidates if o.type == "MESH" and not o.get("mjworld_bake_clone") and
                         ((o.visible_get(view_layer=context.view_layer) and not o.hide_render) or o.get("mjworld_bake_source"))
                         and _included(o)), key=lambda o: o.name)
    objects, notes = skip_faceless(context, objects)
    if not objects:
        if notes:
            raise ExportError("Nothing to export: " + " ".join(notes))
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
        root, report = _build_simple(context, filepath, objects, labels, candidates, unit, plane_mask, masks, assets, asset_prefix,
                                     notes)
        if regroup:
            report = regroup_export(context, root, report)
        guide = (f"MuJoCo 3.15+\n\nOpen with:\n  python -m mujoco.viewer --mjcf={filepath.name}\n\n"
                 f"Keep the XML and asset folder ({asset_prefix}) together. All asset paths are relative.\n"
                 "In the viewer, double-click a movable object/door, then Ctrl+right-drag to apply force.\n"
                 "Visual meshes do not collide. Collision shapes and the analytic floor are in group 3.\n"
                 "For an omniwheel robot, preserve its explicit wheel-capsule/floor contact pairs when merging.\n"
                 "This isolated world does not contain a robot or ROS controller configuration.\n")
        publish(assets, [(filepath, xml_bytes(root)),
                         (filepath.parent / "export_report.json", (json.dumps(report, indent=2) + "\n").encode("utf-8")),
                         (filepath.parent / "OPEN_WORLD.txt", guide.encode("utf-8"))])
    finally:
        assets.close()
    return report


def _build_simple(context, filepath, objects, labels, candidates, unit, plane_mask, masks, assets, asset_prefix, notes=()):
    """Build the ordinary world in memory; asset files go through ``assets``."""
    from .baking import bake_covered
    scene = context.scene
    settings = scene.mujoco_world
    warnings = list(notes)
    covered = bake_covered(scene)
    warnings.extend(dropped_object_notes(context, candidates))
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
        # A bake source's visual is replaced by the atlas, but only while an
        # atlas actually covers it.
        replaced = bool(obj.get("mjworld_bake_source")) and obj.name in covered
        if obj.get("mjworld_bake_source") and not replaced and cfg.role != "COLLIDER":
            warnings.append(uncovered_bake_source(obj))
        if cfg.role != "COLLIDER" and not replaced:
            for material_id, tri in geometry["triangles"]:
                by_material.setdefault(material_id, []).append(tri)
        total_triangles = 0
        for material_id, triangles in sorted(by_material.items()):
            mat = obj.material_slots[material_id].material if material_id < len(obj.material_slots) else None
            where = f"{obj.name} / {mat.name if mat else 'no material'}"
            info = inspect_material(mat)
            for note in info["warnings"]:
                warnings.append(f"{where}: {note}")
            uv_name = info.get("uv_map") or geometry["render_uv"]
            image = info["image"]
            problem = image_problem(image) if image else None
            if problem:
                warnings.append(f"{where}: image '{image.name}' {problem}; export used the flat material colour.")
                image = None
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
                        written = {}
                        image_file = assets.write(image_name + ".png",
                                                  lambda target: written.update(_write_image(image, target, baked=baked, scene=scene, where=where)),
                                                  content_named=True)
                        if written["hdr"]:
                            warnings.append(f"{obj.name}: HDR bake '{image.name}' converted to display PNG using {profile_text(written['profile'])}. Keep that view transform consistent when comparing Blender and MuJoCo.")
                        warnings.extend(f"{where}: {note}" for note in written["notes"])
                        ET.SubElement(asset, "texture", name=image_name, type="2d", file=image_file,
                                      colorspace=texture_colorspace(settings, written["linear"]))
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
    report = {"addon_version": addon_version(), "blender_version": bpy.app.version_string, "file": filepath.name,
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
