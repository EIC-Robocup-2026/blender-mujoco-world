"""Apply supported Blender material edits without changing untouched imports."""

import hashlib
import json
import xml.etree.ElementTree as ET


def _description(obj):
    from .exporter import inspect_material, material_uses_nodes
    material = obj.material_slots[0].material if obj.material_slots else None
    info = inspect_material(material)
    image = info["image"]
    snapshot = {"material": material.name if material else None,
                "color": info["color"], "emission": info["emission"],
                "image": image.name if image else None,
                "image_path": image.filepath if image else None,
                "image_size": list(image.size) if image else None,
                "uv_map": info.get("uv_map"),
                "baked": bool(obj.mujoco_world.baked)}
    if material_uses_nodes(material):
        shader = next((node for node in material.node_tree.nodes
                       if node.type == "BSDF_PRINCIPLED"), None)
        if shader:
            for field in ("Roughness", "Metallic"):
                snapshot[field] = float(shader.inputs[field].default_value)
    return info, snapshot


def capture_material_baselines(scene, root):
    """Record supported appearance after adoption; existing XML remains exact."""
    for obj in scene.objects:
        raw = obj.get("mjworld_mjcf_geom")
        if obj.type != "MESH" or not raw:
            continue
        _info, snapshot = _description(obj)
        obj["mjworld_material_baseline"] = json.dumps(snapshot, sort_keys=True)


def update_authored_materials(scene, root, assets, asset_prefix, geoms):
    """Export changed simple colours/images per object, retaining source physics.

    ``geoms`` maps each exported adopted object to its geom element, and
    ``assets`` is the export's AssetWriter. Complex shader changes still need
    the static bake. A reassigned material gets a new material reference on
    that object rather than changing every object that used the imported
    material.
    """
    from .baking import bake_covered
    from .exporter import _write_image, image_problem, numbers, profile_text, property_numbers, slug, srgb, texture_colorspace
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    warnings = []
    exported = {}
    covered = bake_covered(scene)
    for obj in scene.objects:
        raw = obj.get("mjworld_mjcf_geom")
        previous = obj.get("mjworld_material_baseline")
        if obj.type != "MESH" or not raw or previous is None:
            continue
        geom = geoms.get(obj)
        # A bake source keeps its baseline appearance while an atlas replaces it.
        if geom is None or obj.mujoco_world.role == "COLLIDER" or (obj.get("mjworld_bake_source") and obj.name in covered):
            continue
        info, snapshot = _description(obj)
        current = json.dumps(snapshot, sort_keys=True)
        image = info["image"]
        if current == previous and not (image and image.is_dirty):
            continue
        warnings.extend(f"{obj.name}: {warning}" for warning in info["warnings"])
        if len(obj.material_slots) > 1:
            warnings.append(f"{obj.name}: the adopted primitive uses its first material. Bake static multi-material appearance, or use separate visual parts.")
        key = current
        if key not in exported:
            label = "authored_" + slug(snapshot["material"] or "neutral")[:40]
            label += "_" + hashlib.sha256(key.encode()).hexdigest()[:12]
            rgba = list(info["color"])
            if scene.mujoco_world.texture_mode == "DISPLAY":
                rgba[:3] = [srgb(value) for value in rgba[:3]]
            baked = bool(snapshot["baked"] or info["emission"])
            attributes = {"name": label, "rgba": numbers(rgba), "specular": "0",
                          "emission": property_numbers([scene.mujoco_world.baked_emission if baked else 0])}
            problem = image_problem(image) if image else None
            if problem:
                warnings.append(f"{obj.name}: image '{image.name}' {problem}; export used the flat material colour.")
                image = None
            if image:
                written = {}
                filename = assets.write(label + ".png", lambda target: written.update(
                    _write_image(image, target, baked=baked, scene=scene, where=obj.name)))
                if written["hdr"]:
                    warnings.append(f"{obj.name}: HDR bake '{image.name}' converted to display PNG using "
                                    f"{profile_text(written['profile'])}. Keep that view transform consistent when "
                                    "comparing Blender and MuJoCo.")
                warnings.extend(f"{obj.name}: {note}" for note in written["notes"])
                ET.SubElement(asset, "texture", name=label + "_texture", type="2d",
                              file=asset_prefix + "/" + filename,
                              colorspace=texture_colorspace(scene.mujoco_world, written["linear"]))
                attributes.update(texture=label + "_texture", rgba=f"1 1 1 {rgba[3]:.9g}",
                                  texuniform="false")
            ET.SubElement(asset, "material", **attributes)
            exported[key] = label
        geom.set("material", exported[key])
    return warnings
