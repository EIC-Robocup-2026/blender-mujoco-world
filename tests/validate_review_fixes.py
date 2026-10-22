"""Compile the worlds written by test_review_fixes_blender.py and check their textures.

Run after the Blender fixture:
  python tests/validate_review_fixes.py OUTPUT_DIR
"""
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

if len(sys.argv) != 2:
    raise SystemExit("usage: validate_review_fixes.py OUTPUT_DIR")
OUT = Path(sys.argv[1]).resolve()
summary = json.loads((OUT / "review_fixes.json").read_text())
assert not summary["failures"], f"the Blender fixture failed: {sorted(summary['failures'])}"
LINEAR, SRGB = int(mujoco.mjtColorSpace.mjCOLORSPACE_LINEAR), int(mujoco.mjtColorSpace.mjCOLORSPACE_SRGB)
models = {name: mujoco.MjModel.from_xml_path(path) for name, path in summary["worlds"].items()}
results = {"compiled": len(models)}


def texture_rgb(model, geom_prefix):
    """First texel and colourspace of the texture on the visual geom of a body."""
    for geom in range(model.ngeom):
        body = model.body(model.geom_bodyid[geom]).name
        if body.startswith(geom_prefix) and model.geom_group[geom] == 1:
            texture = model.mat_texid[model.geom_matid[geom]][int(mujoco.mjtTextureRole.mjTEXROLE_RGB)]
            assert texture >= 0, geom_prefix
            start = model.tex_adr[texture]
            channels = model.tex_nchannel[texture]
            return tuple(int(v) for v in model.tex_data[start:start + 3]), int(model.tex_colorspace[texture]), channels
    raise AssertionError(f"no visual geom for {geom_prefix}")


# MuJoCo reads the copied 16-bit PNG as 8-bit texels.
deep, _space, _channels = texture_rgb(models["texture_images"], "A_deep_png")
assert deep[0] == 0 and abs(deep[1] - 117) <= 1, deep
display = models["byte_display"]
data_rgb, data_space, _ = texture_rgb(display, "A_data_albedo")
assert data_space == LINEAR and np.allclose(data_rgb, (188, 137, 99), atol=1), (data_rgb, data_space)
assert texture_rgb(display, "B_srgb_albedo")[:2] == ((128, 64, 32), LINEAR)
srgb_mode = models["byte_srgb"]
assert texture_rgb(srgb_mode, "A_data_albedo")[:2] == ((128, 64, 32), LINEAR)
assert texture_rgb(srgb_mode, "B_srgb_albedo")[:2] == ((128, 64, 32), SRGB)
for mode, space in (("adopted_display", LINEAR), ("adopted_srgb", SRGB)):
    model = models[mode]
    spaces = {model.texture(i).name: int(model.tex_colorspace[i]) for i in range(model.ntex)}
    assert spaces and all(value == space for value in spaces.values()), (mode, spaces)
    results[mode] = spaces
results["byte_display_data_rgb"] = data_rgb
print("REVIEW_FIXES_PHYSICS_OK", json.dumps(results))
