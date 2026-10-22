"""Compile and briefly simulate every MuJoCo world a Blender test wrote.

Run after a Blender test that has no dedicated validator:
  python tests/validate_compile.py TEST OUTPUT_DIR

TEST is the name the Blender test records in its run marker, for example
``bake`` (test_baking_blender.py) or ``shared_assets``
(test_shared_assets_blender.py). Every ``.xml`` file below OUTPUT_DIR with a
``<mujoco>`` root must compile, step 200 times without warnings, and keep
finite positions and velocities.
"""
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness  # noqa: E402

if len(sys.argv) < 2:
    raise SystemExit("usage: python tests/validate_compile.py TEST OUTPUT_DIR")
TEST = sys.argv.pop(1)
OUT, _, _ = harness.completed_output(TEST, "OUTPUT_DIR")

results = {}
for path in sorted(OUT.rglob("*.xml")):
    try:
        if ET.parse(path).getroot().tag != "mujoco":
            continue
    except ET.ParseError:
        continue
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    for _ in range(200):
        mujoco.mj_step(model, data)
    assert np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all(), path
    warnings = [int(data.warning[i].number) for i in range(len(data.warning))]
    assert not any(warnings), (path, warnings)
    results[str(path.relative_to(OUT))] = {"nq": model.nq, "ngeom": model.ngeom, "nmesh": model.nmesh,
                                           "ntex": model.ntex}
assert results, f"no MuJoCo world below {OUT}"
print("COMPILE_OK", json.dumps({"test": TEST, "worlds": results}, indent=2))
