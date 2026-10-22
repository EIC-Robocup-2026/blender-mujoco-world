# Tests

The tests build scenes in Blender, export them with the add-on from this
checkout, and check the written worlds with MuJoCo outside Blender.

## Requirements

- Blender 4.2 or later. The suite is run on Blender 4.5.14 LTS and 5.2.1 LTS.
- A Python 3 interpreter with the packages in [requirements.txt](requirements.txt)
  (`mujoco>=3.15`, `numpy`, `Pillow`). The suite is run with Python 3.12 and
  MuJoCo 3.15.0.
- Bash, `diff` and `cmp`. With `git` available, the runner also checks that the
  checkout is unchanged.

For example, in a virtual environment:

```bash
python3 -m venv /tmp/mujoco-world-venv
/tmp/mujoco-world-venv/bin/pip install -r tests/requirements.txt
```

## Run everything

```bash
BLENDER=/path/to/blender PYTHON=/tmp/mujoco-world-venv/bin/python tests/run_all.sh [OUTPUT_DIR]
```

| Variable | Required | Meaning |
| --- | --- | --- |
| `BLENDER` | no | Blender executable. Default: `blender` on `PATH`. |
| `PYTHON` | no | Python with the packages above. Default: `python3`, if it has them. |
| `WALKIE_ROS_ROOT` | no | A walkie-ros checkout. The smoke and export-fix worlds are then merged with the Walkie robot through a copy of its `robot_bringup.world_loader`, the consumer of these exports. The checkout is only read. |
| `MJWORLD_ROOM_ROOT` | no | An existing room project, adopted and exported by the room tests below. The tests work on a copy inside `OUTPUT_DIR` and check that the copy stays unchanged. |

`OUTPUT_DIR` must be a new or empty directory outside the checkout; without it,
the runner creates one under `$TMPDIR` (or `/tmp`). Each step prints `PASS`,
`FAIL` or `SKIP` with its duration and writes its log to
`OUTPUT_DIR/logs/STEP.log`. Every step runs even after a failure. The exit
status is 0 when every step passed, 1 when a step failed, and 2 when the setup
is invalid (missing Blender, Python packages, or an unusable `OUTPUT_DIR`).

`OUTPUT_DIR` contains one folder per Blender test, `logs/`, `dist/` (the two
reproducibility builds of the ZIP), `install/` and `install_sandbox/` (the
installation check), `blender_user/` (Blender's isolated user directory) and,
with `MJWORLD_ROOM_ROOT`, `room_project/`.

## Steps

Each Blender test writes its worlds to its own folder; the validator after it
reads that folder.

| Step | Blender test | Validator | Covers |
| --- | --- | --- | --- |
| `smoke` | `build_smoke_scene.py` | `validate_export.py` | Transforms, roles, textures, preserved weight, selection-only export, portability, the door's torque-driven movement, the floor plane options, the export operator writing the same world as the export function, and the default export path. With `WALKIE_ROS_ROOT`: merged with the robot through `assemble_world`, the merged world compiles and simulates, robot and world keep their masses and inertias, and the loader's collision classes equal the masks of the Walkie preset. |
| `bake` | `test_baking_blender.py` | `validate_compile.py bake` | Static bake with separate colliders, HDR master, export of the atlas, restore of the sources. |
| `compound` | `test_compound_blender.py` | `validate_compound.py` | Hollow compound bodies, nested hinges, retained imported physics and its live edits. |
| `export_fixes` | `test_export_fixes_blender.py` | `validate_export_fixes.py` | Radian angles, inertia settings that merge cleanly with a robot, contact mask formats, body name clashes, excluded body groups, adopted `fromto` primitives. With `WALKIE_ROS_ROOT`: the door world merged with the robot. |
| `adopt` | `test_adopt_blender.py` | `validate_adopt.py` | All-or-nothing adoption, carried mesh inertia, duplicated adopted objects, in-place export, viewport-disabled objects, Weight on adopted bodies, the hinge cursor on shared meshes, failed exports, shared asset folders (also two adopted projects in one folder, without external data), door contact masks. |
| `review_fixes` | `test_review_fixes_blender.py` | `validate_review_fixes.py` | Metal and emission in the bake, the pinned lightmap profile, stale bake marks, bake failure cleanup, restore edge cases, faceless meshes, clone shading inputs, the bake's early checks, render-active UV maps, 16-bit/float/UDIM/missing/Non-Color images, both texture colour modes, adopted texture colour spaces, the operators' linked-data, restore and error handling. |
| `followup` | `test_followup_fixes_blender.py` | `validate_followup_fixes.py` | Stretched capsules, rejected `<frame>` and `<replicate>`, slide, ball and untyped joints, class-typed geoms, contact classes on moved geoms, body and geom maps, colliding and non-ASCII mesh names, origin offsets, notes for curves and instances, volume-weighted mass, a moved blend file, stale-file cleanup, asset directories with cube textures and height fields, rebake names, repeated registration. |
| `room_compound` | `test_compound_blender.py` with `MJCF BLEND` | `validate_compound.py` | Adoption and edits of the existing room project, exported into a shared asset folder inside the output. |
| `shared_assets` | `test_shared_assets_blender.py` | `validate_compile.py shared_assets` | Both variants of the room project exported twice into one shared asset folder. |
| `room_project_unchanged` | | `diff -r` | The room tests left the project copy unchanged. |
| `package_reproducible` | | `package_addon.py`, `cmp` | Two builds of the ZIP are byte-identical. |
| `install_check` | `install_check.py` | | The ZIP installs into a sandboxed user directory, registers its operators, panel and File > Export entry, exports a world, and disables and re-enables cleanly. |
| `source_tree_unchanged` | | `git status --ignored` | No step wrote into the checkout, ignored files included. |

The room tests expect the layout of the project they were written for:
`scene.xml` and `chula_pat14_6.blend` at the top, `realistic/scene.xml` and
`realistic/realistic.blend` below, and an object named `task_apple_1_00`.
Without `MJWORLD_ROOM_ROOT` they are skipped.

## Run one test

Every script's docstring shows its command. For example:

```bash
blender --background --factory-startup --python-exit-code 1 \
    --python tests/test_adopt_blender.py -- /tmp/mujoco-world-adopt
python tests/validate_adopt.py /tmp/mujoco-world-adopt
```

## Conventions for new tests

- Run Blender with `--background --factory-startup --python-exit-code 1`.
  `--factory-startup` ignores the user's preferences, startup file and
  installed add-ons. Without `--python-exit-code 1`, Blender exits with status 0
  after an exception in the script.
- A Blender test calls `harness.begin(NAME)` first and `harness.complete(OUT, NAME)`
  last. `begin` checks the arguments, clears the output folder, and imports the
  add-on from this checkout, failing if it resolves to an installed copy.
  DeprecationWarnings raised by the add-on are errors.
- `begin` clears a folder only when it is empty or holds earlier test output
  (marked by `.mujoco_world_test_output`), so a mistyped path never deletes
  unrelated files.
- `complete` writes `run_complete.json` with the test name, the Blender version
  and the runner's `MJWORLD_RUN_TOKEN`. A validator calls
  `harness.completed_output(NAME)`, which refuses output without that marker, from
  another test, or from an earlier run, so stale output never passes.
- Write only below the output folder. Blender ignores `PYTHONDONTWRITEBYTECODE`,
  so Blender tests set `sys.dont_write_bytecode = True` before importing
  `harness`; the runner starts validators with `python -I -B`.
- Add the step to `run_all.sh` with `test_pair`, and its row to the table above.
