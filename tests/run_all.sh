#!/usr/bin/env bash
# Run every Blender test of the add-on and its MuJoCo validator.
#
# Usage:
#   tests/run_all.sh [OUTPUT_DIR]
#
# OUTPUT_DIR must be new or empty; by default a temporary directory is
# created. Each step writes its log to OUTPUT_DIR/logs/STEP.log. Every step
# runs even after a failure; the script exits with 1 when any step failed.
#
# Environment:
#   BLENDER            Blender executable (default: blender on PATH).
#   PYTHON             Python with mujoco, numpy and Pillow, see
#                      tests/requirements.txt (default: python3 if it has them).
#   WALKIE_ROS_ROOT    Optional walkie-ros checkout. The smoke and export-fix
#                      worlds are then merged with the Walkie robot through a
#                      copy of its world loader. The checkout is only read.
#                      When its loader cannot be used, the walkie_* steps fail.
#   MJWORLD_ALLOW_WALKIE_SKIP
#                      Set to 1 to report an unusable loader as SKIP instead.
#   MJWORLD_ROOM_ROOT  Optional existing room project (scene.xml,
#                      chula_pat14_6.blend, and baked/scene.xml with
#                      baked/baked.blend or realistic/scene.xml with
#                      realistic/realistic.blend), such as
#                      simulation/mujoco/worlds/chula_pat14_6 in walkie-ros.
#                      The tests use a copy in OUTPUT_DIR and check that the
#                      copy is left unchanged.
set -euo pipefail

TESTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$TESTS")"

fail_setup() {
    echo "run_all.sh: $*" >&2
    exit 2
}

BLENDER="${BLENDER:-$(command -v blender || true)}"
[[ -n "$BLENDER" ]] || fail_setup "no blender on PATH; set BLENDER to the Blender executable"
[[ -x "$BLENDER" ]] || fail_setup "BLENDER=$BLENDER is not an executable file"
if [[ -z "${PYTHON:-}" ]]; then
    PYTHON="$(command -v python3 || true)"
    [[ -n "$PYTHON" ]] && "$PYTHON" -I -c "import mujoco, numpy, PIL" 2>/dev/null \
        || fail_setup "python3 lacks mujoco, numpy or Pillow; set PYTHON (see tests/requirements.txt)"
fi
"$PYTHON" -I -c "import mujoco, numpy, PIL" 2>/dev/null \
    || fail_setup "PYTHON=$PYTHON cannot import mujoco, numpy and Pillow (see tests/requirements.txt)"
WALKIE_ROS_ROOT="${WALKIE_ROS_ROOT:-}"
MJWORLD_ROOM_ROOT="${MJWORLD_ROOM_ROOT:-}"
[[ -z "$WALKIE_ROS_ROOT" || -d "$WALKIE_ROS_ROOT" ]] || fail_setup "WALKIE_ROS_ROOT=$WALKIE_ROS_ROOT is not a directory"
[[ -z "$MJWORLD_ROOM_ROOT" || -d "$MJWORLD_ROOM_ROOT" ]] || fail_setup "MJWORLD_ROOM_ROOT=$MJWORLD_ROOM_ROOT is not a directory"

if [[ $# -gt 1 ]]; then
    fail_setup "usage: tests/run_all.sh [OUTPUT_DIR]"
elif [[ $# -eq 1 ]]; then
    OUT="$1"
    if [[ -e "$OUT" ]]; then
        [[ -d "$OUT" && -z "$(ls -A "$OUT")" ]] || fail_setup "$OUT exists and is not an empty directory"
    fi
    mkdir -p "$OUT"
    OUT="$(cd "$OUT" && pwd -P)"
    case "$OUT/" in
        "$(cd "$ROOT" && pwd -P)/"*) fail_setup "OUTPUT_DIR must be outside the checkout $ROOT" ;;
    esac
else
    OUT="$(mktemp -d "${TMPDIR:-/tmp}/mujoco-world-tests.XXXXXX")"
fi
mkdir -p "$OUT/logs"

# Validators accept only output carrying this run's token, so output left by
# an earlier run never passes as the result of this one.
MJWORLD_RUN_TOKEN="$(date +%s)-$$-$RANDOM"
export MJWORLD_RUN_TOKEN
# Blender reads no user preferences, startup file or installed add-on copy.
export BLENDER_USER_RESOURCES="$OUT/blender_user"

PASSED=()
FAILED=()
SKIPPED=()

# run_step NAME COMMAND...: run COMMAND with its output in logs/NAME.log.
run_step() {
    local name=$1
    shift
    local start=$SECONDS status
    if "$@" >"$OUT/logs/$name.log" 2>&1; then
        status=PASS
        PASSED+=("$name")
    else
        status=FAIL
        FAILED+=("$name")
    fi
    printf '%-26s %s %5ss\n' "$name" "$status" "$((SECONDS - start))"
    [[ $status == PASS ]]
}

skip_step() {
    printf '%-26s SKIP  (%s)\n' "$1" "$2"
    SKIPPED+=("$1: $2")
}

blender_test() {
    local script=$1
    shift
    "$BLENDER" --background --factory-startup --python-exit-code 1 --python "$TESTS/$script" -- "$@"
}

validator() {
    local script=$1
    shift
    "$PYTHON" -I -B "$TESTS/$script" "$@"
}

# test_pair NAME BLENDER_SCRIPT OUTPUT_DIR BLENDER_ARGS... -- VALIDATOR ARGS...
test_pair() {
    local name=$1 script=$2 out=$3
    shift 3
    local blender_args=()
    while [[ $# -gt 0 && $1 != "--" ]]; do
        blender_args+=("$1")
        shift
    done
    shift
    if run_step "$name" blender_test "$script" "$out" ${blender_args[@]+"${blender_args[@]}"}; then
        run_step "validate_$name" validator "$@" || true
    else
        skip_step "validate_$name" "Blender step failed"
    fi
}

passed() {
    local step
    for step in ${PASSED[@]+"${PASSED[@]}"}; do
        [[ $step == "$1" ]] && return 0
    done
    return 1
}

# walkie_check NAME MARKER: the Walkie merge inside validate_NAME. A validator
# prints MARKER and passes when it could not use the walkie-ros world loader;
# with WALKIE_ROS_ROOT set, that is a failure unless MJWORLD_ALLOW_WALKIE_SKIP=1.
walkie_check() {
    local name=walkie_$1 marker=$2 log="$OUT/logs/validate_$1.log" reason
    if [[ -z "$WALKIE_ROS_ROOT" ]]; then
        skip_step "$name" "WALKIE_ROS_ROOT not set"
    elif ! passed "validate_$1"; then
        skip_step "$name" "validate_$1 did not pass"
    elif ! grep -q "^$marker" "$log"; then
        printf '%-26s PASS\n' "$name"
        PASSED+=("$name")
    else
        reason="$(grep -m 1 "^$marker" "$log" | cut -d ' ' -f 2-)"
        if [[ "${MJWORLD_ALLOW_WALKIE_SKIP:-}" == 1 ]]; then
            skip_step "$name" "$reason"
        else
            printf '%-26s FAIL  (%s; set MJWORLD_ALLOW_WALKIE_SKIP=1 to allow)\n' "$name" "$reason"
            FAILED+=("$name")
        fi
    fi
}

tree_state() {
    git -C "$ROOT" status --porcelain --ignored --untracked-files=all
}

package_reproducible() {
    # Two builds of the same sources must be byte-identical, and hold exactly
    # the add-on's files with fixed metadata.
    "$PYTHON" -I -B "$ROOT/package_addon.py" --output "$OUT/dist/first" \
        && "$PYTHON" -I -B "$ROOT/package_addon.py" --output "$OUT/dist/second" \
        && cmp "$OUT"/dist/first/mujoco_world-*.zip "$OUT"/dist/second/mujoco_world-*.zip \
        && "$PYTHON" -I -B "$TESTS/check_package.py" "$OUT"/dist/first/mujoco_world-*.zip
}

install_check() {
    BLENDER_USER_RESOURCES="$OUT/install_sandbox" "$BLENDER" --background --factory-startup \
        --python-exit-code 1 --python "$TESTS/install_check.py" -- "$OUT/install" "$OUT"/dist/first/mujoco_world-*.zip
}

walkie_args=()
[[ -n "$WALKIE_ROS_ROOT" ]] && walkie_args=("$WALKIE_ROS_ROOT")

echo "Blender: $("$BLENDER" --version 2>/dev/null | head -n 1)"
echo "Python:  $("$PYTHON" -I -c 'import sys, mujoco; print(sys.version.split()[0], "mujoco", mujoco.__version__)')"
echo "Output:  $OUT"
echo
# The tests write only to OUTPUT_DIR: the checkout, including ignored files
# such as bytecode caches, must look the same afterwards.
check_tree=false
if git -C "$ROOT" rev-parse --git-dir >/dev/null 2>&1; then
    check_tree=true
    before="$(tree_state)"
fi

test_pair smoke build_smoke_scene.py "$OUT/smoke" -- \
    validate_export.py "$OUT/smoke" ${walkie_args[@]+"${walkie_args[@]}"}
walkie_check smoke WALKIE_CONSUMER_PARITY_SKIPPED
test_pair bake test_baking_blender.py "$OUT/bake" -- validate_compile.py bake "$OUT/bake"
test_pair compound test_compound_blender.py "$OUT/compound" -- validate_compound.py "$OUT/compound"
test_pair export_fixes test_export_fixes_blender.py "$OUT/export_fixes" -- \
    validate_export_fixes.py "$OUT/export_fixes" ${walkie_args[@]+"${walkie_args[@]}"}
walkie_check export_fixes WALKIE_MERGE_SKIPPED
test_pair adopt test_adopt_blender.py "$OUT/adopt" -- validate_adopt.py "$OUT/adopt"
test_pair review_fixes test_review_fixes_blender.py "$OUT/review_fixes" -- \
    validate_review_fixes.py "$OUT/review_fixes"
test_pair followup test_followup_fixes_blender.py "$OUT/followup" -- \
    validate_followup_fixes.py "$OUT/followup"

if [[ -n "$MJWORLD_ROOM_ROOT" ]]; then
    room="$OUT/room_project"
    cp -R "$MJWORLD_ROOM_ROOT" "$room"
    chmod -R u+w "$room"
    test_pair room_compound test_compound_blender.py "$OUT/room_compound" \
        "$room/scene.xml" "$room/chula_pat14_6.blend" -- validate_compound.py "$OUT/room_compound"
    test_pair shared_assets test_shared_assets_blender.py "$OUT/shared_assets" "$room" -- \
        validate_compile.py shared_assets "$OUT/shared_assets"
    run_step room_project_unchanged diff -r "$MJWORLD_ROOM_ROOT" "$room" || true
else
    skip_step room_compound "MJWORLD_ROOM_ROOT not set"
    skip_step shared_assets "MJWORLD_ROOM_ROOT not set"
fi

if run_step package_reproducible package_reproducible; then
    run_step install_check install_check || true
else
    skip_step install_check "packaging failed"
fi

if $check_tree; then
    after="$(tree_state)"
    if [[ "$before" == "$after" ]]; then
        printf '%-26s PASS\n' source_tree_unchanged
        PASSED+=(source_tree_unchanged)
    else
        diff <(echo "$before") <(echo "$after") >"$OUT/logs/source_tree_unchanged.log" || true
        printf '%-26s FAIL  (see logs/source_tree_unchanged.log)\n' source_tree_unchanged
        FAILED+=(source_tree_unchanged)
    fi
fi

echo
echo "${#PASSED[@]} passed, ${#FAILED[@]} failed, ${#SKIPPED[@]} skipped. Output and logs: $OUT"
if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "Failed: ${FAILED[*]}"
    exit 1
fi
