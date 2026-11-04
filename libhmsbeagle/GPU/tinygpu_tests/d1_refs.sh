#!/bin/bash
# TODO.md plan step D1: the references d1_compare.py checks each hardware run against, made offline before the session (the
# output is deterministic), under $BEAGLE_TINYGPU_DATA/d1/refs/. For every d1_runs.txt line: synthetictest with --rsrc 0 in
# single precision (<label>.sp.out) and double precision without SSE (<label>.dp.out; CPU-SSE-Double prints logL nan for
# 3-state partitions); hmctest --tinygpu on the CPU (<label>.sp.out)
# and on the Mac's own GPU through the OpenCL plugin (<label>.gpuref.out; see d1_compare.py). No TinyGPU plugin is on the
# library path, and dlopen searches the cwd, so this runs from the references directory: the plugin, whose constructor
# talks to TinyGPU.app, cannot load (a reference that mentions TinyGPU is refused anyway). A reference with a non-finite
# value or an 'error:' line is refused. <label>.cmd records the line each reference set was made for (run_d1.sh checks it).
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
REFS="$BEAGLE_TINYGPU_DATA/d1/refs"; mkdir -p "$REFS"; cd "$REFS" || exit 2
ref() {   # <name> <extra library dir or ""> <program> <arguments ...>
    local name=$1 libs=$2 prog=$3; shift 3
    env DYLD_LIBRARY_PATH="$libs$BEAGLE_BUILD/libhmsbeagle/CPU:$BEAGLE_BUILD/libhmsbeagle" BEAGLE_TINYGPU_NO_LAUNCH=1 \
        APL_REMOTE_SOCK=/nonexistent/tinygpu.sock "$BEAGLE_BUILD/examples/$prog" "$@" > "$name.out" 2> "$name.err" < /dev/null
    [ $? -eq 0 ] && grep -q "Impl Name" "$name.out" && ! grep -q TinyGPU "$name.out" "$name.err" \
        && ! grep -qiE '(^|[^a-z])-?(nan|inf)([^a-z]|$)|error:' "$name.out" || { echo "FAIL: $REFS/$name.out"; exit 1; }
    echo "$REFS/$name.out: $(grep "Impl Name" "$name.out")"
}
while IFS='|' read -r label cmd kernels; do
    set -- $cmd; prog=$1; shift; args="$*"
    case $prog in
        synthetictest) ref $label.sp "" $prog ${args/--rsrc 1/--rsrc 0}
                       ref $label.dp "" $prog ${args/--rsrc 1/--rsrc 0} --doubleprecision --disablevector ;;
        hmctest) ref $label.sp "" $prog ${args/--gpu 1 /}
                 ref $label.gpuref "$BEAGLE_BUILD/libhmsbeagle/GPU/CMake_OpenCL:" $prog $args
                 grep -q "Impl Name : OpenCL-Single" $label.gpuref.out || { echo "FAIL: $REFS/$label.gpuref.out is not OpenCL"; exit 1; } ;;
    esac
    echo "$cmd" > $label.cmd
done < <(grep -E '^[a-z0-9_]+\|' "$TG_TESTS/d1_runs.txt")
