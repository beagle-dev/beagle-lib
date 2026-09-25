# Sourced by the harness scripts (see README.md). Any of these can be overridden from the environment.
TG_TESTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU_DIR="$(cd "$TG_TESTS/.." && pwd)"
REPO="$(cd "$GPU_DIR/../.." && pwd)"
: "${BEAGLE_BUILD:=$REPO/build}"
: "${TINYGRAD_PATH:=$HOME/Dropbox/Projects/tinygrad-hcq1}"
: "${BEAGLE_PYTHON:=$HOME/Dropbox/Projects/tinygrad/venv/bin/python}"
: "${BEAGLE_TINYGPU_DATA:=$HOME/.beagle/tinygpu}"
: "${TINYGPU_TEST_WORK:=$TG_TESTS/.work}"
export TINYGRAD_PATH BEAGLE_PYTHON BEAGLE_TINYGPU_DATA TINYGPU_TEST_WORK
mkdir -p "$TINYGPU_TEST_WORK"
TEST_BIN="$BEAGLE_BUILD/examples/tinygpuhybridtest"
TEST_LIBS="$BEAGLE_BUILD/libhmsbeagle/GPU/CMake_TinyGPUHybrid:$BEAGLE_BUILD/libhmsbeagle/CPU:$BEAGLE_BUILD/libhmsbeagle"

# Offline scripts call this first: the plugin they load must contain the BEAGLE_TINYGPU_NO_LAUNCH guard, or a failed
# connection to a fake would start the real TinyGPU.app. (A static check: it runs nothing.)
require_no_launch_guard() {
    local so
    for so in "$BEAGLE_BUILD"/libhmsbeagle/GPU/CMake_TinyGPUHybrid/libhmsbeagle-tinygpu-hybrid*.so; do
        [ -f "$so" ] || { echo "no TinyGPU plugin under $BEAGLE_BUILD; build hmsbeagle-tinygpu-hybrid first"; exit 2; }
        grep -aq "BEAGLE_TINYGPU_NO_LAUNCH is set; not starting TinyGPU.app" "$so" || {
            echo "$so lacks the no-launch guard; rebuild hmsbeagle-tinygpu-hybrid before running offline tests"; exit 2; }
    done
}

# run_point.sh's stop rule (TODO.md plan step P3), on the plugin's output (GPUInterfaceTinyGPUHybridNV.cpp nv_report_unload):
# exactly one fini report, WPR2_HI 0 in it, and the teardown line saying the next boot needs no power cycle. Reads a file
# and runs nothing, so run_offline.sh checks it on the fakes' output.
fini_verdict() {
    [ "$(grep -c "TinyGPU/NV: GPU teardown: " "$1")" -eq 1 ] && grep -q "TinyGPU/NV: GPU teardown: .*WPR2_HI=0x00000000)" "$1" \
        && grep -q "TinyGPU/NV: teardown: .*the next boot needs no power cycle" "$1"
}
