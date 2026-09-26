# Sourced by the harness scripts (see README.md). Any of these can be overridden from the environment.
TG_TESTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU_DIR="$(cd "$TG_TESTS/.." && pwd)"
REPO="$(cd "$GPU_DIR/../.." && pwd)"
: "${BEAGLE_BUILD:=$REPO/build}"
: "${TINYGRAD_PATH:=$HOME/Dropbox/Projects/tinygrad-hcq1}"
: "${BEAGLE_PYTHON:=$HOME/Dropbox/Projects/tinygrad/venv/bin/python}"
: "${BEAGLE_TINYGPU_DATA:=$HOME/.beagle/tinygpu}"
: "${TINYGPU_TEST_WORK:=$HOME/Library/Caches/beagle-tinygpu-tests}"   # per computer: the repo may be a synced folder
export TINYGRAD_PATH BEAGLE_PYTHON BEAGLE_TINYGPU_DATA TINYGPU_TEST_WORK
mkdir -p "$TINYGPU_TEST_WORK"
TEST_BIN="$BEAGLE_BUILD/examples/tinygpuhybridtest"
TEST_LIBS="$BEAGLE_BUILD/libhmsbeagle/GPU/CMake_TinyGPUHybrid:$BEAGLE_BUILD/libhmsbeagle/CPU:$BEAGLE_BUILD/libhmsbeagle"
# $BEAGLE_TINYGPU_DATA may be shared between computers (a synced folder), each with its own eGPU: the hardware scripts tag
# their runs/ files with this computer's name and keep their lock on this computer
HW_HOST=$(scutil --get LocalHostName 2>/dev/null || hostname -s)
# plan step V1's L0 recordings (STATUS.md R32): the C++ runtime's cold boot, a warm boot and a warm boot at 64 states on the
# RTX 4060, each with its run and teardown, in $BEAGLE_TINYGPU_DATA/recordings (they hold NVIDIA firmware: never in git)
TG_L0="20260925-204611_mittag-leffler_cold 20260925-204652_mittag-leffler_warm 20260925-204737_mittag-leffler_warm64"

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

# TODO.md plan step D1: a run of a d1_runs.txt line means something only if it booted once, used the TinyGPU resource in the
# C++ runtime with an embedded cubin, had no launch rejected, and launched exactly the line's kernels (the plugin's
# BEAGLE_NV_PROFILE report at fini). Reads files and runs nothing, so run_offline.sh checks it on the fakes' output (where
# stdout and stderr share one file). Prints the first failed check.
d1_verdict() {   # <stdout file> <stderr file> "<kernels, sorted>"
    local got; got=$(sed -nE 's/^TinyGPU\/NV: \[profile\]   kernel ([A-Za-z0-9_]+) n=.*/\1/p' "$2" | xargs)
    [ "$(grep -c "TinyGPU/NV: daemon booted" "$2")" -eq 1 ] || { echo "not exactly one boot"; return 1; }
    grep -q "Rsrc Name : TinyGPU-NV-Hybrid" "$1" || { echo "not the TinyGPU resource"; return 1; }
    grep -q "TinyGPU/NV: C++ runtime: embedded cubin SP_" "$2" || { echo "not the C++ runtime with an embedded cubin"; return 1; }
    ! grep -qE "not launched|TinyGPU/NV: .*failed" "$2" || { echo "a launch was rejected or a step failed"; return 1; }
    [ "$got" = "$3" ] || { echo "launched: $got"; return 1; }
}

# The hardware scripts' shared protections (run_point.sh, run_d1.sh; plan steps P3, D1). hw_begin: nothing else changes what
# the plugin or the daemon does to the GPU, and one hardware script runs at a time on this computer (the lock is removed at
# exit; it is in $TMPDIR, next to tinygrad's nv_usb4.lock, so a computer sharing $BEAGLE_TINYGPU_DATA runs its own eGPU freely).
hw_begin() {
    local v
    # also BEAGLE's dispatch and compile knobs, and tinygrad's that change a boot: PMA/PROFILE/VIZ start its profiler setup (a
    # Blackwell branch that never ran on this card), DISABLE_HTTP_CACHE re-downloads the firmware inside the boot, REMOTE swaps
    # the device list, HCQDEV_WAIT_TIMEOUT_MS the timeline timeout, GMMU every GPU mapping (plan step B1)
    for v in BEAGLE_NV_TEARDOWN BEAGLE_NV_DATA_MB BEAGLE_NV_DISPATCH_DAEMON APL_REMOTE_SOCK BEAGLE_TINYGPU_NO_LAUNCH FAKE_NV_MEM FAKE_TEST_BIN \
             BEAGLE_NV_FILL_LAUNCH_DIMS BEAGLE_NV_CHAIN_LAUNCHES BEAGLE_NV_USE_NVJITLINK PTXAS HCQDEV_WAIT_TIMEOUT_MS \
             DISABLE_HTTP_CACHE PMA PROFILE VIZ REMOTE GMMU \
             BEAGLE_TG_OFFLINE BEAGLE_TG_MUTATE BEAGLE_TG_RECORD BEAGLE_TG_RECORD_LOG BEAGLE_TG_MARKERS BEAGLE_TG_DAEMON_PIDFILE; do   # plan V1's harness
        [ -n "${!v+x}" ] && { echo "$v is set; unset it first; not running"; exit 2; }
    done
    # the firmware is staged, so no boot downloads inside the daemon (decision 5; macOS may purge tinygrad's cache): offline,
    # re-staging from $BEAGLE_TINYGPU_DATA/fw if needed
    "$BEAGLE_PYTHON" "$TG_TESTS/check_firmware.py" > "$TINYGPU_TEST_WORK/check_firmware_hw.log" 2>&1 \
        || { cat "$TINYGPU_TEST_WORK/check_firmware_hw.log"; echo "firmware not staged (check_firmware.py failed); not running"; exit 2; }
    # the one allowed knob: plan step B1's fallback unload (LEVEL_0 when 0), announced so the run's output says so
    [ -n "${BEAGLE_NV_UNLOAD_LEVEL+x}" ] && echo "note: BEAGLE_NV_UNLOAD_LEVEL=$BEAGLE_NV_UNLOAD_LEVEL is set (0: the LEVEL_0 unload, plan step B1's fallback)"
    mkdir -p "$BEAGLE_TINYGPU_DATA/runs"; HW_LOCK="${TMPDIR:-/tmp}"; HW_LOCK="${HW_LOCK%/}/beagle_tinygpu_hw.lock"
    mkdir "$HW_LOCK" 2>/dev/null || { echo "another hardware script holds $HW_LOCK (remove it if none is running); not running"; exit 2; }
    trap 'rmdir "$HW_LOCK" 2>/dev/null; [ -n "$LSP" ] && kill $LSP 2>/dev/null' EXIT
}
# hw_logstream <file>: the filter of every hardware run so far (STATUS.md R17-R20), in its own process group so that a Ctrl-C
# in the terminal cannot end it during the run; with nothing from the eGPU it prints only its header. Sets LSP.
hw_logstream() {
    set -m
    log stream --predicate 'composedMessage CONTAINS[c] "DART" OR composedMessage CONTAINS[c] "apciec" OR process CONTAINS[c] "tinygpu" OR composedMessage CONTAINS[c] "panic"' > "$1" 2>&1 &
    LSP=$!
    set +m
    for i in $(seq 50); do grep -q "^Filtering the log data" "$1" && break; sleep 0.1; done
    grep -q "^Filtering the log data" "$1" || { echo "log stream did not attach; not running"; exit 2; }
}
# hw_logstream_stop: after the run, lets log stream flush and stops it; fails if it had already ended (the check was blind)
hw_logstream_stop() {
    kill -0 $LSP 2>/dev/null || return 1
    sleep 2; kill $LSP 2>/dev/null
}
# hw_hold_check: a remaining daemon or nv_teardown_diag may hold the GPU: keep the Mac awake while it holds, and say what to do
hw_hold_check() {
    local pid p
    pid=$(pgrep -f "$DAEMON_RE") || return 0
    for p in $pid; do nohup caffeinate -ims -w $p > /dev/null 2>&1 & done
    echo "STOP: pid $pid (nv_dispatch_daemon or nv_teardown_diag) is still running and may hold the GPU: unplug the eGPU first, then kill $pid"
    return 1
}
