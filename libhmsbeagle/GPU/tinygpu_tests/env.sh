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
# RTX 4060, each with its run and teardown, in $BEAGLE_TINYGPU_DATA/recordings (they hold NVIDIA firmware: never in git). They
# and the GB205's were made through the daemon, tinygrad's own boot: the C++ boot makes the same requests (test_c11.sh, test_b2.sh)
TG_L0="20260925-204611_mittag-leffler_cold 20260925-204652_mittag-leffler_warm 20260925-204737_mittag-leffler_warm64"
# plan step B2's GB205 recordings (STATUS.md R45-R48): the GB20x L0 (level runtime), then rungs H1 (vram), H2 (sysmem), T
# (teardown, the plugin's COT teardown), H3 (rm, the plugin's NVDevice) and H4 (gsp_hw, the plugin's GSP-RM boot) behind the
# guard, warm boots at 4 states in one enumeration
TG_GB20X="20260927-093033_Marcs-Mac-Studio-490_gb205_l0 20260927-094318_Marcs-Mac-Studio-490_gb205_h1_vram 20260927-094646_Marcs-Mac-Studio-490_gb205_h2_sysmem 20260927-112033_Marcs-Mac-Studio-490_gb205_t 20260927-113001_Marcs-Mac-Studio-490_gb205_h3_rm 20260927-114036_Marcs-Mac-Studio-490_gb205_h4_gsp_hw"
# plan step A2j's AMD L0 recordings (STATUS.md R71, R74): the AMD daemon's boot-only sessions on the RX 7900 XT, warm (a
# partial boot) and cold (a full one, after a power cycle), through tgproxy --guard; each replays exactly to the oracle's
# daemon and to the C++ boot (amd_l0_replay.py, test_a2.sh)
TG_AMD_L0="20261001-125155_Marcs-Mac-Studio-490_amd_l0_warm 20261002-083213_Marcs-Mac-Studio-490_amd_l0_cold"

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
# TODO.md plan step C10: after run_point.sh --kill the plugin printed no fini report, and the crash guard's TinyGPULog lines (<file>,
# this run's) decide as fini_verdict does: the guard saw the plugin's EOF, its teardown confirmed the unload, WPR2 is down and
# the teardown succeeded ("teardown_ok", which the plugin's "the next boot needs no power cycle" reads; on COT no "halted":
# false), it closed the connection, and nothing held.
guard_verdict() {
    grep -q "guard [0-9]*: the plugin went away without fini (EOF)" "$1" \
        && grep -E 'guard: the GPU teardown: \{"unload_ok": true' "$1" | grep -E '"wpr2_hi": 0[,}]' | grep -q '"teardown_ok": true' \
        && ! grep -qE 'guard: the GPU teardown: \{.*"halted": false' "$1" \
        && grep -q "guard [0-9]*: the GPU is torn down; closing the TinyGPU.app connection" "$1" && ! grep -q "HOLDING" "$1"
}

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
    [ "$(grep -c "TinyGPU/NV: level boot: the C++ boot, with no daemon" "$2")" -eq 1 ] || { echo "not exactly one boot"; return 1; }
    grep -q "Rsrc Name : TinyGPU-NV-Hybrid" "$1" || { echo "not the TinyGPU resource"; return 1; }
    grep -q "TinyGPU/NV: C++ runtime: embedded cubin SP_" "$2" || { echo "not the C++ runtime with an embedded cubin"; return 1; }
    ! grep -qE "not launched|TinyGPU/NV: .*failed" "$2" || { echo "a launch was rejected or a step failed"; return 1; }
    [ "$got" = "$3" ] || { echo "launched: $got"; return 1; }
}
# TODO.md plan step A4, D1 on the AMD card: as d1_verdict, from the AMD plugin's lines: one C++ boot, the TinyGPU resource
# with the C++ runtime, no failed step and no lost GPU (plan step A3), and exactly the line's kernels launched (the plugin's
# launch lines, in byte order as d1_runs.txt lists them). Reads files and runs nothing (test_a4.sh checks it on the fake).
amd_d1_verdict() {   # <stdout file> <stderr file> "<kernels, sorted>"
    local got; got=$(sed -nE 's/^TinyGPU\/AMD: launch ([A-Za-z0-9_]+) grid=.*/\1/p' "$2" | LC_ALL=C sort -u | xargs)
    [ "$(grep -c "TinyGPU/AMD: C++ boot done" "$2")" -eq 1 ] || { echo "not exactly one boot"; return 1; }
    grep -q "Rsrc Name : TinyGPU-AMD-Hybrid" "$1" || { echo "not the TinyGPU AMD resource"; return 1; }
    grep -q "TinyGPU/AMD: C++ runtime: handed over after the C++ boot" "$2" || { echo "the C++ runtime never took over"; return 1; }
    ! grep -qE "TinyGPU/AMD: .*(failed|this instance fails|the GPU is lost)" "$2" || { echo "a step failed or the GPU was lost"; return 1; }
    [ "$got" = "$3" ] || { echo "launched: $got"; return 1; }
}

# The hardware scripts' shared protections (run_point.sh, run_d1.sh, run_l0.sh; plan steps P3, D1). hw_begin: nothing else
# changes what the plugin does to the GPU, and one hardware script runs at a time on this computer (the lock is removed at
# exit; it is in $TMPDIR, next to tinygrad's nv_usb4.lock, so a computer sharing $BEAGLE_TINYGPU_DATA runs its own eGPU freely).
hw_begin() {
    local v
    # the plugin's knobs, and the harness's (plan V1's markers and fakes, the log, C10's test kill, run_point.sh --kill, and the
    # guard's path); since plan step C13c no Python runs, so tinygrad's own variables change nothing
    for v in BEAGLE_NV_TEARDOWN BEAGLE_NV_DATA_MB APL_REMOTE_SOCK BEAGLE_TINYGPU_NO_LAUNCH FAKE_TEST_BIN BEAGLE_NV_FILL_LAUNCH_DIMS \
             BEAGLE_TG_MARKERS BEAGLE_TINYGPU_LOG BEAGLE_NV_TEST_KILL BEAGLE_NV_GUARD; do
        [ -n "${!v+x}" ] && { echo "$v is set; unset it first; not running"; exit 2; }
    done
    # the firmware is staged where the boot looks for it (decision 5; macOS may purge tinygrad's cache): offline, re-staging
    # from $BEAGLE_TINYGPU_DATA/fw if needed
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
# The log-stream lines that are not the eGPU, which the hardware scripts' STOP check skips: the filter's header, the Neural
# Engines' DART mappings ("ANE0 ... dartMapBase"; on a two-die Mac also ANE2's, STATUS.md R62), camera lines, and TinyGPU.app's GUI process gaining or losing a visibility
# inheritance (RunningBoard, as its window's visibility changes: STATUS.md R35), with that message's continuation lines. Any
# other line from TinyGPU.app (a RunningBoard termination, say), and every DART, apciec or panic line, still stops the run.
HW_LOG_BENIGN='^Filtering the log data|^Timestamp +Thread|\(AppleH11ANEInterface\) ANE[0-9]+:|H13Cam|TinyGPU: \(RunningBoardServices\) (didChangeInheritances$|\[com\.apple\.runningboard:connection\] (Gained|Lost) inheritances: \{\($)|^    <RBSInheritance[|] |^\)\}$'
# hw_logstream_stop: after the run, lets log stream flush and stops it; fails if it had already ended (the check was blind)
hw_logstream_stop() {
    kill -0 $LSP 2>/dev/null || return 1
    sleep 2; kill $LSP 2>/dev/null
}
# what can hold the GPU after a run: the crash guard (.../beagle-tinygpu-guard, plan step C10), anchored at the end of the
# command line, so a shell or editor that merely mentions it does not match
GUARD_RE="(^|/)beagle-tinygpu-guard$"
# hw_hold_check: a remaining crash guard may hold the GPU: keep the Mac awake while it holds, and say what to do
hw_hold_check() {
    local pid p
    pid=$(pgrep -f "$GUARD_RE") || return 0
    for p in $pid; do nohup caffeinate -ims -w $p > /dev/null 2>&1 & done
    echo "STOP: pid $pid (the crash guard) is still running and may hold the GPU: unplug the eGPU first, then kill $pid"
    return 1
}

# The AMD hardware scripts' protections (run_amd_point.sh, run_amd_smoke.sh, run_amd_discovery.sh; TODO.md plan step A0).
# amd_hw_begin: nothing set that changes the AMD path or forces tinygrad's full reset (AM_RESET), no BEAGLE process, at most
# one TinyGPU.app server, one hardware script at a time (hw_begin's lock), and TinyGPU.app serving an AMD card (tg_probe.py:
# one config read). Sets AMD_PCI (vendor:device).
amd_hw_begin() {
    local v n p
    for v in APL_REMOTE_SOCK BEAGLE_TINYGPU_NO_LAUNCH AM_RESET; do
        [ -n "${!v+x}" ] && { echo "$v is set; unset it first; not running"; exit 2; }
    done
    pgrep -fl "beagle-tinygpu-guard|tinygpuhybridtest|synthetictest|hmctest|amd_dispatch_daemon|nv_dispatch_daemon" \
        && { echo "a BEAGLE process is running; not running"; exit 2; }
    n=$(pgrep -f "TinyGPU.app/Contents/MacOS/TinyGPU server" | wc -l | tr -d ' ')
    [ "$n" -le 1 ] || { echo "$n TinyGPU.app servers are running (a duplicate); not running"; exit 2; }
    mkdir -p "$BEAGLE_TINYGPU_DATA/runs"; HW_LOCK="${TMPDIR:-/tmp}"; HW_LOCK="${HW_LOCK%/}/beagle_tinygpu_hw.lock"
    mkdir "$HW_LOCK" 2>/dev/null || { echo "another hardware script holds $HW_LOCK (remove it if none is running); not running"; exit 2; }
    trap 'rmdir "$HW_LOCK" 2>/dev/null; [ -n "$LSP" ] && kill $LSP 2>/dev/null' EXIT
    p=$("$BEAGLE_PYTHON" "$TG_TESTS/tg_probe.py" 2>&1); echo "probe: $p"
    AMD_PCI=$(echo "$p" | sed -n 's/^TinyGPU.app serves \(1002:[0-9a-f]*\) .*/\1/p')
    [ -n "$AMD_PCI" ] || { echo "TinyGPU.app does not serve an AMD card; not running"; exit 2; }
}
# amd_boot_check: the boot tinygrad's AM driver would give the card, read without writing (amd_state.py, at the bases of the
# table run_amd_discovery.sh captured). Refuses when it would start with an SMU mode1 reset, which plan step A0 aborts on;
# without a captured table it cannot tell, and says so.
amd_boot_check() {
    local tbl rc
    tbl=$(ls "$BEAGLE_TINYGPU_DATA/discovery/${AMD_PCI/:/_}"_*.json 2>/dev/null | head -1)
    [ -n "$tbl" ] || { echo "note: no discovery table for $AMD_PCI in $BEAGLE_TINYGPU_DATA/discovery: the boot is not predicted"; return 0; }
    "$BEAGLE_PYTHON" "$TG_TESTS/amd_state.py" "$tbl"; rc=$?
    [ $rc -eq 0 ] && return 0
    [ $rc -eq 3 ] && { echo "a mode1 reset would follow: power-cycle the card (unplug it, then plug it in again) first; not running"; exit 2; }
    echo "the boot check failed (exit $rc); not running"; exit 2
}
# amd_require_app_zip: stock tinygrad's APLRemotePCIDevice runs ensure_app, which kills TinyGPU and reinstalls the app unless
# its release zip is in tinygrad's download cache (system.py:419-427): run_amd_smoke.sh and run_amd_discovery.sh refuse
# instead. BEAGLE's daemon never calls it (amd_dispatch_daemon.py _install_inherited_tinygpu).
amd_require_app_zip() {
    local zip="${XDG_CACHE_HOME:-$HOME/Library/Caches}/tinygrad/downloads/TinyGPU_c0d024f9ff0e1dc8fdf217f255da7101d91e8323.zip"
    [ -f "$zip" ] && [ -x /Applications/TinyGPU.app/Contents/MacOS/TinyGPU ] \
        || { echo "no $zip or no TinyGPU.app: stock tinygrad's ensure_app would reinstall the app; not running"; exit 2; }
}
# amd_hw_end <output>: after a run, log stream must still be attached and have seen nothing from the eGPU ($LS), and
# tinygrad's DEBUG=2 lines must show no mode1 reset or malformed state (plan step A0's abort conditions). Prints a STOP line
# and returns nonzero otherwise.
amd_hw_end() {
    local events reset
    hw_logstream_stop || { echo "STOP: log stream ended during the run ($LS): the eGPU check was blind: stop all hardware work"; return 1; }
    events=$(grep -cvE "$HW_LOG_BENIGN" "$LS")
    [ "$events" -eq 0 ] || { echo "STOP: log stream saw $events eGPU event line(s) ($LS): stop all hardware work"
                             grep -vE "$HW_LOG_BENIGN" "$LS" | head -5 | cut -c1-200; return 1; }
    reset=$(grep -m1 -E "^am [^:]*: (mode1 reset|Malformed state)" "$1")
    [ -z "$reset" ] || { echo "STOP: tinygrad reset the card ($reset): stop all hardware work"; return 1; }
}
