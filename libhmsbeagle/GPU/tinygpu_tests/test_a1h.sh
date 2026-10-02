#!/bin/bash
# TODO.md plan step A1h, on the C++ boot since plan step A2l: the AMD C++ runtime end to end, offline. The real plugin boots
# fake_amd_device.py's card itself and runs tinygpuhybridtest on it; the fake runs every PM4 and SDMA packet and checks every
# address the GPU would touch (the DART check), the doorbell protocol and, with FAKE_AMD_HSACO, each dispatch's kernel against
# the build's HSACO. Kernels are not run, so logL is wrong by design: a case passes when the plugin booted and handed over, ran
# to the end with no runtime error and the fake saw NO ERRORS (the fault case: when the plugin decoded the fake's SQ MEMVIOL
# and stopped, without hanging). The rings and kernargs are tinygrad's sizes, so the wraps take 64,000 evaluations.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
require_no_launch_guard
[ -x "$TEST_BIN" ] || { echo "no $TEST_BIN; build it first"; exit 2; }
GUARD_BIN="$BEAGLE_BUILD/libhmsbeagle/GPU/CMake_TinyGPUHybrid/beagle-tinygpu-guard"
results=()

run_case() {   # <label> <ok|fault> [VAR=value ...] -- [tinygpuhybridtest args ...]
    local label=$1 expect=$2; shift 2
    local envs=(); while [ $# -gt 0 ] && [ "$1" != "--" ]; do envs+=("$1"); shift; done; [ "$1" = "--" ] && shift
    local sockdir; sockdir=$(mktemp -d /tmp/tga.XXXXXX)
    local sock="$sockdir/dev.sock" mem="$TINYGPU_TEST_WORK/fake_amd_$label" dlog="$TINYGPU_TEST_WORK/fake_amd_$label.log"
    local out="$TINYGPU_TEST_WORK/run_amd_$label.txt"
    rm -rf "$mem" "$dlog" "$out"; mkdir -p "$mem"
    env "${envs[@]}" "$BEAGLE_PYTHON" "$TG_TESTS/fake_amd_device.py" "$sock" "$mem" > "$dlog" 2>&1 &
    local srv=$!
    for i in $(seq 100); do grep -q listening "$dlog" 2>/dev/null && break; sleep 0.1; done
    if ! grep -q listening "$dlog"; then results+=("$label: FAIL (the fake device did not start; $dlog)"); kill $srv 2>/dev/null; return; fi
    env BEAGLE_TINYGPU_NO_LAUNCH=1 BEAGLE_TINYGPU_NO_DOWNLOAD=1 BEAGLE_TINYGPU_LOG="$TINYGPU_TEST_WORK/beagle_tinygpu_offline.log" \
        APL_REMOTE_SOCK="$sock" TMPDIR="$sockdir" BEAGLE_AMD_GUARD="$TG_TESTS/replay/crash_guard_wrap.sh" BEAGLE_TG_GUARD_BIN="$GUARD_BIN" \
        BEAGLE_TG_GUARD_PIDFILE="$sockdir/guard.pid" BEAGLE_AMD_DATA_MB=1024 DYLD_LIBRARY_PATH="$TEST_LIBS" "${envs[@]}" \
        "$TEST_BIN" "$@" > "$out" 2>&1 &
    local tst=$! rc=124
    for i in $(seq 3000); do kill -0 $tst 2>/dev/null || { wait $tst; rc=$?; break; }; sleep 0.1; done
    [ $rc -eq 124 ] && { kill -KILL $tst 2>/dev/null; wait $tst 2>/dev/null; }
    # the crash guard exits at the plugin's clean; one that holds keeps the fake's connection, so it is ended here (offline only)
    local gpid held=""; gpid=$(cat "$sockdir/guard.pid" 2>/dev/null)
    if [ -n "$gpid" ]; then
        for i in $(seq 100); do kill -0 "$gpid" 2>/dev/null || break; sleep 0.1; done
        if kill -0 "$gpid" 2>/dev/null; then held=1; kill -KILL "$gpid"; for i in $(seq 50); do kill -0 "$gpid" 2>/dev/null || break; sleep 0.1; done; fi
    fi
    # the last session's verdict: BEAGLE's resource listing makes a session of its own (one CFG_READ) before the run's
    for i in $(seq 50); do [ "$(grep -c 'client done' "$dlog")" -ge 2 ] && grep -q '"launches"' "$dlog" && break; sleep 0.1; done
    kill $srv 2>/dev/null; wait $srv 2>/dev/null; rm -rf "$sockdir"
    local counts verdict; counts=$(grep "client done" "$dlog" | tail -1)
    verdict=$(grep -E "fake TinyGPU.app \(AMD device\): (NO ERRORS|[0-9]+ ERRORS)" "$dlog" | tail -1)
    n() { echo "$counts" | grep -o "\"$1\": [0-9]*" | grep -o '[0-9]*$'; }
    local why=""
    [ $rc -eq 124 ] && why="the test hung (killed after 300 s)"
    [ -n "$held" ] && why="${why:+$why; }the crash guard held"
    [ "$(grep -c 'client done' "$dlog")" -ge 2 ] || why="${why:+$why; }the run's session never ended"
    echo "$verdict" | grep -q "NO ERRORS" || why="${why:+$why; }the fake saw errors: $(echo "$verdict" | cut -c1-300)"
    grep -q "TinyGPU/AMD: C++ runtime: handed over after the C++ boot" "$out" || why="${why:+$why; }no handoff"
    [ "$(n launches)" -gt 0 ] 2>/dev/null || why="${why:+$why; }no launches reached the fake GPU"
    if [ "$expect" = fault ]; then
        grep -q "TinyGPU/AMD: sq_intr: error (MEMVIOL)" "$out" || why="${why:+$why; }the SQ MEMVIOL was not decoded"
        grep -q "the GPU reported a fault" "$out" || why="${why:+$why; }the fault did not stop the runtime"
    else
        grep -q "^per evaluation:" "$out" || why="${why:+$why; }the test did not finish its evaluations"
        grep -qE "TinyGPU/AMD: .*(failed|refused|fault)" "$out" && why="${why:+$why; }$(grep -m1 -E "TinyGPU/AMD: .*(failed|refused|fault)" "$out" | cut -c1-200)"
        for e in "${envs[@]}"; do   # FAKE_AMD_HSACO: every dispatch was one of the HSACO's kernels
            [[ $e == FAKE_AMD_HSACO=* ]] && ! grep "kernels launched:" "$dlog" | tail -1 | grep -q '": [1-9]' && why="${why:+$why; }no dispatch matched the HSACO's kernels"
        done
        if [ "$expect" = wraps ]; then
            for w in "compute ring wraps" "sdma ring wraps" "kernargs wraps"; do [ "$(n "$w")" -gt 0 ] 2>/dev/null || why="${why:+$why; }no $w"; done
        fi
    fi
    if [ -z "$why" ]; then results+=("$label: PASS ($(n launches) launches; $(echo "$counts" | grep -oE '"(copied bytes|signals|compute ring wraps|sdma ring wraps|kernargs wraps)": [0-9]*' | tr '\n' ' '))")
    else results+=("$label: FAIL ($why; $out, $dlog)"); fi
}

# the build's HSACOs, every dispatch checked against them
run_case n4 ok FAKE_AMD_HSACO=SP_4 -- --state-count 4 --reps 3 --diag-compare-cpu
run_case n64 ok FAKE_AMD_HSACO=SP_64 -- --state-count 64 --reps 3 --diag-compare-cpu
run_case n256 ok FAKE_AMD_HSACO=SP_256 -- --state-count 256 --reps 2
# tinygrad's 16 MiB compute and SDMA rings and kernargs: the compute ring's idle wait at its end, the SDMA tail's zero fill and
# wrap, the kernargs wrap (each about 1.3 KiB, 0.3 KiB and 1.5 KiB an evaluation at 4 states)
run_case wraps wraps -- --state-count 4 --reps 64000
# the GPU stops at the 3rd dispatch with an SQ MEMVIOL in the IH ring: the wait's interrupt check must decode it and fail (a
# cold card: its full boot programs the IH ring, which a fake warm card has never had)
run_case fault fault FAKE_AMD_STATE=cold FAKE_AMD_FAULT=1 HCQDEV_WAIT_TIMEOUT_MS=3000 -- --state-count 4 --reps 1
echo "=== A1h"; printf '%s\n' "${results[@]}"
! printf '%s\n' "${results[@]}" | grep -q FAIL
