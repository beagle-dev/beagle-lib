#!/bin/bash
# TODO.md plan step A1h: the AMD C++ runtime end to end, offline. The real plugin (its default, the C++ runtime) runs tinygpuhybridtest with
# fake_amd_daemon.py in place of amd_dispatch_daemon.py, against fake_amd_device.py, which runs every PM4 and SDMA packet and
# checks every address the GPU would touch (the DART check), the doorbell protocol and each dispatch's kernel. Kernels are
# not run, so logL is wrong by design: a case passes when the plugin handed over, ran to the end with no runtime error and
# the fake saw NO ERRORS (the fault case: when the plugin decoded the fake's SQ MEMVIOL and stopped, without hanging).
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
require_no_launch_guard
[ -x "$TEST_BIN" ] || { echo "no $TEST_BIN; build it first"; exit 2; }
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
        APL_REMOTE_SOCK="$sock" TMPDIR="$sockdir" BEAGLE_AMD_DISPATCH_DAEMON="$TG_TESTS/fake_amd_daemon.py" \
        BEAGLE_NV_SCRIPTS="$GPU_DIR" DYLD_LIBRARY_PATH="$TEST_LIBS" "${envs[@]}" "$TEST_BIN" "$@" > "$out" 2>&1 &
    local tst=$! rc=124
    for i in $(seq 1200); do kill -0 $tst 2>/dev/null || { wait $tst; rc=$?; break; }; sleep 0.1; done
    [ $rc -eq 124 ] && { kill -KILL $tst 2>/dev/null; wait $tst 2>/dev/null; }
    # the last session's verdict: BEAGLE's resource listing makes a session of its own (one CFG_READ) before the run's
    for i in $(seq 50); do [ "$(grep -c 'client done' "$dlog")" -ge 2 ] && grep -q '"launches"' "$dlog" && break; sleep 0.1; done
    kill $srv 2>/dev/null; wait $srv 2>/dev/null; rm -rf "$sockdir"
    local launches verdict; launches=$(grep -o '"launches": [0-9]*' "$dlog" | tail -1 | grep -o '[0-9]*$')
    verdict=$(grep -E "fake TinyGPU.app \(AMD device\): (NO ERRORS|[0-9]+ ERRORS)" "$dlog" | tail -1)
    local why=""
    [ $rc -eq 124 ] && why="the test hung (killed after 120 s)"
    [ "$(grep -c 'client done' "$dlog")" -ge 2 ] || why="${why:+$why; }the run's session never ended"
    echo "$verdict" | grep -q "NO ERRORS" || why="${why:+$why; }the fake saw errors: $(echo "$verdict" | cut -c1-300)"
    grep -q "TinyGPU/AMD: C++ runtime: handed over" "$out" || why="${why:+$why; }no handoff"
    if [ "$expect" = ok ]; then
        grep -q "^per evaluation:" "$out" || why="${why:+$why; }the test did not finish its evaluations"
        grep -qE "TinyGPU/AMD: .*(failed|refused|fault)|TinyGPU/AMD: handoff:" "$out" && why="${why:+$why; }$(grep -m1 -E "TinyGPU/AMD: .*(failed|refused|fault)|TinyGPU/AMD: handoff:" "$out" | cut -c1-200)"
        [ "${launches:-0}" -gt 0 ] || why="${why:+$why; }no launches reached the fake GPU"
        local aot=1; for e in "${envs[@]}"; do [ "$e" = BEAGLE_AMD_AOT=0 ] && aot=0; done
        if [ $aot = 1 ]; then grep -q "ahead-of-time HSACO .*: no run-time compile" "$out" || why="${why:+$why; }not the build's HSACO"
        else grep -q "precompile_all_kernels — loaded" "$out" || why="${why:+$why; }not the daemon's compile"; fi
    else
        grep -q "TinyGPU/AMD: sq_intr: error (MEMVIOL)" "$out" || why="${why:+$why; }the SQ MEMVIOL was not decoded"
        grep -q "the GPU reported a fault" "$out" || why="${why:+$why; }the fault did not stop the runtime"
    fi
    if [ -z "$why" ]; then results+=("$label: PASS ($launches launches; $(grep -oE '"(copied bytes|signals|compute ring wraps|sdma ring wraps)": [0-9]*' "$dlog" | tail -4 | tr '\n' ' '))")
    else results+=("$label: FAIL ($why; $out, $dlog)"); fi
}

# the build's HSACOs (plan step A1j: no run-time compile), and once the daemon's compile instead (BEAGLE_AMD_AOT=0)
run_case n4 ok -- --state-count 4 --reps 3 --diag-compare-cpu
run_case n4_daemon_compile ok BEAGLE_AMD_AOT=0 -- --state-count 4 --reps 3
run_case n64 ok -- --state-count 64 --reps 3 --diag-compare-cpu
run_case n256 ok -- --state-count 256 --reps 2
# small rings and kernargs: the compute ring's idle wait at its end, the SDMA tail's zero fill and wrap, the kernargs wrap
run_case wraps ok FAKE_AMD_RING_KB=16 FAKE_AMD_SDMA_RING_KB=4 FAKE_AMD_KARGS_KB=4 -- --state-count 4 --reps 40
# the GPU stops at the 3rd dispatch with an SQ MEMVIOL in the IH ring: the wait's interrupt check must decode it and fail
run_case fault fault FAKE_AMD_FAULT=1 HCQDEV_WAIT_TIMEOUT_MS=3000 -- --state-count 4 --reps 1
echo "=== A1h"; printf '%s\n' "${results[@]}"
! printf '%s\n' "${results[@]}" | grep -q FAIL
