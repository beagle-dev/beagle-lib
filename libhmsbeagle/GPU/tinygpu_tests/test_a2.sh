#!/bin/bash
# TODO.md plan step A2, end to end with no eGPU, on fake_amd_device.py's register-level card (fake_am_gpu.py), which runs every
# PM4 and SDMA packet through the GMC page tables and audits every system address the GPU could reach (the DART check):
#   - A2a: the real amd_dispatch_daemon.py (amd_daemon_on_fake.py) boots the card with tinygrad's AMDev, cold (a full boot),
#     warm (a partial one) and dirty (a mode1 reset first), and the plugin's C++ runtime (A1) runs tinygpuhybridtest on what
#     it hands over;
#   - A2h: the plugin boots the card itself (BEAGLE_AMD_CPP_BOOT=1, no daemon), cold and warm, and the whole instance session's
#     requests equal the daemon-booted run's byte for byte; a dirty card is refused before the mode1 reset.
# Kernels are not run, so logL is wrong by design: a case passes when the plugin handed over, ran to the end with no runtime
# error, and the fake saw NO ERRORS.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
require_no_launch_guard
[ -x "$TEST_BIN" ] || { echo "no $TEST_BIN; build it first"; exit 2; }
results=()

run_case() {   # <label> <daemon> [VAR=value ...] -- [tinygpuhybridtest args ...]
    local label=$1 daemon=$2; shift 2
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
        APL_REMOTE_SOCK="$sock" TMPDIR="$sockdir" BEAGLE_AMD_DISPATCH_DAEMON="$TG_TESTS/$daemon" \
        BEAGLE_NV_SCRIPTS="$GPU_DIR" DYLD_LIBRARY_PATH="$TEST_LIBS" "${envs[@]}" "$TEST_BIN" "$@" > "$out" 2>&1 &
    local tst=$! rc=124
    for i in $(seq 1800); do kill -0 $tst 2>/dev/null || { wait $tst; rc=$?; break; }; sleep 0.1; done
    [ $rc -eq 124 ] && { kill -KILL $tst 2>/dev/null; wait $tst 2>/dev/null; }
    # the last session's verdict: BEAGLE's resource listing makes a session of its own (one CFG_READ) before the run's
    for i in $(seq 100); do [ "$(grep -c 'client done' "$dlog")" -ge 2 ] && grep -q '"launches"' "$dlog" && break; sleep 0.1; done
    kill $srv 2>/dev/null; wait $srv 2>/dev/null; rm -rf "$sockdir"
    local launches verdict; launches=$(grep -o '"launches": [0-9]*' "$dlog" | tail -1 | grep -o '[0-9]*$')
    verdict=$(grep -E "fake TinyGPU.app \(AMD device\): (NO ERRORS|[0-9]+ ERRORS)" "$dlog" | tail -1)
    local why=""
    [ $rc -eq 124 ] && why="the test hung (killed after 180 s)"
    [ "$(grep -c 'client done' "$dlog")" -ge 2 ] || why="${why:+$why; }the run's session never ended"
    echo "$verdict" | grep -q "NO ERRORS" || why="${why:+$why; }the fake saw errors: $(echo "$verdict" | cut -c1-300)"
    grep -q "TinyGPU/AMD: C++ runtime: handed over" "$out" || why="${why:+$why; }no handoff"
    grep -q "^per evaluation:" "$out" || why="${why:+$why; }the test did not finish its evaluations"
    grep -qE "TinyGPU/AMD: .*(failed|refused|fault)|TinyGPU/AMD: handoff:" "$out" && why="${why:+$why; }$(grep -m1 -E "TinyGPU/AMD: .*(failed|refused|fault)|TinyGPU/AMD: handoff:" "$out" | cut -c1-200)"
    [ "${launches:-0}" -gt 0 ] || why="${why:+$why; }no launches reached the fake GPU"
    if [ -z "$why" ]; then results+=("$label: PASS ($launches launches; $(grep -oE '"(copied bytes|signals|psp cmd 0x6|tlb flushes|system PTEs audited|compute queues activated|sdma queues activated)": [0-9]*' "$dlog" | tail -7 | tr '\n' ' '))")
    else results+=("$label: FAIL ($why; $out, $dlog)"); fi
}

# A2a: tinygrad's own boot on the fake card, then the A1 C++ runtime on its queues
run_case a2a_daemon_cold amd_daemon_on_fake.py FAKE_AMD_STATE=cold -- --state-count 4 --reps 3 --diag-compare-cpu
run_case a2a_daemon_warm amd_daemon_on_fake.py FAKE_AMD_STATE=warm -- --state-count 64 --reps 3
# an AM session that did not finalize (SCRATCH_REG6 1): tinygrad's full boot after an SMU mode1 reset
run_case a2a_daemon_dirty amd_daemon_on_fake.py FAKE_AMD_STATE=dirty -- --state-count 4 --reps 2

# A2h: the plugin boots the card itself (BEAGLE_AMD_CPP_BOOT=1, no daemon), cold and warm; each run's instance session must
# send the requests the daemon-booted run sends, byte for byte (FAKE_AMD_RECORD: the second session, after the resource listing's)
for st in cold warm; do
    rec_d="$TINYGPU_TEST_WORK/a2h_rec_daemon_$st" rec_c="$TINYGPU_TEST_WORK/a2h_rec_cpp_$st"
    rm -f "$rec_d".* "$rec_c".*
    run_case a2h_daemon_$st amd_daemon_on_fake.py FAKE_AMD_STATE=$st FAKE_AMD_RECORD="$rec_d" -- --state-count 4 --reps 3 --diag-compare-cpu
    run_case a2h_cpp_boot_$st amd_daemon_on_fake.py BEAGLE_AMD_CPP_BOOT=1 FAKE_AMD_STATE=$st FAKE_AMD_RECORD="$rec_c" -- --state-count 4 --reps 3 --diag-compare-cpu
    if ! grep -q "C++ runtime: handed over after the C++ boot" "$TINYGPU_TEST_WORK/run_amd_a2h_cpp_boot_$st.txt"; then
        results+=("a2h_session_$st: FAIL (the C++ boot did not run: $TINYGPU_TEST_WORK/run_amd_a2h_cpp_boot_$st.txt)")
    elif grep -q "spawning amd_dispatch_daemon" "$TINYGPU_TEST_WORK/run_amd_a2h_cpp_boot_$st.txt"; then
        results+=("a2h_session_$st: FAIL (a daemon was spawned)")
    elif cmp -s "$rec_d.1" "$rec_c.1"; then
        results+=("a2h_session_$st: PASS (the whole instance session, $(wc -c < "$rec_c.1" | tr -d ' ') bytes of requests, equals the daemon-booted run's)")
    else
        results+=("a2h_session_$st: FAIL ($(cmp "$rec_d.1" "$rec_c.1" 2>&1 | head -1))")
    fi
done
# a card an earlier session left unclean: the C++ boot refuses before tinygrad's mode1 reset, the run fails cleanly
label=a2h_cpp_boot_dirty sockdir=$(mktemp -d /tmp/tga.XXXXXX)
dlog="$TINYGPU_TEST_WORK/fake_amd_$label.log" out="$TINYGPU_TEST_WORK/run_amd_$label.txt" mem="$TINYGPU_TEST_WORK/fake_amd_$label"
rm -rf "$mem"; mkdir -p "$mem"
FAKE_AMD_STATE=dirty "$BEAGLE_PYTHON" "$TG_TESTS/fake_amd_device.py" "$sockdir/dev.sock" "$mem" > "$dlog" 2>&1 & srv=$!
for i in $(seq 100); do grep -q listening "$dlog" 2>/dev/null && break; sleep 0.1; done
env BEAGLE_TINYGPU_NO_LAUNCH=1 BEAGLE_TINYGPU_NO_DOWNLOAD=1 BEAGLE_TINYGPU_LOG="$TINYGPU_TEST_WORK/beagle_tinygpu_offline.log" APL_REMOTE_SOCK="$sockdir/dev.sock" \
    TMPDIR="$sockdir" BEAGLE_AMD_CPP_BOOT=1 BEAGLE_NV_SCRIPTS="$GPU_DIR" DYLD_LIBRARY_PATH="$TEST_LIBS" "$TEST_BIN" --state-count 4 --reps 1 > "$out" 2>&1
rc=$?
for i in $(seq 50); do [ "$(grep -c 'client done' "$dlog")" -ge 2 ] && break; sleep 0.1; done
kill $srv 2>/dev/null; wait $srv 2>/dev/null; rm -rf "$sockdir"
if [ $rc -ne 0 ] && grep -q "needs an SMU mode1 reset" "$out" && grep -q "NO ERRORS" "$dlog" && ! grep -q '"mode1 resets"' "$dlog" \
   && ! grep -q '"compute queues activated"' "$dlog"; then results+=("$label: PASS (refused before the mode1 reset; exit $rc, no queue set up)")
else results+=("$label: FAIL (exit $rc; $out, $dlog)"); fi
echo "=== A2"; printf '%s\n' "${results[@]}"
! printf '%s\n' "${results[@]}" | grep -q FAIL
