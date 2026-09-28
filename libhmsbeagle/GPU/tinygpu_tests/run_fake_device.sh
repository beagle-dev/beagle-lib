#!/bin/bash
# Offline end-to-end run of the real plugin, which boots the GPU itself (plan steps C11-C13), against fake_nv_device.py, a
# fake TinyGPU.app playing an AD107 (or with FAKE_NV_CHIP=gb205 a GB205) at the register level (TODO.md plan step V1). No GPU
# and no TinyGPU.app: BEAGLE_TINYGPU_NO_LAUNCH=1 and a short per-run socket. FAKE_TG_PROXY=<new recording dir> puts the
# recording proxy in between. FAKE_TG_GUARD=1 runs that proxy in guard mode (replay/tgguard.py): a
# refusal ends the run at once (the proxy holds, as it would on the eGPU; offline everything is then ended), and the run
# passes only if FAKE_EXPECT_TRIP=<regex> is set and matches the guard's reason. FAKE_SIGINT_AFTER=<regex> (plan step C12) runs
# the test in its own process group and, once its output matches, sends that group SIGINT, as a terminal's Ctrl-C would.
# FAKE_SECOND_AFTER=<regex> (plan step P5) starts a second test process (--reps 1) against the same fake and lock once the
# output matches, and gives it 30 s. FAKE_TEST_BIN=<another BEAGLE example> runs that instead (plan step D1).
#   run_fake_device.sh <label> [VAR=value ...] -- [tinygpuhybridtest args ...]
# Exit status 0 only if the test booted once, loaded its programs, the teardown says the next boot needs no power cycle
# (fini_verdict), and the fake device reports NO ERRORS (and the proxy, if any, ended every session cleanly); tinygpuhybridtest
# must also have timed its evaluations. Kernels are not run by the fake, so logL is wrong by design.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
require_no_launch_guard
LABEL=$1; shift
ENVS=(); while [ $# -gt 0 ] && [ "$1" != "--" ]; do ENVS+=("$1"); shift; done; [ "$1" = "--" ] && shift
TEST_BIN=${FAKE_TEST_BIN:-$TEST_BIN}
[ -x "$TEST_BIN" ] || { echo "no $TEST_BIN; build it first"; exit 2; }
SOCKDIR=$(mktemp -d /tmp/tgd.XXXXXX); SOCK="$SOCKDIR/dev.sock"
MEM="$TINYGPU_TEST_WORK/fake_device_$LABEL"; rm -rf "$MEM"; mkdir -p "$MEM"
DLOG="$TINYGPU_TEST_WORK/fake_device_$LABEL.log"; OUT="$TINYGPU_TEST_WORK/run_device_$LABEL.txt"; rm -f "$DLOG" "$OUT"
SRV=""; TST=""; T2=""; PRX=""
cleanup() { [ -n "$TST" ] && kill -KILL "$TST" 2>/dev/null; [ -n "$T2" ] && kill -KILL "$T2" 2>/dev/null
            [ -n "$PRX" ] && { kill -KILL "$PRX" 2>/dev/null; wait "$PRX" 2>/dev/null; }
            [ -n "$SRV" ] && { kill "$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null; }; rm -rf "$SOCKDIR"; }
trap cleanup EXIT
trap 'echo "[$LABEL] interrupted"; exit 130' INT TERM
"$BEAGLE_PYTHON" "$TG_TESTS/fake_nv_device.py" "$SOCK" "$MEM" > "$DLOG" 2>&1 &
SRV=$!
for i in $(seq 100); do grep -q listening "$DLOG" 2>/dev/null && break; sleep 0.1; done
grep -q listening "$DLOG" || { echo "the fake device did not start:"; cat "$DLOG"; exit 2; }
CLIENT_SOCK=$SOCK
if [ -n "$FAKE_TG_PROXY" ]; then
    PLOG="$TINYGPU_TEST_WORK/proxy_$LABEL.log"; CLIENT_SOCK="$SOCKDIR/px.sock"
    "$BEAGLE_PYTHON" "$TG_TESTS/replay/tgproxy.py" --listen "$CLIENT_SOCK" --upstream "$SOCK" --out "$FAKE_TG_PROXY" --label "$LABEL" \
        $([ "${FAKE_TG_GUARD:-0}" = 1 ] && echo --guard) > "$PLOG" 2>&1 &
    PRX=$!
    for i in $(seq 100); do grep -q "tgproxy listening" "$PLOG" 2>/dev/null && break; sleep 0.1; done
    grep -q "tgproxy listening" "$PLOG" || { echo "the proxy did not start:"; cat "$PLOG"; exit 2; }
    # what ran, for run_replay.sh: the same test and variables replay the recording
    "$BEAGLE_PYTHON" -c 'import json, sys; json.dump(dict(test_bin=sys.argv[1], envs=[e for e in sys.argv[2].split("\x1f") if e], args=sys.argv[3:]), sys.stdout)' \
        "$(basename "$TEST_BIN")" "$(IFS=$'\x1f'; echo "${ENVS[*]}")" "$@" > "$SOCKDIR/run.json"
fi
# TMPDIR: the run's own nv_usb4.lock (plan step P5). exec, so a background job's pid ($!) is the test's own, which the SIGINT and
# the watchdog's SIGKILL must reach (with FAKE_SIGINT_AFTER, perl comes before env: macOS strips DYLD_LIBRARY_PATH from a system
# binary's environment).
run_test() {
    exec ${FAKE_SIGINT_AFTER:+perl -e 'setpgrp(0, 0); exec @ARGV or die "exec: $!"'} env BEAGLE_TINYGPU_NO_LAUNCH=1 BEAGLE_TINYGPU_LOG="$TINYGPU_TEST_WORK/beagle_tinygpu_offline.log" APL_REMOTE_SOCK="$CLIENT_SOCK" \
        BEAGLE_NV_GUARD="$TG_TESTS/replay/crash_guard_wrap.sh" BEAGLE_TG_GUARD_BIN="$BEAGLE_BUILD/libhmsbeagle/GPU/CMake_TinyGPUHybrid/beagle-tinygpu-guard" BEAGLE_TG_GUARD_PIDFILE="$SOCKDIR/guard.pid" \
        BEAGLE_NV_PROFILE=1 DYLD_LIBRARY_PATH="$TEST_LIBS" TMPDIR="$SOCKDIR" "${ENVS[@]}" "$TEST_BIN" "$@"
}
run_test "$@" > "$OUT" 2>&1 &
TST=$!
TRIP=""
for i in $(seq ${FAKE_RUN_TIMEOUT:-300}); do
    kill -0 $TST 2>/dev/null || break
    if [ -n "$FAKE_SIGINT_AFTER" ] && grep -qE "$FAKE_SIGINT_AFTER" "$OUT"; then   # its group is its own pid (setpgrp above)
        FAKE_SIGINT_AFTER=; kill -INT -- -$TST; echo "[$LABEL] SIGINT sent to the test's process group"
    fi
    if [ -n "$FAKE_SECOND_AFTER" ] && grep -qE "$FAKE_SECOND_AFTER" "$OUT"; then
        FAKE_SECOND_AFTER=; OUT2="$TINYGPU_TEST_WORK/run_device_${LABEL}_second.txt"; s0=$(date +%s)
        run_test --reps 1 > "$OUT2" 2>&1 &
        T2=$!
        for j in $(seq 30); do kill -0 $T2 2>/dev/null || break; sleep 1; done
        if kill -0 $T2 2>/dev/null; then echo "[$LABEL] second process still running after 30 s; killing it (fake device only)"; kill -KILL $T2; fi
        wait $T2; rc2=$?; T2=""; echo "[$LABEL] second process exit=$rc2 after $(( $(date +%s) - s0 )) s (output: $OUT2)"
    fi
    if [ -n "$FAKE_TG_PROXY" ] && grep -q "FAIL-STOP" "$PLOG" 2>/dev/null; then   # the guard (or the proxy) stopped forwarding
        TRIP=$(grep -m1 "ended: failstop" "$PLOG" | sed 's/.*ended: failstop: //'); kill -KILL $TST 2>/dev/null; break
    fi
    sleep 1
done
if kill -0 $TST 2>/dev/null; then echo "[$LABEL] timed out; killing the test (fake device only)"; kill -KILL $TST; fi
wait $TST; RC=$?; TST=""
# the crash guard (plan steps C10, C11): it exits at the plugin's clean, or once its own decision tore the GPU down; one that
# holds (its log says whom to kill) keeps the fake's connection as it would the eGPU's, so it is ended here
GPID=$(cat "$SOCKDIR/guard.pid" 2>/dev/null)
if [ -n "$GPID" ]; then
    for i in $(seq 600); do
        kill -0 "$GPID" 2>/dev/null || break
        grep -q "then kill $GPID\." "$TINYGPU_TEST_WORK/beagle_tinygpu_offline.log" 2>/dev/null && break
        sleep 0.1
    done
    if kill -0 "$GPID" 2>/dev/null; then
        echo "[$LABEL] the guard (pid $GPID) held the fake connection; ending it"; kill -KILL "$GPID"
        for i in $(seq 50); do kill -0 "$GPID" 2>/dev/null || break; sleep 0.1; done   # not this shell's child: polled, not waited for
    fi
fi
if [ -n "$PRX" ]; then
    kill -TERM $PRX 2>/dev/null
    for i in $(seq 100); do kill -0 $PRX 2>/dev/null || break; sleep 0.1; done
    kill -0 $PRX 2>/dev/null && { echo "[$LABEL] the proxy did not end its recording"; kill -KILL $PRX; }
    wait $PRX 2>/dev/null; PRX=""
fi
sleep 0.3; kill $SRV 2>/dev/null; wait $SRV 2>/dev/null; SRV=""
[ -n "$FAKE_TG_PROXY" ] && [ -d "$FAKE_TG_PROXY" ] && cp "$SOCKDIR/run.json" "$FAKE_TG_PROXY/run.json"
echo "[$LABEL] tinygpuhybridtest exit=$RC (output: $OUT)"
grep -E "fake TinyGPU.app \((AD107|GB205) device\): " "$DLOG" | tail -2 | cut -c1-600
if [ -n "$TRIP" ]; then   # the run stopped at a refusal: its verdict is whether that refusal was the one expected
    echo "[$LABEL] the proxy stopped forwarding: $TRIP"
    [ -n "$FAKE_EXPECT_TRIP" ] && echo "$TRIP" | grep -qE "$FAKE_EXPECT_TRIP" && { echo "[$LABEL] PASS (the expected refusal)"; exit 0; }
    echo "[$LABEL] FAIL"; exit 1
fi
[ -n "$FAKE_EXPECT_TRIP" ] && { echo "[$LABEL] FAIL: expected a refusal matching '$FAKE_EXPECT_TRIP'; none came"; exit 1; }
missing=()
need() { grep -qE "$1" "$OUT" || missing+=("$2"); }
need "TinyGPU/NV: C\+\+ runtime: built the NVDevice after the C\+\+ boot, with no daemon" "a boot"
need "C\+\+ runtime: [1-9][0-9]* kernels loaded" "C++ program loading"
[ -n "$FAKE_TEST_BIN" ] || need "^per evaluation:" "timed evaluations"
fini_verdict "$OUT" || missing+=("a clean fini report")
grep -E "fake TinyGPU.app \((AD107|GB205) device\): " "$DLOG" | tail -1 | grep -q "NO ERRORS" || missing+=("fake device NO ERRORS")
if [ -n "$FAKE_TG_PROXY" ]; then
    grep -q "tgproxy: recording ended after" "$PLOG" && ! grep "tgproxy: session [0-9]* ended:" "$PLOG" | grep -qv " ended: eof;" \
        || missing+=("the proxy's clean sessions")
fi
if [ ${#missing[@]} -eq 0 ]; then echo "[$LABEL] PASS"; exit 0; fi
echo "[$LABEL] FAIL: missing ${missing[*]}"; exit 1
