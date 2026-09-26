#!/bin/bash
# Offline end-to-end run of the real plugin AND the real daemon (tinygrad's boot, nv_init_helper's patches, the P2 teardown)
# against fake_nv_device.py, a fake TinyGPU.app playing an AD107 at the register level (TODO.md plan step V1). No GPU and no
# TinyGPU.app: BEAGLE_TINYGPU_NO_LAUNCH=1, a short per-run socket, and the daemon started through replay/tgdaemon.py with
# BEAGLE_TG_OFFLINE=1 (tgharness_py's pre-connect patches). FAKE_TG_PROXY=<new recording dir> puts the recording proxy in
# between; BEAGLE_TG_RECORD=1 adds the recording shim. FAKE_TG_GUARD=1 runs that proxy in guard mode (replay/tgguard.py): a
# refusal ends the run at once (the proxy holds, as it would on the eGPU; offline everything is then ended), and the run
# passes only if FAKE_EXPECT_TRIP=<regex> is set and matches the guard's reason.
#   run_fake_device.sh <label> [VAR=value ...] -- [tinygpuhybridtest args ...]
# Exit status 0 only if the test booted once, ran in the C++ runtime, the teardown says the next boot needs no power cycle
# (fini_verdict), and the fake device reports NO ERRORS (and the proxy, if any, ended every session cleanly). Kernels are not
# run by the fake, so logL is wrong by design.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
require_no_launch_guard
LABEL=$1; shift
ENVS=(); while [ $# -gt 0 ] && [ "$1" != "--" ]; do ENVS+=("$1"); shift; done; [ "$1" = "--" ] && shift
[ -x "$TEST_BIN" ] || { echo "no $TEST_BIN; build tinygpuhybridtest first"; exit 2; }
SOCKDIR=$(mktemp -d /tmp/tgd.XXXXXX); SOCK="$SOCKDIR/dev.sock"
MEM="$TINYGPU_TEST_WORK/fake_device_$LABEL"; rm -rf "$MEM"; mkdir -p "$MEM"
DLOG="$TINYGPU_TEST_WORK/fake_device_$LABEL.log"; OUT="$TINYGPU_TEST_WORK/run_device_$LABEL.txt"; rm -f "$DLOG" "$OUT"
SRV=""; TST=""; PRX=""
cleanup() { [ -n "$TST" ] && kill -KILL "$TST" 2>/dev/null; [ -n "$PRX" ] && { kill -KILL "$PRX" 2>/dev/null; wait "$PRX" 2>/dev/null; }
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
env BEAGLE_TINYGPU_NO_LAUNCH=1 APL_REMOTE_SOCK="$CLIENT_SOCK" BEAGLE_NV_DISPATCH_DAEMON="$TG_TESTS/replay/tgdaemon.py" BEAGLE_TG_OFFLINE=1 BEAGLE_TG_DAEMON_PIDFILE="$SOCKDIR/daemon.pid" \
    BEAGLE_NV_PROFILE=1 BEAGLE_NV_SCRIPTS="$GPU_DIR" DYLD_LIBRARY_PATH="$TEST_LIBS" TMPDIR="$SOCKDIR" BEAGLE_NV_USE_DAEMON=0 "${ENVS[@]}" \
    "$TEST_BIN" "$@" > "$OUT" 2>&1 &
TST=$!
TRIP=""
for i in $(seq ${FAKE_RUN_TIMEOUT:-300}); do
    kill -0 $TST 2>/dev/null || break
    if [ -n "$FAKE_TG_PROXY" ] && grep -q "FAIL-STOP" "$PLOG" 2>/dev/null; then   # the guard (or the proxy) stopped forwarding
        TRIP=$(grep -m1 "ended: failstop" "$PLOG" | sed 's/.*ended: failstop: //'); kill -KILL $TST 2>/dev/null; break
    fi
    sleep 1
done
if kill -0 $TST 2>/dev/null; then echo "[$LABEL] timed out; killing the test (fake device only)"; kill -KILL $TST; fi
wait $TST; RC=$?; TST=""
# this run's daemon (its pid file): it exits after fini; one still running holds the fake's connection, as it would hold the
# eGPU's after a failure (unplug first, then kill), so it is ended here: offline it holds nothing but this run's fake
DPID=$(cat "$SOCKDIR/daemon.pid" 2>/dev/null)
if [ -n "$DPID" ]; then
    for i in $(seq 50); do kill -0 "$DPID" 2>/dev/null || break; sleep 0.1; done
    kill -0 "$DPID" 2>/dev/null && { echo "[$LABEL] the daemon (pid $DPID) held the fake connection; ending it"; kill -KILL "$DPID"; }
fi   # the daemon exits after fini
if [ -n "$PRX" ]; then
    kill -TERM $PRX 2>/dev/null
    for i in $(seq 100); do kill -0 $PRX 2>/dev/null || break; sleep 0.1; done
    kill -0 $PRX 2>/dev/null && { echo "[$LABEL] the proxy did not end its recording"; kill -KILL $PRX; }
    wait $PRX 2>/dev/null; PRX=""
fi
sleep 0.3; kill $SRV 2>/dev/null; wait $SRV 2>/dev/null; SRV=""
[ -n "$FAKE_TG_PROXY" ] && [ -d "$FAKE_TG_PROXY" ] && cp "$SOCKDIR/run.json" "$FAKE_TG_PROXY/run.json"
cp ~/Library/Logs/nv_dispatch_daemon.log "$TINYGPU_TEST_WORK/run_device_${LABEL}_daemon.log" 2>/dev/null
echo "[$LABEL] tinygpuhybridtest exit=$RC (output: $OUT, daemon log: run_device_${LABEL}_daemon.log)"
grep -E "fake TinyGPU.app \(AD107 device\): " "$DLOG" | tail -2 | cut -c1-600
if [ -n "$TRIP" ]; then   # the run stopped at a refusal: its verdict is whether that refusal was the one expected
    echo "[$LABEL] the proxy stopped forwarding: $TRIP"
    [ -n "$FAKE_EXPECT_TRIP" ] && echo "$TRIP" | grep -qE "$FAKE_EXPECT_TRIP" && { echo "[$LABEL] PASS (the expected refusal)"; exit 0; }
    echo "[$LABEL] FAIL"; exit 1
fi
[ -n "$FAKE_EXPECT_TRIP" ] && { echo "[$LABEL] FAIL: expected a refusal matching '$FAKE_EXPECT_TRIP'; none came"; exit 1; }
missing=()
need() { grep -qE "$1" "$OUT" || missing+=("$2"); }
need "TinyGPU/NV: daemon booted" "daemon boot"
need "C\+\+ runtime: [1-9][0-9]* kernels loaded" "C++ program loading"
need "^per evaluation:" "timed evaluations"
fini_verdict "$OUT" || missing+=("a clean fini report")
grep "fake TinyGPU.app (AD107 device): " "$DLOG" | tail -1 | grep -q "NO ERRORS" || missing+=("fake device NO ERRORS")
if [ -n "$FAKE_TG_PROXY" ]; then
    grep -q "tgproxy: recording ended after" "$PLOG" && ! grep "tgproxy: session [0-9]* ended:" "$PLOG" | grep -qv " ended: eof;" \
        || missing+=("the proxy's clean sessions")
fi
if [ ${#missing[@]} -eq 0 ]; then echo "[$LABEL] PASS"; exit 0; fi
echo "[$LABEL] FAIL: missing ${missing[*]}"; exit 1
