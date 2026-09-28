#!/bin/bash
# Replays a V1 recording to the unmodified plugin (TODO.md plan step V1), which boots the GPU itself (plan steps C11-C13):
# replay/tgreplay.py serves the recording on a private socket, and the recorded test runs again with the recorded variables
# (run.json in the recording; the harness's own, and the mode variables plan step C13c removed, are not replayed). Nothing
# reaches TinyGPU.app or the eGPU: BEAGLE_TINYGPU_NO_LAUNCH=1, the plugin's lock in a private TMPDIR. A recording made through
# the daemon (the L0 recordings, before plan step C11) replays to the C++ boot as it is: the same requests.
#   run_replay.sh <recording dir> [label] [--mutate NAME] [--guard [--guard-defect NAME]] [--record] [--out <replay recording dir>]
# --record adds the plugin's markers (for a recording made with them: the markers are then compared). Exit status 0 only if
# every recorded session replayed exactly (tgreplay's verdict); the test's own result is not used (the replay does not
# reproduce what the GPU wrote into big buffers, e.g. results, so logL can be wrong).
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
require_no_launch_guard
REC=$1; shift; [ -f "$REC/events.bin" ] && [ -f "$REC/run.json" ] || { echo "usage: run_replay.sh <recording dir with events.bin and run.json> [label] [--mutate NAME] [--guard] [--record] [--out DIR]"; exit 2; }
LABEL=replay; [ $# -gt 0 ] && [[ "$1" != --* ]] && { LABEL=$1; shift; }
XARGS=(); RECORD_ENV=()
while [ $# -gt 0 ]; do
    case $1 in
        --mutate) XARGS+=(--mutate "$2"); shift ;;
        --guard) XARGS+=(--guard) ;;
        --guard-defect) XARGS+=(--guard-defect "$2"); shift ;;
        --out) XARGS+=(--out "$2"); shift ;;
        --record) RECORD_ENV=(BEAGLE_TG_MARKERS=1) ;;
        *) echo "unknown option $1"; exit 2 ;;
    esac
    shift
done
RUN=(); while IFS= read -r line; do RUN+=("$line"); done < <("$BEAGLE_PYTHON" -c 'import json, sys; r = json.load(open(sys.argv[1])); print(r["test_bin"]); print(len(r["envs"])); [print(x) for x in r["envs"] + r["args"]]' "$REC/run.json")   # bash 3.2: no readarray
BIN="$BEAGLE_BUILD/examples/${RUN[0]}"; NENV=${RUN[1]}; ARGS=("${RUN[@]:$((2 + NENV))}")
ENVS=(); for e in "${RUN[@]:2:$NENV}"; do
    case $e in BEAGLE_TG_*|BEAGLE_NV_CPP_LEVEL=*|BEAGLE_NV_USE_DAEMON=*|BEAGLE_NV_CPP_DISPATCH=*) ;; *) ENVS+=("$e") ;; esac
done
[ -x "$BIN" ] || { echo "no $BIN"; exit 2; }
SOCKDIR=$(mktemp -d /tmp/tgr.XXXXXX); SOCK="$SOCKDIR/rp.sock"
MEM="$TINYGPU_TEST_WORK/replay_mem_$LABEL"; rm -rf "$MEM"; mkdir -p "$MEM"
RLOG="$TINYGPU_TEST_WORK/replay_$LABEL.log"; OUT="$TINYGPU_TEST_WORK/run_replay_$LABEL.txt"; rm -f "$RLOG" "$OUT"
SRV=""; TST=""
cleanup() { [ -n "$TST" ] && kill -KILL "$TST" 2>/dev/null; [ -n "$SRV" ] && { kill -KILL "$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null; }; rm -rf "$SOCKDIR"; }
trap cleanup EXIT
trap 'echo "[$LABEL] interrupted"; exit 130' INT TERM
"$BEAGLE_PYTHON" "$TG_TESTS/replay/tgreplay.py" --listen "$SOCK" --rec "$REC" --mem "$MEM" "${XARGS[@]}" > "$RLOG" 2>&1 &
SRV=$!
for i in $(seq 300); do grep -q "tgreplay listening" "$RLOG" 2>/dev/null && break; kill -0 $SRV 2>/dev/null || break; sleep 0.1; done
grep -q "tgreplay listening" "$RLOG" || { echo "tgreplay did not start:"; cat "$RLOG"; exit 2; }
env BEAGLE_TINYGPU_NO_LAUNCH=1 BEAGLE_TINYGPU_LOG="$TINYGPU_TEST_WORK/beagle_tinygpu_offline.log" APL_REMOTE_SOCK="$SOCK" \
    BEAGLE_NV_GUARD="$TG_TESTS/replay/crash_guard_wrap.sh" BEAGLE_TG_GUARD_BIN="$BEAGLE_BUILD/libhmsbeagle/GPU/CMake_TinyGPUHybrid/beagle-tinygpu-guard" BEAGLE_TG_GUARD_PIDFILE="$SOCKDIR/guard.pid" \
    BEAGLE_NV_PROFILE=1 DYLD_LIBRARY_PATH="$TEST_LIBS" TMPDIR="$SOCKDIR" "${ENVS[@]}" "${RECORD_ENV[@]}" \
    "$BIN" "${ARGS[@]}" > "$OUT" 2>&1 &
TST=$!
for i in $(seq ${FAKE_RUN_TIMEOUT:-300}); do kill -0 $TST 2>/dev/null || break; sleep 1; done
if kill -0 $TST 2>/dev/null; then echo "[$LABEL] timed out; killing the test (replay only)"; kill -KILL $TST; fi
wait $TST; RC=$?; TST=""
# the crash guard: it exits at the plugin's clean; one that holds after a failed replay keeps the replay's connection, as it
# would the eGPU's, so it is ended here: offline it holds nothing but this run's replay
GPID=$(cat "$SOCKDIR/guard.pid" 2>/dev/null)
if [ -n "$GPID" ]; then
    for i in $(seq 100); do
        kill -0 "$GPID" 2>/dev/null || break
        grep -q "then kill $GPID\." "$TINYGPU_TEST_WORK/beagle_tinygpu_offline.log" 2>/dev/null && break   # it said it holds
        sleep 0.1
    done
    kill -0 "$GPID" 2>/dev/null && { echo "[$LABEL] the guard (pid $GPID) held the replay's connection; ending it"; kill -KILL "$GPID"; }
fi
for i in $(seq 100); do kill -0 $SRV 2>/dev/null || break; sleep 0.1; done   # tgreplay exits after the recording's last session
kill -0 $SRV 2>/dev/null && { echo "[$LABEL] tgreplay still waiting for a session the client never opened"; kill -KILL $SRV; }
wait $SRV; VRC=$?; SRV=""
echo "[$LABEL] test exit=$RC (output: $OUT); replay log: $RLOG"
grep -E "^replay session|^tgreplay: (PASS|FAIL)|first divergence|    seq " "$RLOG" | head -40
[ $VRC -eq 0 ] && echo "[$LABEL] PASS" || echo "[$LABEL] FAIL"
exit $VRC
