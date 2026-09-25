#!/bin/bash
# Offline end-to-end run of the real plugin against fake_nv_daemon.py + fake_tinygpu_server.py. No GPU and no
# TinyGPU.app: BEAGLE_TINYGPU_NO_LAUNCH=1 makes the plugin fail rather than start the real app, and the test
# starts only once the fake is listening on a short per-run socket path (macOS sun_path is 104 bytes).
#   run_fake_runtime.sh <label> [VAR=value ...] -- [tinygpuhybridtest args ...]
# e.g. run_fake_runtime.sh runtime BEAGLE_NV_USE_DAEMON=0 -- --reps 20
# FAKE_TEST_BIN=<another BEAGLE example> runs that instead, with its own arguments (plan step D1; run_offline.sh), e.g.
#   FAKE_TEST_BIN=$BEAGLE_BUILD/examples/synthetictest run_fake_runtime.sh st BEAGLE_NV_USE_DAEMON=0 -- --rsrc 1 --manualscale
# The mode follows the variables (BEAGLE_NV_USE_DAEMON=0: C++ runtime; BEAGLE_NV_CPP_DISPATCH=1: C++ dispatch;
# otherwise the daemon path). Exit status 0 only if the run reached every stage that mode must reach and the fake
# server reports NO ERRORS. logL is wrong by design (kernels are not emulated), so the test's own exit status is
# not used.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
require_no_launch_guard
TEST_BIN=${FAKE_TEST_BIN:-$TEST_BIN}
LABEL=$1; shift
ENVS=(); while [ $# -gt 0 ] && [ "$1" != "--" ]; do ENVS+=("$1"); shift; done; [ "$1" = "--" ] && shift
[ -x "$TEST_BIN" ] || { echo "no $TEST_BIN; build tinygpuhybridtest first"; exit 2; }
MODE=daemon
for e in "${ENVS[@]}"; do
    [ "$e" = BEAGLE_NV_CPP_DISPATCH=1 ] && [ $MODE = daemon ] && MODE=dispatch
    [ "$e" = BEAGLE_NV_USE_DAEMON=0 ] && MODE=runtime
done

SOCKDIR=$(mktemp -d "${TMPDIR:-/tmp}/tg.XXXXXX"); SOCK="$SOCKDIR/fk.sock"
[ ${#SOCK} -lt 100 ] || { rmdir "$SOCKDIR"; SOCKDIR=$(mktemp -d /tmp/tg.XXXXXX); SOCK="$SOCKDIR/fk.sock"; }
MEM="$TINYGPU_TEST_WORK/fake_mem_$LABEL"; rm -rf "$MEM"; mkdir -p "$MEM"
SLOG="$TINYGPU_TEST_WORK/fake_server_$LABEL.log"; OUT="$TINYGPU_TEST_WORK/run_fake_$LABEL.txt"; rm -f "$SLOG" "$OUT"
SRV=""; TST=""
cleanup() { [ -n "$TST" ] && kill -KILL "$TST" 2>/dev/null; [ -n "$SRV" ] && { kill "$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null; }; rm -rf "$SOCKDIR"; }
trap cleanup EXIT
trap 'echo "[$LABEL] interrupted"; exit 130' INT TERM

"$BEAGLE_PYTHON" "$TG_TESTS/fake_tinygpu_server.py" "$SOCK" "$MEM" > "$SLOG" 2>&1 &
SRV=$!
for i in $(seq 100); do grep -q listening "$SLOG" 2>/dev/null && break; sleep 0.1; done
grep -q listening "$SLOG" || { echo "fake server did not start:"; cat "$SLOG"; exit 2; }

env BEAGLE_TINYGPU_NO_LAUNCH=1 APL_REMOTE_SOCK="$SOCK" FAKE_NV_MEM="$MEM" BEAGLE_NV_DISPATCH_DAEMON="$TG_TESTS/fake_nv_daemon.py" \
    BEAGLE_NV_PROFILE=1 BEAGLE_NV_SCRIPTS="$GPU_DIR" DYLD_LIBRARY_PATH="$TEST_LIBS" "${ENVS[@]}" \
    "$TEST_BIN" "$@" > "$OUT" 2>&1 &
TST=$!
for i in $(seq ${FAKE_RUN_TIMEOUT:-300}); do   # watchdog: only our own processes
    kill -0 $TST 2>/dev/null || break
    # FAKE_SIGINT_AFTER=<regex>: one SIGINT to the test 2 s after its output matches (plan step P3's handler)
    [ -n "$FAKE_SIGINT_AFTER" ] && grep -qE "$FAKE_SIGINT_AFTER" "$OUT" && { sleep 2; kill -INT $TST; FAKE_SIGINT_AFTER=; }
    sleep 1
done
# SIGKILL: the test turns SIGTERM into an orderly stop, which a hung test never reaches
if kill -0 $TST 2>/dev/null; then echo "[$LABEL] timed out; killing the test (fake GPU only)"; kill -KILL $TST; fi
wait $TST; RC=$?; TST=""
sleep 0.5; kill $SRV 2>/dev/null; wait $SRV 2>/dev/null; SRV=""
echo "[$LABEL] mode=$MODE tinygpuhybridtest exit=$RC (output: $OUT)"
cat "$SLOG"

missing=()
need() { grep -qE "$1" "$OUT" || missing+=("$2"); }
need "TinyGPU/NV: daemon booted" "daemon boot"
[ $MODE = runtime ] || need "TinyGPU/NV: compile_all — loaded [1-9]" "compile_all"
need "TinyGPU/NV: \[profile\] +launch_batch +n= +[1-9]" "C++ launch_batch profile"
[ -n "$FAKE_TEST_BIN" ] || need "^per evaluation:" "timed evaluations"
grep -qE "TinyGPU/NV: .*failed" "$OUT" && missing+=("(a 'failed' line was printed)")
case $MODE in
    daemon) need "\[nv_dispatch_daemon\] \[profile\] +cmd\.launch_batch +n= +[1-9]" "daemon launch_batch" ;;
    dispatch) need "C\+\+ dispatch: [1-9][0-9]* kernels handed over" "handoff" ;;
    runtime) need "C\+\+ runtime: embedded cubin SP_[0-9]+ sm_[0-9]+ \([0-9]+ bytes, [1-9][0-9]* kernels" "embedded cubin (plan step C1)"
             grep -q "compiling all kernels" "$OUT" && missing+=("(compile_all in the C++ runtime)")
             need "C\+\+ runtime: [1-9][0-9]* kernels loaded" "C++ program loading" ;;
esac
if [ $MODE != daemon ]; then
    grep "client done" "$SLOG" | tail -1 | grep -qE '"launches": [1-9]' || missing+=("launches seen by the fake GPU")
    # the state page at fini (plan step P3): nothing in flight, and the last value submitted is both the C++ timeline and
    # the fake GPU's release count (it releases 1, 2, ... with no gaps)
    rel=$(grep "client done" "$SLOG" | tail -1 | sed -nE 's/.*"releases": ([0-9]+).*/\1/p')
    need "C\+\+ state page: phase 1, frame_in_flight 0, last_submitted $rel, C\+\+ timeline signal $rel\$" "state page at fini"
else
    grep -q "state page" "$OUT" && missing+=("(a state page in daemon mode)")
fi
grep "fake TinyGPU.app: " "$SLOG" | tail -1 | grep -q "NO ERRORS" || missing+=("fake server NO ERRORS")
if [ ${#missing[@]} -eq 0 ]; then echo "[$LABEL] PASS"; exit 0; fi
echo "[$LABEL] FAIL: missing ${missing[*]}"; exit 1
