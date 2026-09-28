#!/bin/bash
# HARDWARE: one L0 recording (TODO.md plan step V1): the plugin's real boot, run and teardown on the eGPU (the C++ boot, with no
# daemon: plan steps C11-C13), recorded through the recording proxy (replay/tgproxy.py, which starts TinyGPU.app if nothing
# listens, as tinygrad does), with the plugin's markers (BEAGLE_TG_MARKERS=1). Everything else is run_point.sh's: the eGPU must
# be cold (power-cycled) or torn down by the previous run; never Ctrl-C or kill a run; a hung or holding GPU (or a proxy that
# stopped forwarding) is unplugged before anything is killed. Run it under caffeinate -ims, with the lid open.
#   run_l0.sh <label> <state-count> [reps] [--poison] [--guard]
# (--guard runs the proxy in guard mode, replay/tgguard.py, for a new client's first run: a trigger it refuses is not forwarded
# and the proxy holds, so unplug the eGPU)
# L0_DRY_RUN=1 runs the same script offline, against fake_nv_device.py instead of TinyGPU.app, recording under
# $TINYGPU_TEST_WORK: a check of the script itself, which touches no eGPU.
# The recording goes to $BEAGLE_TINYGPU_DATA/recordings/<date>-<time>_<computer>_<label>/, with the test's output and run.json
# (for run_replay.sh, which replays it offline). It holds NVIDIA firmware (the GSP image, the falcon ucodes) and the VBIOS:
# never in git. Exits 0 only if run_point.sh would (the test passed, the fini report says the next boot needs no power cycle,
# the crash guard exited, log stream saw nothing from the eGPU) and the proxy ended every session at its client's close; 1
# stops the session's chain of runs; 2 means nothing was started.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
USAGE="usage: run_l0.sh <label> <state-count> [reps] [--poison] [--guard]"
LABEL=$1; N=$2; REPS=${3:-5}; FLAGS=(); GUARD=()
set -- "${@:4}"
while [ $# -gt 0 ]; do
    case $1 in
        --poison) FLAGS+=(--poison) ;;
        --guard) GUARD=(--guard) ;;
        *) echo "$USAGE"; exit 2 ;;
    esac
    shift
done
[[ "$LABEL" =~ ^[a-z0-9_]+$ ]] && [[ "$N" =~ ^[0-9]+$ ]] && [[ "$REPS" =~ ^[1-9][0-9]*$ ]] || { echo "$USAGE"; exit 2; }
# what can hold the GPU: the crash guard (plan step C10), or a proxy that stopped forwarding
pgrep -f "$GUARD_RE" > /dev/null && { echo "a crash guard is still running (it may hold the GPU); not running"; exit 2; }
pgrep -f "replay/tgproxy\.py" > /dev/null && { echo "a recording proxy is still running (it may hold the GPU); not running"; exit 2; }
hw_begin
DRY=${L0_DRY_RUN:-0}; OFFLINE_ENV=()
# the plugin skips its TinyGPU.app check when APL_REMOTE_SOCK names another server (the proxy): this is that check (decision 25)
APP=/Applications/TinyGPU.app
[ "$DRY" = 1 ] || for rel in Contents/MacOS/TinyGPU Contents/Library/SystemExtensions/org.tinygrad.tinygpu.driver2.dext/org.tinygrad.tinygpu.driver2; do
    want=$(grep -A1 "\"/$rel\"" "$GPU_DIR/TinyGPUTransport.h" | grep -oE '"[0-9a-f]{64}"' | tr -d '"')
    got=$(shasum -a 256 "$APP/$rel" 2>/dev/null | cut -d' ' -f1)
    [ -n "$want" ] && [ "$got" = "$want" ] || { echo "$APP/$rel is not TinyGPU release c0d024f9's (see TinyGPUTransport.h check_app); not running"; exit 2; }
done
if [ "$DRY" != 1 ]; then
    for i in $(seq 1 30); do [ "$(ioreg -l -w0 2>/dev/null | grep -c de100000)" -gt 0 ] && break; sleep 2; done
    [ "$(ioreg -l -w0 2>/dev/null | grep -c de100000)" -gt 0 ] || { echo "eGPU not enumerated; not running"; exit 2; }
    RUNS="$BEAGLE_TINYGPU_DATA/runs"; RECS="$BEAGLE_TINYGPU_DATA/recordings"
else RUNS="$TINYGPU_TEST_WORK/l0_dry/runs"; RECS="$TINYGPU_TEST_WORK/l0_dry/recordings"; fi
mkdir -p "$RUNS" "$RECS"
STAMP=$(date +%Y%m%d-%H%M%S)_$HW_HOST
REC="$RECS/${STAMP}_$LABEL"; [ -e "$REC" ] && { echo "$REC exists; not running"; exit 2; }
OUT="$RUNS/${STAMP}_L0_$LABEL.txt"; LS="$RUNS/${STAMP}_L0_${LABEL}_logstream.txt"; PLOG="$RUNS/${STAMP}_L0_${LABEL}_proxy.log"
PRIV=$(mktemp -d /tmp/tgl0.XXXXXX); PSOCK="$PRIV/px.sock"
UPSTREAM="${TMPDIR:-/tmp}"; UPSTREAM="${UPSTREAM%/}/tinygpu.sock"   # tinygrad's temp("tinygpu.sock"), where TinyGPU.app listens
START_APP=--start-app; FAKE=""
if [ "$DRY" = 1 ]; then   # the fake AD107 instead of TinyGPU.app, and this run's crash guard ended if it holds (its pid file)
    UPSTREAM="$PRIV/dev.sock"; START_APP=""
    OFFLINE_ENV=(BEAGLE_NV_GUARD="$TG_TESTS/replay/crash_guard_wrap.sh" BEAGLE_TG_GUARD_BIN="$BEAGLE_BUILD/libhmsbeagle/GPU/CMake_TinyGPUHybrid/beagle-tinygpu-guard"
                 BEAGLE_TG_GUARD_PIDFILE="$PRIV/guard.pid")
    "$BEAGLE_PYTHON" "$TG_TESTS/fake_nv_device.py" "$UPSTREAM" "$PRIV/mem" > "$RUNS/${STAMP}_L0_${LABEL}_fake_device.log" 2>&1 &
    FAKE=$!
    for i in $(seq 100); do grep -q listening "$RUNS/${STAMP}_L0_${LABEL}_fake_device.log" 2>/dev/null && break; sleep 0.1; done
fi
hw_logstream "$LS"
# the proxy in its own process group (a terminal's Ctrl-C cannot reach it; it ignores SIGINT and SIGHUP too)
set -m
"$BEAGLE_PYTHON" "$TG_TESTS/replay/tgproxy.py" --listen "$PSOCK" --upstream "$UPSTREAM" --out "$REC" $START_APP --label "$LABEL" "${GUARD[@]}" > "$PLOG" 2>&1 &
PRX=$!
set +m
for i in $(seq 100); do grep -q "tgproxy listening" "$PLOG" 2>/dev/null && break; kill -0 $PRX 2>/dev/null || break; sleep 0.1; done
grep -q "tgproxy listening" "$PLOG" || { echo "the proxy did not start (nothing was sent to the GPU):"; cat "$PLOG"; kill $PRX 2>/dev/null; exit 2; }
cd "$REPO"
# the harness's variables only here, never exported (hw_begin)
env APL_REMOTE_SOCK="$PSOCK" BEAGLE_TINYGPU_NO_LAUNCH=1 BEAGLE_TG_MARKERS=1 BEAGLE_NV_PROFILE=1 DYLD_LIBRARY_PATH="$TEST_LIBS" "${OFFLINE_ENV[@]}" \
    "$TEST_BIN" --state-count "$N" --reps "$REPS" --diag-compare-cpu "${FLAGS[@]}" > "$OUT" 2>&1
rc=$?
for i in $(seq 60); do pgrep -f "$GUARD_RE" > /dev/null || break; sleep 1; done   # it exits at the plugin's clean, or decides at EOF
if [ "$DRY" = 1 ] && [ -f "$PRIV/guard.pid" ] && kill -0 "$(cat "$PRIV/guard.pid")" 2>/dev/null; then
    echo "(dry run) the crash guard held the fake connection; ending it"; kill -KILL "$(cat "$PRIV/guard.pid")"
fi
hw_hold_check || { echo "the proxy (pid $PRX) keeps its connections too: after unplugging, kill -9 $PRX"; nohup caffeinate -ims -w $PRX > /dev/null 2>&1 & exit 1; }
# the proxy ends its recording at SIGTERM once the plugin's last session has ended; if it stopped forwarding it holds instead
kill -TERM $PRX 2>/dev/null
for i in $(seq 30); do kill -0 $PRX 2>/dev/null || break; sleep 1; done
if kill -0 $PRX 2>/dev/null; then
    nohup caffeinate -ims -w $PRX > /dev/null 2>&1 &
    grep -E "FAIL-STOP|ended: failstop" "$PLOG" | head -3
    echo "STOP: the recording proxy (pid $PRX) still holds the TinyGPU.app connection (log: $PLOG): unplug the eGPU first, then kill -9 $PRX"
    exit 1
fi
hw_logstream_stop; ls_ok=$?
if [ -d "$REC" ]; then
    cp "$OUT" "$REC/test_output.txt"
    "$BEAGLE_PYTHON" -c 'import json, sys; json.dump(dict(test_bin="tinygpuhybridtest", envs=[], args=sys.argv[1:]), sys.stdout)' \
        --state-count "$N" --reps "$REPS" --diag-compare-cpu "${FLAGS[@]}" > "$REC/run.json"
fi
[ -n "$FAKE" ] && { kill $FAKE 2>/dev/null; wait $FAKE 2>/dev/null; }
rm -rf "$PRIV"
echo "L0 $LABEL: N=$N reps=$REPS ${FLAGS[*]} exit=$rc output=$OUT recording=$REC"
grep -E "C\+\+ runtime:|per evaluation|maxAbsDiff|CPU-reference logL|^PASS|^FAIL|timed out|failed|rror" "$OUT" | grep -v "^  \[" | head -12
grep -E "GPU teardown|TinyGPU/NV: teardown:|no teardown result|keeps the TinyGPU.app" "$OUT"
grep -E "recording ended|session [0-9]+ ended" "$PLOG" | cut -c1-250
[ $ls_ok -eq 0 ] || { echo "STOP: log stream ended during the run ($LS): the eGPU check was blind: stop all hardware work"; exit 1; }
EVENTS=$(grep -cvE "$HW_LOG_BENIGN" "$LS")
[ "$EVENTS" -eq 0 ] || { echo "STOP: log stream saw $EVENTS eGPU event line(s) ($LS): stop all hardware work"; exit 1; }
fini_verdict "$OUT" || { echo "STOP: bad fini report (lines above): replug the eGPU before the next run"; exit 1; }
grep -q "tgproxy: recording ended after" "$PLOG" && ! grep "tgproxy: session [0-9]* ended:" "$PLOG" | grep -qv " ended: eof;" \
    || { echo "STOP: the proxy did not end every session at its client's close ($PLOG)"; exit 1; }
[ $rc -eq 0 ] || [ "$DRY" = 1 ] || { echo "STOP: the test failed (exit $rc); the teardown was clean"; exit 1; }   # the fake runs no kernels
echo "OK: PASS, recorded; WPR2 is down, the next boot needs no power cycle. Replay it offline: run_replay.sh $REC l0_$LABEL --record"
