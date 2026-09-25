#!/bin/bash
# HARDWARE: one tinygpuhybridtest run on the real eGPU. Boots the GPU, so the eGPU must be cold (power-cycled) or torn
# down by the previous run, as the teardown does by default (TODO.md plan step P3); a warm GPU is refused with nothing
# written. Never Ctrl-C or kill a run; a hung or holding GPU must be unplugged before anything is killed.
#   run_point.sh <state-count> [cpp|daemon|runtime] [reps] [--poison]
# Waits for the eGPU to enumerate, runs from the build tree with --diag-compare-cpu under log stream, keeps the output,
# the daemon log and the log stream under $BEAGLE_TINYGPU_DATA/runs/, and prints a summary. Exits 0 only if the test
# passed, the fini report says the next boot needs no power cycle (fini_verdict, env.sh), the daemon exited and log
# stream saw nothing from the eGPU; 1 stops a chain of runs after a run, 2 means nothing was started.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
N=$1; MODE=${2:-runtime}; REPS=${3:-200}; POISON=$4
[ -n "$N" ] && [[ -z "$POISON" || "$POISON" = --poison ]] || { echo "usage: run_point.sh <state-count> [cpp|daemon|runtime] [reps] [--poison]"; exit 2; }
case $MODE in
    cpp) MODE_ENV=BEAGLE_NV_CPP_DISPATCH=1 ;;
    runtime) MODE_ENV=BEAGLE_NV_USE_DAEMON=0 ;;
    daemon) MODE_ENV=BEAGLE_NV_CPP_DISPATCH=0 ;;
    *) echo "unknown mode $MODE"; exit 2 ;;
esac
# what can hold the GPU: the daemon (<python> .../nv_dispatch_daemon.py <fd> [<fd>]) or a holding nv_teardown_diag.py
DAEMON_RE="nv_dispatch_daemon.py [0-9]|nv_teardown_diag.py"
pgrep -f "$DAEMON_RE" > /dev/null && { echo "an nv_dispatch_daemon or nv_teardown_diag is still running (it may hold the GPU); not running"; exit 2; }
for i in $(seq 1 30); do [ "$(ioreg -l -w0 2>/dev/null | grep -c de100000)" -gt 0 ] && break; sleep 2; done
if [ "$(ioreg -l -w0 2>/dev/null | grep -c de100000)" -eq 0 ]; then echo "eGPU not enumerated; not running"; exit 2; fi
RUNS="$BEAGLE_TINYGPU_DATA/runs"; mkdir -p "$RUNS"
STAMP=$(date +%Y%m%d-%H%M%S); OUT="$RUNS/${STAMP}_N${N}_${MODE}.txt"; LS="$RUNS/${STAMP}_N${N}_${MODE}_logstream.txt"
# the filter of every hardware run so far (STATUS.md R17, R18): with nothing from the eGPU it prints only its header
log stream --predicate 'composedMessage CONTAINS[c] "DART" OR composedMessage CONTAINS[c] "apciec" OR process CONTAINS[c] "tinygpu" OR composedMessage CONTAINS[c] "panic"' > "$LS" 2>&1 &
LSP=$!; trap 'kill $LSP 2>/dev/null' EXIT
for i in $(seq 50); do grep -q "^Filtering the log data" "$LS" && break; sleep 0.1; done
grep -q "^Filtering the log data" "$LS" || { echo "log stream did not attach; not running"; exit 2; }
cd "$REPO"
env $MODE_ENV BEAGLE_NV_PROFILE=1 BEAGLE_NV_SCRIPTS="$GPU_DIR" DYLD_LIBRARY_PATH="$TEST_LIBS" \
    "$TEST_BIN" --state-count "$N" --reps "$REPS" --diag-compare-cpu $POISON > "$OUT" 2>&1
rc=$?
for i in $(seq 60); do pgrep -f "$DAEMON_RE" > /dev/null || break; sleep 1; done   # it exits after fini, or decides at EOF
sleep 2; kill $LSP 2>/dev/null   # let log stream flush
cp ~/Library/Logs/nv_dispatch_daemon.log "$RUNS/${STAMP}_N${N}_${MODE}_daemon.log" 2>/dev/null
echo "N=$N mode=$MODE $POISON exit=$rc output=$OUT"
grep -E "C\+\+ dispatch:|C\+\+ runtime:|load_programs|\] +(launch_batch|h2d|d2h) |launches in|maxAbsDiff|First mismatching|CPU-reference logL|per evaluation|repeats|^PASS|^FAIL|timed out|failed|rror" "$OUT" | grep -v "^  \[" | head -30
grep -E "GPU teardown|TinyGPU/NV: teardown:|no teardown result|keeps the TinyGPU.app|fini round trip" "$OUT"
if pid=$(pgrep -f "$DAEMON_RE"); then echo "STOP: pid $pid (nv_dispatch_daemon or nv_teardown_diag) is still running and may hold the GPU: unplug the eGPU first, then kill $pid"; exit 1; fi
# eGPU events: every line but the filter's header, the column header log stream prints before its first event, and the
# Apple Neural Engine's and camera's own buffer messages, which match "DART" (dartMapBase) and have nothing to do with the eGPU
EVENTS=$(grep -cvE "^Filtering the log data|^Timestamp +Thread|\(AppleH11ANEInterface\) ANE0:|H13Cam" "$LS")
[ "$EVENTS" -eq 0 ] || { echo "STOP: log stream saw $EVENTS eGPU event line(s) ($LS): stop all hardware work"; exit 1; }
fini_verdict "$OUT" || { echo "STOP: bad fini report (lines above): replug the eGPU before the next run"; exit 1; }
[ $rc -eq 0 ] || { echo "STOP: the test failed (exit $rc); the teardown was clean"; exit 1; }
echo "OK: PASS; WPR2 is down, the next boot needs no power cycle"
