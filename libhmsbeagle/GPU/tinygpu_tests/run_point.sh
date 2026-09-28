#!/bin/bash
# HARDWARE: one tinygpuhybridtest run on the real eGPU. Boots the GPU, so the eGPU must be cold (power-cycled) or torn
# down by the previous run, as the teardown does by default (TODO.md plan step P3); a warm GPU is refused with nothing
# written. Never Ctrl-C or kill a run; a hung or holding GPU must be unplugged before anything is killed.
#   run_point.sh <state-count>[,<state-count>...] [default] [reps] [--poison] [--instances K] [--threads] [--cycles C] [--kill idle] [--exit-after MS]
# (default, the only mode since plan step C13c: the plugin's C++ boot and runtime, with no daemon and no Python; the slot stays
# so that earlier command lines keep their shape, and a removed mode is refused. The list, --instances, --threads and
# --cycles: several instances in one process, TODO.md plan step P5; --exit-after MS: MS into the evaluations another thread
# calls exit(0), as a host's shutdown would, plan step C12; --kill idle: the plugin SIGKILLs itself at fini once the GPU is
# idle, as a crash would (BEAGLE_NV_TEST_KILL=idle), and the crash guard tears the GPU down, plan step C10; the run then passes
# if the test died of the SIGKILL and the guard's TinyGPULog lines say its teardown left the next boot needing no power cycle,
# guard_verdict, env.sh)
# Waits for the eGPU to enumerate, runs from the build tree with --diag-compare-cpu under log stream, keeps the output and
# the log stream under $BEAGLE_TINYGPU_DATA/runs/, and prints a summary. Exits 0 only if the test passed, the fini report
# says the next boot needs no power cycle (fini_verdict, env.sh), the crash guard exited and log stream saw nothing from the
# eGPU; 1 stops a chain of runs after a run, 2 means nothing was started.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
N=$1; MODE=${2:-default}; REPS=${3:-200}; shift $(( $# < 3 ? $# : 3 ))
USAGE="usage: run_point.sh <state-count>[,<state-count>...] [default] [reps] [--poison] [--instances K] [--threads] [--cycles C] [--kill idle] [--exit-after MS]"
FLAGS=(); KILL=
while [ $# -gt 0 ]; do
    case $1 in
        --poison|--threads) FLAGS+=("$1") ;;
        --instances|--cycles) [[ "$2" =~ ^[1-9][0-9]*$ ]] || { echo "$USAGE"; exit 2; }; FLAGS+=("$1" "$2"); shift ;;
        --exit-after) [[ "$2" =~ ^[0-9]+$ ]] || { echo "$USAGE"; exit 2; }; FLAGS+=("$1" "$2"); shift ;;
        --kill) [ "$2" = idle ] || { echo "$USAGE"; exit 2; }; KILL=$2; shift ;;
        *) echo "$USAGE"; exit 2 ;;
    esac
    shift
done
[[ "$N" =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo "$USAGE"; exit 2; }
[ "$MODE" = default ] || { echo "mode $MODE was removed in TODO.md plan step C13c: the plugin's only path is the C++ boot (default)"; exit 2; }
KILL_ENV=(); [ -n "$KILL" ] && KILL_ENV=(BEAGLE_NV_TEST_KILL=$KILL)
pgrep -f "$GUARD_RE" > /dev/null && { echo "a crash guard is still running (it may hold the GPU); not running"; exit 2; }
hw_begin
for i in $(seq 1 30); do [ "$(ioreg -l -w0 2>/dev/null | grep -c de100000)" -gt 0 ] && break; sleep 2; done
if [ "$(ioreg -l -w0 2>/dev/null | grep -c de100000)" -eq 0 ]; then echo "eGPU not enumerated; not running"; exit 2; fi
RUNS="$BEAGLE_TINYGPU_DATA/runs"; mkdir -p "$RUNS"
STAMP=$(date +%Y%m%d-%H%M%S)_$HW_HOST; OUT="$RUNS/${STAMP}_N${N}_${MODE}.txt"; LS="$RUNS/${STAMP}_N${N}_${MODE}_logstream.txt"
hw_logstream "$LS"
TGLOG="$HOME/Library/Logs/beagle_tinygpu.log"; TGLOG_N=$(cat "$TGLOG" 2>/dev/null | wc -l)   # this run's TinyGPULog lines follow
cd "$REPO"
env "${KILL_ENV[@]}" BEAGLE_NV_PROFILE=1 DYLD_LIBRARY_PATH="$TEST_LIBS" \
    "$TEST_BIN" --state-count "$N" --reps "$REPS" --diag-compare-cpu "${FLAGS[@]}" > "$OUT" 2>&1
rc=$?
for i in $(seq 60); do pgrep -f "$GUARD_RE" > /dev/null || break; sleep 1; done   # it exits at the plugin's clean, or decides at EOF
hw_logstream_stop; ls_ok=$?
echo "N=$N mode=$MODE ${FLAGS[*]} exit=$rc output=$OUT"
grep -E "C\+\+ runtime:|load_programs|\] +(launch_batch|h2d|d2h) |launches in|maxAbsDiff|First mismatching|CPU-reference logL|per evaluation|repeats|^instance [0-9]+ \(|^tips:|^PASS|^FAIL|timed out|failed|rror" "$OUT" | grep -v "^  \[" | head -30
grep -E "GPU teardown|TinyGPU/NV: teardown:|no teardown result|keeps the TinyGPU.app" "$OUT"
hw_hold_check || exit 1
[ $ls_ok -eq 0 ] || { echo "STOP: log stream ended during the run ($LS): the eGPU check was blind: stop all hardware work"; exit 1; }
# eGPU events: every line but the filter's header, the column header log stream prints before its first event, and the
# Apple Neural Engine's and camera's own buffer messages, which match "DART" (dartMapBase) and have nothing to do with the eGPU
EVENTS=$(grep -cvE "$HW_LOG_BENIGN" "$LS")
[ "$EVENTS" -eq 0 ] || { echo "STOP: log stream saw $EVENTS eGPU event line(s) ($LS): stop all hardware work"; exit 1; }
if [ -n "$KILL" ]; then   # plan step C10: the plugin died before its fini report; the crash guard's report decides
    tail -n +$((TGLOG_N + 1)) "$TGLOG" > "$RUNS/${STAMP}_N${N}_${MODE}_tinygpulog.txt"
    grep -E "BEAGLE_NV_TEST_KILL|guard" "$RUNS/${STAMP}_N${N}_${MODE}_tinygpulog.txt" | cut -c1-400
    [ $rc -eq 137 ] || { echo "STOP: the test did not die of the SIGKILL (exit $rc)"; exit 1; }
    guard_verdict "$RUNS/${STAMP}_N${N}_${MODE}_tinygpulog.txt" \
        || { echo "STOP: the crash guard's teardown report is not clean (lines above): replug the eGPU before the next run"; exit 1; }
    echo "OK: killed $KILL; the crash guard tore the GPU down: WPR2 is down, the next boot needs no power cycle"
    exit 0
fi
fini_verdict "$OUT" || { echo "STOP: bad fini report (lines above): replug the eGPU before the next run"; exit 1; }
[ $rc -eq 0 ] || { echo "STOP: the test failed (exit $rc); the teardown was clean"; exit 1; }
echo "OK: PASS; WPR2 is down, the next boot needs no power cycle"
