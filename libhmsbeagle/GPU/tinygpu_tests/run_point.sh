#!/bin/bash
# HARDWARE: one tinygpuhybridtest run on the real eGPU. Boots the GPU, so the eGPU must be cold (power-cycled) or torn
# down by the previous run, as the teardown does by default (TODO.md plan step P3); a warm GPU is refused with nothing
# written. Never Ctrl-C or kill a run; a hung or holding GPU must be unplugged before anything is killed.
#   run_point.sh <state-count>[,<state-count>...] [default] [reps] [--poison] [--double] [--instances K] [--threads] [--cycles C] [--kill idle] [--exit-after MS] [--oom-pool MB] [--leave-warm] [--recover]
# (default, the only mode since plan step C13c: the plugin's C++ boot and runtime, with no daemon and no Python; the slot stays
# so that earlier command lines keep their shape, and a removed mode is refused. The list, --instances, --threads and
# --cycles: several instances in one process, TODO.md plan step P5; --exit-after MS: MS into the evaluations another thread
# calls exit(0), as a host's shutdown would, plan step C12; --kill idle: the plugin SIGKILLs itself at fini once the GPU is
# idle, as a crash would (BEAGLE_NV_TEST_KILL=idle), and the crash guard tears the GPU down, plan step C10; the run then passes
# if the test died of the SIGKILL and the guard's TinyGPULog lines say its teardown left the next boot needing no power cycle,
# guard_verdict, env.sh); --double: the TinyGPU instances in double precision, plan step C16; --oom-pool MB: plan step M1's check, a
# VRAM pool of MB MiB (BEAGLE_NV_DATA_MB, for this run only) that the GPU cannot hold, so the run passes only if
# beagleCreateInstance returns BEAGLE_ERROR_OUT_OF_MEMORY, said, and the GPU is torn down cleanly; --leave-warm and --recover,
# plan step P4 on Ada: --leave-warm runs with BEAGLE_NV_TEARDOWN=0 and passes only if the GSP confirmed its unload and WPR2
# stayed up, the GPU left warm on purpose; --recover runs with BEAGLE_NV_RECOVER=1 and passes only if the boot tore that warm
# GPU down first, then everything else held; not the two together: BEAGLE_NV_TEARDOWN=0 leaves out the teardown's images, so
# the boot refuses the recovery, and a second recovery needs a --leave-warm run between)
# Waits for the eGPU to enumerate, runs from the build tree with --diag-compare-cpu under log stream, keeps the output and
# the log stream under $BEAGLE_TINYGPU_DATA/runs/, and prints a summary. Exits 0 only if the test passed, the fini report
# says the next boot needs no power cycle (fini_verdict, env.sh), the crash guard exited and log stream saw nothing from the
# eGPU; 1 stops a chain of runs after a run, 2 means nothing was started, 3 an --oom-pool run without its refusal.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
N=$1; MODE=${2:-default}; REPS=${3:-200}; shift $(( $# < 3 ? $# : 3 ))
USAGE="usage: run_point.sh <state-count>[,<state-count>...] [default] [reps] [--poison] [--double] [--instances K] [--threads] [--cycles C] [--kill idle] [--exit-after MS] [--oom-pool MB] [--leave-warm] [--recover]"
FLAGS=(); KILL=; OOM_POOL=; LEAVE_WARM=; RECOVER=
while [ $# -gt 0 ]; do
    case $1 in
        --poison|--threads|--double) FLAGS+=("$1") ;;
        --instances|--cycles) [[ "$2" =~ ^[1-9][0-9]*$ ]] || { echo "$USAGE"; exit 2; }; FLAGS+=("$1" "$2"); shift ;;
        --exit-after) [[ "$2" =~ ^[0-9]+$ ]] || { echo "$USAGE"; exit 2; }; FLAGS+=("$1" "$2"); shift ;;
        --kill) [ "$2" = idle ] || { echo "$USAGE"; exit 2; }; KILL=$2; shift ;;
        --oom-pool) [[ "$2" =~ ^[1-9][0-9]*$ ]] || { echo "$USAGE"; exit 2; }; OOM_POOL=$2; shift ;;
        --leave-warm) LEAVE_WARM=1 ;;
        --recover) RECOVER=1 ;;
        *) echo "$USAGE"; exit 2 ;;
    esac
    shift
done
[[ "$N" =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo "$USAGE"; exit 2; }
[ "$MODE" = default ] || { echo "mode $MODE was removed in TODO.md plan step C13c: the plugin's only path is the C++ boot (default)"; exit 2; }
TEST_ENV=(); [ -n "$KILL" ] && TEST_ENV=(BEAGLE_NV_TEST_KILL=$KILL)
[ -n "$OOM_POOL" ] && TEST_ENV+=(BEAGLE_NV_DATA_MB=$OOM_POOL)   # hw_begin refuses these set outside: only the options set them
[ -n "$LEAVE_WARM" ] && TEST_ENV+=(BEAGLE_NV_TEARDOWN=0)
[ -n "$RECOVER" ] && TEST_ENV+=(BEAGLE_NV_RECOVER=1)
[ -n "$LEAVE_WARM$RECOVER" ] && { [ -z "$KILL$OOM_POOL" ] || { echo "--leave-warm and --recover take no --kill or --oom-pool"; exit 2; }; }
[ -n "$LEAVE_WARM" ] && [ -n "$RECOVER" ] && { echo "--leave-warm and --recover cannot go together: the boot refuses a recovery under BEAGLE_NV_TEARDOWN=0"; exit 2; }
pgrep -f "$GUARD_RE" > /dev/null && { echo "a crash guard is still running (it may hold the GPU); not running"; exit 2; }
hw_begin
for i in $(seq 1 30); do [ "$(ioreg -l -w0 2>/dev/null | grep -c de100000)" -gt 0 ] && break; sleep 2; done
if [ "$(ioreg -l -w0 2>/dev/null | grep -c de100000)" -eq 0 ]; then echo "eGPU not enumerated; not running"; exit 2; fi
RUNS="$BEAGLE_TINYGPU_DATA/runs"; mkdir -p "$RUNS"
STAMP=$(date +%Y%m%d-%H%M%S)_$HW_HOST; OUT="$RUNS/${STAMP}_N${N}_${MODE}.txt"; LS="$RUNS/${STAMP}_N${N}_${MODE}_logstream.txt"
hw_logstream "$LS"
TGLOG="$HOME/Library/Logs/beagle_tinygpu.log"; TGLOG_N=$(cat "$TGLOG" 2>/dev/null | wc -l)   # this run's TinyGPULog lines follow
cd "$REPO"
env "${TEST_ENV[@]}" BEAGLE_NV_PROFILE=1 DYLD_LIBRARY_PATH="$TEST_LIBS" \
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
if [ -n "$RECOVER" ]; then   # plan step P4: the boot must have found the GPU warm and torn it down before booting it
    grep -E "TinyGPU/NV: (a warm GPU|the teardown at boot)" "$OUT" | cut -c1-240
    grep -q "TinyGPU/NV: a warm GPU (WPR2_HI=" "$OUT" && grep -q "TinyGPU/NV: the teardown at boot: done: .*WPR2 is down, so the boot goes on" "$OUT" \
        || { if fini_verdict "$OUT"; then echo "FAIL (P4): the boot recovered no warm GPU (exit $rc)"; exit 3; fi
             echo "STOP: the boot did not recover the warm GPU (lines above): power-cycle the eGPU before the next run"; exit 1; }
fi
if [ -n "$LEAVE_WARM" ]; then   # plan step P4: the GSP unloaded, NVIDIA's teardown not run, so WPR2 stays up for --recover
    [ "$(grep -c "TinyGPU/NV: GPU teardown: unload confirmed (GSP MAILBOX0=0x80000000" "$OUT")" -eq 1 ] \
        && grep -q "TinyGPU/NV: no teardown result (WPR2 is still up); power-cycle the eGPU before the next boot" "$OUT" \
        || { echo "STOP: not the warm exit --leave-warm expects (lines above): power-cycle the eGPU before the next run"; exit 1; }
    [ $rc -eq 0 ] || { echo "STOP: the test failed (exit $rc); the GPU was left warm"; exit 1; }
    echo "OK: PASS; the GPU left warm on purpose (the GSP suspended, WPR2 up): the next run needs --recover, or a power cycle first"
    exit 0
fi
fini_verdict "$OUT" || { echo "STOP: bad fini report (lines above): replug the eGPU before the next run"; exit 1; }
if [ -n "$OOM_POOL" ]; then   # plan step M1: the refusal is the expected outcome
    grep -E "out of GPU memory|beagleCreateInstance failed" "$OUT" | cut -c1-240
    [ $rc -ne 0 ] && grep -q "beagleCreateInstance failed (error -2)" "$OUT" && grep -q "TinyGPU/NV: out of GPU memory: " "$OUT" \
        || { echo "FAIL (M1): no BEAGLE_ERROR_OUT_OF_MEMORY for a ${OOM_POOL} MiB pool (exit $rc); the teardown was clean"; exit 3; }
    echo "OK: BEAGLE_ERROR_OUT_OF_MEMORY for a ${OOM_POOL} MiB pool, said; WPR2 is down, the next boot needs no power cycle"
    exit 0
fi
[ $rc -eq 0 ] || { echo "STOP: the test failed (exit $rc); the teardown was clean"; exit 1; }
echo "OK: PASS; WPR2 is down, the next boot needs no power cycle"
