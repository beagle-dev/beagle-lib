#!/bin/bash
# HARDWARE: one tinygpuhybridtest run on the real eGPU. Boots the GPU, so the eGPU must be cold (power-cycled) or torn
# down by the previous run, as the teardown does by default (TODO.md plan step P3); a warm GPU is refused with nothing
# written. Never Ctrl-C or kill a run; a hung or holding GPU must be unplugged before anything is killed.
#   run_point.sh <state-count>[,<state-count>...] [cpp|daemon|runtime|teardown|vram|sysmem|rm|gsp_hw|flcn_hw|default] [reps] [--poison] [--instances K] [--threads] [--cycles C]
# (the list, --instances, --threads and --cycles: several instances in one process, TODO.md plan step P5; default: no mode
# variable, the plugin's own choice: the C++ runtime on Ada, plan decision 16, with its own GSP unload and teardown at fini,
# plan step C5, its own memory manager, plan step C6, NVDevice, plan step C7, and GSP-RM boot, plan step C8 (level gsp_hw); runtime: the C++ runtime with the daemon's teardown, BEAGLE_NV_CPP_LEVEL=runtime; teardown: the plugin's,
# BEAGLE_NV_CPP_LEVEL=teardown; vram and sysmem: also the plugin's own memory manager, plan step C6's rungs H1 and H2; rm: also
# the NVDevice, which the plugin builds with its own RM client after the daemon's NVDev-only boot, plan step C7's rung H3; gsp_hw:
# also GSP-RM's init_hw and the golden image, after a daemon boot that stops once GSP-RM started, plan step C8's rung H4;
# flcn_hw: also the falcons' init_hw (FWSEC-FRTS, booter_load), after a daemon boot that stops after both init_sw, plan step C9)
# Waits for the eGPU to enumerate, runs from the build tree with --diag-compare-cpu under log stream, keeps the output,
# the daemon log and the log stream under $BEAGLE_TINYGPU_DATA/runs/, and prints a summary. Exits 0 only if the test
# passed, the fini report says the next boot needs no power cycle (fini_verdict, env.sh), the daemon exited and log
# stream saw nothing from the eGPU; 1 stops a chain of runs after a run, 2 means nothing was started.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
N=$1; MODE=${2:-runtime}; REPS=${3:-200}; shift $(( $# < 3 ? $# : 3 ))
USAGE="usage: run_point.sh <state-count>[,<state-count>...] [cpp|daemon|runtime|teardown|vram|sysmem|rm|gsp_hw|flcn_hw|default] [reps] [--poison] [--instances K] [--threads] [--cycles C]"
FLAGS=()
while [ $# -gt 0 ]; do
    case $1 in
        --poison|--threads) FLAGS+=("$1") ;;
        --instances|--cycles) [[ "$2" =~ ^[1-9][0-9]*$ ]] || { echo "$USAGE"; exit 2; }; FLAGS+=("$1" "$2"); shift ;;
        *) echo "$USAGE"; exit 2 ;;
    esac
    shift
done
[[ "$N" =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo "$USAGE"; exit 2; }
# the other modes boot once per instance (a second one is refused) and so tear down once per cycle, which fini_verdict
# would call a bad report
[[ " runtime teardown vram sysmem rm gsp_hw flcn_hw " == *" $MODE "* ]] || { [[ "$N" != *,* ]] && [[ " ${FLAGS[*]} " != *" --instances "* ]] && [[ " ${FLAGS[*]} " != *" --threads "* ]] \
    && [[ " ${FLAGS[*]} " != *" --cycles "* ]]; } || { echo "a state-count list, --instances, --threads and --cycles need the runtime mode"; exit 2; }
case $MODE in
    cpp) MODE_ENV=BEAGLE_NV_CPP_DISPATCH=1 ;;
    runtime) MODE_ENV="BEAGLE_NV_USE_DAEMON=0 BEAGLE_NV_CPP_LEVEL=runtime" ;;
    teardown|vram|sysmem|rm|gsp_hw|flcn_hw) MODE_ENV="BEAGLE_NV_USE_DAEMON=0 BEAGLE_NV_CPP_LEVEL=$MODE" ;;
    daemon) MODE_ENV=BEAGLE_NV_CPP_DISPATCH=0 ;;
    default) MODE_ENV= ;;
    *) echo "unknown mode $MODE"; exit 2 ;;
esac
# what can hold the GPU: the daemon (<python> .../nv_dispatch_daemon.py <fd> [<fd>]) or a holding <python> .../nv_teardown_diag.py;
# anchored at the end of the command line, so a shell or editor that merely mentions either script does not match
DAEMON_RE="nv_dispatch_daemon\.py [0-9]+( [0-9]+)?$|nv_teardown_diag\.py$"
pgrep -f "$DAEMON_RE" > /dev/null && { echo "an nv_dispatch_daemon or nv_teardown_diag is still running (it may hold the GPU); not running"; exit 2; }
hw_begin
for i in $(seq 1 30); do [ "$(ioreg -l -w0 2>/dev/null | grep -c de100000)" -gt 0 ] && break; sleep 2; done
if [ "$(ioreg -l -w0 2>/dev/null | grep -c de100000)" -eq 0 ]; then echo "eGPU not enumerated; not running"; exit 2; fi
RUNS="$BEAGLE_TINYGPU_DATA/runs"; mkdir -p "$RUNS"
STAMP=$(date +%Y%m%d-%H%M%S)_$HW_HOST; OUT="$RUNS/${STAMP}_N${N}_${MODE}.txt"; LS="$RUNS/${STAMP}_N${N}_${MODE}_logstream.txt"
hw_logstream "$LS"
cd "$REPO"
# -u: the mode is the one named here, whatever the shell exports
env -u BEAGLE_NV_USE_DAEMON -u BEAGLE_NV_CPP_DISPATCH -u BEAGLE_NV_CPP_LEVEL $MODE_ENV BEAGLE_NV_PROFILE=1 BEAGLE_NV_SCRIPTS="$GPU_DIR" DYLD_LIBRARY_PATH="$TEST_LIBS" \
    "$TEST_BIN" --state-count "$N" --reps "$REPS" --diag-compare-cpu "${FLAGS[@]}" > "$OUT" 2>&1
rc=$?
for i in $(seq 60); do pgrep -f "$DAEMON_RE" > /dev/null || break; sleep 1; done   # it exits after fini, or decides at EOF
hw_logstream_stop; ls_ok=$?
cp ~/Library/Logs/nv_dispatch_daemon.log "$RUNS/${STAMP}_N${N}_${MODE}_daemon.log" 2>/dev/null
echo "N=$N mode=$MODE ${FLAGS[*]} exit=$rc output=$OUT"
grep -E "C\+\+ dispatch:|C\+\+ runtime:|load_programs|\] +(launch_batch|h2d|d2h) |launches in|maxAbsDiff|First mismatching|CPU-reference logL|per evaluation|repeats|^instance [0-9]+ \(|^tips:|^PASS|^FAIL|timed out|failed|rror" "$OUT" | grep -v "^  \[" | head -30
grep -E "GPU teardown|TinyGPU/NV: teardown:|no teardown result|keeps the TinyGPU.app|fini round trip" "$OUT"
[ "$MODE" = default ] && echo "the default chose $(grep -q "the C++ runtime, the default on this GPU" "$OUT" && echo "the C++ runtime" || echo "the daemon path")"
hw_hold_check || exit 1
[ $ls_ok -eq 0 ] || { echo "STOP: log stream ended during the run ($LS): the eGPU check was blind: stop all hardware work"; exit 1; }
# eGPU events: every line but the filter's header, the column header log stream prints before its first event, and the
# Apple Neural Engine's and camera's own buffer messages, which match "DART" (dartMapBase) and have nothing to do with the eGPU
EVENTS=$(grep -cvE "$HW_LOG_BENIGN" "$LS")
[ "$EVENTS" -eq 0 ] || { echo "STOP: log stream saw $EVENTS eGPU event line(s) ($LS): stop all hardware work"; exit 1; }
fini_verdict "$OUT" || { echo "STOP: bad fini report (lines above): replug the eGPU before the next run"; exit 1; }
[ $rc -eq 0 ] || { echo "STOP: the test failed (exit $rc); the teardown was clean"; exit 1; }
echo "OK: PASS; WPR2 is down, the next boot needs no power cycle"
