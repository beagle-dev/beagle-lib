#!/bin/bash
# HARDWARE: one tinygpuhybridtest run on the real eGPU. Boots the GPU, so the eGPU must be freshly power-cycled
# (unplugged and replugged) first. Never Ctrl-C or kill a run; a hung GPU must be unplugged before anything is killed.
#   run_point.sh <state-count> [cpp|daemon|runtime] [reps]
# Waits for the eGPU to enumerate, runs from the build tree with --diag-compare-cpu, keeps the output and the daemon
# log under $BEAGLE_TINYGPU_DATA/runs/, and prints a summary.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
N=$1; MODE=${2:-runtime}; REPS=${3:-200}
[ -n "$N" ] || { echo "usage: run_point.sh <state-count> [cpp|daemon|runtime] [reps]"; exit 2; }
case $MODE in
    cpp) MODE_ENV=BEAGLE_NV_CPP_DISPATCH=1 ;;
    runtime) MODE_ENV=BEAGLE_NV_USE_DAEMON=0 ;;
    daemon) MODE_ENV=BEAGLE_NV_CPP_DISPATCH=0 ;;
    *) echo "unknown mode $MODE"; exit 2 ;;
esac
for i in $(seq 1 30); do [ "$(ioreg -l -w0 2>/dev/null | grep -c de100000)" -gt 0 ] && break; sleep 2; done
if [ "$(ioreg -l -w0 2>/dev/null | grep -c de100000)" -eq 0 ]; then echo "eGPU not enumerated; not running"; exit 2; fi
RUNS="$BEAGLE_TINYGPU_DATA/runs"; mkdir -p "$RUNS"
STAMP=$(date +%Y%m%d-%H%M%S); OUT="$RUNS/${STAMP}_N${N}_${MODE}.txt"
cd "$REPO"
env $MODE_ENV BEAGLE_NV_PROFILE=1 BEAGLE_NV_SCRIPTS="$GPU_DIR" DYLD_LIBRARY_PATH="$TEST_LIBS" \
    "$TEST_BIN" --state-count "$N" --reps "$REPS" --diag-compare-cpu > "$OUT" 2>&1
rc=$?
cp ~/Library/Logs/nv_dispatch_daemon.log "$RUNS/${STAMP}_N${N}_${MODE}_daemon.log" 2>/dev/null
echo "N=$N mode=$MODE exit=$rc output=$OUT"
grep -E "C\+\+ dispatch:|C\+\+ runtime:|load_programs|\] +(launch_batch|h2d|d2h) |launches in|maxAbsDiff|First mismatching|CPU-reference logL|per evaluation|repeats|^PASS|^FAIL|timed out|failed|rror" "$OUT" | grep -v "^  \[" | head -30
