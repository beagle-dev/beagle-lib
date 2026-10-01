#!/bin/bash
# TODO.md plan step A0: one tinygpuhybridtest run on the AMD eGPU through BEAGLE's AMD path (the plugin, and
# amd_dispatch_daemon.py on the pinned tinygrad), compared with the CPU, at DEBUG=2 so tinygrad's boot lines say which boot it
# was. With amd_hw_begin's protections and amd_boot_check (env.sh); log stream watched; the Mac kept awake; never killed. The
# daemon's log is kept beside the output.
#   run_amd_point.sh <state-count> [reps] [tinygpuhybridtest args ...]
# Exits 0 only if the test passed, the daemon exited, log stream saw nothing from the eGPU and tinygrad reset nothing; 1 is a
# STOP (stop all hardware work), 2 a refusal before anything ran, 3 a clean run whose test failed.
source "$(dirname "$0")/env.sh"
[ $# -ge 1 ] || { echo "usage: $0 <state-count> [reps] [tinygpuhybridtest args ...]"; exit 2; }
N=$1; REPS=${2:-5}; shift; [ $# -gt 0 ] && shift
[ -x "$TEST_BIN" ] || { echo "no $TEST_BIN (BEAGLE_BUILD=$BEAGLE_BUILD); not running"; exit 2; }
amd_hw_begin
amd_boot_check
STAMP=$(date +%Y%m%d-%H%M%S)_$HW_HOST; OUT="$BEAGLE_TINYGPU_DATA/runs/${STAMP}_amd_N${N}.txt"; LS="${OUT%.txt}_logstream.txt"
hw_logstream "$LS"
cd "$REPO"
caffeinate -ims env BEAGLE_NV_SCRIPTS="$GPU_DIR" BEAGLE_AMD_PROFILE=1 DEBUG=2 DYLD_LIBRARY_PATH="$TEST_LIBS" \
    "$TEST_BIN" --state-count "$N" --reps "$REPS" --diag-compare-cpu "$@" > "$OUT" 2>&1
rc=$?
for i in $(seq 30); do pgrep -f "amd_dispatch_daemon.py" > /dev/null || break; sleep 1; done
amd_hw_end "$OUT"; hw=$?
cp ~/Library/Logs/amd_dispatch_daemon.log "${OUT%.txt}_daemon.log" 2>/dev/null
echo "N=$N reps=$REPS $* exit=$rc output=$OUT"
grep -E "Rsrc Name|TinyGPU/AMD: (daemon booted|.*failed)|^PASS|^FAIL|CPU-reference logL|per evaluation|repeats|rror" "$OUT" \
    | grep -v "^  \[" | cut -c1-160 | head -20
echo "boot: $(grep -oE "^am [^:]*: AM_[A-Z0-9]+ initialized" "$OUT" | sed -E 's/.*AM_([A-Z0-9]+) .*/\1/' | xargs)"
grep -hE "plugin's TinyGPU|launch_batch:" "${OUT%.txt}_daemon.log" 2>/dev/null
pgrep -fl "amd_dispatch_daemon.py" && { echo "STOP: the AMD daemon is still running (it may hold the GPU): unplug the eGPU first, then kill it"; exit 1; }
[ $hw -eq 0 ] || exit 1
[ $rc -eq 0 ] || { echo "FAIL: the test failed (exit $rc); log stream clean"; exit 3; }
echo "OK: PASS, log stream clean"
