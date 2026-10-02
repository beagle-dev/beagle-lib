#!/bin/bash
# TODO.md plan step A0, since plan step A2l on the C++ boot (no Python): one tinygpuhybridtest run on the AMD eGPU through BEAGLE's
# AMD path, compared with the CPU, at DEBUG=2 so the C++ boot's lines ("am usb4: ...", tinygrad's) say which boot it was. With
# amd_hw_begin's protections and amd_boot_check (env.sh); log stream watched; the Mac kept awake; never killed. The crash
# guard exits at the plugin's clean; one that holds is a STOP. run_amd_cpp_point.sh is the same run through tgproxy --guard,
# recorded.
#   run_amd_point.sh <state-count> [reps] [tinygpuhybridtest args ...]
# Exits 0 only if the test passed, the crash guard exited at the plugin's clean, log stream saw nothing from the eGPU and the
# boot reset nothing; 1 is a STOP (stop all hardware work), 2 a refusal before anything ran, 3 a clean run whose test failed.
source "$(dirname "$0")/env.sh"
[ $# -ge 1 ] || { echo "usage: $0 <state-count> [reps] [tinygpuhybridtest args ...]"; exit 2; }
N=$1; REPS=${2:-5}; shift; [ $# -gt 0 ] && shift
[ -x "$TEST_BIN" ] || { echo "no $TEST_BIN (BEAGLE_BUILD=$BEAGLE_BUILD); not running"; exit 2; }
amd_hw_begin
amd_boot_check
STAMP=$(date +%Y%m%d-%H%M%S)_$HW_HOST; OUT="$BEAGLE_TINYGPU_DATA/runs/${STAMP}_amd_N${N}.txt"; LS="${OUT%.txt}_logstream.txt"
TLOG="${OUT%.txt}_tinygpulog.txt"   # the plugin's and the crash guard's TinyGPULog lines
hw_logstream "$LS"
cd "$REPO"
caffeinate -ims env BEAGLE_AMD_PROFILE=1 DEBUG=2 BEAGLE_TINYGPU_LOG="$TLOG" DYLD_LIBRARY_PATH="$TEST_LIBS" \
    "$TEST_BIN" --state-count "$N" --reps "$REPS" --diag-compare-cpu "$@" > "$OUT" 2>&1
rc=$?
for i in $(seq 60); do pgrep -f "$GUARD_RE" > /dev/null || break; grep -q "HOLDING" "$TLOG" 2>/dev/null && break; sleep 1; done
amd_hw_end "$OUT"; hw=$?
echo "N=$N reps=$REPS $* exit=$rc output=$OUT"
grep -E "Rsrc Name|TinyGPU/AMD: (C\+\+ boot|C\+\+ runtime|.*failed|.*crash guard)|^PASS|^FAIL|CPU-reference logL|per evaluation|repeats|rror" "$OUT" \
    | grep -v "^  \[" | cut -c1-160 | head -20
echo "boot: $(grep -oE "^am [^:]*: AM_[A-Z0-9]+ initialized" "$OUT" | sed -E 's/.*AM_([A-Z0-9]+) .*/\1/' | xargs)"
grep -E "guard [0-9]+: (the plugin|the GPU is|no queue)|HOLDING" "$TLOG" | cut -c1-240
hw_hold_check || exit 1
[ $hw -eq 0 ] || exit 1
grep -q "the plugin finalized the GPU itself; exiting" "$TLOG" || { echo "FAIL: the crash guard did not take the plugin's clean ($TLOG); log stream clean"; exit 3; }
[ $rc -eq 0 ] || { echo "FAIL: the test failed (exit $rc); log stream clean"; exit 3; }
echo "OK: PASS, the crash guard exited at the plugin's clean, log stream clean"
