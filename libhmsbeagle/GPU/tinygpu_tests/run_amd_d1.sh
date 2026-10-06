#!/bin/bash
# HARDWARE: TODO.md plan step A4, one D1 run on the AMD eGPU: the d1_runs.txt line <label> (plan step D1's), on the plugin's C++
# boot and runtime, with the AMD hardware scripts' protections (amd_hw_begin, amd_boot_check: env.sh). Never Ctrl-C or kill
# a run; a crash guard that holds is a STOP (unplug the eGPU before anything is killed).
#   [D1_DOUBLE=1] run_amd_d1.sh <label>
# D1_DOUBLE=1 (TODO.md plan step A7): a synthetictest line in double precision (--doubleprecision), compared at double
# precision's tolerance (d1_compare.py --double); its runs are named apart (amd_d1dp_), as their VRAM results differ.
# Needs the line's CPU references (d1_refs.sh). Keeps stdout and stderr apart (the plugin's lines would otherwise land inside
# synthetictest's buffered 10000-value lines) and both, the log stream and the TinyGPULog lines under $BEAGLE_TINYGPU_DATA/runs/.
# A line that already ran on this card may find its own results in VRAM (the same data at the same addresses; VRAM
# survives the partial boot), so a kernel that silently did not run would pass: it reruns only after a power cycle
# (D1_REPLUGGED=1 says so). Exits 0 only if the program exited 0, the crash guard exited at the plugin's clean, log stream saw
# nothing from the eGPU and the boot reset nothing, the run launched exactly the line's kernels on the TinyGPU resource with
# no step failed and nothing lost (amd_d1_verdict) and its numbers match the references (d1_compare.py); 1 is a STOP (stop
# all hardware work), 2 means nothing was started, 3 a clean run whose kernels or numbers are wrong (recorded; the next run
# may go ahead).
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
LABEL=$1
IFS='|' read -r _ CMD KERNELS < <(grep "^$LABEL|" "$TG_TESTS/d1_runs.txt")
[ -n "$LABEL" ] && [ -n "$CMD" ] || { echo "usage: run_amd_d1.sh <label from d1_runs.txt>"; exit 2; }
set -- $CMD; PROG=$1; shift
DP=""; [ "$D1_DOUBLE" = 1 ] && { [ $PROG = synthetictest ] || { echo "$LABEL is not a synthetictest line: no double precision"; exit 2; }; DP=dp; set -- "$@" --doubleprecision; }
[ -x "$BEAGLE_BUILD/examples/$PROG" ] || { echo "no $BEAGLE_BUILD/examples/$PROG (BEAGLE_BUILD=$BEAGLE_BUILD); not running"; exit 2; }
REFS="$BEAGLE_TINYGPU_DATA/d1/refs"; REF3="$REFS/$LABEL.dp.out"; [ $PROG = hmctest ] && REF3="$REFS/$LABEL.gpuref.out"
[ -s "$REFS/$LABEL.sp.out" ] && [ -s "$REF3" ] || { echo "no CPU references for $LABEL; run d1_refs.sh first"; exit 2; }
[ "$(cat "$REFS/$LABEL.cmd" 2>/dev/null)" = "$CMD" ] || { echo "the references for $LABEL are from another d1_runs.txt line; rerun d1_refs.sh"; exit 2; }
RUNS="$BEAGLE_TINYGPU_DATA/runs"
prev=$(ls -t "$RUNS"/*_${HW_HOST}_amd_d1${DP}_$LABEL.out 2>/dev/null | head -1)
[ -z "$prev" ] || [ "$D1_REPLUGGED" = 1 ] || { echo "$LABEL already ran on this card ($prev): its results may still be in VRAM; power-cycle the eGPU, then rerun with D1_REPLUGGED=1; not running"; exit 2; }
amd_hw_begin
amd_boot_check
STAMP=$(date +%Y%m%d-%H%M%S)_$HW_HOST; OUT="$RUNS/${STAMP}_amd_d1${DP}_$LABEL.out"; ERR="${OUT%.out}.err"; LS="${OUT%.out}_logstream.txt"
TLOG="${OUT%.out}_tinygpulog.txt"   # the plugin's and the crash guard's TinyGPULog lines
# from here a stray Ctrl-C or a closed terminal cannot end the run: the program inherits these as ignored (log stream, which
# handles SIGINT itself, runs in its own process group: hw_logstream)
trap '' INT HUP
hw_logstream "$LS"
cd "$REPO"
caffeinate -ims env DEBUG=2 BEAGLE_AMD_PROFILE=1 BEAGLE_TINYGPU_LOG="$TLOG" DYLD_LIBRARY_PATH="$TEST_LIBS" "$BEAGLE_BUILD/examples/$PROG" "$@" \
    > "$OUT" 2> "$ERR" < /dev/null
rc=$?
for i in $(seq 60); do pgrep -f "$GUARD_RE" > /dev/null || break; grep -q "HOLDING" "$TLOG" 2>/dev/null && break; sleep 1; done
amd_hw_end "$ERR"; hw=$?
echo "amd d1${DP:+ (double)} $LABEL exit=$rc output=$OUT $ERR"
grep -E "Rsrc Name|Impl Name|^logL|^now:|^error" "$OUT" | cut -c1-120
grep -E "TinyGPU/AMD: (C\+\+ boot|C\+\+ runtime|.*failed|.*fails|.*lost|.*crash guard)|rror" "$ERR" | cut -c1-200 | head -20
echo "boot: $(grep -oE "^am [^:]*: AM_[A-Z0-9]+ initialized" "$ERR" | sed -E 's/.*AM_([A-Z0-9]+) .*/\1/' | xargs)"
grep -E "guard [0-9]+: (the plugin|the GPU is|no queue)|HOLDING" "$TLOG" | cut -c1-240
hw_hold_check || exit 1
[ $hw -eq 0 ] || exit 1
grep -q "the plugin finalized the GPU itself; exiting" "$TLOG" || { echo "STOP: the crash guard did not take the plugin's clean ($TLOG): stop all hardware work"; exit 1; }
[ $rc -eq 0 ] || { echo "STOP: the program failed (exit $rc); the card was finalized"; exit 1; }
why=$(amd_d1_verdict "$OUT" "$ERR" "$KERNELS") || { echo "FAIL (A4): $why; the card was finalized"; exit 3; }
"$BEAGLE_PYTHON" "$TG_TESTS/d1_compare.py" $PROG "$OUT" "$REFS/$LABEL.sp.out" "$REF3" ${DP:+--double} || { echo "FAIL (A4): numbers (above); the card was finalized"; exit 3; }
echo "OK: PASS, the crash guard exited at the plugin's clean, log stream clean"
