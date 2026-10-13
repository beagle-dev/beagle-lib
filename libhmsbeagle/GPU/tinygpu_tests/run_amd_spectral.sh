#!/bin/bash
# HARDWARE: TODO.md plan step S5 (PLAN-spectral-tinygpu-degree2.md), one spectral run on the AMD eGPU: the s1_runs.txt line
# <label>, on the plugin's C++ boot and runtime, with run_amd_d1.sh's protections (amd_hw_begin, amd_boot_check: env.sh).
# Never Ctrl-C or kill a run; a crash guard that holds is a STOP (unplug the eGPU before anything is killed).
#   [S1_REPLUGGED=1] run_amd_spectral.sh <label>
# A line that already ran on this card may find its own results in VRAM (VRAM survives the partial boot; the NaN fill covers
# partials destinations only, and every input is seeded), so it runs again only after a power cycle (S1_REPLUGGED=1 says
# so). "This card": an earlier run of the line on this host whose boot was the card's arch. Keeps stdout and stderr apart,
# and both, the log stream and the TinyGPULog lines under $BEAGLE_TINYGPU_DATA/runs/. Exit codes:
#   0  PASS: the program exited 0 with its PASS (gpuspectraltest's last line, at its tolerance: 1e-09 in double, 1e-03 in
#      single; adjointtest4's every check), the crash guard exited at the plugin's own fini, log stream was clean, the boot
#      the predicted one, and the run launched exactly the line's kernels, on the SPEC_ variants only (amd_spectral_verdict)
#   1  STOP all hardware work: a crash guard still holds; the log stream died or saw eGPU events; a mode1 reset or a boot
#      other than the predicted one; no plugin finalize; or the program killed by a signal
#   2  nothing was started
#   3  a clean teardown, but the program exited non-zero or failed the verdict (recorded; the next run may go ahead)
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
LABEL=$1
IFS='|' read -r _ CMD KERNELS < <(grep "^$LABEL|" "$TG_TESTS/s1_runs.txt")
[ -n "$LABEL" ] && [ -n "$CMD" ] || { echo "usage: run_amd_spectral.sh <label from s1_runs.txt>"; exit 2; }
set -- $CMD; PROG=$1; shift
[ -x "$BEAGLE_BUILD/examples/$PROG" ] || { echo "no $BEAGLE_BUILD/examples/$PROG (BEAGLE_BUILD=$BEAGLE_BUILD); not running"; exit 2; }
RUNS="$BEAGLE_TINYGPU_DATA/runs"
amd_hw_begin
arch=$(sed -nE "s/^    \{0x${AMD_PCI#1002:}, \"(gfx[0-9a-f]+)\".*/\1/p" "$GPU_DIR/TinyGPUAMDBootTables.h")   # am::kChips: the card's arch
prev=$(grep -l "C++ boot done .* arch=$arch\$" $(ls -t "$RUNS"/*_${HW_HOST}_amd_spec_$LABEL.err 2>/dev/null) /dev/null 2>/dev/null | head -1)
[ -z "$prev" ] || [ "$S1_REPLUGGED" = 1 ] || { echo "$LABEL already ran on this card ($arch: $prev): its results may still be in VRAM; power-cycle the eGPU, then rerun with S1_REPLUGGED=1; not running"; exit 2; }
amd_boot_check
STAMP=$(date +%Y%m%d-%H%M%S)_$HW_HOST; OUT="$RUNS/${STAMP}_amd_spec_$LABEL.out"; ERR="${OUT%.out}.err"; LS="${OUT%.out}_logstream.txt"
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
echo "amd spectral $LABEL exit=$rc output=$OUT $ERR"
grep -E "Rsrc Name|Impl Name|against|^(PASS|FAIL)|FAIL" "$OUT" | cut -c1-140 | head -40
grep -E "TinyGPU/AMD: (C\+\+ boot|C\+\+ runtime|.*failed|.*fails|.*lost|.*crash guard)|not found in the build|rror" "$ERR" | cut -c1-200 | head -30
echo "boot: $(grep -oE "^am [^:]*: AM_[A-Z0-9]+ initialized" "$ERR" | sed -E 's/.*AM_([A-Z0-9]+) .*/\1/' | xargs)"
grep -E "guard [0-9]+: (the plugin|the GPU is|no queue)|HOLDING" "$TLOG" | cut -c1-240
hw_hold_check || exit 1
[ $hw -eq 0 ] || exit 1
grep -q "the plugin finalized the GPU itself; exiting" "$TLOG" || { echo "STOP: the crash guard did not take the plugin's clean ($TLOG): stop all hardware work"; exit 1; }
[ $rc -lt 128 ] || { echo "STOP: the program was killed by a signal (exit $rc): stop all hardware work"; exit 1; }
[ $rc -eq 0 ] || { echo "FAIL (S5): the program exited $rc; the card was finalized"; exit 3; }
if [ $PROG = gpuspectraltest ]; then
    tol=1e-09; grep -q -- "--single" <<< "$CMD" && tol=1e-03
    tail -1 "$OUT" | grep -q "^PASS (tolerance $tol" || { echo "FAIL (S5): no 'PASS (tolerance $tol' last line; the card was finalized"; exit 3; }
else
    grep -q "PASS" "$OUT" && ! grep -q "FAIL" "$OUT" || { echo "FAIL (S5): $PROG reported a FAIL, or no PASS; the card was finalized"; exit 3; }
fi
why=$(amd_spectral_verdict "$OUT" "$ERR" "$KERNELS" $PROG) || { echo "FAIL (S5): $why; the card was finalized"; exit 3; }
echo "OK: PASS, the crash guard exited at the plugin's clean, log stream clean"
