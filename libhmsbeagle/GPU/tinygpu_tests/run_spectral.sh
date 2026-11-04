#!/bin/bash
# HARDWARE: TODO.md plan step S5 (PLAN-spectral-tinygpu-degree2.md), one spectral run on the NVIDIA eGPU: the s1_runs.txt line
# <label>, in the C++ runtime, with run_d1.sh's protections. Boots the GPU, so the eGPU must be cold (power-cycled) or torn
# down by the previous run; a warm GPU is refused with nothing written, unless it is an Ada GPU whose GSP-RM was unloaded,
# which the boot tears down first (plan step P4, the default). Never Ctrl-C or kill a run; a hung or holding GPU must be
# unplugged before anything is killed.
#   [S1_REPLUGGED=1] run_spectral.sh <label>
# A line that already ran on this computer may find its own results in VRAM (VRAM survives a warm boot; the NaN fill covers
# partials destinations only, and every input is seeded), so it runs again only after a power cycle (S1_REPLUGGED=1 says
# so). Keeps stdout and stderr apart and the log stream under $BEAGLE_TINYGPU_DATA/runs/. Exit codes:
#   0  PASS: the program exited 0 with its PASS (gpuspectraltest's last line, at its tolerance: 1e-09 in double, 1e-03 in
#      single; adjointtest4's every check), the teardown was clean, and the run launched exactly the line's kernels, on the
#      spectral modules only (spectral_verdict, env.sh)
#   1  STOP all hardware work: a crash guard still holds; the log stream died or saw eGPU events; a bad fini report; no boot
#      line; or the program killed by a signal
#   2  nothing was started
#   3  a clean teardown, but the program exited non-zero or failed the verdict (recorded; the next run may go ahead)
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
LABEL=$1
IFS='|' read -r _ CMD KERNELS < <(grep "^$LABEL|" "$TG_TESTS/s1_runs.txt")
[ -n "$LABEL" ] && [ -n "$CMD" ] || { echo "usage: run_spectral.sh <label from s1_runs.txt>"; exit 2; }
set -- $CMD; PROG=$1; shift
[ -x "$BEAGLE_BUILD/examples/$PROG" ] || { echo "no $BEAGLE_BUILD/examples/$PROG (BEAGLE_BUILD=$BEAGLE_BUILD); not running"; exit 2; }
RUNS="$BEAGLE_TINYGPU_DATA/runs"
prev=$(ls -t "$RUNS"/*_${HW_HOST}_spec_$LABEL.out 2>/dev/null | head -1)
[ -z "$prev" ] || [ "$S1_REPLUGGED" = 1 ] || { echo "$LABEL already ran on this card ($prev): its results may still be in VRAM; power-cycle the eGPU, then rerun with S1_REPLUGGED=1; not running"; exit 2; }
pgrep -f "$GUARD_RE" > /dev/null && { echo "a crash guard is still running (it may hold the GPU); not running"; exit 2; }
# TinyGPU.app serves one client: a second BEAGLE example would hang in tg_cfg_read (plan decision 8's ps check)
pgrep -f "^[^ ]*/examples/(tinygputest|synthetictest|hmctest|gpuspectraltest|adjointtest4)( |$)" > /dev/null && { echo "another BEAGLE example is running; not running"; exit 2; }
hw_begin
for i in $(seq 1 30); do [ "$(ioreg -l -w0 2>/dev/null | grep -c de100000)" -gt 0 ] && break; sleep 2; done
if [ "$(ioreg -l -w0 2>/dev/null | grep -c de100000)" -eq 0 ]; then echo "eGPU not enumerated; not running"; exit 2; fi
STAMP=$(date +%Y%m%d-%H%M%S); OUT="$RUNS/${STAMP}_${HW_HOST}_spec_$LABEL.out"; ERR="${OUT%.out}.err"; LS="${OUT%.out}_logstream.txt"
# from here a stray Ctrl-C or a closed terminal cannot end the run: the program inherits these as ignored (log stream, which
# handles SIGINT itself, runs in its own process group: hw_logstream)
trap '' INT HUP
hw_logstream "$LS"
cd "$REPO"
env -u BEAGLE_NV_FILL_LAUNCH_DIMS BEAGLE_NV_PROFILE=1 DYLD_LIBRARY_PATH="$TEST_LIBS" \
    "$BEAGLE_BUILD/examples/$PROG" "$@" > "$OUT" 2> "$ERR"
rc=$?
for i in $(seq 60); do pgrep -f "$GUARD_RE" > /dev/null || break; sleep 1; done   # it exits at the plugin's clean, or decides at EOF
hw_logstream_stop; ls_ok=$?
echo "spectral $LABEL exit=$rc output=$OUT $ERR"
grep -E "Rsrc Name|Impl Name|against|^(PASS|FAIL)|FAIL" "$OUT" | cut -c1-140 | head -40
grep -E "C\+\+ runtime:|launches in|\[profile\]   kernel|not launched|kernel not found|timed out|failed|rror" "$ERR" | head -40
grep -E "GPU teardown|TinyGPU/NV: teardown:|no teardown result|keeps the TinyGPU.app" "$ERR"
hw_hold_check || exit 1
[ $ls_ok -eq 0 ] || { echo "STOP: log stream ended during the run ($LS): the eGPU check was blind: stop all hardware work"; exit 1; }
# eGPU events: as run_d1.sh, every line but HW_LOG_BENIGN's (the filter's header, the Neural Engine's DART lines, ...)
EVENTS=$(grep -cvE "$HW_LOG_BENIGN" "$LS")
[ "$EVENTS" -eq 0 ] || { echo "STOP: log stream saw $EVENTS eGPU event line(s) ($LS): stop all hardware work"; exit 1; }
grep -q "TinyGPU/NV: level boot: the C++ boot, with no daemon" "$ERR" || { echo "STOP: NOT RUN: the GPU was never booted (see $OUT, $ERR)"; exit 1; }
fini_verdict "$ERR" || { echo "STOP: bad fini report (lines above): replug the eGPU before the next run"; exit 1; }
[ $rc -lt 128 ] || { echo "STOP: the program was killed by a signal (exit $rc): stop all hardware work"; exit 1; }
[ $rc -eq 0 ] || { echo "FAIL (S5): the program exited $rc; the teardown was clean"; exit 3; }
if [ $PROG = gpuspectraltest ]; then
    tol=1e-09; grep -q -- "--single" <<< "$CMD" && tol=1e-03
    tail -1 "$OUT" | grep -q "^PASS (tolerance $tol" || { echo "FAIL (S5): no 'PASS (tolerance $tol' last line; the teardown was clean"; exit 3; }
else
    grep -q "PASS" "$OUT" && ! grep -q "FAIL" "$OUT" || { echo "FAIL (S5): $PROG reported a FAIL, or no PASS; the teardown was clean"; exit 3; }
fi
why=$(spectral_verdict "$OUT" "$ERR" "$KERNELS" $PROG) || { echo "FAIL (S5): $why; the teardown was clean"; exit 3; }
echo "OK: PASS; WPR2 is down, the next boot needs no power cycle"
