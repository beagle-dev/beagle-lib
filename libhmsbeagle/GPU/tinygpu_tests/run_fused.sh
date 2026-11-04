#!/bin/bash
# HARDWARE: fused-handoff/PLAN-fused-uppartials.md section 6 item 10 (sessions F7, F8), one fusedbench run on the NVIDIA eGPU:
# the f1_runs.txt line <label> (hw or both), in the C++ runtime, with run_spectral.sh's protections. Boots the GPU, so the
# eGPU must be cold (power-cycled) or torn down by the previous run; a warm GPU is refused with nothing written, unless it is
# an Ada GPU whose GSP-RM was unloaded, which the boot tears down first (plan step P4, the default). Never Ctrl-C or kill a
# run; a hung or holding GPU must be unplugged before anything is killed.
#   run_fused.sh <label>
# A line may run again without a power cycle (README's answer to Q11): fusedbench poisons every internal partials buffer
# with NaN before its checks and compares B with A in the same process, so a result left in VRAM cannot pass for a new one.
# Keeps stdout and stderr apart, the CSV next to them (fusedbench --csv; bench lines only) and the log stream under
# $BEAGLE_TINYGPU_DATA/runs/. Exit codes:
#   0  PASS: fusedbench exited 0 with PASS as its last line, every CSV row is ok, the teardown was clean, and the run passed
#      fused_verdict (a B-only line: one kernelUpPartialsFused launch per B updatePartials call, no standard pruning or
#      scaling kernel) or fused_run_verdict (any other: the standard modules only, kernelUpPartialsFused launched; env.sh)
#   1  STOP all hardware work: a crash guard still holds; the log stream died or saw eGPU events; a bad fini report; no boot
#      line; or the program killed by a signal
#   2  nothing was started
#   3  a clean teardown, but fusedbench exited non-zero or failed the verdict (recorded; the next run may go ahead)
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
LABEL=$1
IFS='|' read -r _ ARGS WHERE < <(grep "^$LABEL|" "$TG_TESTS/f1_runs.txt")
[ -n "$LABEL" ] && [ -n "$ARGS" ] && [ "$WHERE" != fake ] || { echo "usage: run_fused.sh <label of an hw or both line in f1_runs.txt>"; exit 2; }
ARGS=${ARGS//\$REPO/$REPO}   # --configs paths are absolute: the run below starts in $REPO
[ -x "$BEAGLE_BUILD/examples/fusedbench" ] || { echo "no $BEAGLE_BUILD/examples/fusedbench (BEAGLE_BUILD=$BEAGLE_BUILD; a BUILD_FUSED build); not running"; exit 2; }
RUNS="$BEAGLE_TINYGPU_DATA/runs"
pgrep -f "$GUARD_RE" > /dev/null && { echo "a crash guard is still running (it may hold the GPU); not running"; exit 2; }
# TinyGPU.app serves one client: a second BEAGLE example would hang in tg_cfg_read (plan decision 8's ps check)
pgrep -f "^[^ ]*/examples/(tinygputest|synthetictest|hmctest|gpuspectraltest|adjointtest4|fusedbench)( |$)" > /dev/null && { echo "another BEAGLE example is running; not running"; exit 2; }
hw_begin
for i in $(seq 1 30); do [ "$(ioreg -l -w0 2>/dev/null | grep -c de100000)" -gt 0 ] && break; sleep 2; done
if [ "$(ioreg -l -w0 2>/dev/null | grep -c de100000)" -eq 0 ]; then echo "eGPU not enumerated; not running"; exit 2; fi
STAMP=$(date +%Y%m%d-%H%M%S); OUT="$RUNS/${STAMP}_${HW_HOST}_fused_$LABEL.out"; ERR="${OUT%.out}.err"; LS="${OUT%.out}_logstream.txt"
CSV="${OUT%.out}.csv"
# from here a stray Ctrl-C or a closed terminal cannot end the run: the program inherits these as ignored (log stream, which
# handles SIGINT itself, runs in its own process group: hw_logstream)
trap '' INT HUP
hw_logstream "$LS"
cd "$REPO"
env -u BEAGLE_NV_FILL_LAUNCH_DIMS BEAGLE_NV_PROFILE=1 DYLD_LIBRARY_PATH="$TEST_LIBS" \
    "$BEAGLE_BUILD/examples/fusedbench" $ARGS --csv "$CSV" > "$OUT" 2> "$ERR"
rc=$?
for i in $(seq 60); do pgrep -f "$GUARD_RE" > /dev/null || break; sleep 1; done   # it exits at the plugin's clean, or decides at EOF
hw_logstream_stop; ls_ok=$?
echo "fused $LABEL exit=$rc output=$OUT $ERR"
grep -E "^(config|fusedbench --check|stress)|^  (REF|A|A2|B) |CHECK FAILED|B unavailable|B updatePartials calls|^(PASS|FAIL)" "$OUT" | cut -c1-160 | tail -40
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
[ $rc -eq 0 ] || { echo "FAIL (fused): fusedbench exited $rc (77: B unavailable); the teardown was clean"; exit 3; }
[ "$(tail -1 "$OUT")" = PASS ] || { echo "FAIL (fused): fusedbench's last line is not PASS; the teardown was clean"; exit 3; }
if [[ " $ARGS " != *" --check "* ]]; then   # bench lines: one CSV row per timed instance and configuration, each ok
    awk -F, '/^(#|machine,)/ { next } { n++; if ($NF != 1) bad++ } END { exit !(n > 0 && !bad) }' "$CSV" 2>/dev/null \
        || { echo "FAIL (fused): $CSV has no rows, or a row that is not ok; the teardown was clean"; exit 3; }
fi
if [[ " $ARGS " == *" --impls B "* ]]; then why=$(fused_verdict "$OUT" "$ERR")
else why=$(fused_run_verdict "$OUT" "$ERR"); fi || { echo "FAIL (fused): $why; the teardown was clean"; exit 3; }
echo "OK: PASS; WPR2 is down, the next boot needs no power cycle"
