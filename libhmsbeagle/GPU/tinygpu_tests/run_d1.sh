#!/bin/bash
# HARDWARE: one TODO.md plan step D1 run on the real eGPU: the d1_runs.txt line <label>, in the C++ runtime, with
# run_point.sh's protections (run_point.sh stays tinygpuhybridtest's). Boots the GPU, so the eGPU must be cold
# (power-cycled) or torn down by the previous run; a warm GPU is refused with nothing written. The user starts each run
# (plan decision 8). Never Ctrl-C or kill a run; a hung or holding GPU must be unplugged before anything is killed.
#   run_d1.sh <label>
# Needs the line's CPU references (d1_refs.sh). Keeps stdout and stderr apart (the plugin's lines would otherwise land inside
# synthetictest's buffered 10000-value lines), the daemon log and the log stream under $BEAGLE_TINYGPU_DATA/runs/. Exits 0
# only if the program exited 0, the fini report says the next boot needs no power cycle (fini_verdict), the daemon exited,
# log stream saw nothing from the eGPU, the run launched exactly the line's kernels on the TinyGPU resource (d1_verdict) and
# its numbers match the references (d1_compare.py); 1 stops the session's chain of runs, 2 means nothing was started, 3 is a
# clean run whose kernels or numbers are wrong (recorded; the next run may go ahead).
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
LABEL=$1
IFS='|' read -r _ CMD KERNELS < <(grep "^$LABEL|" "$TG_TESTS/d1_runs.txt")
[ -n "$LABEL" ] && [ -n "$CMD" ] || { echo "usage: run_d1.sh <label from d1_runs.txt>"; exit 2; }
set -- $CMD; PROG=$1; shift
REFS="$BEAGLE_TINYGPU_DATA/d1/refs"; REF3="$REFS/$LABEL.dp.out"; [ $PROG = hmctest ] && REF3="$REFS/$LABEL.gpuref.out"
[ -s "$REFS/$LABEL.sp.out" ] && [ -s "$REF3" ] || { echo "no CPU references for $LABEL; run d1_refs.sh first"; exit 2; }
[ "$(cat "$REFS/$LABEL.cmd" 2>/dev/null)" = "$CMD" ] || { echo "the references for $LABEL are from another d1_runs.txt line; rerun d1_refs.sh"; exit 2; }
RUNS="$BEAGLE_TINYGPU_DATA/runs"
# a rerun of a line finds its own earlier results in VRAM (the same data at the same addresses; VRAM survives a warm boot),
# so a kernel that silently did not run would pass: a line runs again only after a power cycle (D1_REPLUGGED=1 says so).
# Only this computer's runs count (runs/ may be shared; a card moved between computers is power-cycled on the way). Files
# with no computer tag predate it (2026-09-24, the old Mac's RTX 4060) and are not checked.
prev=$(ls -t "$RUNS"/*_${HW_HOST}_d1_$LABEL.out 2>/dev/null | head -1)
[ -z "$prev" ] || [ "$D1_REPLUGGED" = 1 ] || { echo "$LABEL already ran on this card ($prev): its results may still be in VRAM; power-cycle the eGPU, then rerun with D1_REPLUGGED=1; not running"; exit 2; }
# what can hold the GPU: the daemon (<python> .../nv_dispatch_daemon.py <fd> [<fd>]), a holding <python> .../nv_teardown_diag.py,
# or the crash guard (.../beagle-tinygpu-guard, plan step C10); anchored at the end of the command line, so a shell or editor
# that merely mentions one of them does not match
DAEMON_RE="nv_dispatch_daemon\.py [0-9]+( [0-9]+)?$|nv_teardown_diag\.py$|(^|/)beagle-tinygpu-guard$"
pgrep -f "$DAEMON_RE" > /dev/null && { echo "an nv_dispatch_daemon, nv_teardown_diag or crash guard is still running (it may hold the GPU); not running"; exit 2; }
# TinyGPU.app serves one client: a second BEAGLE example would hang in tg_cfg_read (plan decision 8's ps check)
pgrep -f "^[^ ]*/examples/(tinygpuhybridtest|synthetictest|hmctest)( |$)" > /dev/null && { echo "another BEAGLE example is running; not running"; exit 2; }
hw_begin
for i in $(seq 1 30); do [ "$(ioreg -l -w0 2>/dev/null | grep -c de100000)" -gt 0 ] && break; sleep 2; done
if [ "$(ioreg -l -w0 2>/dev/null | grep -c de100000)" -eq 0 ]; then echo "eGPU not enumerated; not running"; exit 2; fi
STAMP=$(date +%Y%m%d-%H%M%S); OUT="$RUNS/${STAMP}_${HW_HOST}_d1_$LABEL.out"; ERR="${OUT%.out}.err"; LS="${OUT%.out}_logstream.txt"
# from here a stray Ctrl-C or a closed terminal cannot end the run: the program inherits these as ignored (log stream, which
# handles SIGINT itself, runs in its own process group: hw_logstream)
trap '' INT HUP
hw_logstream "$LS"
cd "$REPO"
env -u BEAGLE_NV_FILL_LAUNCH_DIMS BEAGLE_NV_USE_DAEMON=0 BEAGLE_NV_PROFILE=1 BEAGLE_NV_SCRIPTS="$GPU_DIR" DYLD_LIBRARY_PATH="$TEST_LIBS" \
    "$BEAGLE_BUILD/examples/$PROG" "$@" > "$OUT" 2> "$ERR"
rc=$?
for i in $(seq 60); do pgrep -f "$DAEMON_RE" > /dev/null || break; sleep 1; done   # it exits after fini, or decides at EOF
hw_logstream_stop; ls_ok=$?
cp ~/Library/Logs/nv_dispatch_daemon.log "${OUT%.out}_daemon.log" 2>/dev/null
echo "d1 $LABEL exit=$rc output=$OUT $ERR"
grep -E "Rsrc Name|Impl Name|^logL|^now:|^error" "$OUT" | cut -c1-120
grep -E "C\+\+ runtime:|\] +(launch_batch|h2d|d2h) |launches in|\[profile\]   kernel|not launched|timed out|failed|rror" "$ERR" | head -30
grep -E "GPU teardown|TinyGPU/NV: teardown:|no teardown result|keeps the TinyGPU.app|fini round trip" "$ERR"
hw_hold_check || exit 1
[ $ls_ok -eq 0 ] || { echo "STOP: log stream ended during the run ($LS): the eGPU check was blind: stop all hardware work"; exit 1; }
# eGPU events: every line but the filter's header, the column header log stream prints before its first event, and the
# Apple Neural Engine's and camera's own buffer messages, which match "DART" (dartMapBase) and have nothing to do with the eGPU
EVENTS=$(grep -cvE "$HW_LOG_BENIGN" "$LS")
[ "$EVENTS" -eq 0 ] || { echo "STOP: log stream saw $EVENTS eGPU event line(s) ($LS): stop all hardware work"; exit 1; }
grep -qE "TinyGPU/NV: (daemon booted|boot failed)" "$ERR" || { echo "STOP: NOT RUN: the GPU was never booted (see $OUT, $ERR)"; exit 1; }
fini_verdict "$ERR" || { echo "STOP: bad fini report (lines above): replug the eGPU before the next run"; exit 1; }
[ $rc -eq 0 ] || { echo "STOP: the program failed (exit $rc); the teardown was clean"; exit 1; }
why=$(d1_verdict "$OUT" "$ERR" "$KERNELS") || { echo "FAIL (D1): $why; the teardown was clean"; exit 3; }
"$BEAGLE_PYTHON" "$TG_TESTS/d1_compare.py" $PROG "$OUT" "$REFS/$LABEL.sp.out" "$REF3" || { echo "FAIL (D1): numbers (above); the teardown was clean"; exit 3; }
echo "OK: PASS; WPR2 is down, the next boot needs no power cycle"
