#!/bin/bash
# HARDWARE: BEAST's post-order likelihood benchmark (run_beast_cpu.sh's XMLs, STATUS.md R95) on the TinyGPU resource, one
# precision per run (one boot), with run_point.sh's protections (env.sh): hw_begin, the eGPU enumerated, log stream, the crash
# guard's exit, fini_verdict. Never Ctrl-C or kill it; a hung or holding GPU must be unplugged before anything is killed.
#   run_beast_gpu.sh double|single <xml>
# -beagle_GPU -beagle_order 1 (TinyGPU is resource 1 under $TEST_LIBS) and -seed 666, as run_beast_cpu.sh. BEAST falls back to
# the CPU if the TinyGPU instance fails, so the run must say it used resource 1. BEAST_JAVA and BEAST_JAR as run_beast_cpu.sh.
# Exits 0 if BEAST ran on resource 1, the fini report says the next boot needs no power cycle, the guard exited and log stream
# saw nothing from the eGPU; 1 stops all hardware work; 2 means nothing was started; 3 is a clean teardown without that run
# (another resource, a failed run, or the crash guard's teardown instead of the plugin's).
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
: "${BEAST_JAVA:=$(/usr/libexec/java_home)/bin/java}"
: "${BEAST_JAR:=$HOME/Dropbox/Projects/BEAST/build/dist/beast.jar}"
PREC=$1; XML=$2
{ [ "$PREC" = double ] || [ "$PREC" = single ]; } && [ -f "$XML" ] && [ $# -eq 2 ] \
    || { echo "usage: run_beast_gpu.sh double|single <xml>"; exit 2; }
XML=$(cd "$(dirname "$XML")" && pwd)/$(basename "$XML")
FLAG=; [ "$PREC" = single ] && FLAG=-beagle_single
pgrep -f "$GUARD_RE" > /dev/null && { echo "a crash guard is still running (it may hold the GPU); not running"; exit 2; }
hw_begin
for i in $(seq 1 30); do [ "$(ioreg -l -w0 2>/dev/null | grep -c de100000)" -gt 0 ] && break; sleep 2; done
if [ "$(ioreg -l -w0 2>/dev/null | grep -c de100000)" -eq 0 ]; then echo "eGPU not enumerated; not running"; exit 2; fi
RUNS="$BEAGLE_TINYGPU_DATA/runs"; mkdir -p "$RUNS" "$TINYGPU_TEST_WORK/beast"
BASE="$RUNS/$(date +%Y%m%d-%H%M%S)_${HW_HOST}_beast_$(basename "$XML" .xml)_gpu_$PREC"
OUT="$BASE.txt"; LS="${BASE}_logstream.txt"
hw_logstream "$LS"
TGLOG="$HOME/Library/Logs/beagle_tinygpu.log"; TGLOG_N=$(cat "$TGLOG" 2>/dev/null | wc -l)   # this run's TinyGPULog lines follow
cd "$TINYGPU_TEST_WORK/beast"
env DYLD_LIBRARY_PATH="$TEST_LIBS" "$BEAST_JAVA" -Djava.library.path="$BEAGLE_BUILD/libhmsbeagle/JNI" -jar "$BEAST_JAR" \
    -beagle_GPU -beagle_order 1 $FLAG -seed 666 "$XML" > "$OUT" 2>&1
rc=$?
for i in $(seq 60); do pgrep -f "$GUARD_RE" > /dev/null || break; sleep 1; done   # it exits at the plugin's clean, or decides at EOF
hw_logstream_stop; ls_ok=$?
tail -n +$((TGLOG_N + 1)) "$TGLOG" > "${BASE}_tinygpulog.txt"
echo "BEAST gpu $PREC exit=$rc output=$OUT"
grep -E "Using BEAGLE resource|with instance flags|unique site patterns|rescaling|Underflow|TreeDataLikelihood\(|^Benchmark |TinyGPU/NV: (C\+\+ runtime|level boot)|failed|not launched|rror" "$OUT" | head -30
grep -E "GPU teardown|TinyGPU/NV: teardown:|no teardown result|keeps the TinyGPU.app|TinyGPU/NV: (a warm GPU|the teardown at boot)" "$OUT"
hw_hold_check || exit 1
[ $ls_ok -eq 0 ] || { echo "STOP: log stream ended during the run ($LS): the eGPU check was blind: stop all hardware work"; exit 1; }
EVENTS=$(grep -cvE "$HW_LOG_BENIGN" "$LS")
[ "$EVENTS" -eq 0 ] || { echo "STOP: log stream saw $EVENTS eGPU event line(s) ($LS): stop all hardware work"; exit 1; }
if ! fini_verdict "$OUT"; then
    guard_verdict "${BASE}_tinygpulog.txt" || { echo "STOP: bad fini report (lines above): replug the eGPU before the next run"; exit 1; }
    echo "no fini report from the plugin, but the crash guard tore the GPU down cleanly: WPR2 is down, the next boot needs no power cycle"
    exit 3
fi
grep -qE "Using BEAGLE resource 1: NVIDIA .*\(TinyGPU\)" "$OUT" || { echo "FAIL: BEAST did not run on resource 1 (TinyGPU); the teardown was clean"; exit 3; }
[ $rc -eq 0 ] || { echo "FAIL: BEAST failed (exit $rc); the teardown was clean"; exit 3; }
echo "OK: PASS; WPR2 is down, the next boot needs no power cycle"
