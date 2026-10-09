#!/bin/bash
# HARDWARE: BEAST's post-order likelihood benchmark (run_beast_cpu.sh's XMLs) on the AMD eGPU through the TinyGPU resource, one
# precision per run (one boot), with run_amd_point.sh's protections (env.sh): amd_hw_begin, amd_boot_check, log stream, DEBUG=2
# so the C++ boot's lines say which boot it was, the crash guard's exit at the plugin's clean, amd_hw_end. Never Ctrl-C or kill
# it; a hung or holding card must be unplugged before anything is killed.
#   run_beast_amd.sh double|single <xml>
# -beagle_GPU -beagle_order 1 and -seed 666, as run_beast_gpu.sh. BEAST falls back to the CPU if the TinyGPU instance fails, so
# the run must say it used resource 1. BEAST_JAVA and BEAST_JAR as run_beast_cpu.sh.
# Exits 0 if BEAST ran on resource 1, the crash guard exited at the plugin's clean, log stream saw nothing from the eGPU and the
# boot reset nothing; 1 stops all hardware work; 2 means nothing was started; 3 is a clean run without the expected result
# (another resource, or BEAST failed).
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
: "${BEAST_JAVA:=$(/usr/libexec/java_home)/bin/java}"
: "${BEAST_JAR:=$HOME/Dropbox/Projects/BEAST/build/dist/beast.jar}"
PREC=$1; XML=$2
{ [ "$PREC" = double ] || [ "$PREC" = single ]; } && [ -f "$XML" ] && [ $# -eq 2 ] \
    || { echo "usage: run_beast_amd.sh double|single <xml>"; exit 2; }
XML=$(cd "$(dirname "$XML")" && pwd)/$(basename "$XML")
FLAG=; [ "$PREC" = single ] && FLAG=-beagle_single
amd_hw_begin
amd_boot_check
mkdir -p "$TINYGPU_TEST_WORK/beast"
BASE="$BEAGLE_TINYGPU_DATA/runs/$(date +%Y%m%d-%H%M%S)_${HW_HOST}_beast_$(basename "$XML" .xml)_amd_$PREC"
OUT="$BASE.txt"; LS="${BASE}_logstream.txt"; TLOG="${BASE}_tinygpulog.txt"   # TLOG: the plugin's and the crash guard's lines
hw_logstream "$LS"
cd "$TINYGPU_TEST_WORK/beast"
env DEBUG=2 BEAGLE_TINYGPU_LOG="$TLOG" DYLD_LIBRARY_PATH="$TEST_LIBS" "$BEAST_JAVA" -Djava.library.path="$BEAGLE_BUILD/libhmsbeagle/JNI" \
    -jar "$BEAST_JAR" -beagle_GPU -beagle_order 1 $FLAG -seed 666 "$XML" > "$OUT" 2>&1
rc=$?
for i in $(seq 60); do pgrep -f "$GUARD_RE" > /dev/null || break; grep -q "HOLDING" "$TLOG" 2>/dev/null && break; sleep 1; done
amd_hw_end "$OUT"; hw=$?
echo "BEAST amd $PREC exit=$rc output=$OUT"
grep -E "Using BEAGLE resource|with instance flags|unique site patterns|rescaling|Underflow|TreeDataLikelihood\(|^Benchmark |TinyGPU/AMD: (C\+\+ boot|C\+\+ runtime|.*failed|.*crash guard)|rror" "$OUT" \
    | cut -c1-200 | head -30
echo "boot: $(grep -oE "^am [^:]*: AM_[A-Z0-9]+ initialized" "$OUT" | sed -E 's/.*AM_([A-Z0-9]+) .*/\1/' | xargs)"
grep -E "guard [0-9]+: (the plugin|the GPU is|no queue)|HOLDING" "$TLOG" | cut -c1-240
hw_hold_check || exit 1
[ $hw -eq 0 ] || exit 1
grep -q "the plugin finalized the GPU itself; exiting" "$TLOG" || { echo "FAIL: the crash guard did not take the plugin's clean ($TLOG); log stream clean"; exit 3; }
grep -qE "Using BEAGLE resource 1: AMD .*\(TinyGPU\)" "$OUT" || { echo "FAIL: BEAST did not run on resource 1 (TinyGPU); log stream clean"; exit 3; }
[ $rc -eq 0 ] || { echo "FAIL: BEAST failed (exit $rc); log stream clean"; exit 3; }
echo "OK: PASS, the crash guard exited at the plugin's clean, log stream clean"
