#!/bin/bash
# BEAST's post-order likelihood benchmark on BeagleCPUImpl, double then single precision (STATUS.md R95): an XML that times
# its likelihood with <benchmarker>s (BEAST/benchmark/rabies_phylogeo_mat_exp_timing.xml, 1 site, and
# rabies_phylogeo_mat_exp_1000sites_timing.xml, 1000 simulated sites), under -beagle_CPU -beagle_SSE_off (VECTOR_NONE:
# BeagleCPUImpl) and -seed 666, so an XML that simulates its sites gets the same ones in every run. One thread by default
# (-beagle_threads 0: THREADING_NONE); --threads N passes -beagle_threads N instead (BEAST's own default: a thread per core).
#   run_beast_cpu.sh <xml> [--threads N]
# The build's JNI library comes through -Djava.library.path, its plugins through DYLD_LIBRARY_PATH: libhmsbeagle opens them by
# bare name and the build's copy has no rpath. The TinyGPU plugin's directory is left out, so nothing touches the eGPU.
# BEAST_JAVA (a JDK's java: /usr/bin/java drops DYLD_ variables) and BEAST_JAR can be set. Outputs go to $BEAGLE_TINYGPU_DATA/runs/.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
: "${BEAST_JAVA:=$(/usr/libexec/java_home)/bin/java}"
: "${BEAST_JAR:=$HOME/Dropbox/Projects/BEAST/build/dist/beast.jar}"
USAGE="usage: run_beast_cpu.sh <xml> [--threads N]"
XML=$1; THREADS=(-beagle_threads 0); TAG=
[ -f "$XML" ] || { echo "$USAGE"; exit 2; }
if [ $# -gt 1 ]; then
    [ "$2" = --threads ] && [[ "$3" =~ ^[0-9]+$ ]] && [ $# -eq 3 ] || { echo "$USAGE"; exit 2; }
    THREADS=(-beagle_threads "$3"); TAG=_t$3
fi
XML=$(cd "$(dirname "$XML")" && pwd)/$(basename "$XML")
RUNS="$BEAGLE_TINYGPU_DATA/runs"; mkdir -p "$RUNS" "$TINYGPU_TEST_WORK/beast"
cd "$TINYGPU_TEST_WORK/beast"
for prec in double single; do
    FLAG=; [ $prec = single ] && FLAG=-beagle_single
    OUT="$RUNS/$(date +%Y%m%d-%H%M%S)_${HW_HOST}_beast_$(basename "$XML" .xml)_cpu_$prec$TAG.txt"
    /usr/bin/env DYLD_LIBRARY_PATH="$BEAGLE_BUILD/libhmsbeagle:$BEAGLE_BUILD/libhmsbeagle/CPU" \
        "$BEAST_JAVA" -Djava.library.path="$BEAGLE_BUILD/libhmsbeagle/JNI" -jar "$BEAST_JAR" \
        -beagle_CPU -beagle_SSE_off "${THREADS[@]}" $FLAG -seed 666 "$XML" > "$OUT" 2>&1
    echo "BEAST cpu $prec${TAG:+ ${THREADS[*]}} exit=$? output=$OUT"
    grep -E "Using BEAGLE resource|with instance flags|for CPU|unique site patterns|rescaling|Underflow|TreeDataLikelihood\(|^Benchmark " "$OUT"
done
