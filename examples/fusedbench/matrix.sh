#!/bin/bash
# matrix.sh <machine> <group...>: fusedbench on this checkout's build-tree OpenCL plugins, one group file per fresh
# process, once with every instance in fast completion mode and once in slow mode (fusedbench --mode; on the M1 Max
# the mode is fixed when each OpenCL context is created). CSVs and consoles go to
# ~/Dropbox/Projects/beagle-lib/fused-handoff/results/<machine>/<date>/<group>-p<k>.{csv,log}, k counting up from 1.
# FUSEDBENCH_EXTRA adds options to every fusedbench command line (the F2 dry run: --b-require none);
# FUSEDBENCH_MODES replaces the mode list "fast slow" (slow mode makes every dependent launch slow: hours at n = 1000).
#
# macOS SIP strips DYLD_* from scripts like this one, so the plugin path is computed from this script's location and
# passed with env on each command line. A run fails on exit 77 (B unavailable) or on a CSV without B's implName
# (-Fused unless --b-require none).

here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
lib="$root/build/libhmsbeagle"
plugins="$lib:$lib/CPU:$lib/GPU/CMake_OpenCL:$lib/GPU/CMake_OpenCLSpectral:$lib/GPU/CMake_OpenCLFused"
bench="$root/build/examples/fusedbench"

if [ $# -lt 2 ]; then
    echo "usage: matrix.sh <machine> <group...>   (groups: $(cd "$here/groups" && ls *.txt | sed 's/\.txt//' | tr '\n' ' '))" >&2
    exit 2
fi
machine=$1
shift
out="$HOME/Dropbox/Projects/beagle-lib/fused-handoff/results/$machine/$(date +%Y-%m-%d)"
mkdir -p "$out" || exit 1
case " ${FUSEDBENCH_EXTRA:-} " in
    *" --b-require none "*) want='^# implName B: ' ;;
    *) want='^# implName B: .*-Fused$' ;;
esac

status=0
for group in "$@"; do
    configs="$here/groups/$group.txt"
    [ -f "$configs" ] || { echo "matrix.sh: no group file $configs" >&2; status=1; continue; }
    for mode in ${FUSEDBENCH_MODES:-fast slow}; do
        k=1
        while [ -e "$out/$group-p$k.csv" ] || [ -e "$out/$group-p$k.log" ]; do k=$((k + 1)); done
        csv="$out/$group-p$k.csv"
        log="$out/$group-p$k.log"
        echo "matrix.sh: $group --mode $mode -> $csv"
        # shellcheck disable=SC2086
        env DYLD_LIBRARY_PATH="$plugins" "$bench" --framework opencl --machine "$machine" --mode $mode \
            --configs "$configs" --csv "$csv" ${FUSEDBENCH_EXTRA:-} > "$log" 2>&1
        rc=$?
        tail -1 "$log"
        if [ $rc -ne 0 ]; then
            echo "matrix.sh: FAIL: $group --mode $mode exited $rc (77: B unavailable)" >&2
            status=1
        elif ! grep -q "$want" "$csv"; then
            echo "matrix.sh: FAIL: $csv has no implName line for B matching '$want'" >&2
            status=1
        fi
    done
done
exit $status
