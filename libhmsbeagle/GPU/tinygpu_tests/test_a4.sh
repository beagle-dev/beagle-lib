#!/bin/bash
# TODO.md plan step A4, NV's plan step D1 on the AMD path, offline: every d1_runs.txt line (synthetictest's derivatives,
# scaling, complex eigensystems and partitions, and hmctest) on fake_amd_device.py's card, through the crash guard. The
# fake checks every dispatch against the build's HSACO for the line's variant (FAKE_AMD_HSACO: SP_ and BeagleGPUImpl's
# padded state count) and every address against the page tables; it runs no kernel, so the numbers are wrong by design
# (run_amd_d1.sh compares them on the card). A line passes if its program exits 0, launched exactly the line's kernels
# (amd_d1_verdict), the guard exited at the plugin's clean and the fake reports NO ERRORS. Which kernels run depends only on
# the arguments, so run_amd_d1.sh's hardware run must launch the same. One PASS or FAIL line per check; exit 0 only if all
# pass.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
require_no_launch_guard
GUARD_BIN="$BEAGLE_BUILD/libhmsbeagle/GPU/CMake_TinyGPU/beagle-tinygpu-guard"
[ -x "$GUARD_BIN" ] || { echo "no $GUARD_BIN; build beagle-tinygpu-guard first"; exit 2; }
W="$TINYGPU_TEST_WORK/a4"; rm -rf "$W"; mkdir -p "$W"
TL="$W/beagle_tinygpu.log"   # the plugin's and the guard's TinyGPULog lines in these runs
fails=0
pass() { echo "PASS $1"; }
fail() { echo "FAIL $1"; fails=$((fails + 1)); }
glog() { sed -n "/a4 run $1 starts/,/a4 run .* starts/p" "$TL"; }   # one run's TinyGPULog lines
verdict() { grep -E "fake TinyGPU.app \(AMD device\): (NO ERRORS|[0-9]+ ERRORS)" "$W/$1.dev" | tail -1; }
padded() {   # BeagleGPUImpl's padded state count
    local n=$1 p; for p in 4 16 32 48 64 80 128 192 256; do [ $n -le $p ] && { echo $p; return; }; done; echo $((n + n % 16))
}

run() {   # <label> <variant> <program> [args ...]: one run on a fresh fake card, then its guard (waited for, or ended if it holds)
    local l=$1 v=$2 prog=$3; shift 3
    local d; d=$(mktemp -d /tmp/tg4.XXXXXX)
    mkdir -p "$W/mem_$l"
    echo "a4 run $l starts" >> "$TL"
    FAKE_AMD_HSACO=$v "$BEAGLE_PYTHON" "$TG_TESTS/fake_amd_device.py" "$d/dev.sock" "$W/mem_$l" > "$W/$l.dev" 2>&1 &
    local srv=$!
    for i in $(seq 100); do grep -q listening "$W/$l.dev" 2>/dev/null && break; sleep 0.1; done
    env BEAGLE_TINYGPU_NO_LAUNCH=1 BEAGLE_TINYGPU_NO_DOWNLOAD=1 BEAGLE_TINYGPU_LOG="$TL" APL_REMOTE_SOCK="$d/dev.sock" TMPDIR="$d" \
        BEAGLE_AMD_GUARD="$TG_TESTS/replay/crash_guard_wrap.sh" BEAGLE_TG_GUARD_BIN="$GUARD_BIN" BEAGLE_TG_GUARD_PIDFILE="$d/guard.pid" \
        BEAGLE_AMD_DATA_MB=512 BEAGLE_AMD_PROFILE=1 DYLD_LIBRARY_PATH="$TEST_LIBS" "$BEAGLE_BUILD/examples/$prog" "$@" > "$W/$l.out" 2> "$W/$l.err" < /dev/null
    echo $? > "$W/$l.rc"
    local gpid; gpid=$(cat "$d/guard.pid" 2>/dev/null)
    if [ -n "$gpid" ]; then   # it exits at the plugin's clean; one that holds keeps the fake's connection
        for i in $(seq 600); do kill -0 "$gpid" 2>/dev/null || break; grep -q "then kill $gpid\." "$TL" && break; sleep 0.1; done
        if kill -0 "$gpid" 2>/dev/null; then
            echo "[$l] the guard (pid $gpid) held the fake connection; ending it" > "$W/$l.held"; kill -KILL "$gpid"
            for i in $(seq 50); do kill -0 "$gpid" 2>/dev/null || break; sleep 0.1; done   # not this shell's child: polled
        fi
    fi
    for i in $(seq 100); do [ "$(grep -c 'client done' "$W/$l.dev")" -ge 2 ] && break; sleep 0.1; done
    kill $srv 2>/dev/null; wait $srv 2>/dev/null; rm -rf "$d" "$W/mem_$l"
}

while IFS='|' read -r label cmd kernels; do
    set -- $cmd; prog=$1; shift
    states=4; [ $prog = synthetictest ] && states=$(echo "$cmd" | sed -nE 's/.*--states ([0-9]+).*/\1/p')
    v=SP_$(padded $states)
    run $label $v $prog "$@"
    why=$(amd_d1_verdict "$W/$label.out" "$W/$label.err" "$kernels")
    if [ -z "$why" ] && [ "$(cat $W/$label.rc)" = 0 ] && glog $label | grep -q "the plugin finalized the GPU itself; exiting" \
       && verdict $label | grep -q "NO ERRORS"; then
        pass "D1 $label ($v): exactly the line's kernels, each against the build's HSACO; the card finalized (NO ERRORS)"
    else
        fail "D1 $label ($v): ${why:-exit $(cat $W/$label.rc); $(verdict $label | cut -c1-200)} (see $W/$label.out, $W/$label.err)"
    fi
done < <(grep -E '^[a-z0-9_]+\|' "$TG_TESTS/d1_runs.txt")

left=$(ps -axo command= | awk '$1 ~ /beagle-tinygpu-guard$/ || $0 ~ /fake_amd_device\.py/' | wc -l | tr -d ' ')
if [ $left -eq 0 ]; then pass "no fake or guard is left running"; else fail "no fake or guard is left running ($left)"; fi
echo; [ $fails -eq 0 ] && echo "test_a4: PASS" || echo "test_a4: $fails FAILED"
[ $fails -eq 0 ]
