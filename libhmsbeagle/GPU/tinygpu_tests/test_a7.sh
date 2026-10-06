#!/bin/bash
# TODO.md plan step A7, double precision on the AMD card, offline: the plugin offers it when the build's HSACOs include the DP_
# variants (they do with comgr), and BEAGLE's double-precision instances run on them. On fake_amd_device.py's card, through
# the crash guard, each dispatch checked against the build's HSACO of the run's variant (FAKE_AMD_HSACO):
#   - tinygputest --double at 4 and 64 states, and two instances at 4 and 64 in one process: the resource lists DOUBLE,
#     each instance is TinyGPU-Double on its DP_ variant's programs, the card finalized at exit (NO ERRORS);
#   - without --double the test stays in single precision (SP_4), as before;
#   - every synthetictest line of d1_runs.txt with --doubleprecision: exactly the line's kernels, as in single precision
#     (amd_d1_verdict), against the DP_ variant. (hmctest's --tinygpu is single precision only.)
# The fake runs no kernel, so results are wrong by design: tinygputest exits 1 here, and run_amd_point.sh --double and
# run_amd_d1.sh's D1_DOUBLE=1 compare the numbers on the card. One PASS or FAIL line per check; exit 0 only if all pass.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
require_no_launch_guard
GUARD_BIN="$BEAGLE_BUILD/libhmsbeagle/GPU/CMake_TinyGPU/beagle-tinygpu-guard"
[ -x "$GUARD_BIN" ] || { echo "no $GUARD_BIN; build beagle-tinygpu-guard first"; exit 2; }
W="$TINYGPU_TEST_WORK/a7"; rm -rf "$W"; mkdir -p "$W"
TL="$W/beagle_tinygpu.log"   # the plugin's and the guard's TinyGPULog lines in these runs
fails=0
pass() { echo "PASS $1"; }
fail() { echo "FAIL $1"; fails=$((fails + 1)); }
check() { if eval "$2"; then pass "$1"; else fail "$1"; fi; }
glog() { sed -n "/a7 run $1 starts/,/a7 run .* starts/p" "$TL"; }   # one run's TinyGPULog lines
verdict() { grep -E "fake TinyGPU.app \(AMD device\): (NO ERRORS|[0-9]+ ERRORS)" "$W/$1.dev" | tail -1; }
CLEAN="the plugin finalized the GPU itself; exiting"
padded() {   # BeagleGPUImpl's padded state count
    local n=$1 p; for p in 4 16 32 48 64 80 128 192 256; do [ $n -le $p ] && { echo $p; return; }; done; echo $((n + n % 16))
}

run() {   # <label> <variant> <program> [args ...]: one run on a fresh fake card, then its guard (waited for, or ended if it holds)
    local l=$1 v=$2 prog=$3; shift 3
    local d; d=$(mktemp -d /tmp/tg7.XXXXXX)
    mkdir -p "$W/mem_$l"
    echo "a7 run $l starts" >> "$TL"
    FAKE_AMD_HSACO=$v "$BEAGLE_PYTHON" "$TG_TESTS/fake_amd_device.py" "$d/dev.sock" "$W/mem_$l" > "$W/$l.dev" 2>&1 &
    local srv=$!
    for i in $(seq 100); do grep -q listening "$W/$l.dev" 2>/dev/null && break; sleep 0.1; done
    env BEAGLE_TINYGPU_NO_LAUNCH=1 BEAGLE_TINYGPU_NO_DOWNLOAD=1 BEAGLE_TINYGPU_LOG="$TL" APL_REMOTE_SOCK="$d/dev.sock" TMPDIR="$d" \
        BEAGLE_AMD_GUARD="$TG_TESTS/replay/crash_guard_wrap.sh" BEAGLE_TG_GUARD_BIN="$GUARD_BIN" BEAGLE_TG_GUARD_PIDFILE="$d/guard.pid" \
        BEAGLE_AMD_DATA_MB=${POOL_MB:-512} BEAGLE_AMD_PROFILE=1 DYLD_LIBRARY_PATH="$TEST_LIBS" "$BEAGLE_BUILD/examples/$prog" "$@" > "$W/$l.out" 2> "$W/$l.err" < /dev/null
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

loaded() {   # <label> <variants, comma-separated>: exactly these variants' programs loaded, each once
    local v n=0; for v in ${2//,/ }; do [ "$(grep -c "C++ runtime: $v's programs loaded" "$W/$1.err")" -eq 1 ] || return 1; n=$((n + 1)); done
    [ "$(grep -c "C++ runtime: [SD]P_[0-9]*'s programs loaded" "$W/$1.err")" -eq $n ]
}
runs_on() {   # <label> <variants> <implementation>: the run's instances, its variants and the card's end
    grep -q "flags: GPU DOUBLE SINGLE TINYGPU" "$W/$1.out" && [ "$(cat $W/$1.rc)" -lt 128 ] && loaded $1 $2 \
        && ! grep -q "Implementation : TinyGPU-$([ $3 = Double ] && echo Single || echo Double)" "$W/$1.out" \
        && grep -qE "(Implementation : |\()TinyGPU-$3" "$W/$1.out" && glog $1 | grep -q "$CLEAN" && verdict $1 | grep -q "NO ERRORS"
}

# 1. tinygputest in double precision
run dp4 DP_4 tinygputest --double --state-count 4 --reps 3 --diag-compare-cpu
check "--double at 4 states: the resource lists DOUBLE, a TinyGPU-Double instance on DP_4, the card finalized (NO ERRORS)" "runs_on dp4 DP_4 Double"
run dp64 DP_64 tinygputest --double --state-count 64 --reps 3 --diag-compare-cpu
check "--double at 64 states: TinyGPU-Double on DP_64, the card finalized (NO ERRORS)" "runs_on dp64 DP_64 Double"
run dp2 DP_4,DP_64 tinygputest --double --state-count 4,64 --reps 3
check "--double, two instances at 4 and 64 states: one boot, DP_4 and DP_64 each loaded once, every instance's tips read back exactly (NO ERRORS)" \
    "runs_on dp2 DP_4,DP_64 Double && [ \"\$(grep -c 'TinyGPU/AMD: C++ boot done' $W/dp2.err)\" -eq 1 ] \
     && grep -q '^tips: every instance read back its own tip partials exactly' $W/dp2.out"

# 2. the default stays single precision
run sp4 SP_4 tinygputest --state-count 4 --reps 3
check "without --double: TinyGPU-Single on SP_4, as before" "runs_on sp4 SP_4 Single"

# 3. D1's synthetictest lines in double precision
while IFS='|' read -r label cmd kernels; do
    set -- $cmd; prog=$1; shift
    [ $prog = synthetictest ] || continue
    states=$(echo "$cmd" | sed -nE 's/.*--states ([0-9]+).*/\1/p'); v=DP_$(padded $states)
    POOL_MB=2048 run d1_$label $v $prog "$@" --doubleprecision   # at 63 states in double, 655 MB of buffers besides DP_64's 318 MiB of scratch
    why=$(amd_d1_verdict "$W/d1_$label.out" "$W/d1_$label.err" "$kernels")
    if [ -z "$why" ] && [ "$(cat $W/d1_$label.rc)" = 0 ] && grep -q "Impl Name : TinyGPU-Double" "$W/d1_$label.out" \
       && glog d1_$label | grep -q "$CLEAN" && verdict d1_$label | grep -q "NO ERRORS"; then
        pass "D1 $label in double precision ($v): exactly the line's kernels, each against the build's HSACO; the card finalized (NO ERRORS)"
    else
        fail "D1 $label in double precision ($v): ${why:-exit $(cat $W/d1_$label.rc); $(verdict d1_$label | cut -c1-200)} (see $W/d1_$label.out, $W/d1_$label.err)"
    fi
done < <(grep -E '^[a-z0-9_]+\|' "$TG_TESTS/d1_runs.txt")

left=$(ps -axo command= | awk '$1 ~ /beagle-tinygpu-guard$/ || $0 ~ /fake_amd_device\.py/' | wc -l | tr -d ' ')
check "no fake or guard is left running" "[ $left -eq 0 ]"
echo; [ $fails -eq 0 ] && echo "test_a7: PASS" || echo "test_a7: $fails FAILED"
[ $fails -eq 0 ]
