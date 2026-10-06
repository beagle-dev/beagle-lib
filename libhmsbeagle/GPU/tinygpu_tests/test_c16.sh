#!/bin/bash
# TODO.md plan step C16, double precision on NV, offline: the plugin offers it when the build's cubins include the DP_ modules
# (they do since plan step C16, for sm_86, sm_89 and sm_120), and BEAGLE's double-precision instances run on them. On
# fake_nv_device.py's AD107 and GB205, each run through run_fake_device.sh (one boot, its programs, the teardown's clean
# report, NO ERRORS):
#   - tinygputest --double at 4 and 64 states, and two instances at 4 and 64 in one process: the resource lists DOUBLE,
#     each instance is TinyGPU-Double on the DP_ cubin for the GPU's architecture (sm_89, sm_120);
#   - without --double the test stays in single precision (SP_ cubins), as before;
#   - on the AD107, every synthetictest line of d1_runs.txt with --doubleprecision: exactly the line's kernels (d1_verdict).
# The fakes run no kernel, so the numbers are wrong by design (run_point.sh ... --double compares them on the GPU). One PASS
# or FAIL line per check; exit 0 only if all pass.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
require_no_launch_guard
W="$TINYGPU_TEST_WORK/c16"; rm -rf "$W"; mkdir -p "$W"
fails=0
pass() { echo "PASS $1"; }
fail() { echo "FAIL $1"; fails=$((fails + 1)); }
check() { if eval "$2"; then pass "$1"; else fail "$1"; fi; }
out() { echo "$TINYGPU_TEST_WORK/run_device_$1.txt"; }   # the plugin's and the test's output (run_fake_device.sh's)
run() {   # <label> [VAR=value ...] [-- test args]: one fake run; its verdict is run_fake_device.sh's
    local l=$1 envs=(); shift
    while [ $# -gt 0 ] && [ "$1" != "--" ]; do envs+=("$1"); shift; done
    [ "$1" = "--" ] && shift
    "$TG_TESTS/run_fake_device.sh" $l "${envs[@]}" -- "$@" > "$W/$l.txt" 2>&1 < /dev/null
}
on() {   # <label> <run's exit> <Double|Single> <cubins, space-separated>: the run's verdict, its instances and its cubins
    local l=$1 r=$2 impl=$3 c; shift 3
    [ $r -eq 0 ] && grep -q "flags: GPU DOUBLE SINGLE TINYGPU" "$(out $l)" || return 1
    grep -qE "(Implementation : |\()TinyGPU-$impl" "$(out $l)" && ! grep -qE "(Implementation : |\()TinyGPU-$([ $impl = Double ] && echo Single || echo Double)" "$(out $l)" || return 1
    for c in "$@"; do grep -q "TinyGPU/NV: C++ runtime: embedded cubin $c " "$(out $l)" || return 1; done
}

for chip in ad107 gb205; do
    if [ $chip = gb205 ]; then export FAKE_NV_CHIP=gb205; arch=sm_120; else unset FAKE_NV_CHIP; arch=sm_89; fi
    C=$(echo $chip | tr a-z A-Z)
    run c16_${chip}_dp4 -- --double --state-count 4 --reps 3 --diag-compare-cpu; r=$?
    check "$C: --double at 4 states: the resource lists DOUBLE, a TinyGPU-Double instance on DP_4 $arch, the GPU torn down cleanly" "on c16_${chip}_dp4 $r Double 'DP_4 $arch'"
    run c16_${chip}_dp64 -- --double --state-count 64 --reps 3 --diag-compare-cpu; r=$?
    check "$C: --double at 64 states: TinyGPU-Double on DP_64 $arch" "on c16_${chip}_dp64 $r Double 'DP_64 $arch'"
    run c16_${chip}_dp2 -- --double --state-count 4,64 --reps 3; r=$?
    check "$C: --double, two instances at 4 and 64 states: DP_4 and DP_64 $arch, one boot, every instance's tips read back exactly" \
        "on c16_${chip}_dp2 $r Double 'DP_4 $arch' 'DP_64 $arch' && grep -q '^tips: every instance read back its own tip partials exactly' '$(out c16_${chip}_dp2)'"
    run c16_${chip}_sp4 -- --state-count 4 --reps 3; r=$?
    check "$C: without --double: TinyGPU-Single on SP_4 $arch, as before" "on c16_${chip}_sp4 $r Single 'SP_4 $arch'"
done
unset FAKE_NV_CHIP

# D1's synthetictest lines in double precision, on the AD107
while IFS='|' read -r label cmd kernels; do
    set -- $cmd; bin=$1; shift
    [ $bin = synthetictest ] || continue
    FAKE_TEST_BIN="$BEAGLE_BUILD/examples/$bin" run c16_d1_$label -- "$@" --doubleprecision; r=$?
    rm -rf "$TINYGPU_TEST_WORK/fake_device_c16_d1_$label"
    if why=$(d1_verdict "$(out c16_d1_$label)" "$(out c16_d1_$label)" "$kernels") && [ $r -eq 0 ] && grep -q "Impl Name : TinyGPU-Double" "$(out c16_d1_$label)"; then
        pass "D1 $label in double precision: PASS, exactly the line's kernels"
    else fail "D1 $label in double precision: ${why:-the run failed (see $W/c16_d1_$label.txt)}"; fi
done < <(grep -E '^[a-z0-9_]+\|' "$TG_TESTS/d1_runs.txt")

left=$(ps -axo command | grep -cE "Python .*(tgproxy|tgreplay|fake_nv_device)\.py|[b]eagle-tinygpu-guard")
check "no harness process or guard is left running" "[ $left -eq 0 ]"
echo; [ $fails -eq 0 ] && echo "test_c16: PASS" || echo "test_c16: $fails FAILED"
[ $fails -eq 0 ]
