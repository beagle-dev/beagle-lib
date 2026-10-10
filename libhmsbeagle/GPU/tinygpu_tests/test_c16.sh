#!/bin/bash
# TODO.md plan step C16, double precision on NV, offline: the plugin offers it when the build's cubins include the DP_ modules
# (they do since plan step C16, for sm_86, sm_89 and sm_120), and BEAGLE's double-precision instances run on them. On
# fake_nv_device.py's AD107, GA104 (Ampere, plan step G1) and GB205, each run through run_fake_device.sh (one boot, its
# programs, the teardown's clean report, NO ERRORS):
#   - tinygputest --double at 4 and 64 states, and two instances at 4 and 64 in one process: the resource lists DOUBLE,
#     each instance is TinyGPU-Double on the DP_ cubin for the GPU's architecture (sm_89, sm_86, sm_120);
#   - without --double the test stays in single precision (SP_ cubins), as before;
#   - on the AD107, every synthetictest line of d1_runs.txt with --doubleprecision: exactly the line's kernels (d1_verdict).
# The fakes run no kernel, so the numbers are wrong by design (run_point.sh ... --double compares them on the GPU). The runs
# go in parallel (env.sh's pjob), the longest first; then one PASS or FAIL line per check, in this order; exit 0 only if all pass.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
require_no_launch_guard
unset FAKE_NV_CHIP   # each run names its own chip, for its fake alone (the AD107's runs and D1's: none)
W="$TINYGPU_TEST_WORK/c16"; rm -rf "$W"; mkdir -p "$W"
export TG_PIDS="$W/pids"   # the fakes and guards the runs start (env.sh's track_pid), for the check that none is left running
pjob_init "$W"
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
chiprun() {   # <chip> <label> [-- test args]: a run on that chip's fake (FAKE_NV_CHIP reaches run_fake_device.sh, so the fake)
    local chip=$1; shift
    if [ $chip = ad107 ]; then run "$@"; else FAKE_NV_CHIP=$chip run "$@"; fi
}
d1run() {   # <example> <label> [-- test args]: a D1 line on the AD107 with that BEAGLE example; its fake's memory then removed
    local bin=$1 l=$2 r; shift
    FAKE_TEST_BIN="$BEAGLE_BUILD/examples/$bin" run "$@"; r=$?
    rm -rf "$TINYGPU_TEST_WORK/fake_device_$l"
    return $r
}
d1job() {   # <label> <command line>: d1run's job for a d1_runs.txt line, in double precision
    local l=$1; set -- $2
    pjob c16_d1_$l d1run $1 c16_d1_$l -- "${@:2}" --doubleprecision
}
arch_of() { case $1 in ad107) echo sm_89;; ga104) echo sm_86;; gb205) echo sm_120;; esac; }
# D1's synthetictest lines (in d1_runs.txt's order), in double precision on the AD107
D1=(); while IFS='|' read -r label cmd kernels; do
    set -- $cmd; [ $1 = synthetictest ] && D1+=("$label|$cmd|$kernels")
done < <(grep -E '^[a-z0-9_]+\|' "$TG_TESTS/d1_runs.txt")

# the runs, the longest first: D1's 63-state lines, then the 64-state runs, then the rest
for l in "${D1[@]}"; do IFS='|' read -r label cmd kernels <<< "$l"; case $label in st63_*) d1job $label "$cmd";; esac; done
for chip in ad107 ga104 gb205; do
    pjob c16_${chip}_dp2 chiprun $chip c16_${chip}_dp2 -- --double --state-count 4,64 --reps 3
    pjob c16_${chip}_dp64 chiprun $chip c16_${chip}_dp64 -- --double --state-count 64 --reps 3 --diag-compare-cpu
done
for l in "${D1[@]}"; do IFS='|' read -r label cmd kernels <<< "$l"; case $label in st63_*) ;; *) d1job $label "$cmd";; esac; done
for chip in ad107 ga104 gb205; do
    pjob c16_${chip}_dp4 chiprun $chip c16_${chip}_dp4 -- --double --state-count 4 --reps 3 --diag-compare-cpu
    pjob c16_${chip}_sp4 chiprun $chip c16_${chip}_sp4 -- --state-count 4 --reps 3
done
pwait

for chip in ad107 ga104 gb205; do
    arch=$(arch_of $chip)
    C=$(echo $chip | tr a-z A-Z)
    r=$(pjob_rc c16_${chip}_dp4)
    check "$C: --double at 4 states: the resource lists DOUBLE, a TinyGPU-Double instance on DP_4 $arch, the GPU torn down cleanly" "on c16_${chip}_dp4 $r Double 'DP_4 $arch'"
    r=$(pjob_rc c16_${chip}_dp64)
    check "$C: --double at 64 states: TinyGPU-Double on DP_64 $arch" "on c16_${chip}_dp64 $r Double 'DP_64 $arch'"
    r=$(pjob_rc c16_${chip}_dp2)
    check "$C: --double, two instances at 4 and 64 states: DP_4 and DP_64 $arch, one boot, every instance's tips read back exactly" \
        "on c16_${chip}_dp2 $r Double 'DP_4 $arch' 'DP_64 $arch' && grep -q '^tips: every instance read back its own tip partials exactly' '$(out c16_${chip}_dp2)'"
    r=$(pjob_rc c16_${chip}_sp4)
    check "$C: without --double: TinyGPU-Single on SP_4 $arch, as before" "on c16_${chip}_sp4 $r Single 'SP_4 $arch'"
done

for l in "${D1[@]}"; do
    IFS='|' read -r label cmd kernels <<< "$l"
    r=$(pjob_rc c16_d1_$label)
    if why=$(d1_verdict "$(out c16_d1_$label)" "$(out c16_d1_$label)" "$kernels") && [ $r -eq 0 ] && grep -q "Impl Name : TinyGPU-Double" "$(out c16_d1_$label)"; then
        pass "D1 $label in double precision: PASS, exactly the line's kernels"
    else fail "D1 $label in double precision: ${why:-the run failed (see $W/c16_d1_$label.txt)}"; fi
done

left=$(tracked_alive)   # this stage's own processes only: other stages' fakes and guards may be running
check "no harness process or guard is left running" "[ -z \"\$left\" ]"
echo; [ $fails -eq 0 ] && echo "test_c16: PASS" || echo "test_c16: $fails FAILED"
[ $fails -eq 0 ]
