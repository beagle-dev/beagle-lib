#!/bin/bash
# TODO.md plan step S3.4 (PLAN-spectral-tinygpu-degree2.md), offline: the TinyGPU spectral implementation's requests on NV,
# through spectral_probe.cpp on fake_nv_device.py's AD107 (run_fake_device.sh with FAKE_TEST_BIN; its own verdict is
# ignored for the requests that never boot):
#   - the refusal: a standard request (single precision, GPU, TinyGPU required) that prefers SPECTRAL is refused by the
#     standard factory: BEAGLE_ERROR_NO_RESOURCE (-6), no boot, and no spectral kernel looked up in a standard module;
#   - the routing (plan step S8's baseline): BEAST's pre-order requests at 61 states with no resource list, preferring
#     GPU and DOUBLE and requiring PREORDER_TRANSPOSE_AUTO, then the same also requiring SPECTRAL: neither picks TinyGPU
#     while the plugin has no spectral modules (GetSupportsSpectral).
# The runs go in parallel (env.sh pjob), then one PASS or FAIL line per check; exit 0 only if all pass, 2 if nothing ran.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
require_no_launch_guard
W="$TINYGPU_TEST_WORK/s1"; rm -rf "$W"; mkdir -p "$W"
export TG_PIDS="$W/pids"   # the fakes and guards the runs start (env.sh's track_pid), for the check that none is left running
pjob_init "$W"
fails=0
pass() { echo "PASS $1"; }
fail() { echo "FAIL $1"; fails=$((fails + 1)); }
check() { if eval "$2"; then pass "$1"; else fail "$1"; fi; }

c++ -std=c++17 -O1 -Wall -Wextra -I"$REPO" -o "$W/spectral_probe" "$TG_TESTS/spectral_probe.cpp" \
    -L"$BEAGLE_BUILD/libhmsbeagle" -lhmsbeagle -Wl,-rpath,"$BEAGLE_BUILD/libhmsbeagle" || exit 2

out() { echo "$TINYGPU_TEST_WORK/run_device_$1.txt"; }   # the plugin's and the probe's output (run_fake_device.sh's)
probe() {   # <label> [VAR=value ...] [-- probe args]: one run of the probe on the fake AD107
    local l=$1 envs=(); shift
    while [ $# -gt 0 ] && [ "$1" != "--" ]; do envs+=("$1"); shift; done
    [ "$1" = "--" ] && shift
    FAKE_TEST_BIN="$W/spectral_probe" "$TG_TESTS/run_fake_device.sh" $l "${envs[@]}" -- "$@" > "$W/$l.txt" 2>&1 < /dev/null
}
booted() { grep -qE "level boot: the C\+\+ boot|C\+\+ boot done" "$(out $1)"; }

pjob s1_refusal probe s1_refusal -- --resource 1 --req 'GPU|TINYGPU|SINGLE' --pref SPECTRAL 4:s:std
pjob s1_route probe s1_route -- --pref 'GPU|DOUBLE' --req TRANSPOSE_AUTO 61:d:std 61:d:spec
pwait

check "the refusal: a standard request preferring SPECTRAL gets BEAGLE_ERROR_NO_RESOURCE, with no boot and no spectral kernel looked up" \
    "grep -q '^4:s:std rc=-6 impl=none' '$(out s1_refusal)' && ! booted s1_refusal && ! grep -q 'kernel not found' '$(out s1_refusal)'"
check "the routing (S8): BEAST's pre-order requests, with and without SPECTRAL, pick no TinyGPU implementation" \
    "grep -q '^61:d:std rc=' '$(out s1_route)' && grep -q '^61:d:spec rc=' '$(out s1_route)' \
     && ! grep -E '^61:d:(std|spec) rc=' '$(out s1_route)' | grep -q TinyGPU"

left=$(tracked_alive)   # this stage's own processes only: other stages' fakes and guards may be running
check "no harness process or guard is left running" "[ -z \"\$left\" ]"
echo; [ $fails -eq 0 ] && echo "test_s1: PASS" || echo "test_s1: $fails FAILED"
[ $fails -eq 0 ]
