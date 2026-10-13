#!/bin/bash
# TODO.md plan steps S3.4 and S4.5 (PLAN-spectral-tinygpu-degree2.md), offline: the TinyGPU spectral implementation on NV.
# Through spectral_probe.cpp on fake_nv_device.py's AD107 (run_fake_device.sh with FAKE_TEST_BIN; its own verdict is
# ignored for the requests that never boot):
#   - the refusal: a standard request (single precision, GPU, TinyGPU and DYNAMIC scaling required, which no spectral
#     factory offers) that prefers SPECTRAL is refused by the standard factory: BEAGLE_ERROR_NO_RESOURCE (-6), no boot, and
#     no spectral kernel looked up in a standard module;
#   - the routing (S8): BEAST's pre-order requests at 61 states with no resource list, preferring GPU and DOUBLE and
#     requiring PREORDER_TRANSPOSE_AUTO, then the same also requiring SPECTRAL: without spectral modules neither picks
#     TinyGPU; with them the first still does not (the standard factory hides the transposes) and the second picks
#     TinyGPU-Double-Spectral.
# With BUILD_SPECTRAL in $BEAGLE_BUILD's cache, also:
#   - every s1_runs.txt line on the fake AD107, GA104 and GB205 (spectral_verdict, env.sh): one boot, the SPEC_ modules of
#     sm_89, sm_86 and sm_120 only, no launch rejected, no kernel missing, exactly the line's kernels (the fakes compute
#     nothing: the examples' numbers and exit codes are ignored);
#   - S2, mixed modules: standard and spectral instances at 4 and 64 states in both precisions alive at once: one boot,
#     each of the 8 modules loaded once;
#   - S5, out of memory in a spectral pool: with 1024 eigen buffers at 64 states each spectral pool is 16 MiB; the smallest
#     VRAM pool P that holds the instance is found by bisection, and at P - 8 MiB beagleCreateInstance returns -2 after
#     SPEC_SP_64's programs loaded, with an allocation of 16.0 MiB the line that failed;
#   - S6, the GPU lost during the adjoint: TinyGPU.app gone at half the doorbells of a 200-adjoint run: the first nonzero
#     return code is -1, and none returns to 0;
#   - S7, the refusal under failure: S8's request with a 1 MiB pool: -2, with exactly one failed setup (the standard
#     factory does not retry it).
# The runs go in parallel (env.sh pjob), then one PASS or FAIL line per check; exit 0 only if all pass, 2 if nothing ran.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
require_no_launch_guard
W="$TINYGPU_TEST_WORK/s1"; rm -rf "$W"; mkdir -p "$W"
export TG_PIDS="$W/pids"   # the fakes and guards the runs start (env.sh's track_pid), for the check that none is left running
pjob_init "$W"
SPECTRAL=0; grep -q '^BUILD_SPECTRAL:BOOL=ON$' "$BEAGLE_BUILD/CMakeCache.txt" 2>/dev/null && SPECTRAL=1
if [ $SPECTRAL = 1 ]; then
    for p in gpuspectraltest adjointtest4; do
        [ -x "$BEAGLE_BUILD/examples/$p" ] || { echo "no $BEAGLE_BUILD/examples/$p; build it first"; exit 2; }
    done
fi
fails=0
pass() { echo "PASS $1"; }
fail() { echo "FAIL $1"; fails=$((fails + 1)); }
check() { if eval "$2"; then pass "$1"; else fail "$1"; fi; }

c++ -std=c++17 -O1 -Wall -Wextra -I"$REPO" -o "$W/spectral_probe" "$TG_TESTS/spectral_probe.cpp" \
    -L"$BEAGLE_BUILD/libhmsbeagle" -lhmsbeagle -Wl,-rpath,"$BEAGLE_BUILD/libhmsbeagle" || exit 2

out() { echo "$TINYGPU_TEST_WORK/run_device_$1.txt"; }   # the plugin's and the program's output (run_fake_device.sh's)
dev() { echo "$TINYGPU_TEST_WORK/fake_device_$1.log"; }   # the fake's
fake() {   # <label> <chip: ad107, ga104 or gb205> <drop at, or 0> <program> [VAR=value ...] [-- args]: one run on that fake
    local l=$1 chip=$2 drop=$3 bin=$4 envs=(); shift 4
    while [ $# -gt 0 ] && [ "$1" != "--" ]; do envs+=("$1"); shift; done
    [ "$1" = "--" ] && shift
    FAKE_NV_CHIP=${chip#ad107} FAKE_DROP_AT=$drop FAKE_TEST_BIN="$bin" FAKE_RUN_TIMEOUT=${FAKE_RUN_TIMEOUT:-60} \
        "$TG_TESTS/run_fake_device.sh" $l "${envs[@]}" -- "$@" > "$W/$l.txt" 2>&1 < /dev/null
}
probe() { local l=$1; shift; fake $l ad107 0 "$W/spectral_probe" "$@"; }   # on the AD107
booted() { grep -qE "level boot: the C\+\+ boot|C\+\+ boot done" "$(out $1)"; }
loaded_once() {   # <label> <module ...>: exactly these modules' cubins loaded, each once
    local m n=0; for m in "${@:2}"; do [ "$(grep -c "embedded cubin $m " "$(out $1)")" -eq 1 ] || return 1; n=$((n + 1)); done
    [ "$(grep -c "embedded cubin " "$(out $1)")" -eq $n ]
}
s5() {   # the smallest pool P (bisection) that holds the instance, then P - 8
    local lo=1 hi=1024 m
    s5_fits() { probe s1_s5_$1 BEAGLE_NV_DATA_MB=$1 -- --resource 1 --eigen 1024 64:s:spec; grep -q '^64:s:spec rc=0 ' "$(out s1_s5_$1)"; }
    s5_fits $hi || { echo 0 > "$W/s5.P"; return; }
    while [ $((hi - lo)) -gt 1 ]; do m=$(( (lo + hi) / 2 )); if s5_fits $m; then hi=$m; else lo=$m; fi; done
    echo $hi > "$W/s5.P"
    probe s1_s5_short BEAGLE_NV_DATA_MB=$((hi - 8)) -- --resource 1 --eigen 1024 64:s:spec
}
s6() {   # 200 adjoints undropped, then TinyGPU.app gone at half their doorbells
    probe s1_s6_full -- --resource 1 --adjoint-reps 200 4:s:spec
    local n; n=$(grep -oE '"doorbells": [0-9]+' "$(dev s1_s6_full)" | tail -1 | grep -oE '[0-9]+$')
    echo "${n:-0}" > "$W/s6.doorbells"
    fake s1_s6_drop ad107 $(( ${n:-0} / 2 )) "$W/spectral_probe" -- --resource 1 --adjoint-reps 200 4:s:spec
}
adjoint_lost() {   # <label>: the first nonzero adjoint return code is -1, and none returns to 0 after it
    awk '/^4:s:spec adjoint [0-9]+: rc=/ { split($0, a, "rc="); rc = a[2] + 0
             if (seen && rc == 0) bad = 1; if (rc != 0 && !seen) { seen = 1; if (rc != -1) bad = 1 } }
         END { exit !(seen && !bad) }' "$(out $1)"
}

# the runs, the longest first
if [ $SPECTRAL = 1 ]; then
    while IFS='|' read -r label cmd kernels; do
        for chip in ad107 ga104 gb205; do
            pjob s1_${chip}_$label fake s1_${chip}_$label $chip 0 "$BEAGLE_BUILD/examples/${cmd%% *}" -- ${cmd#* }
        done
    done < <(grep -E '^[a-z0-9_]+\|' "$TG_TESTS/s1_runs.txt")
    pjob s1_s5 s5
    pjob s1_s6 s6
    pjob s1_mixed probe s1_mixed -- --resource 1 4:s:std 4:s:spec 64:s:std 64:s:spec 4:d:std 4:d:spec 64:d:std 64:d:spec
    pjob s1_s7 probe s1_s7 BEAGLE_NV_DATA_MB=1 -- --resource 1 --req 'GPU|TINYGPU|SINGLE' --pref SPECTRAL 4:s:std
fi
pjob s1_refusal probe s1_refusal -- --resource 1 --req 'GPU|TINYGPU|SINGLE|DYNAMIC' --pref SPECTRAL 4:s:std
pjob s1_route probe s1_route -- --pref 'GPU|DOUBLE' --req TRANSPOSE_AUTO 61:d:std 61:d:spec
pwait

check "the refusal: a standard request preferring SPECTRAL gets BEAGLE_ERROR_NO_RESOURCE, with no boot and no spectral kernel looked up" \
    "grep -q '^4:s:std rc=-6 impl=none' '$(out s1_refusal)' && ! booted s1_refusal && ! grep -q 'kernel not found' '$(out s1_refusal)'"
if [ $SPECTRAL = 1 ]; then
    check "the routing (S8): BEAST's pre-order request picks no TinyGPU implementation, and with SPECTRAL required TinyGPU-Double-Spectral" \
        "grep -q '^61:d:std rc=' '$(out s1_route)' && ! grep '^61:d:std rc=' '$(out s1_route)' | grep -q TinyGPU \
         && grep -q '^61:d:spec rc=[0-9]* impl=TinyGPU-Double-Spectral$' '$(out s1_route)'"
    while IFS='|' read -r label cmd kernels; do
        for chip in ad107 ga104 gb205; do
            l=s1_${chip}_$label
            why=$(spectral_verdict "$(out $l)" "$(out $l)" "$kernels" ${cmd%% *})
            check "$label on the fake $(echo $chip | tr a-z A-Z): one boot, SPEC_ modules only, exactly the line's kernels${why:+ ($why)}" \
                "spectral_verdict '$(out $l)' '$(out $l)' '$kernels' ${cmd%% *} > /dev/null"
        done
    done < <(grep -E '^[a-z0-9_]+\|' "$TG_TESTS/s1_runs.txt")
    check "S2, mixed modules: 4 standard and 4 spectral instances alive at once, one boot, SP_4, SPEC_SP_4, SP_64, SPEC_SP_64 and their DP_ twins each loaded once" \
        "[ \"\$(grep -c 'level boot: the C++ boot, with no daemon' '$(out s1_mixed)')\" -eq 1 ] \
         && loaded_once s1_mixed SP_4 SPEC_SP_4 SP_64 SPEC_SP_64 DP_4 SPEC_DP_4 DP_64 SPEC_DP_64 \
         && [ \"\$(grep -cE '^(4|64):[sd]:std rc=[0-9]+ impl=TinyGPU-(Single|Double)$' '$(out s1_mixed)')\" -eq 4 ] \
         && [ \"\$(grep -cE '^(4|64):[sd]:spec rc=[0-9]+ impl=TinyGPU-(Single|Double)-Spectral$' '$(out s1_mixed)')\" -eq 4 ] \
         && ! grep -q 'kernel not found' '$(out s1_mixed)'"
    P=$(cat "$W/s5.P" 2>/dev/null)
    check "S5, out of memory in a spectral pool: created at a ${P:-?} MiB pool; at $(( ${P:-0} - 8 )) MiB -2, after SPEC_SP_64's programs loaded, on an allocation of 16.0 MiB" \
        "[ \"${P:-0}\" -gt 8 ] && grep -q '^64:s:spec rc=-2 impl=none' '$(out s1_s5_short)' \
         && awk '/embedded cubin SPEC_SP_64 /{c=NR} /kernels loaded/{if (c) k=NR} /out of GPU memory: an allocation of 16.0 MiB/{if (k) o=NR} END{exit !o}' '$(out s1_s5_short)'"
    check "S6, the GPU lost during the adjoint ($(cat "$W/s6.doorbells" 2>/dev/null) doorbells undropped): every adjoint 0 before; after the drop the first nonzero rc is -1, and none returns to 0" \
        "grep -q '^4:s:spec rc=[0-9]* impl=TinyGPU-Single-Spectral$' '$(out s1_s6_full)' \
         && [ \"\$(grep -c '^4:s:spec adjoint' '$(out s1_s6_full)')\" -eq 1 ] && grep -q '^4:s:spec adjoint 0: rc=0$' '$(out s1_s6_full)' \
         && adjoint_lost s1_s6_drop"
    check "S7, the refusal under failure: -2 with exactly one failed setup (the spectral attempt's; the standard factory does not retry)" \
        "grep -q '^4:s:std rc=-2 impl=none' '$(out s1_s7)' && [ \"\$(grep -c \"this instance's setup failed\" '$(out s1_s7)')\" -eq 1 ]"
else
    check "the routing (S8): BEAST's pre-order requests, with and without SPECTRAL, pick no TinyGPU implementation" \
        "grep -q '^61:d:std rc=' '$(out s1_route)' && grep -q '^61:d:spec rc=' '$(out s1_route)' \
         && ! grep -E '^61:d:(std|spec) rc=' '$(out s1_route)' | grep -q TinyGPU"
fi

left=$(tracked_alive)   # this stage's own processes only: other stages' fakes and guards may be running
check "no harness process or guard is left running" "[ -z \"\$left\" ]"
echo; [ $fails -eq 0 ] && echo "test_s1: PASS" || echo "test_s1: $fails FAILED"
[ $fails -eq 0 ]
