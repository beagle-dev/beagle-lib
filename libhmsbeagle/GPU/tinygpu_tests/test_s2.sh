#!/bin/bash
# TODO.md plan steps S3.4 and S8.3 (PLAN-spectral-tinygpu-degree2.md), offline: test_s1.sh's checks on the AMD card, through
# spectral_probe.cpp and the s1_runs.txt lines on fake_amd_device.py's card (FAKE_AMD_CHIP, gfx1100 by default), through the
# crash guard, each dispatch checked against the build's HSACOs of the run's variants (FAKE_AMD_HSACO):
#   - the refusal: a standard request (single precision, GPU, TinyGPU and DYNAMIC scaling required, which no spectral
#     factory offers) that prefers SPECTRAL is refused by the standard factory: BEAGLE_ERROR_NO_RESOURCE (-6), no boot, and
#     no spectral kernel looked up in a standard variant;
#   - the routing (S8): BEAST's pre-order requests at 61 states with no resource list, preferring GPU and DOUBLE and
#     requiring PREORDER_TRANSPOSE_AUTO, then the same also requiring SPECTRAL: without spectral variants neither picks
#     TinyGPU; with them the first still does not and the second picks TinyGPU-Double-Spectral.
# With the build's SPEC_ HSACOs, also:
#   - every s1_runs.txt line: one C++ boot, its SPEC_ variants' programs only, nothing missing, no failed step and no lost
#     GPU, exactly the line's kernels (amd_spectral_verdict, env.sh), the guard exiting at the plugin's own fini and the fake
#     reporting NO ERRORS (the fake computes nothing: the examples' numbers and exit codes are ignored);
#   - S2, mixed variants: standard and spectral instances at 4 and 64 states in both precisions alive at once: one boot,
#     each of the 8 variants' programs loaded once;
#   - S5, out of memory in a spectral pool: with 1024 eigen buffers at 64 states each spectral pool is 16 MiB; the smallest
#     VRAM pool P that holds the instance is found by bisection, and at P - 8 MiB beagleCreateInstance returns -2 after
#     SPEC_SP_64's programs loaded, with an allocation of 16.0 MiB the line that failed;
#   - S6, the GPU lost during the adjoint: TinyGPU.app gone at half the requests of a 200-adjoint run: the first nonzero
#     return code is -1, and none returns to 0;
#   - S7, the refusal under failure: S8's request with a 1 MiB pool: -2, with exactly one failed instance (the standard
#     factory does not retry it).
# The runs go in parallel (env.sh pjob), each with its own fake and its own TinyGPULog file, then one PASS or FAIL line per
# check; exit 0 only if all pass, 2 if nothing ran.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
require_no_launch_guard
GUARD_BIN="$BEAGLE_BUILD/libhmsbeagle/GPU/CMake_TinyGPU/beagle-tinygpu-guard"
[ -x "$GUARD_BIN" ] || { echo "no $GUARD_BIN; build beagle-tinygpu-guard first"; exit 2; }
CHIP=${FAKE_AMD_CHIP:-gfx1100}
W="$TINYGPU_TEST_WORK/s2_$CHIP"; rm -rf "$W"; mkdir -p "$W"   # the chip's own: both chips' passes may run at once
export TG_PIDS="$W/pids"   # each fake and guard these runs start (env.sh track_pid), for the last check
pjob_init "$W"
SPECTRAL=0; grep -q '"SPEC_' "$GPU_DIR/kernels/TinyGPUAMDHsaco.h" 2>/dev/null && SPECTRAL=1   # the build's SPEC_ HSACOs
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

# test_a7.sh's run(), for any program: one run on a fresh fake card, then its guard (waited for, or ended if it holds)
run() {   # <label> <variants, comma-separated> <drop at, or 0> <program> [args ...]
    local l=$1 v=$2 drop=$3 bin=$4; shift 4
    local d; d=$(mktemp -d /tmp/tgs2.XXXXXX)
    mkdir -p "$W/mem_$l"; : > "$W/$l.tglog"
    FAKE_AMD_HSACO=$v FAKE_AMD_DROP_AT=$drop "$BEAGLE_PYTHON" "$TG_TESTS/fake_amd_device.py" "$d/dev.sock" "$W/mem_$l" > "$W/$l.dev" 2>&1 &
    local srv=$!; track_pid $srv fake_amd_device
    for i in $(seq 300); do grep -q listening "$W/$l.dev" 2>/dev/null && break; kill -0 $srv 2>/dev/null || break; sleep 0.1; done
    env BEAGLE_TINYGPU_NO_LAUNCH=1 BEAGLE_TINYGPU_NO_DOWNLOAD=1 BEAGLE_TINYGPU_LOG="$W/$l.tglog" APL_REMOTE_SOCK="$d/dev.sock" TMPDIR="$d" \
        BEAGLE_AMD_GUARD="$TG_TESTS/replay/crash_guard_wrap.sh" BEAGLE_TG_GUARD_BIN="$GUARD_BIN" BEAGLE_TG_GUARD_PIDFILE="$d/guard.pid" \
        BEAGLE_AMD_DATA_MB=${POOL_MB:-512} BEAGLE_AMD_PROFILE=1 DYLD_LIBRARY_PATH="$TEST_LIBS" "$bin" "$@" > "$W/$l.out" 2> "$W/$l.err" < /dev/null
    echo $? > "$W/$l.rc"
    local gpid; gpid=$(cat "$d/guard.pid" 2>/dev/null); track_pid "$gpid" beagle-tinygpu-guard
    if [ -n "$gpid" ]; then   # it exits at the plugin's clean; one that holds keeps the fake's connection
        for i in $(seq 600); do kill -0 "$gpid" 2>/dev/null || break; grep -q "then kill $gpid\." "$W/$l.tglog" && break; sleep 0.1; done
        if kill -0 "$gpid" 2>/dev/null; then
            echo "[$l] the guard (pid $gpid) held the fake connection; ending it" > "$W/$l.held"; kill -KILL "$gpid"
            for i in $(seq 50); do kill -0 "$gpid" 2>/dev/null || break; sleep 0.1; done   # not this shell's child: polled
        fi
    fi
    # the fake's verdict on each session it served, which it prints after their "client done": the resource listing's, and
    # the run's if the program booted the card; then the fake is ended
    local need=1; booted $l && need=2
    for i in $(seq 100); do
        awk -v n=$need '/ device\): client done: /{c++} / device\): (NO ERRORS|[0-9]+ ERRORS)/{v++} END{exit !(c == v && v >= n)}' "$W/$l.dev" && break
        sleep 0.1
    done
    kill $srv 2>/dev/null; wait $srv 2>/dev/null; rm -rf "$d" "$W/mem_$l"
}
probe() { local l=$1 v=$2; shift 2; run $l $v 0 "$W/spectral_probe" "$@"; }
booted() { cat "$W/$1.err" "$W/$1.tglog" | grep -qE "C\+\+ boot|programs loaded"; }
CLEAN="the plugin finalized the GPU itself; exiting"
clean() { grep -q "$CLEAN" "$W/$1.tglog" && grep -E "fake TinyGPU.app \(AMD device\): (NO ERRORS|[0-9]+ ERRORS)" "$W/$1.dev" | tail -1 | grep -q "NO ERRORS"; }
loaded_once() {   # <label> <variant ...>: exactly these variants' programs loaded, each once
    local v n=0; for v in "${@:2}"; do [ "$(grep -c "C++ runtime: $v's programs loaded" "$W/$1.err")" -eq 1 ] || return 1; n=$((n + 1)); done
    [ "$(grep -c "C++ runtime: .*'s programs loaded" "$W/$1.err")" -eq $n ]
}
padded() {   # BeagleGPUImpl's padded state count
    local n=$1 p; for p in 4 16 32 48 64 80 128 192 256; do [ $n -le $p ] && { echo $p; return; }; done; echo $((n + n % 16))
}
variants() {   # <s1_runs.txt command>: the SPEC_ variants its instances use, comma-separated
    local cmd=$1 p=DP s states v=""
    case $cmd in
        adjointtest4*) echo SPEC_SP_4,SPEC_SP_16,SPEC_SP_32,SPEC_SP_64; return ;;   # --all: 4, 16, 17 and 56 states, single
    esac
    grep -q -- "--single" <<< "$cmd" && p=SP
    states=$(sed -nE 's/.*--states ([0-9 ]+).*/\1/p' <<< "$cmd"); [ -n "$states" ] || states="4 17 61 70 150 256"   # its default
    for s in $states; do v="$v,SPEC_${p}_$(padded $s)"; done
    echo "${v#,}"
}
s5() {   # the smallest pool P (bisection) that holds the instance, then P - 8
    local lo=1 hi=1024 m
    s5_fits() { POOL_MB=$1 probe s2_s5_$1 SPEC_SP_64 --resource 1 --eigen 1024 64:s:spec; grep -q '^64:s:spec rc=0 ' "$W/s2_s5_$1.out"; }
    s5_fits $hi || { echo 0 > "$W/s5.P"; return; }
    while [ $((hi - lo)) -gt 1 ]; do m=$(( (lo + hi) / 2 )); if s5_fits $m; then hi=$m; else lo=$m; fi; done
    echo $hi > "$W/s5.P"
    POOL_MB=$((hi - 8)) probe s2_s5_short SPEC_SP_64 --resource 1 --eigen 1024 64:s:spec
}
requests() {   # <label>: the run's session's requests, its counts' "cmd <n>" entries summed (FAKE_AMD_DROP_AT counts these)
    grep "client done" "$W/$1.dev" | tail -1 | grep -oE '"cmd [0-9]+": [0-9]+' | awk -F': ' '{s += $2} END {print s + 0}'
}
s6() {   # 1 and 200 adjoints undropped, then TinyGPU.app gone halfway through the 200 (the boot's requests come first)
    probe s2_s6_one SPEC_SP_4 --resource 1 --adjoint-reps 1 4:s:spec
    probe s2_s6_full SPEC_SP_4 --resource 1 --adjoint-reps 200 4:s:spec
    local one full; one=$(requests s2_s6_one); full=$(requests s2_s6_full)
    echo "$full" > "$W/s6.requests"
    run s2_s6_drop SPEC_SP_4 $(( one + (full - one) / 2 )) "$W/spectral_probe" --resource 1 --adjoint-reps 200 4:s:spec
}
# an s1_runs.txt line: a 2 GiB pool, as test_a7.sh's D1 lines in double: SPEC_DP_192's scratch alone is 297 MiB on gfx1100
line() { local l=$1 v=$2 bin=$3; shift 3; POOL_MB=2048 run $l $v 0 "$bin" "$@"; }
s7() { POOL_MB=1 probe s2_s7 SPEC_SP_4 --resource 1 --req 'GPU|TINYGPU|SINGLE' --pref SPECTRAL 4:s:std; }   # a 1 MiB pool
adjoint_lost() {   # <label>: the first nonzero adjoint return code is -1, and none returns to 0 after it
    awk '/^4:s:spec adjoint [0-9]+: rc=/ { split($0, a, "rc="); rc = a[2] + 0
             if (seen && rc == 0) bad = 1; if (rc != 0 && !seen) { seen = 1; if (rc != -1) bad = 1 } }
         END { exit !(seen && !bad) }' "$W/$1.out"
}

# the runs, the longest first
if [ $SPECTRAL = 1 ]; then
    while IFS='|' read -r label cmd kernels; do
        pjob s2_$label line s2_$label $(variants "$cmd") "$BEAGLE_BUILD/examples/${cmd%% *}" ${cmd#* }
    done < <(grep -E '^[a-z0-9_]+\|' "$TG_TESTS/s1_runs.txt")
    pjob s2_s5 s5
    pjob s2_s6 s6
    pjob s2_mixed probe s2_mixed SP_4,SPEC_SP_4,SP_64,SPEC_SP_64,DP_4,SPEC_DP_4,DP_64,SPEC_DP_64 \
        --resource 1 4:s:std 4:s:spec 64:s:std 64:s:spec 4:d:std 4:d:spec 64:d:std 64:d:spec
    pjob s2_s7 s7
fi
pjob s2_refusal probe s2_refusal SP_4 --resource 1 --req 'GPU|TINYGPU|SINGLE|DYNAMIC' --pref SPECTRAL 4:s:std
pjob s2_route probe s2_route DP_64,SPEC_DP_64 --pref 'GPU|DOUBLE' --req TRANSPOSE_AUTO 61:d:std 61:d:spec
pwait

check "the refusal on $CHIP: a standard request preferring SPECTRAL gets BEAGLE_ERROR_NO_RESOURCE, with no boot and no spectral kernel looked up" \
    "grep -q '^4:s:std rc=-6 impl=none' $W/s2_refusal.out && ! booted s2_refusal && ! grep -q 'not found in the build' $W/s2_refusal.err"
if [ $SPECTRAL = 1 ]; then
    check "the routing (S8) on $CHIP: BEAST's pre-order request picks no TinyGPU implementation, and with SPECTRAL required TinyGPU-Double-Spectral" \
        "grep -q '^61:d:std rc=' $W/s2_route.out && ! grep '^61:d:std rc=' $W/s2_route.out | grep -q TinyGPU \
         && grep -q '^61:d:spec rc=[0-9]* impl=TinyGPU-Double-Spectral$' $W/s2_route.out"
    while IFS='|' read -r label cmd kernels; do
        l=s2_$label
        why=$(amd_spectral_verdict "$W/$l.out" "$W/$l.err" "$kernels" ${cmd%% *})
        check "$label on $CHIP ($(variants "$cmd")): one boot, its SPEC_ variants only, exactly the line's kernels, the card finalized${why:+ ($why)}" \
            "amd_spectral_verdict '$W/$l.out' '$W/$l.err' '$kernels' ${cmd%% *} > /dev/null && clean $l"
    done < <(grep -E '^[a-z0-9_]+\|' "$TG_TESTS/s1_runs.txt")
    check "S2 on $CHIP, mixed variants: 4 standard and 4 spectral instances alive at once, one boot, SP_4, SPEC_SP_4, SP_64, SPEC_SP_64 and their DP_ twins each loaded once" \
        "[ \"\$(grep -c 'TinyGPU/AMD: C++ boot done' $W/s2_mixed.err)\" -eq 1 ] \
         && loaded_once s2_mixed SP_4 SPEC_SP_4 SP_64 SPEC_SP_64 DP_4 SPEC_DP_4 DP_64 SPEC_DP_64 \
         && [ \"\$(grep -cE '^(4|64):[sd]:std rc=[0-9]+ impl=TinyGPU-(Single|Double)$' $W/s2_mixed.out)\" -eq 4 ] \
         && [ \"\$(grep -cE '^(4|64):[sd]:spec rc=[0-9]+ impl=TinyGPU-(Single|Double)-Spectral$' $W/s2_mixed.out)\" -eq 4 ] \
         && ! grep -q 'not found in the build' $W/s2_mixed.err && clean s2_mixed"
    P=$(cat "$W/s5.P" 2>/dev/null)
    check "S5 on $CHIP, out of memory in a spectral pool: created at a ${P:-?} MiB pool; at $(( ${P:-0} - 8 )) MiB -2, after SPEC_SP_64's programs loaded, on an allocation of 16.0 MiB" \
        "[ \"${P:-0}\" -gt 8 ] && grep -q '^64:s:spec rc=-2 impl=none' $W/s2_s5_short.out \
         && awk '/SPEC_SP_64.s programs loaded/{k=NR} /out of GPU memory: an allocation of 16.0 MiB/{if (k) o=NR} END{exit !o}' $W/s2_s5_short.err"
    check "S6 on $CHIP, the GPU lost during the adjoint ($(cat "$W/s6.requests" 2>/dev/null) requests undropped): every adjoint 0 before; after the drop the first nonzero rc is -1, and none returns to 0" \
        "grep -q '^4:s:spec rc=[0-9]* impl=TinyGPU-Single-Spectral$' $W/s2_s6_full.out \
         && [ \"\$(grep -c '^4:s:spec adjoint' $W/s2_s6_full.out)\" -eq 1 ] && grep -q '^4:s:spec adjoint 0: rc=0$' $W/s2_s6_full.out \
         && adjoint_lost s2_s6_drop"
    check "S7 on $CHIP, the refusal under failure: -2 with exactly one failed instance (the spectral attempt's; the standard factory does not retry)" \
        "grep -q '^4:s:std rc=-2 impl=none' $W/s2_s7.out && [ \"\$(grep -c 'this instance fails' $W/s2_s7.err)\" -eq 1 ]"
else
    check "the routing (S8) on $CHIP: BEAST's pre-order requests, with and without SPECTRAL, pick no TinyGPU implementation" \
        "grep -q '^61:d:std rc=' $W/s2_route.out && grep -q '^61:d:spec rc=' $W/s2_route.out \
         && ! grep -E '^61:d:(std|spec) rc=' $W/s2_route.out | grep -q TinyGPU"
fi

left=$(tracked_alive)   # this stage's own processes only: other stages' fakes and guards may be running
check "no harness process or guard is left running" "[ -z \"\$left\" ]"
echo; [ $fails -eq 0 ] && echo "test_s2 ($CHIP): PASS" || echo "test_s2 ($CHIP): $fails FAILED"
[ $fails -eq 0 ]
