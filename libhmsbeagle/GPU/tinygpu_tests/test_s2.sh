#!/bin/bash
# TODO.md plan step S3.4 (PLAN-spectral-tinygpu-degree2.md), offline: test_s1.sh's checks on the AMD card, through
# spectral_probe.cpp on fake_amd_device.py's card (FAKE_AMD_CHIP, gfx1100 by default), through the crash guard:
#   - the refusal: a standard request (single precision, GPU, TinyGPU and DYNAMIC scaling required, which no spectral
#     factory offers) that prefers SPECTRAL is refused by the standard factory: BEAGLE_ERROR_NO_RESOURCE (-6), no boot, and no spectral kernel looked up in a standard variant;
#   - the routing (plan step S8's baseline): BEAST's pre-order requests at 61 states with no resource list, preferring
#     GPU and DOUBLE and requiring PREORDER_TRANSPOSE_AUTO, then the same also requiring SPECTRAL: neither picks TinyGPU
#     while the plugin has no spectral variants (GetSupportsSpectral).
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
fails=0
pass() { echo "PASS $1"; }
fail() { echo "FAIL $1"; fails=$((fails + 1)); }
check() { if eval "$2"; then pass "$1"; else fail "$1"; fi; }

c++ -std=c++17 -O1 -Wall -Wextra -I"$REPO" -o "$W/spectral_probe" "$TG_TESTS/spectral_probe.cpp" \
    -L"$BEAGLE_BUILD/libhmsbeagle" -lhmsbeagle -Wl,-rpath,"$BEAGLE_BUILD/libhmsbeagle" || exit 2

# test_a7.sh's run(), for the probe: one run on a fresh fake card, then its guard (waited for, or ended if it holds)
run() {   # <label> <variant> [probe args ...]
    local l=$1 v=$2; shift 2
    local d; d=$(mktemp -d /tmp/tgs2.XXXXXX)
    mkdir -p "$W/mem_$l"; : > "$W/$l.tglog"
    FAKE_AMD_HSACO=$v "$BEAGLE_PYTHON" "$TG_TESTS/fake_amd_device.py" "$d/dev.sock" "$W/mem_$l" > "$W/$l.dev" 2>&1 &
    local srv=$!; track_pid $srv fake_amd_device
    for i in $(seq 300); do grep -q listening "$W/$l.dev" 2>/dev/null && break; kill -0 $srv 2>/dev/null || break; sleep 0.1; done
    env BEAGLE_TINYGPU_NO_LAUNCH=1 BEAGLE_TINYGPU_NO_DOWNLOAD=1 BEAGLE_TINYGPU_LOG="$W/$l.tglog" APL_REMOTE_SOCK="$d/dev.sock" TMPDIR="$d" \
        BEAGLE_AMD_GUARD="$TG_TESTS/replay/crash_guard_wrap.sh" BEAGLE_TG_GUARD_BIN="$GUARD_BIN" BEAGLE_TG_GUARD_PIDFILE="$d/guard.pid" \
        BEAGLE_AMD_DATA_MB=${POOL_MB:-512} BEAGLE_AMD_PROFILE=1 DYLD_LIBRARY_PATH="$TEST_LIBS" "$W/spectral_probe" "$@" > "$W/$l.out" 2> "$W/$l.err" < /dev/null
    echo $? > "$W/$l.rc"
    local gpid; gpid=$(cat "$d/guard.pid" 2>/dev/null); track_pid "$gpid" beagle-tinygpu-guard
    if [ -n "$gpid" ]; then   # it exits at the plugin's clean; one that holds keeps the fake's connection
        for i in $(seq 600); do kill -0 "$gpid" 2>/dev/null || break; grep -q "then kill $gpid\." "$W/$l.tglog" && break; sleep 0.1; done
        if kill -0 "$gpid" 2>/dev/null; then
            echo "[$l] the guard (pid $gpid) held the fake connection; ending it" > "$W/$l.held"; kill -KILL "$gpid"
            for i in $(seq 50); do kill -0 "$gpid" 2>/dev/null || break; sleep 0.1; done   # not this shell's child: polled
        fi
    fi
    # the fake's verdict on each session it served (it prints one after each "client done"), then the fake is ended
    for i in $(seq 100); do
        awk '/ device\): client done: /{c++} / device\): (NO ERRORS|[0-9]+ ERRORS)/{v++} END{exit c != v}' "$W/$l.dev" && break
        sleep 0.1
    done
    kill $srv 2>/dev/null; wait $srv 2>/dev/null; rm -rf "$d" "$W/mem_$l"
}
booted() { cat "$W/$1.err" "$W/$1.tglog" | grep -qE "C\+\+ boot|programs loaded"; }

pjob s2_refusal run s2_refusal SP_4 --resource 1 --req 'GPU|TINYGPU|SINGLE|DYNAMIC' --pref SPECTRAL 4:s:std
pjob s2_route run s2_route DP_64 --pref 'GPU|DOUBLE' --req TRANSPOSE_AUTO 61:d:std 61:d:spec
pwait

check "the refusal on $CHIP: a standard request preferring SPECTRAL gets BEAGLE_ERROR_NO_RESOURCE, with no boot and no spectral kernel looked up" \
    "grep -q '^4:s:std rc=-6 impl=none' $W/s2_refusal.out && ! booted s2_refusal && ! grep -q 'not found in the build' $W/s2_refusal.err"
check "the routing (S8) on $CHIP: BEAST's pre-order requests, with and without SPECTRAL, pick no TinyGPU implementation" \
    "grep -q '^61:d:std rc=' $W/s2_route.out && grep -q '^61:d:spec rc=' $W/s2_route.out \
     && ! grep -E '^61:d:(std|spec) rc=' $W/s2_route.out | grep -q TinyGPU"

left=$(tracked_alive)   # this stage's own processes only: other stages' fakes and guards may be running
check "no harness process or guard is left running" "[ -z \"\$left\" ]"
echo; [ $fails -eq 0 ] && echo "test_s2 ($CHIP): PASS" || echo "test_s2 ($CHIP): $fails FAILED"
[ $fails -eq 0 ]
