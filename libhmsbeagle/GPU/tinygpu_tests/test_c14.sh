#!/bin/bash
# TODO.md plan step C14, offline: GPUInterface::FreeMemory, on both vendors' VRAM pools (TinyGPUPool.h).
#   - test_c14_pool.cpp: the free list under both pool allocators (no frees: the addresses before C14; random allocations
#     and frees: first fit, nothing overlapping or lost; instance cycles at the first cycle's addresses);
#   - on fake_nv_device.py's AD107 (run_fake_device.sh) and fake_amd_device.py's card (through the crash guard, each dispatch
#     checked against the build's HSACOs of the run's variants, so an image that reused memory overwrote fails it): 20 cycles
#     of two instances at 4 and 64 states in one process on a pool that holds a few instances at once (BEAGLE_NV_DATA_MB=64,
#     BEAGLE_AMD_DATA_MB=64) all run, every cycle's instances reading back their own tips (on NV, every cycle's programs
#     at the first cycle's addresses), and the GPU is torn down cleanly; the same 40 instances alive at once on that pool
#     fail with BEAGLE_ERROR_OUT_OF_MEMORY, so the cycles ran on memory the earlier ones freed.
# The fakes run no kernel. The runs go in parallel (env.sh's pjob; test_c14_pool as two processes, its NV and its AMD half),
# the longest first; then one PASS or FAIL line per check, in this order; exit 0 only if all pass.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
require_no_launch_guard
GUARD_BIN="$BEAGLE_BUILD/libhmsbeagle/GPU/CMake_TinyGPU/beagle-tinygpu-guard"
[ -x "$GUARD_BIN" ] || { echo "no $GUARD_BIN; build beagle-tinygpu-guard first"; exit 2; }
W="$TINYGPU_TEST_WORK/c14"; rm -rf "$W"; mkdir -p "$W"
export TG_PIDS="$W/pids"   # the fakes and guards the runs start (env.sh's track_pid), for the check that none is left running
pjob_init "$W"
CHIP=${FAKE_AMD_CHIP:-gfx1100}   # the fake AMD card's (fake_amd_device.py reads FAKE_AMD_CHIP), in its runs' labels
fails=0
pass() { echo "PASS $1"; }
fail() { echo "FAIL $1"; fails=$((fails + 1)); }
check() { if eval "$2"; then pass "$1"; else fail "$1"; fi; }

# 1. the free list
c++ -std=c++17 -O1 -Wall -Wextra -I"$REPO" -o "$W/test_c14_pool" "$TG_TESTS/test_c14_pool.cpp" || exit 2
pool() { "$W/test_c14_pool" $1 > "$W/test_c14_pool_$1.txt"; }   # <nv|amd>: one vendor's half of its checks

# 2. NV, on the fake AD107
out() { echo "$TINYGPU_TEST_WORK/run_device_$1.txt"; }   # the plugin's and the test's output (run_fake_device.sh's)
nv() {   # <label> [VAR=value ...] [-- test args]: one fake run; its verdict is run_fake_device.sh's
    local l=$1 envs=(); shift
    while [ $# -gt 0 ] && [ "$1" != "--" ]; do envs+=("$1"); shift; done
    [ "$1" = "--" ] && shift
    "$TG_TESTS/run_fake_device.sh" $l "${envs[@]}" -- "$@" > "$W/$l.txt" 2>&1 < /dev/null
}

# 3. AMD, on the fake card
glog() { cat "$W/$1.tglog"; }   # one run's TinyGPULog lines (the plugin's and its guard's)
verdict() { grep -E "fake TinyGPU.app \(AMD device\): (NO ERRORS|[0-9]+ ERRORS)" "$W/$1.dev" | tail -1; }
CLEAN="the plugin finalized the GPU itself; exiting"
amd() {   # <label> <variants, comma-separated> [args ...]: one run on a fresh fake card, then its guard (waited for, or ended if it holds)
    local l=$1 v=$2; shift 2
    local d; d=$(mktemp -d /tmp/tg14.XXXXXX)
    rm -f "$W/$l.tglog"; mkdir -p "$W/mem_$l"
    FAKE_AMD_HSACO=$v "$BEAGLE_PYTHON" "$TG_TESTS/fake_amd_device.py" "$d/dev.sock" "$W/mem_$l" > "$W/$l.dev" 2>&1 &
    local srv=$!; track_pid $srv fake_amd_device
    for i in $(seq 300); do grep -q listening "$W/$l.dev" 2>/dev/null && break; sleep 0.1; done
    env BEAGLE_TINYGPU_NO_LAUNCH=1 BEAGLE_TINYGPU_NO_DOWNLOAD=1 BEAGLE_TINYGPU_LOG="$W/$l.tglog" APL_REMOTE_SOCK="$d/dev.sock" TMPDIR="$d" \
        BEAGLE_AMD_GUARD="$TG_TESTS/replay/crash_guard_wrap.sh" BEAGLE_TG_GUARD_BIN="$GUARD_BIN" BEAGLE_TG_GUARD_PIDFILE="$d/guard.pid" \
        BEAGLE_AMD_DATA_MB=64 DYLD_LIBRARY_PATH="$TEST_LIBS" "$TEST_BIN" "$@" > "$W/$l.txt" 2>&1 < /dev/null
    echo $? > "$W/$l.rc"
    local gpid; gpid=$(cat "$d/guard.pid" 2>/dev/null); track_pid "$gpid" beagle-tinygpu-guard
    if [ -n "$gpid" ]; then   # it exits at the plugin's clean; one that holds keeps the fake's connection
        for i in $(seq 600); do kill -0 "$gpid" 2>/dev/null || break; grep -q "then kill $gpid\." "$W/$l.tglog" 2>/dev/null && break; sleep 0.1; done
        if kill -0 "$gpid" 2>/dev/null; then
            echo "[$l] the guard (pid $gpid) held the fake connection; ending it" > "$W/$l.held"; kill -KILL "$gpid"
            for i in $(seq 50); do kill -0 "$gpid" 2>/dev/null || break; sleep 0.1; done   # not this shell's child: polled
        fi
    fi
    # the fake's verdict on both sessions (it prints it after each one's "client done"), then the fake is ended
    for i in $(seq 300); do [ "$(grep -cE 'fake TinyGPU.app \(AMD device\): (NO ERRORS|[0-9]+ ERRORS)' "$W/$l.dev")" -ge 2 ] && break; sleep 0.1; done
    kill $srv 2>/dev/null; wait $srv 2>/dev/null; rm -rf "$d" "$W/mem_$l"
}
loaded() {   # <label> <variants, comma-separated>: exactly these variants' programs loaded, each once
    local v n=0; for v in ${2//,/ }; do [ "$(grep -c "C++ runtime: $v's programs loaded" "$W/$1.txt")" -eq 1 ] || return 1; n=$((n + 1)); done
    [ "$(grep -c "C++ runtime: [SD]P_[0-9]*'s programs loaded" "$W/$1.txt")" -eq $n ]
}

# the runs, the longest first
pjob c14_pool_amd pool amd
pjob c14_pool_nv pool nv
pjob c14_nv_cycles nv c14_nv_cycles BEAGLE_NV_DATA_MB=64 -- --cycles 20 --state-count 4,64 --reps 3
pjob c14_nv_live nv c14_nv_live BEAGLE_NV_DATA_MB=64 -- --instances 40 --state-count 4,64 --reps 3
pjob c14_${CHIP}_cycles amd c14_${CHIP}_cycles SP_4,SP_64 --cycles 20 --state-count 4,64 --reps 3
pjob c14_${CHIP}_live amd c14_${CHIP}_live SP_4,SP_64 --instances 40 --state-count 4,64 --reps 3
pwait

# 1. the free list: the two halves' lines in the order one process prints them (NV, AMD, NV, ...)
paste -d '\n' <(grep -E "^(PASS|FAIL) " "$W/test_c14_pool_nv.txt") <(grep -E "^(PASS|FAIL) " "$W/test_c14_pool_amd.txt") | grep -v '^$' > "$W/test_c14_pool.txt"
cat "$W/test_c14_pool.txt"
fails=$((fails + $(grep -c "^FAIL " "$W/test_c14_pool.txt")))
r=0; for h in nv amd; do [ "$(pjob_rc c14_pool_$h)" -eq 0 ] || r=1; done
check "test_c14_pool: exit 0" "[ $r -eq 0 ]"

# 2. NV
r=$(pjob_rc c14_nv_cycles)
check "NV: 20 cycles of two instances (4 and 64 states) on a 64 MiB pool: all run, each cycle's tips read back exactly, every cycle's programs at the first cycle's two addresses, the GPU torn down cleanly" \
    "[ $r -eq 0 ] && [ \"\$(grep -c '^tips: every instance read back its own tip partials exactly' '$(out c14_nv_cycles)')\" -eq 20 ] \
     && [ \"\$(grep -c 'kernels loaded (image' '$(out c14_nv_cycles)')\" -eq 40 ] \
     && [ \"\$(grep -o 'kernels loaded (image [0-9]* bytes at 0x[0-9a-f]*' '$(out c14_nv_cycles)' | sort -u | wc -l)\" -eq 2 ] \
     && ! grep -q 'out of GPU memory' '$(out c14_nv_cycles)'"
check "NV: the same 40 instances alive at once on that pool: BEAGLE_ERROR_OUT_OF_MEMORY, said, and the GPU torn down cleanly (NO ERRORS)" \
    "grep -q 'out of GPU memory: an allocation of [0-9.]* MiB, with [0-9.]* MiB left of the 64 MiB VRAM pool' '$(out c14_nv_live)' \
     && grep -q 'beagleCreateInstance failed (error -2)' '$(out c14_nv_live)' && fini_verdict '$(out c14_nv_live)' \
     && grep -q 'fake TinyGPU.app (AD107 device): NO ERRORS' '$TINYGPU_TEST_WORK/fake_device_c14_nv_live.log'"

# 3. AMD
l=c14_${CHIP}_cycles
check "AMD: 20 cycles of two instances (4 and 64 states) on a 64 MiB pool: all run, each cycle's tips read back exactly, SP_4 and SP_64 loaded once, every dispatch's image intact, the card finalized (NO ERRORS)" \
    "[ \"\$(cat $W/$l.rc)\" -lt 128 ] && [ \"\$(grep -c '^tips: every instance read back its own tip partials exactly' $W/$l.txt)\" -eq 20 ] \
     && loaded $l SP_4,SP_64 && ! grep -q 'out of GPU memory' $W/$l.txt \
     && glog $l | grep -q '$CLEAN' && verdict $l | grep -q 'NO ERRORS'"
l=c14_${CHIP}_live
check "AMD: the same 40 instances alive at once on that pool: BEAGLE_ERROR_OUT_OF_MEMORY, said, and the card finalized (NO ERRORS)" \
    "grep -q 'out of GPU memory: an allocation of [0-9.]* MiB, with [0-9.]* MiB left of the 64 MiB VRAM pool' $W/$l.txt \
     && grep -q 'beagleCreateInstance failed (error -2)' $W/$l.txt \
     && glog $l | grep -q '$CLEAN' && verdict $l | grep -q 'NO ERRORS'"

left=$(tracked_alive)   # this stage's own processes only: other stages' fakes and guards may be running
check "no harness process or guard is left running" "[ -z \"\$left\" ]"
echo; [ $fails -eq 0 ] && echo "test_c14: PASS" || echo "test_c14: $fails FAILED"
[ $fails -eq 0 ]
