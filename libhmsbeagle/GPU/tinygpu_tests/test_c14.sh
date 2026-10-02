#!/bin/bash
# TODO.md plan step C14, offline: GPUInterface::FreeMemory, on both vendors' VRAM pools (TinyGPUPool.h).
#   - test_c14_pool.cpp: the free list under both pool allocators (no frees: the addresses before C14; random allocations
#     and frees: first fit, nothing overlapping or lost; instance cycles at the first cycle's addresses);
#   - on fake_nv_device.py's AD107 (run_fake_device.sh) and fake_amd_device.py's card (through the crash guard, each dispatch
#     checked against the build's HSACOs of the run's variants, so an image that reused memory overwrote fails it): 20 cycles
#     of two instances at 4 and 64 states in one process on a pool that holds a few instances at once (BEAGLE_NV_DATA_MB=64,
#     BEAGLE_AMD_DATA_MB=192) all run, every cycle's instances reading back their own tips (on NV, every cycle's programs
#     at the first cycle's addresses), and the GPU is torn down cleanly; the same 40 instances alive at once on that pool
#     fail with BEAGLE_ERROR_OUT_OF_MEMORY, so the cycles ran on memory the earlier ones freed.
# The fakes run no kernel. One PASS or FAIL line per check; exit 0 only if all pass.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
require_no_launch_guard
GUARD_BIN="$BEAGLE_BUILD/libhmsbeagle/GPU/CMake_TinyGPUHybrid/beagle-tinygpu-guard"
[ -x "$GUARD_BIN" ] || { echo "no $GUARD_BIN; build beagle-tinygpu-guard first"; exit 2; }
W="$TINYGPU_TEST_WORK/c14"; rm -rf "$W"; mkdir -p "$W"
TL="$W/beagle_tinygpu.log"   # the AMD runs' TinyGPULog lines
fails=0
pass() { echo "PASS $1"; }
fail() { echo "FAIL $1"; fails=$((fails + 1)); }
check() { if eval "$2"; then pass "$1"; else fail "$1"; fi; }

# 1. the free list
c++ -std=c++17 -O1 -Wall -Wextra -I"$REPO" -o "$W/test_c14_pool" "$TG_TESTS/test_c14_pool.cpp" || exit 2
"$W/test_c14_pool" > "$W/test_c14_pool.txt"; r=$?
grep -E "^(PASS|FAIL) " "$W/test_c14_pool.txt"
fails=$((fails + $(grep -c "^FAIL " "$W/test_c14_pool.txt")))
check "test_c14_pool: exit 0" "[ $r -eq 0 ]"

# 2. NV, on the fake AD107
out() { echo "$TINYGPU_TEST_WORK/run_device_$1.txt"; }   # the plugin's and the test's output (run_fake_device.sh's)
nv() {   # <label> [VAR=value ...] [-- test args]: one fake run; its verdict is run_fake_device.sh's
    local l=$1 envs=(); shift
    while [ $# -gt 0 ] && [ "$1" != "--" ]; do envs+=("$1"); shift; done
    [ "$1" = "--" ] && shift
    "$TG_TESTS/run_fake_device.sh" $l "${envs[@]}" -- "$@" > "$W/$l.txt" 2>&1 < /dev/null
}
nv c14_nv_cycles BEAGLE_NV_DATA_MB=64 -- --cycles 20 --state-count 4,64 --reps 3; r=$?
check "NV: 20 cycles of two instances (4 and 64 states) on a 64 MiB pool: all run, each cycle's tips read back exactly, every cycle's programs at the first cycle's two addresses, the GPU torn down cleanly" \
    "[ $r -eq 0 ] && [ \"\$(grep -c '^tips: every instance read back its own tip partials exactly' '$(out c14_nv_cycles)')\" -eq 20 ] \
     && [ \"\$(grep -c 'kernels loaded (image' '$(out c14_nv_cycles)')\" -eq 40 ] \
     && [ \"\$(grep -o 'kernels loaded (image [0-9]* bytes at 0x[0-9a-f]*' '$(out c14_nv_cycles)' | sort -u | wc -l)\" -eq 2 ] \
     && ! grep -q 'out of GPU memory' '$(out c14_nv_cycles)'"
nv c14_nv_live BEAGLE_NV_DATA_MB=64 -- --instances 40 --state-count 4,64 --reps 3
check "NV: the same 40 instances alive at once on that pool: BEAGLE_ERROR_OUT_OF_MEMORY, said, and the GPU torn down cleanly (NO ERRORS)" \
    "grep -q 'out of GPU memory: an allocation of [0-9.]* MiB, with [0-9.]* MiB left of the 64 MiB VRAM pool' '$(out c14_nv_live)' \
     && grep -q 'beagleCreateInstance failed (error -2)' '$(out c14_nv_live)' && fini_verdict '$(out c14_nv_live)' \
     && grep -q 'fake TinyGPU.app (AD107 device): NO ERRORS' '$TINYGPU_TEST_WORK/fake_device_c14_nv_live.log'"

# 3. AMD, on the fake card
glog() { sed -n "/c14 run $1 starts/,/c14 run .* starts/p" "$TL"; }   # one run's TinyGPULog lines
verdict() { grep -E "fake TinyGPU.app \(AMD device\): (NO ERRORS|[0-9]+ ERRORS)" "$W/$1.dev" | tail -1; }
CLEAN="the plugin finalized the GPU itself; exiting"
amd() {   # <label> <variants, comma-separated> [args ...]: one run on a fresh fake card, then its guard (waited for, or ended if it holds)
    local l=$1 v=$2; shift 2
    local d; d=$(mktemp -d /tmp/tg14.XXXXXX)
    mkdir -p "$W/mem_$l"
    echo "c14 run $l starts" >> "$TL"
    FAKE_AMD_HSACO=$v "$BEAGLE_PYTHON" "$TG_TESTS/fake_amd_device.py" "$d/dev.sock" "$W/mem_$l" > "$W/$l.dev" 2>&1 &
    local srv=$!
    for i in $(seq 100); do grep -q listening "$W/$l.dev" 2>/dev/null && break; sleep 0.1; done
    env BEAGLE_TINYGPU_NO_LAUNCH=1 BEAGLE_TINYGPU_NO_DOWNLOAD=1 BEAGLE_TINYGPU_LOG="$TL" APL_REMOTE_SOCK="$d/dev.sock" TMPDIR="$d" \
        BEAGLE_AMD_GUARD="$TG_TESTS/replay/crash_guard_wrap.sh" BEAGLE_TG_GUARD_BIN="$GUARD_BIN" BEAGLE_TG_GUARD_PIDFILE="$d/guard.pid" \
        BEAGLE_AMD_DATA_MB=192 DYLD_LIBRARY_PATH="$TEST_LIBS" "$TEST_BIN" "$@" > "$W/$l.txt" 2>&1 < /dev/null
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
    local v n=0; for v in ${2//,/ }; do [ "$(grep -c "C++ runtime: $v's programs loaded" "$W/$1.txt")" -eq 1 ] || return 1; n=$((n + 1)); done
    [ "$(grep -c "C++ runtime: [SD]P_[0-9]*'s programs loaded" "$W/$1.txt")" -eq $n ]
}
amd c14_amd_cycles SP_4,SP_64 --cycles 20 --state-count 4,64 --reps 3
check "AMD: 20 cycles of two instances (4 and 64 states) on a 192 MiB pool: all run, each cycle's tips read back exactly, SP_4 and SP_64 loaded once, every dispatch's image intact, the card finalized (NO ERRORS)" \
    "[ \"\$(cat $W/c14_amd_cycles.rc)\" -lt 128 ] && [ \"\$(grep -c '^tips: every instance read back its own tip partials exactly' $W/c14_amd_cycles.txt)\" -eq 20 ] \
     && loaded c14_amd_cycles SP_4,SP_64 && ! grep -q 'out of GPU memory' $W/c14_amd_cycles.txt \
     && glog c14_amd_cycles | grep -q '$CLEAN' && verdict c14_amd_cycles | grep -q 'NO ERRORS'"
amd c14_amd_live SP_4,SP_64 --instances 40 --state-count 4,64 --reps 3
check "AMD: the same 40 instances alive at once on that pool: BEAGLE_ERROR_OUT_OF_MEMORY, said, and the card finalized (NO ERRORS)" \
    "grep -q 'out of GPU memory: an allocation of [0-9.]* MiB, with [0-9.]* MiB left of the 192 MiB VRAM pool' $W/c14_amd_live.txt \
     && grep -q 'beagleCreateInstance failed (error -2)' $W/c14_amd_live.txt \
     && glog c14_amd_live | grep -q '$CLEAN' && verdict c14_amd_live | grep -q 'NO ERRORS'"

left=$(ps -axo command | grep -cE "Python .*(tgproxy|tgreplay|fake_nv_device|fake_amd_device)\.py|[b]eagle-tinygpu-guard")
check "no harness process or guard is left running" "[ $left -eq 0 ]"
echo; [ $fails -eq 0 ] && echo "test_c14: PASS" || echo "test_c14: $fails FAILED"
[ $fails -eq 0 ]
