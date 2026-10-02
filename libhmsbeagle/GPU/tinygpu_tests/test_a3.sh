#!/bin/bash
# TODO.md plan step A3, end to end with no eGPU: NV's error returns (plan step C12) on the AMD path. BEAGLE returns errors instead
# of exiting its host, and nothing of a failed instance, or of one whose GPU is lost, reaches the GPU. On fake_amd_device.py's card:
#   - a dirty card: the boot is refused before the mode1 reset, beagleCreateInstance returns an error and the test exits
#     normally; the guard exits at the plugin's 'N';
#   - a 1 MiB VRAM pool (BEAGLE_AMD_DATA_MB=1): the program upload fails after the queues are set up; beagleCreateInstance
#     returns an error, and the plugin still finalizes the card at exit (the guard sees its clean; NO ERRORS);
#   - a pool the programs fill exactly (132 MiB at 64 states): the instance's first allocation fails, so it fails instead of
#     handing address 0 to the GPU; an error from beagleCreateInstance, the card finalized;
#   - a GPU fault mid-run (FAKE_AMD_FAULT=1, on a cold card): the SQ MEMVIOL is decoded, the GPU is lost to the process,
#     read-backs are NaN and BEAGLE returns errors; the exit still finalizes the card (NO ERRORS);
#   - a GPU that hangs mid-run (FAKE_AMD_HANG=1, a 3 s wait timeout): the wait times out, BEAGLE returns errors, and the
#     exit's fini dequeues the hung queue (its waves reset) and finalizes the card (NO ERRORS);
#   - TinyGPU.app gone mid-run (FAKE_AMD_DROP_AT): the next request fails (EPIPE, no SIGPIPE death), BEAGLE returns errors, the
#     test exits normally, and the guard holds: the plugin cannot finalize the card over a dead connection;
#   - SIGINT to the test's process group mid-run, as a terminal's Ctrl-C: the test stops (exit 130) and finalizes, the plugin
#     finalizes the card (NO ERRORS), and the guard, in its own session, sees the clean;
#   - a hold, then another instance in the same process (FAKE_AMD_WEDGED=1, --cycles 2): the first instance's boot fails once
#     its queues are live (a pool larger than the VRAM, BEAGLE_AMD_DATA_MB=1000000), its finalize cannot see the compute
#     queue off, and the guard holds; the second beagleCreateInstance is refused at once (Initialize does not connect, so
#     BeagleGPUImpl sees no device: BEAGLE_ERROR_NO_RESOURCE) instead of waiting forever on TinyGPU.app. (Since plan step A5
#     every other hold comes at exit: the card outlives its instances.)
# The test exits 1 on the fake (its logL is wrong by design): "exits normally" is a status below 128 that the test's own
# error handling gave, not the plugin's exit. A guard that holds keeps the fake's connection as it would the eGPU's, so it is
# ended here (offline only). One PASS or FAIL line per check; exit 0 only if all pass.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
require_no_launch_guard
[ -x "$TEST_BIN" ] || { echo "no $TEST_BIN; build it first"; exit 2; }
GUARD_BIN="$BEAGLE_BUILD/libhmsbeagle/GPU/CMake_TinyGPUHybrid/beagle-tinygpu-guard"
[ -x "$GUARD_BIN" ] || { echo "no $GUARD_BIN; build beagle-tinygpu-guard first"; exit 2; }
W="$TINYGPU_TEST_WORK/a3"; rm -rf "$W"; mkdir -p "$W"
TL="$W/beagle_tinygpu.log"   # the plugin's and the guard's TinyGPULog lines in these runs
fails=0
pass() { echo "PASS $1"; }
fail() { echo "FAIL $1"; fails=$((fails + 1)); }
check() { if eval "$2"; then pass "$1"; else fail "$1"; fi; }
glog() { sed -n "/a3 run $1 starts/,/a3 run .* starts/p" "$TL"; }   # one run's TinyGPULog lines
verdict() { grep -E "fake TinyGPU.app \(AMD device\): (NO ERRORS|[0-9]+ ERRORS)" "$W/$1.dev" | tail -1; }
status() { cat "$W/$1.rc"; }
CLEAN="the plugin finalized the GPU itself; exiting"

run() {   # <label> [VAR=value ...] [-- tinygpuhybridtest args]: one run on a fresh fake card, then its guard: waited for, or
          # ended if it holds. SIGINT_AFTER=<regex> among the VARs: SIGINT to the test's process group once its output matches.
    local l=$1 envs=() args=(--state-count 4 --reps 3) sigint=""; shift
    while [ $# -gt 0 ] && [ "$1" != "--" ]; do
        if [[ $1 == SIGINT_AFTER=* ]]; then sigint=${1#SIGINT_AFTER=}; else envs+=("$1"); fi
        shift
    done
    [ "$1" = "--" ] && { shift; args=("$@"); }
    local d; d=$(mktemp -d /tmp/tg3.XXXXXX)
    mkdir -p "$W/mem_$l"
    echo "a3 run $l starts" >> "$TL"
    env "${envs[@]}" "$BEAGLE_PYTHON" "$TG_TESTS/fake_amd_device.py" "$d/dev.sock" "$W/mem_$l" > "$W/$l.dev" 2>&1 &
    local srv=$!
    for i in $(seq 100); do grep -q listening "$W/$l.dev" 2>/dev/null && break; sleep 0.1; done
    # its own process group (perl before env: macOS strips DYLD_LIBRARY_PATH from a system binary's environment)
    perl -e 'setpgrp(0, 0); exec @ARGV or die "exec: $!"' env BEAGLE_TINYGPU_NO_LAUNCH=1 BEAGLE_TINYGPU_NO_DOWNLOAD=1 BEAGLE_TINYGPU_LOG="$TL" \
        APL_REMOTE_SOCK="$d/dev.sock" TMPDIR="$d" BEAGLE_AMD_GUARD="$TG_TESTS/replay/crash_guard_wrap.sh" BEAGLE_TG_GUARD_BIN="$GUARD_BIN" \
        BEAGLE_TG_GUARD_PIDFILE="$d/guard.pid" BEAGLE_AMD_DATA_MB=512 DYLD_LIBRARY_PATH="$TEST_LIBS" "${envs[@]}" \
        "$TEST_BIN" "${args[@]}" > "$W/$l.txt" 2>&1 &
    local tst=$! rc=124
    for i in $(seq 1800); do
        kill -0 $tst 2>/dev/null || { wait $tst; rc=$?; break; }
        if [ -n "$sigint" ] && grep -qE "$sigint" "$W/$l.txt"; then sigint=""; kill -INT -- -$tst; echo "SIGINT sent" > "$W/$l.sigint"; fi
        sleep 0.1
    done
    [ $rc -eq 124 ] && { kill -KILL $tst 2>/dev/null; wait $tst 2>/dev/null; echo "[$l] the test hung; killed (fake device only)" > "$W/$l.hung"; }
    echo $rc > "$W/$l.rc"
    local gpid; gpid=$(cat "$d/guard.pid" 2>/dev/null)
    if [ -n "$gpid" ]; then   # it exits at the plugin's clean or 'N'; one that holds keeps the fake's connection
        for i in $(seq 600); do kill -0 "$gpid" 2>/dev/null || break; grep -q "then kill $gpid\." "$TL" && break; sleep 0.1; done
        if kill -0 "$gpid" 2>/dev/null; then
            echo "[$l] the guard (pid $gpid) held the fake connection; ending it" > "$W/$l.held"; kill -KILL "$gpid"
            for i in $(seq 50); do kill -0 "$gpid" 2>/dev/null || break; sleep 0.1; done   # not this shell's child: polled
        fi
    fi
    for i in $(seq 100); do [ "$(grep -c 'client done' "$W/$l.dev")" -ge 2 ] && break; sleep 0.1; done
    kill $srv 2>/dev/null; wait $srv 2>/dev/null; rm -rf "$d"
}
normal() { [ "$(status $1)" -lt 128 ] && [ ! -f "$W/$1.hung" ] && ! grep -q "TinyGPU/AMD: .*exiting$" "$W/$1.txt"; }

# 1. a dirty card
run dirty FAKE_AMD_STATE=dirty
check "a dirty card: refused before the mode1 reset, beagleCreateInstance returns an error, the test exits normally, the guard exits ('N')" \
    "normal dirty && grep -q 'needs an SMU mode1 reset' $W/dirty.txt && grep -q 'beagleCreateInstance failed (error -1)' $W/dirty.txt \
     && glog dirty | grep -q 'boot ended with no queue ever live' && verdict dirty | grep -q 'NO ERRORS' && [ ! -f $W/dirty.held ]"

# 2. a 1 MiB VRAM pool: the program upload fails once the queues are live
run pool1 BEAGLE_AMD_DATA_MB=1
check "a 1 MiB VRAM pool: beagleCreateInstance returns an error, and the plugin still finalizes the card at exit (NO ERRORS)" \
    "normal pool1 && grep -q 'too small' $W/pool1.txt && grep -q 'beagleCreateInstance failed (error -1)' $W/pool1.txt \
     && glog pool1 | grep -q '$CLEAN' && verdict pool1 | grep -q 'NO ERRORS'"

# 3. a pool the programs fill: the instance's allocation fails instead of handing address 0 to the GPU
run pool132 BEAGLE_AMD_DATA_MB=132 -- --state-count 64 --reps 3
check "a pool the programs fill (132 MiB at 64 states): the instance's allocation fails, an error from beagleCreateInstance, the card finalized" \
    "normal pool132 && grep -q 'alloc(.*): the VRAM pool has 0 bytes left .*; this instance fails' $W/pool132.txt \
     && grep -q 'beagleCreateInstance failed (error -1)' $W/pool132.txt && glog pool132 | grep -q '$CLEAN' && verdict pool132 | grep -q 'NO ERRORS'"

# 4. a GPU fault mid-run
run fault FAKE_AMD_STATE=cold FAKE_AMD_FAULT=1 HCQDEV_WAIT_TIMEOUT_MS=3000
check "a GPU fault mid-run: decoded, the GPU lost to the process, BEAGLE returns errors, and the exit finalizes the card (NO ERRORS)" \
    "normal fault && grep -q 'sq_intr: error (MEMVIOL)' $W/fault.txt && grep -q 'the GPU is lost to this process' $W/fault.txt \
     && grep -q 'calculateRootLogLikelihoods failed: -1' $W/fault.txt && glog fault | grep -q '$CLEAN' && verdict fault | grep -q 'NO ERRORS'"

# 5. a GPU that hangs mid-run
run hang FAKE_AMD_HANG=1 HCQDEV_WAIT_TIMEOUT_MS=3000
check "a GPU that hangs mid-run: the wait times out, BEAGLE returns errors, and the exit's fini dequeues the hung queue and finalizes the card (NO ERRORS)" \
    "normal hang && grep -q 'Wait timeout: 3000 ms' $W/hang.txt && grep -q 'the GPU is lost to this process' $W/hang.txt \
     && grep -q 'calculateRootLogLikelihoods failed: -1' $W/hang.txt && glog hang | grep -q '$CLEAN' && verdict hang | grep -q 'NO ERRORS'"

# 6. TinyGPU.app gone mid-run, about 80 evaluations into 2,000 (the boot and the programs take about 43,400 requests, an
#    evaluation about 20)
run drop FAKE_AMD_DROP_AT=45000 -- --state-count 4 --reps 2000
check "TinyGPU.app gone mid-run: the next request fails (no SIGPIPE death), BEAGLE returns errors, the test exits normally, the guard holds" \
    "normal drop && grep -qE 'Connection closed|connection lost' $W/drop.txt && grep -q 'the GPU is lost to this process' $W/drop.txt \
     && grep -q -- '--reps: evaluation [0-9]* failed' $W/drop.txt && glog drop | grep -q 'HOLDING the TinyGPU.app connection' && [ -f $W/drop.held ]"

# 7. SIGINT to the test's process group mid-run
run sigint 'SIGINT_AFTER=C\+\+ runtime: handed over' -- --state-count 4 --reps 20000
check "SIGINT to the test's process group mid-run: the test stops (exit 130) and finalizes, the card is finalized (NO ERRORS), the guard sees the clean" \
    "[ -f $W/sigint.sigint ] && [ \"\$(status sigint)\" = 130 ] && grep -q 'interrupted by signal 2' $W/sigint.txt && glog sigint | grep -q '$CLEAN' && verdict sigint | grep -q 'NO ERRORS'"

# 8. a hold, then another instance in the same process
run held FAKE_AMD_WEDGED=1 BEAGLE_AMD_DATA_MB=1000000 -- --state-count 4 --reps 3 --cycles 2
check "a hold, then another instance in the same process: the guard holds, and the second beagleCreateInstance is refused at once" \
    "normal held && grep -q 'after the C++ boot: MemoryError' $W/held.txt && grep -q '=== cycle 2 of 2' $W/held.txt \
     && grep -q 'the crash guard (pid [0-9]*) holds the eGPU, since an earlier instance' $W/held.txt \
     && grep -q 'Error: No GPU devices' $W/held.txt && grep -q 'beagleCreateInstance failed (error -6)' $W/held.txt && glog held | grep -q 'HOLDING the TinyGPU.app connection' && [ -f $W/held.held ]"

left=$(ps -axo command= | awk '$1 ~ /beagle-tinygpu-guard$/ || $0 ~ /fake_amd_device\.py/' | wc -l | tr -d ' ')
check "no fake or guard is left running" "[ $left -eq 0 ]"
echo; [ $fails -eq 0 ] && echo "test_a3: PASS" || echo "test_a3: $fails FAILED"
[ $fails -eq 0 ]
