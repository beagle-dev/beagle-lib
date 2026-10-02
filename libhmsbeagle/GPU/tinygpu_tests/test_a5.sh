#!/bin/bash
# TODO.md plan step A5, NV's plan step P5 on the AMD path, end to end on fake_amd_device.py's card: the first instance in a
# process boots the card, every later one shares that boot until exit, when the plugin finalizes the card, and each HSACO
# variant an instance uses is loaded once (the fake checks every dispatch against the build's HSACO of its variant:
# FAKE_AMD_HSACO lists them). Through the crash guard:
#   - two instances at 4 and 64 states; four on threads at 4, 64, 16 and 128; two cycles of 4 and 64; two at 4 states on
#     threads; and a child forked after the boot that exits at once (its atexit leaves the card to its parent): one boot, one
#     connection besides the resource listing's, each variant's programs loaded once, every instance's tip partials read back
#     exactly, the card finalized at exit (the guard's clean, NO ERRORS);
#   - exit() from another thread mid-run: the plugin's atexit finalizes the card under its timed lock;
#   - a second process while the first runs: it fails at once on nv_usb4.lock, before it connects (TinyGPU.app serves one
#     client at a time: a connection would wait forever), and the first runs on;
#   - a GPU lost in the first of two cycles (FAKE_AMD_FAULT=1): the second cycle's beagleCreateInstance is refused at once
#     (the GPU was lost earlier in this process), and the card is still finalized at exit (NO ERRORS).
# The test exits 1 on the fake (its logL is wrong by design): "exits normally" is a status below 128. One PASS or FAIL line
# per check; exit 0 only if all pass.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
require_no_launch_guard
[ -x "$TEST_BIN" ] || { echo "no $TEST_BIN; build it first"; exit 2; }
GUARD_BIN="$BEAGLE_BUILD/libhmsbeagle/GPU/CMake_TinyGPUHybrid/beagle-tinygpu-guard"
[ -x "$GUARD_BIN" ] || { echo "no $GUARD_BIN; build beagle-tinygpu-guard first"; exit 2; }
W="$TINYGPU_TEST_WORK/a5"; rm -rf "$W"; mkdir -p "$W"
TL="$W/beagle_tinygpu.log"   # the plugin's and the guard's TinyGPULog lines in these runs
fails=0
pass() { echo "PASS $1"; }
fail() { echo "FAIL $1"; fails=$((fails + 1)); }
check() { if eval "$2"; then pass "$1"; else fail "$1"; fi; }
glog() { sed -n "/a5 run $1 starts/,/a5 run .* starts/p" "$TL"; }   # one run's TinyGPULog lines
verdict() { grep -E "fake TinyGPU.app \(AMD device\): (NO ERRORS|[0-9]+ ERRORS)" "$W/$1.dev" | tail -1; }
CLEAN="the plugin finalized the GPU itself; exiting"

run() {   # <label> <variants, comma-separated> [VAR=value ...] [-- tinygpuhybridtest args]: one run on a fresh fake card, then
          # its guard (waited for, or ended if it holds). SECOND_AFTER=<regex> among the VARs: once the output matches, a second
          # tinygpuhybridtest (--reps 1) against the same fake, given 30 s.
    local l=$1 v=$2 envs=() args=(--state-count 4 --reps 3) second=""; shift 2
    while [ $# -gt 0 ] && [ "$1" != "--" ]; do
        if [[ $1 == SECOND_AFTER=* ]]; then second=${1#SECOND_AFTER=}; else envs+=("$1"); fi
        shift
    done
    [ "$1" = "--" ] && { shift; args=("$@"); }
    local d; d=$(mktemp -d /tmp/tg5.XXXXXX)
    mkdir -p "$W/mem_$l"
    echo "a5 run $l starts" >> "$TL"
    env FAKE_AMD_HSACO=$v "${envs[@]}" "$BEAGLE_PYTHON" "$TG_TESTS/fake_amd_device.py" "$d/dev.sock" "$W/mem_$l" > "$W/$l.dev" 2>&1 &
    local srv=$!
    for i in $(seq 100); do grep -q listening "$W/$l.dev" 2>/dev/null && break; sleep 0.1; done
    local test_env=(BEAGLE_TINYGPU_NO_LAUNCH=1 BEAGLE_TINYGPU_NO_DOWNLOAD=1 BEAGLE_TINYGPU_LOG="$TL" APL_REMOTE_SOCK="$d/dev.sock" TMPDIR="$d"
                    BEAGLE_AMD_GUARD="$TG_TESTS/replay/crash_guard_wrap.sh" BEAGLE_TG_GUARD_BIN="$GUARD_BIN" BEAGLE_TG_GUARD_PIDFILE="$d/guard.pid"
                    BEAGLE_AMD_DATA_MB=512 DYLD_LIBRARY_PATH="$TEST_LIBS")
    env "${test_env[@]}" "${envs[@]}" "$TEST_BIN" "${args[@]}" > "$W/$l.txt" 2>&1 &
    local tst=$! rc=124
    for i in $(seq 1800); do
        kill -0 $tst 2>/dev/null || { wait $tst; rc=$?; break; }
        if [ -n "$second" ] && grep -qE "$second" "$W/$l.txt"; then   # its own pidfile: the first's guard stays the one waited for
            second=""; local s0=$(date +%s)
            env "${test_env[@]}" BEAGLE_TG_GUARD_PIDFILE="$d/guard2.pid" "${envs[@]}" "$TEST_BIN" --state-count 4 --reps 1 > "$W/${l}_second.txt" 2>&1 &
            local t2=$!
            for j in $(seq 300); do kill -0 $t2 2>/dev/null || break; sleep 0.1; done
            if kill -0 $t2 2>/dev/null; then echo "the second process still runs after 30 s; killed (fake device only)" > "$W/${l}_second.hung"; kill -KILL $t2; fi
            wait $t2; echo "$? $(( $(date +%s) - s0 ))" > "$W/${l}_second.rc"
        fi
        sleep 0.1
    done
    [ $rc -eq 124 ] && { kill -KILL $tst 2>/dev/null; wait $tst 2>/dev/null; echo "[$l] the test hung; killed (fake device only)" > "$W/$l.hung"; }
    echo $rc > "$W/$l.rc"
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
normal() { [ "$(cat $W/$1.rc)" -lt 128 ] && [ ! -f "$W/$1.hung" ]; }
loaded_once() {   # <label> <variants, comma-separated>: each variant's programs loaded exactly once, and no other
    local v n=0; for v in ${2//,/ }; do [ "$(grep -c "C++ runtime: $v's programs loaded" "$W/$1.txt")" -eq 1 ] || return 1; n=$((n + 1)); done
    [ "$(grep -c "C++ runtime: [SD]P_[0-9]*'s programs loaded" "$W/$1.txt")" -eq $n ]
}

# 1. several instances in one process
p5() {   # <label> <variants> -- <tinygpuhybridtest args>
    local l=$1 v=$2 why=""; shift 3
    run $l $v -- "$@"
    normal $l || why="$why exit($(cat $W/$l.rc))"
    [ "$(grep -c "TinyGPU/AMD: C++ boot done" "$W/$l.txt")" -eq 1 ] || why="$why boots"
    [ "$(grep -c "client done" "$W/$l.dev")" -eq 2 ] || why="$why connections"
    loaded_once $l $v || why="$why programs"
    grep -q "^tips: every instance read back its own tip partials exactly" "$W/$l.txt" || why="$why tips"
    glog $l | grep -q "$CLEAN" || why="$why clean"
    verdict $l | grep -q "NO ERRORS" || why="$why device"
    if [ -z "$why" ]; then pass "A5 $l: one boot, one connection besides the listing's, each variant loaded once, every instance's tips, finalized at exit (NO ERRORS)"
    else fail "A5 $l:$why (see $W/$l.txt)"; fi
}
p5 two SP_4,SP_64 -- --state-count 4,64 --reps 5
p5 threads SP_4,SP_64,SP_16,SP_128 -- --instances 4 --threads --state-count 4,64,16,128 --reps 300
p5 cycles SP_4,SP_64 -- --cycles 2 --state-count 4,64 --reps 20
p5 same SP_4 -- --state-count 4,4 --threads --reps 20
p5 fork SP_4,SP_64 -- --state-count 4,64 --fork-exit --reps 5
check "the child forked after the boot exited normally, and the parent's card stayed up (the run above)" \
    "grep -q '^forked child [0-9]* exited with status 0' $W/fork.txt"

# 2. exit() from another thread, 300 ms into the evaluations
run exit SP_4 -- --state-count 4 --reps 20000 --exit-after 300
check "exit() from another thread mid-run: the plugin's atexit finalizes the card (the guard's clean, NO ERRORS)" \
    "grep -q '^exit() from another thread, 300 ms into the evaluations' $W/exit.txt && [ ! -f $W/exit.hung ] \
     && glog exit | grep -q '$CLEAN' && verdict exit | grep -q 'NO ERRORS'"

# 3. a second process while the first runs
run lock SP_4 'SECOND_AFTER=C\+\+ runtime: SP_4.s programs loaded' -- --state-count 4 --reps 3000
check "a second process fails at once on nv_usb4.lock, before it connects, while the first runs on (the card finalized, NO ERRORS)" \
    "normal lock && [ ! -f $W/lock_second.hung ] && read r2 t2 < $W/lock_second.rc && [ \$r2 -ne 0 ] && [ \$t2 -le 5 ] \
     && grep -q 'Failed to acquire lock file nv_usb4.lock' $W/lock_second.txt && ! grep -q 'TinyGPU/AMD:' $W/lock_second.txt \
     && glog lock | grep -q '$CLEAN' && verdict lock | grep -q 'NO ERRORS'"

# 4. a GPU lost in the first of two cycles
run lost SP_4 FAKE_AMD_STATE=cold FAKE_AMD_FAULT=1 HCQDEV_WAIT_TIMEOUT_MS=3000 -- --state-count 4 --reps 3 --cycles 2
check "a GPU lost in the first cycle: the second beagleCreateInstance is refused at once, and the card is finalized at exit (NO ERRORS)" \
    "normal lost && grep -q '=== cycle 2 of 2' $W/lost.txt && grep -q 'the GPU was lost earlier in this process; no instance can use it' $W/lost.txt \
     && [ \"\$(grep -c 'TinyGPU/AMD: C++ boot done' $W/lost.txt)\" -eq 1 ] && glog lost | grep -q '$CLEAN' && verdict lost | grep -q 'NO ERRORS'"

left=$(ps -axo command= | awk '$1 ~ /beagle-tinygpu-guard$/ || $0 ~ /fake_amd_device\.py/' | wc -l | tr -d ' ')
check "no fake or guard is left running" "[ $left -eq 0 ]"
echo; [ $fails -eq 0 ] && echo "test_a5: PASS" || echo "test_a5: $fails FAILED"
[ $fails -eq 0 ]
