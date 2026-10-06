#!/bin/bash
# TODO.md plan step A2k, end to end with no eGPU: the crash guard of the AMD C++ boot (beagle-tinygpu-guard's amd_guard), on
# fake_amd_device.py's card, which counts a session that ends with a queue live as an error (TinyGPU.app then
# unwires the sysmem the queue polls: on the Mac, a DART fault):
#   - a normal run passes, the guard has the AMDev's fini state before any queue is set up, and exits at the plugin's clean;
#   - a dirty card: the boot is refused before the mode1 reset, and the guard exits at the plugin's 'N';
#   - killed right after the guard started: the guard closes, and the boot never wrote to the card;
#   - killed once the AMDev booted, before any queue: the guard finalizes the card and closes, and the next run's boot is a
#     partial one;
#   - killed with a launch batch on the GPU, and idle at fini: the guard finalizes the GPU (NO ERRORS: no queue live at the
#     session's end); killed idle, it sends what the plugin's own fini sends: the whole session equals the normal run's, byte
#     for byte (that fini is the oracle daemon's exit: golden_amd_boot.py);
#   - killed with a request in flight, and in the plugin's own fini: the guard holds, sending nothing;
#   - a compute queue that stays active after its dequeue (FAKE_AMD_WEDGED=1): the plugin's own fini cannot see it off and
#     says hold, and killed with a batch on the GPU, the guard's own fini cannot either: both hold.
# A guard that holds keeps the fake's connection as it would the eGPU's, so it is ended here (offline only). One PASS or FAIL
# line per check; exit 0 only if all pass.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
require_no_launch_guard
[ -x "$TEST_BIN" ] || { echo "no $TEST_BIN; build it first"; exit 2; }
GUARD_BIN="$BEAGLE_BUILD/libhmsbeagle/GPU/CMake_TinyGPU/beagle-tinygpu-guard"
[ -x "$GUARD_BIN" ] || { echo "no $GUARD_BIN; build beagle-tinygpu-guard first"; exit 2; }
W="$TINYGPU_TEST_WORK/a2k"; rm -rf "$W"; mkdir -p "$W"
TL="$W/beagle_tinygpu.log"   # the plugin's and the guard's TinyGPULog lines in these runs
fails=0
pass() { echo "PASS $1"; }
fail() { echo "FAIL $1"; fails=$((fails + 1)); }
check() { if eval "$2"; then pass "$1"; else fail "$1"; fi; }
glog() { sed -n "/a2k run $1 starts/,/a2k run .* starts/p" "$TL"; }   # one run's TinyGPULog lines
verdict() { grep -E "fake TinyGPU.app \(AMD device\): (NO ERRORS|[0-9]+ ERRORS)" "$W/$1.dev" | tail -1; }
counts() { grep "fake TinyGPU.app (AMD device): client done: " "$W/$1.dev" | tail -1; }
# a run that went to the end with no runtime error (test_a2.sh's: kernels are not run, so logL is wrong by design)
ran() { grep -q "^per evaluation:" "$W/$1.txt" && ! grep -qE "TinyGPU/AMD: .*(failed|refused|fault)|TinyGPU/AMD: handoff:" "$W/$1.txt"; }

SRV=""; SOCKDIR=""; DEV=""; NSESS=0
start_fake() {   # <label> [VAR=value ...]: a fresh fake card, its log $W/<label>.dev, every session recorded to $W/<label>.rec.<n>
    local l=$1; shift
    SOCKDIR=$(mktemp -d /tmp/tgk.XXXXXX); DEV="$W/$l.dev"; NSESS=0
    mkdir -p "$W/mem_$l"
    env "$@" FAKE_AMD_RECORD="$W/$l.rec" "$BEAGLE_PYTHON" "$TG_TESTS/fake_amd_device.py" "$SOCKDIR/dev.sock" "$W/mem_$l" > "$DEV" 2>&1 &
    SRV=$!
    for i in $(seq 100); do grep -q listening "$DEV" 2>/dev/null && break; sleep 0.1; done
}
stop_fake() { kill $SRV 2>/dev/null; wait $SRV 2>/dev/null; rm -rf "$SOCKDIR"; SRV=""; }
run_test() {   # <label> [VAR=value ...]: tinygputest on the fake, then its guard: waited for, or ended if it holds
    local l=$1; shift
    echo "a2k run $l starts" >> "$TL"
    rm -f "$SOCKDIR/guard.pid"
    env BEAGLE_TINYGPU_NO_LAUNCH=1 BEAGLE_TINYGPU_NO_DOWNLOAD=1 BEAGLE_TINYGPU_LOG="$TL" APL_REMOTE_SOCK="$SOCKDIR/dev.sock" TMPDIR="$SOCKDIR" \
        BEAGLE_AMD_GUARD="$TG_TESTS/replay/crash_guard_wrap.sh" BEAGLE_TG_GUARD_BIN="$GUARD_BIN" BEAGLE_TG_GUARD_PIDFILE="$SOCKDIR/guard.pid" \
        BEAGLE_AMD_DATA_MB=512 DYLD_LIBRARY_PATH="$TEST_LIBS" "$@" "$TEST_BIN" --state-count 4 --reps 2 > "$W/$l.txt" 2>&1
    local gpid; gpid=$(cat "$SOCKDIR/guard.pid" 2>/dev/null)
    if [ -n "$gpid" ]; then
        for i in $(seq 600); do kill -0 "$gpid" 2>/dev/null || break; grep -q "then kill $gpid\." "$TL" && break; sleep 0.1; done
        if kill -0 "$gpid" 2>/dev/null; then
            echo "[$l] the guard (pid $gpid) held the fake connection; ending it" > "$W/$l.held"; kill -KILL "$gpid"
            for i in $(seq 50); do kill -0 "$gpid" 2>/dev/null || break; sleep 0.1; done   # not this shell's child: polled
        fi
    fi
    NSESS=$((NSESS + 2))   # BEAGLE's resource listing makes a session of its own (one CFG_READ) before the run's
    for i in $(seq 600); do [ "$(grep -c 'client done' "$DEV")" -ge $NSESS ] && break; sleep 0.1; done
}
run() { start_fake "$@"; run_test "$@"; stop_fake; }   # one run on a fresh card

# 1. a normal run
run normal
check "a normal run PASS with the C++ boot, the guard had the fini state before any queue and exited at the plugin's clean" \
    "ran normal && grep -q 'C++ runtime: handed over after the C++ boot' $W/normal.txt && glog normal | grep -q 'the setup.s rest (AMD)' \
     && glog normal | grep -q 'the plugin finalized the GPU itself; exiting' && verdict normal | grep -q 'NO ERRORS' && [ ! -f $W/normal.held ]"

# 2. a dirty card: refused before the mode1 reset
run dirty FAKE_AMD_STATE=dirty
check "a dirty card: refused before the mode1 reset, and the guard exits at the plugin's 'N'" \
    "grep -q 'needs an SMU mode1 reset' $W/dirty.txt && glog dirty | grep -q 'boot ended with no queue ever live' && verdict dirty | grep -q 'NO ERRORS' \
     && ! counts dirty | grep -q 'mode1 resets' && [ ! -f $W/dirty.held ]"

# 3. killed right after the guard started
run boot_guard BEAGLE_AMD_TEST_KILL=boot_guard
check "killed right after the guard started: it closes, and the boot never wrote to the card" \
    "glog boot_guard | grep -q 'no queue was ever live, so closing is safe' && ! glog boot_guard | grep -q 'its fini first' \
     && verdict boot_guard | grep -q 'NO ERRORS' && ! counts boot_guard | grep -q '\"cmd 7\"' && [ ! -f $W/boot_guard.held ]"

# 4. killed once the AMDev booted, before any queue: the guard finalizes the card, and the next boot is a partial one
start_fake boot_rest FAKE_AMD_STATE=cold
run_test boot_rest BEAGLE_AMD_TEST_KILL=boot_rest
run_test boot_rest_next
stop_fake
check "killed once the AMDev booted, before any queue: the guard finalizes the card and closes" \
    "glog boot_rest | grep -q 'its fini first' && glog boot_rest | grep -q 'the fini is done' && glog boot_rest | grep -q 'closing is safe' \
     && grep -c 'NO ERRORS' $W/boot_rest.dev | grep -q '^4$' && [ ! -f $W/boot_rest.held ]"
check "... and the next run's boot is a partial one, and passes" \
    "grep -q 'C++ boot done (partial boot)' $W/boot_rest_next.txt && ran boot_rest_next && verdict boot_rest | grep -q 'NO ERRORS'"

# 5, 6. killed with a batch on the GPU, and idle at fini: the guard's fini
for k in batch idle; do
    run $k BEAGLE_AMD_TEST_KILL=$k
    check "killed $([ $k = batch ] && echo 'with a batch on the GPU' || echo 'idle at fini'): the guard finalizes the GPU, no queue live at the end (NO ERRORS)" \
        "glog $k | grep -q 'the GPU is finalized, every queue off; closing' && verdict $k | grep -q 'NO ERRORS' && [ ! -f $W/$k.held ]"
done
check "... and killed idle it sends what the plugin's own fini sends: the whole session equals the normal run's, byte for byte" \
    "[ -s $W/idle.rec.1 ] && cmp -s $W/normal.rec.1 $W/idle.rec.1"

# 7, 8. a request in flight, and the plugin's own fini: hold, sending nothing
for k in frame teardown; do
    run $k BEAGLE_AMD_TEST_KILL=$k
    why=$([ $k = frame ] && echo 'a request may be cut mid-send, or its reply unread' || echo "the plugin's own GPU teardown did not finish")
    check "killed $([ $k = frame ] && echo 'with a request in flight' || echo "in the plugin's own fini"): the guard holds, sending nothing" \
        "glog $k | grep -qF \"HOLDING the TinyGPU.app connection ($why)\" && [ -f $W/$k.held ] && ! counts $k | grep -q 'hqd dequeue requests' \
         && verdict $k | grep -q 'the session ended with 2 queue(s) live (compute, sdma)'"
done

# 9, 10. a compute queue that survives its dequeue
run wedged FAKE_AMD_WEDGED=1
check "a queue that stays active after its dequeue: the plugin's own fini cannot see it off, says hold, and the guard holds" \
    "grep -q 'not seen inactive after their dequeue' $W/wedged.txt && grep -q 'holds the TinyGPU.app connection' $W/wedged.txt \
     && glog wedged | grep -q 'HOLDING the TinyGPU.app connection (the plugin.s own GPU teardown was not confirmed)' && [ -f $W/wedged.held ]"
run wedged_batch FAKE_AMD_WEDGED=1 BEAGLE_AMD_TEST_KILL=batch
check "... and killed with a batch on the GPU, the guard's own fini cannot see it off either: it holds" \
    "glog wedged_batch | grep -q 'not seen inactive after their dequeue' && glog wedged_batch | grep -qF 'HOLDING the TinyGPU.app connection (the GPU did not confirm its queues off)' \
     && [ -f $W/wedged_batch.held ]"

left=$(ps -axo command= | awk '$1 ~ /beagle-tinygpu-guard$/ || $0 ~ /fake_amd_device\.py/' | wc -l | tr -d ' ')
check "no fake or guard is left running" "[ $left -eq 0 ]"
echo; [ $fails -eq 0 ] && echo "test_a2k: PASS" || echo "test_a2k: $fails FAILED"
[ $fails -eq 0 ]
