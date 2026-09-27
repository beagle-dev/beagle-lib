#!/bin/bash
# TODO.md plan step C10, end to end with no eGPU: the crash guard (beagle-tinygpu-guard, tinygpu_guard.cpp) at level flcn_hw,
# where the plugin spawns it once the NVDevice and its timeline exist and sets the state page's keeper word, and the daemon hands
# the guard its keeper role (cmd_release) and exits. On the fake AD107 and the fake GB205 (fake_nv_device.py, which the real
# daemon boots):
#   - a normal run passes, the daemon is gone after the release, the guard exits at the plugin's "clean", and the device
#     receives exactly the bytes it receives with the daemon as the keeper (BEAGLE_NV_GUARD=0);
#   - the plugin killed (BEAGLE_NV_TEST_KILL) while idle: the guard tears the GPU down (the fake sees its unload and teardown and
#     reports NO ERRORS), sending what the plugin's own teardown sends;
#   - killed mid-frame, or inside its own teardown: the guard holds and sends nothing (the harness ends it);
#   - killed after the guard's "ready" but before the keeper word: the daemon decides (it tears down), the guard exits;
#   - killed with the keeper word set, before the release: the daemon exits without a word, the guard decides;
#   - killed right after the release: the guard decides;
#   - a GSP that never answers the unload RPC (FAKE_GSP_SILENT_UNLOAD=1), the plugin killed idle: the guard holds;
#   - killed right after the first launch batch's or copy's submission, the GPU still running it (FAKE_GPU_LAG_MS=500): the
#     guard waits for the C++ timeline, then tears the GPU down (the fake calls an unload before that work ran an error).
# One PASS or FAIL line per check; exit 0 only if all pass.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
require_no_launch_guard
W="$TINYGPU_TEST_WORK/c10"; rm -rf "$W"; mkdir -p "$W"
fails=0
pass() { echo "PASS $1"; }
fail() { echo "FAIL $1"; fails=$((fails + 1)); }
check() { if eval "$2"; then pass "$1"; else fail "$1"; fi; }
TL="$TINYGPU_TEST_WORK/beagle_tinygpu_offline.log"   # the plugin's and the guard's TinyGPULog lines in these runs
out() { echo "$TINYGPU_TEST_WORK/run_device_$1.txt"; }
dlog() { echo "$TINYGPU_TEST_WORK/run_device_$1_daemon.log"; }
device() { grep -E "fake TinyGPU.app \((AD107|GB205) device\): " "$TINYGPU_TEST_WORK/fake_device_$1.log" | tail -1; }
glog() { sed -n "/c10 run $1 starts/,\$p" "$TL"; }   # this run's TinyGPULog lines
behind() {   # the guard's state-page line says the C++ timeline was behind the last submission
    glog $1 | sed -nE 's/.*last_submitted ([0-9]+), seq [0-9]+, C\+\+ timeline ([0-9]+).*/\1 \2/p' | awk '$2 < $1 {ok = 1} END {exit !ok}'
}
run() {   # <label> [VAR=value ...]: one fake run at flcn_hw, its TinyGPULog lines marked
    local l=$1; shift
    echo "c10 run $l starts" >> "$TL"
    FAKE_TG_RECORD="$W/$l.bin" "$TG_TESTS/run_fake_device.sh" $l BEAGLE_NV_CPP_LEVEL=flcn_hw "$@" -- --state-count 4 --reps 3 > "$W/$l.txt" 2>&1
}
UNLOAD='"rpc NV_VGPU_MSG_FUNCTION_UNLOADING_GUEST_DRIVER": 1'

for chip in ad107 gb205; do
    if [ $chip = gb205 ]; then export FAKE_NV_CHIP=gb205; else unset FAKE_NV_CHIP; fi
    C=$(echo $chip | tr a-z A-Z)

    # 1. a normal run, with the guard and with the daemon as the keeper: the same bytes at the device
    run c10_${chip}_guard; r1=$?
    run c10_${chip}_daemon BEAGLE_NV_GUARD=0; r2=$?
    check "$C: a normal run with the guard PASS; the daemon released the keeper role and exited; the guard exited at the plugin's clean" \
        "[ $r1 -eq 0 ] && grep -q 'keeper role handed to the C++ side.s guard' '$(dlog c10_${chip}_guard)' \
         && glog c10_${chip}_guard | grep -q 'the plugin tore the GPU down itself; exiting' && grep -q 'the guard (pid [0-9]*) keeps the keeper role' '$(out c10_${chip}_guard)'"
    check "$C: with BEAGLE_NV_GUARD=0 the daemon stays the keeper; the device received the same bytes both ways" \
        "[ $r2 -eq 0 ] && ! grep -q 'keeper role handed' '$(dlog c10_${chip}_daemon)' && cmp -s '$W/c10_${chip}_guard.bin' '$W/c10_${chip}_daemon.bin'"

    # 2. killed idle: the guard's teardown, what the plugin's own sends
    run c10_${chip}_idle BEAGLE_NV_TEST_KILL=idle
    check "$C: killed idle, the guard unloads and tears the GPU down (device NO ERRORS), sending the plugin's own teardown's bytes" \
        "device c10_${chip}_idle | grep -q 'NO ERRORS' && glog c10_${chip}_idle | grep -q 'the GPU is torn down; closing' \
         && cmp -s '$W/c10_${chip}_guard.bin' '$W/c10_${chip}_idle.bin'"

    # 3, 4. killed mid-frame and inside its own teardown: hold, nothing sent
    for k in frame teardown; do
        run c10_${chip}_$k BEAGLE_NV_TEST_KILL=$k
        why=$([ $k = frame ] && echo 'a frame may be cut mid-send' || echo 'the plugin.s own GPU teardown did not finish')
        check "$C: killed $([ $k = frame ] && echo mid-frame || echo 'in its own teardown'), the guard holds and sends nothing" \
            "glog c10_${chip}_$k | grep -q 'HOLDING the TinyGPU.app connection ($why)' && grep -q 'the guard (pid [0-9]*) held the fake connection' '$W/c10_${chip}_$k.txt' \
             && ! grep -q '$UNLOAD' '$TINYGPU_TEST_WORK/fake_device_c10_${chip}_$k.log'"
    done

    # 5. killed between the guard's ready and the keeper word: the daemon decides; 6. with the word set, before the release, and
    # 7. right after the release: the guard does
    run c10_${chip}_ready BEAGLE_NV_TEST_KILL=ready
    check "$C: killed before the keeper word, the daemon tears the GPU down (device NO ERRORS) and the guard exits" \
        "device c10_${chip}_ready | grep -q 'NO ERRORS' && grep -q 'command socket closed without fini' '$(dlog c10_${chip}_ready)' \
         && glog c10_${chip}_ready | grep -q 'before it handed the keeper role over'"
    run c10_${chip}_handover BEAGLE_NV_TEST_KILL=handover
    check "$C: killed with the keeper word set, before the release: the daemon exits without a word, the guard tears the GPU down (device NO ERRORS)" \
        "device c10_${chip}_handover | grep -q 'NO ERRORS' && grep -q 'the keeper word names the C++ side.s guard' '$(dlog c10_${chip}_handover)' \
         && glog c10_${chip}_handover | grep -q 'the GPU is torn down; closing'"
    run c10_${chip}_released BEAGLE_NV_TEST_KILL=released
    check "$C: killed right after the release, the guard tears the GPU down (device NO ERRORS)" \
        "device c10_${chip}_released | grep -q 'NO ERRORS' && glog c10_${chip}_released | grep -q 'the GPU is torn down; closing'"

    # 8. a GSP that never answers the unload RPC: the guard holds
    export FAKE_GSP_SILENT_UNLOAD=1   # the fake device's knob, so in run_fake_device.sh's environment
    run c10_${chip}_silent BEAGLE_NV_TEST_KILL=idle
    unset FAKE_GSP_SILENT_UNLOAD
    check "$C: a silent GSP, the plugin killed idle: the guard's unload RPC times out and it holds" \
        "glog c10_${chip}_silent | grep -q 'Timeout waiting for RPC response for command 47' \
         && glog c10_${chip}_silent | grep -q 'HOLDING the TinyGPU.app connection (the GPU did not confirm its teardown)' \
         && grep -q 'unload RPC left unanswered' '$TINYGPU_TEST_WORK/fake_device_c10_${chip}_silent.log'"

    # 9, 10. killed mid-batch and mid-copy, the GPU behind: the guard waits for the timeline, then tears the GPU down
    export FAKE_GPU_LAG_MS=500   # the fake device's knob
    for k in batch copy; do
        run c10_${chip}_$k BEAGLE_NV_TEST_KILL=$k
        check "$C: killed mid-$k with the GPU behind, the guard waits for the C++ timeline, then tears the GPU down (device NO ERRORS)" \
            "behind c10_${chip}_$k && device c10_${chip}_$k | grep -q 'NO ERRORS' && glog c10_${chip}_$k | grep -q 'the GPU is torn down; closing'"
    done
    unset FAKE_GPU_LAG_MS
done
unset FAKE_NV_CHIP

left=$(ps -axo command | grep -cE "Python .*(tgdaemon|tgproxy|tgreplay|fake_nv_device)\.py|[b]eagle-tinygpu-guard")
check "no harness process or guard is left running" "[ $left -eq 0 ]"
echo; [ $fails -eq 0 ] && echo "test_c10: PASS" || echo "test_c10: $fails FAILED"
[ $fails -eq 0 ]
