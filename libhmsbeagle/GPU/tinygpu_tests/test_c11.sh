#!/bin/bash
# TODO.md plan step C11, end to end with no eGPU: level boot, where the plugin boots the GPU itself (TinyGPUNVBoot.h,
# golden_boot.py) with no daemon, and the crash guard keeps the GPU from before the plugin's first request to it: at first it
# can only hold, and once the NVDevice is built it has the queues and the timeline too. On the fake AD107, the fake GA104
# (Ampere, plan step G1) and the fake GB205 (fake_nv_device.py):
#   - a normal run passes with no daemon, and the guard exits at the plugin's clean;
#   - a warm GPU (FAKE_WPR2_UP=1) is refused after RESIZE_BAR, MAP_BAR and reads only (plan step P4's checks: 4 on the AD107
#     and the GA104, 2 on the GB205), nothing written, and the guard closes;
#   - killed right after the guard started, and after the software half, before any falcon ran: the guard closes, and the
#     device reports NO ERRORS;
#   - killed once the NVDevice is built but before the guard has the rest of its setup: the guard holds and sends nothing;
#   - killed idle: the guard tears the GPU down (NO ERRORS), sending what the plugin's own teardown sends;
#   - a GSP that never posts INIT_DONE (FAKE_NO_INIT_DONE=1): the boot fails while GSP-RM may run, and the guard holds;
#   - on the AD107, FWSEC-FRTS leaving WPR2 down (FAKE_FALCON_FAIL=frts): the boot fails before GSP-RM started, the plugin
#     says so ('N') and the guard closes;
#   - the RTX 4060's L0 recordings (TG_L0) replay exactly at level boot under the guard: on the card's own replies, the whole
#     boot in C++ makes the requests tinygrad's boot made (the GB205's, TG_GB20X, are test_b2.sh's). Without the markers: those
#     recordings hold the daemon's.
# One PASS or FAIL line per check; exit 0 only if all pass.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
require_no_launch_guard
W="$TINYGPU_TEST_WORK/c11"; rm -rf "$W"; mkdir -p "$W"
fails=0
pass() { echo "PASS $1"; }
fail() { echo "FAIL $1"; fails=$((fails + 1)); }
check() { if eval "$2"; then pass "$1"; else fail "$1"; fi; }
TL="$TINYGPU_TEST_WORK/beagle_tinygpu_offline.log"   # the plugin's and the guard's TinyGPULog lines in these runs
out() { echo "$TINYGPU_TEST_WORK/run_device_$1.txt"; }   # the plugin's output ($W/<label>.txt: run_fake_device.sh's)
dev() { echo "$TINYGPU_TEST_WORK/fake_device_$1.log"; }
device() { grep -E "fake TinyGPU.app \((AD107|GA104|GB205) device\): " "$(dev $1)" | tail -1; }
counts() { grep -E "fake TinyGPU.app \((AD107|GA104|GB205) device\): client done: " "$(dev $1)" | tail -1; }
glog() { sed -n "/c11 run $1 starts/,\$p" "$TL"; }   # this run's TinyGPULog lines
run() {   # <label> [VAR=value ...]: one fake run, its TinyGPULog lines marked
    local l=$1; shift
    echo "c11 run $l starts" >> "$TL"
    FAKE_TG_RECORD="$W/$l.bin" "$TG_TESTS/run_fake_device.sh" $l "$@" -- --state-count 4 --reps 3 > "$W/$l.txt" 2>&1
}
UNLOAD='"rpc NV_VGPU_MSG_FUNCTION_UNLOADING_GUEST_DRIVER": 1'

for chip in ad107 ga104 gb205; do
    if [ $chip = ad107 ]; then unset FAKE_NV_CHIP; else export FAKE_NV_CHIP=$chip; fi
    C=$(echo $chip | tr a-z A-Z)

    # 1. a normal run
    run c11_${chip}_boot; r1=$?
    check "$C: level boot PASS with no daemon, and the guard exited at the plugin's clean" \
        "[ $r1 -eq 0 ] && grep -q 'built the NVDevice after the C++ boot, with no daemon' '$(out c11_${chip}_boot)' \
         && glog c11_${chip}_boot | grep -q 'the setup.s rest' && glog c11_${chip}_boot | grep -q 'the plugin tore the GPU down itself; exiting'"

    # 2. a warm GPU: refused with nothing written
    export FAKE_WPR2_UP=1
    run c11_${chip}_warm
    unset FAKE_WPR2_UP
    reads=4; [ $chip = gb205 ] && reads=2   # plan step P4's: WPR2, BOOT_42 (Ampere and Ada only), then the GSP's MAILBOX0 and RISCV_CPUCTL
    check "$C: a warm GPU is refused after RESIZE_BAR, MAP_BAR and $reads reads, nothing written, and the guard closes" \
        "grep -q 'WarmGPUError: WARM GPU: WPR2 is up' '$(out c11_${chip}_warm)' && counts c11_${chip}_warm | grep -q '\"cmd 1\": 1, \"cmd 11\": 1, \"cmd 3\": 2, \"cmd 6\": '$reads'}' \
         && glog c11_${chip}_warm | grep -q 'closing is safe'"

    # 3, 4. killed before any falcon ran: the guard closes
    for k in boot_guard boot_sw; do
        run c11_${chip}_$k BEAGLE_NV_TEST_KILL=$k
        check "$C: killed at $k, before any falcon ran: the guard closes (device NO ERRORS)" \
            "device c11_${chip}_$k | grep -q 'NO ERRORS' && glog c11_${chip}_$k | grep -q 'closing is safe' && ! grep -q '$UNLOAD' '$(dev c11_${chip}_$k)'"
    done

    # 5. killed once the NVDevice is built, before the guard has the rest of its setup: hold, nothing sent
    run c11_${chip}_built BEAGLE_NV_TEST_KILL=boot_built
    check "$C: killed before the guard had the rest of its setup: it holds and sends nothing" \
        "glog c11_${chip}_built | grep -q 'HOLDING the TinyGPU.app connection (a frame may be cut mid-send)' \
         && grep -q 'the guard (pid [0-9]*) held the fake connection' '$W/c11_${chip}_built.txt' && ! grep -q '$UNLOAD' '$(dev c11_${chip}_built)'"

    # 6. killed idle: the guard's own teardown
    run c11_${chip}_idle BEAGLE_NV_TEST_KILL=idle
    check "$C: killed idle, the guard unloads and tears the GPU down (device NO ERRORS), sending the plugin's own teardown's bytes" \
        "device c11_${chip}_idle | grep -q 'NO ERRORS' && glog c11_${chip}_idle | grep -q 'the GPU is torn down; closing' \
         && cmp -s '$W/c11_${chip}_boot.bin' '$W/c11_${chip}_idle.bin'"

    # 7. no INIT_DONE: a failure while GSP-RM may run holds
    export FAKE_NO_INIT_DONE=1
    run c11_${chip}_noinit
    unset FAKE_NO_INIT_DONE
    check "$C: a GSP that never posts INIT_DONE: the boot fails while GSP-RM may run, and the guard holds" \
        "grep -q 'level boot: building the NVDevice: ' '$(out c11_${chip}_noinit)' && glog c11_${chip}_noinit | grep -q 'HOLDING the TinyGPU.app connection (the plugin did not finish booting GSP-RM)'"
done
unset FAKE_NV_CHIP

# 8. FWSEC-FRTS that leaves WPR2 down (Ada's falcon boot): a failure before GSP-RM started closes
export FAKE_FALCON_FAIL=frts
run c11_ad107_frts
unset FAKE_FALCON_FAIL
check "AD107: FWSEC-FRTS leaving WPR2 down: the boot fails before GSP-RM started, and the guard closes (device NO ERRORS)" \
    "grep -q 'WPR2 is not initialized' '$(out c11_ad107_frts)' && device c11_ad107_frts | grep -q 'NO ERRORS' \
     && glog c11_ad107_frts | grep -q 'closing is safe; exiting'"

# 9. the L0 recordings, replayed at level boot under the guard
replay_line() { grep -E "^replay session 2" "$TINYGPU_TEST_WORK/replay_$1.log"; }
RECS=(); for r in $TG_L0; do [ -f "$BEAGLE_TINYGPU_DATA/recordings/$r/events.bin" ] && RECS+=("$BEAGLE_TINYGPU_DATA/recordings/$r"); done
if [ ${#RECS[@]} -gt 0 ]; then
    for R in "${RECS[@]}"; do
        l=c11_boot_${R##*_}
        "$TG_TESTS/run_replay.sh" "$R" $l --guard > "$W/replay_$l.txt" 2>&1
        check "${R##*/}: replays exactly at level boot, the whole boot in C++ with no daemon, under the guard" \
            "replay_line $l | grep -q 'PASS' && grep -q 'built the NVDevice after the C++ boot, with no daemon' '$TINYGPU_TEST_WORK/run_replay_$l.txt'"
    done
else
    echo "(no L0 recordings in $BEAGLE_TINYGPU_DATA/recordings: their checks skipped)"
fi

left=$(ps -axo command | grep -cE "Python .*(tgproxy|tgreplay|fake_nv_device)\.py|[b]eagle-tinygpu-guard")
check "no harness process or guard is left running" "[ $left -eq 0 ]"
echo; [ $fails -eq 0 ] && echo "test_c11: PASS" || echo "test_c11: $fails FAILED"
[ $fails -eq 0 ]
