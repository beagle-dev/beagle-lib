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
#     says so ('N') and the guard closes.
# (The RTX 4060's L0 recordings, TG_L0, no longer replay here (STATUS.md R118): since e52d494b and 4135b050 the plugin
# writes other data than it did, and the daemon they were made through is gone. test_v1.sh still uses them.)
# Every run is a job of its own (env.sh's pjob), started longest first; the checks then read the runs' files in this order.
# One PASS or FAIL line per check; exit 0 only if all pass.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
require_no_launch_guard
W="$TINYGPU_TEST_WORK/c11"; rm -rf "$W"; mkdir -p "$W"
fails=0
pass() { echo "PASS $1"; }
fail() { echo "FAIL $1"; fails=$((fails + 1)); }
check() { if eval "$2"; then pass "$1"; else fail "$1"; fi; }
out() { echo "$TINYGPU_TEST_WORK/run_device_$1.txt"; }   # the plugin's output ($W/<label>.txt: run_fake_device.sh's)
dev() { echo "$TINYGPU_TEST_WORK/fake_device_$1.log"; }
device() { grep -E "fake TinyGPU.app \((AD107|GA104|GB205) device\): " "$(dev $1)" | tail -1; }
counts() { grep -E "fake TinyGPU.app \((AD107|GA104|GB205) device\): client done: " "$(dev $1)" | tail -1; }
glog() { cat "$TINYGPU_TEST_WORK/tglog_$1.log"; }   # this run's TinyGPULog lines, the plugin's and its guard's (run_fake_device.sh's)
run() {   # <label> [VAR=value ...]: one fake run; the fake's own settings come with the call (FAKE_X=1 run ...), for it alone
    local l=$1; shift
    rm -f "$W/$l.bin"   # the fake appends to it
    FAKE_TG_RECORD="$W/$l.bin" "$TG_TESTS/run_fake_device.sh" $l "$@" -- --state-count 4 --reps 3 > "$W/$l.txt" 2>&1
}
case_run() {   # <chip> <case>: that case's fake run (the AD107 is the fake's default chip: FAKE_NV_CHIP empty)
    local chip=$1 k=$2 nv=$1 l=c11_$1_$2; [ $chip = ad107 ] && nv=
    case $k in
        boot) FAKE_NV_CHIP=$nv run $l ;;
        warm) FAKE_NV_CHIP=$nv FAKE_WPR2_UP=1 run $l ;;
        boot_guard|boot_sw) FAKE_NV_CHIP=$nv run $l BEAGLE_NV_TEST_KILL=$k ;;
        built) FAKE_NV_CHIP=$nv run $l BEAGLE_NV_TEST_KILL=boot_built ;;
        idle) FAKE_NV_CHIP=$nv run $l BEAGLE_NV_TEST_KILL=idle ;;
        noinit) FAKE_NV_CHIP=$nv FAKE_NO_INIT_DONE=1 run $l ;;
        frts) FAKE_NV_CHIP=$nv FAKE_FALCON_FAIL=frts run $l ;;
    esac
}
UNLOAD='"rpc NV_VGPU_MSG_FUNCTION_UNLOADING_GUEST_DRIVER": 1'
CHIPS="ad107 ga104 gb205"

# the runs, longest first: the 10 s INIT_DONE timeouts, then the full boots
pjob_init "$W"; export TG_PIDS="$W/pids"   # every fake and guard the runs start (tracked_alive)
for k in noinit idle boot built; do for chip in $CHIPS; do pjob c11_${chip}_$k case_run $chip $k; done; done
pjob c11_ad107_frts case_run ad107 frts
for k in boot_sw boot_guard warm; do for chip in $CHIPS; do pjob c11_${chip}_$k case_run $chip $k; done; done
pwait

for chip in $CHIPS; do
    C=$(echo $chip | tr a-z A-Z)

    # 1. a normal run
    r1=$(pjob_rc c11_${chip}_boot)
    check "$C: level boot PASS with no daemon, and the guard exited at the plugin's clean" \
        "[ $r1 -eq 0 ] && grep -q 'built the NVDevice after the C++ boot, with no daemon' '$(out c11_${chip}_boot)' \
         && glog c11_${chip}_boot | grep -q 'the setup.s rest' && glog c11_${chip}_boot | grep -q 'the plugin tore the GPU down itself; exiting'"

    # 2. a warm GPU (FAKE_WPR2_UP=1): refused with nothing written
    reads=4; [ $chip = gb205 ] && reads=2   # plan step P4's: WPR2, BOOT_42 (Ampere and Ada only), then the GSP's MAILBOX0 and RISCV_CPUCTL
    check "$C: a warm GPU is refused after RESIZE_BAR, MAP_BAR and $reads reads, nothing written, and the guard closes" \
        "grep -q 'WarmGPUError: WARM GPU: WPR2 is up' '$(out c11_${chip}_warm)' && counts c11_${chip}_warm | grep -q '\"cmd 1\": 1, \"cmd 11\": 1, \"cmd 3\": 2, \"cmd 6\": '$reads'}' \
         && glog c11_${chip}_warm | grep -q 'closing is safe'"

    # 3, 4. killed before any falcon ran: the guard closes
    for k in boot_guard boot_sw; do
        check "$C: killed at $k, before any falcon ran: the guard closes (device NO ERRORS)" \
            "device c11_${chip}_$k | grep -q 'NO ERRORS' && glog c11_${chip}_$k | grep -q 'closing is safe' && ! grep -q '$UNLOAD' '$(dev c11_${chip}_$k)'"
    done

    # 5. killed once the NVDevice is built, before the guard has the rest of its setup: hold, nothing sent
    check "$C: killed before the guard had the rest of its setup: it holds and sends nothing" \
        "glog c11_${chip}_built | grep -q 'HOLDING the TinyGPU.app connection (a frame may be cut mid-send)' \
         && grep -q 'the guard (pid [0-9]*) held the fake connection' '$W/c11_${chip}_built.txt' && ! grep -q '$UNLOAD' '$(dev c11_${chip}_built)'"

    # 6. killed idle: the guard's own teardown
    check "$C: killed idle, the guard unloads and tears the GPU down (device NO ERRORS), sending the plugin's own teardown's bytes" \
        "device c11_${chip}_idle | grep -q 'NO ERRORS' && glog c11_${chip}_idle | grep -q 'the GPU is torn down; closing' \
         && cmp -s '$W/c11_${chip}_boot.bin' '$W/c11_${chip}_idle.bin'"

    # 7. no INIT_DONE (FAKE_NO_INIT_DONE=1): a failure while GSP-RM may run holds
    check "$C: a GSP that never posts INIT_DONE: the boot fails while GSP-RM may run, and the guard holds" \
        "grep -q 'level boot: building the NVDevice: ' '$(out c11_${chip}_noinit)' && glog c11_${chip}_noinit | grep -q 'HOLDING the TinyGPU.app connection (the plugin did not finish booting GSP-RM)'"
done

# 8. FWSEC-FRTS that leaves WPR2 down (FAKE_FALCON_FAIL=frts; Ada's falcon boot): a failure before GSP-RM started closes
check "AD107: FWSEC-FRTS leaving WPR2 down: the boot fails before GSP-RM started, and the guard closes (device NO ERRORS)" \
    "grep -q 'WPR2 is not initialized' '$(out c11_ad107_frts)' && device c11_ad107_frts | grep -q 'NO ERRORS' \
     && glog c11_ad107_frts | grep -q 'closing is safe; exiting'"

left=$(tracked_alive)   # this stage's own fakes and guards only: other stages run at the same time
check "no harness process or guard is left running" "[ -z \"\$left\" ]"
echo; [ $fails -eq 0 ] && echo "test_c11: PASS" || echo "test_c11: $fails FAILED"
[ $fails -eq 0 ]
