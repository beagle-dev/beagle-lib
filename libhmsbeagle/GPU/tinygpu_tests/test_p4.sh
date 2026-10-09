#!/bin/bash
# TODO.md plan step P4, offline: the recovery of a warm GPU at boot (the default; BEAGLE_NV_RECOVER=0 turns it off). When
# WPR2 is up and the GSP is suspended (MAILBOX0 0x80000000) or its RISC-V core halted, the boot runs NVIDIA's teardown (the
# GSP reset, FWSEC-SB, the SEC2 reset, Booter Unload) on its own images before GSP-RM's boot, then boots. On
# fake_nv_device.py (run_fake_device.sh):
#   - the AD107 as an unload without its teardown leaves it (FAKE_WPR2_UP=suspended: BEAGLE_NV_TEARDOWN=0's exit), and with its
#     GSP core halted instead (FAKE_WPR2_UP=halted): FWSEC-SB and Booter Unload at boot bring WPR2 down, then FWSEC-FRTS,
#     booter_load, the whole run and the teardown at exit (NO ERRORS); the same on the GA104 with its GSP suspended (Ampere,
#     recovered since plan step G1: the user's choice, untested on hardware);
#   - refused before any write, as with BEAGLE_NV_RECOVER=0, after reads only: a GSP that may still run (FAKE_WPR2_UP=1), with
#     BEAGLE_NV_TEARDOWN=0, and the GB205 (Ampere and Ada only); with BEAGLE_NV_RECOVER=0 the suspended AD107 is refused as
#     before;
#   - a Booter Unload that leaves WPR2 up (FAKE_FALCON_FAIL=unload): the boot stops before GSP-RM starts, and the guard closes.
# One PASS or FAIL line per check; exit 0 only if all pass.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
require_no_launch_guard
unset FAKE_NV_CHIP FAKE_WPR2_UP FAKE_FALCON_FAIL BEAGLE_NV_RECOVER BEAGLE_NV_TEARDOWN
W="$TINYGPU_TEST_WORK/p4"; rm -rf "$W"; mkdir -p "$W"
fails=0
pass() { echo "PASS $1"; }
fail() { echo "FAIL $1"; fails=$((fails + 1)); }
check() { if eval "$2"; then pass "$1"; else fail "$1"; fi; }
TL="$TINYGPU_TEST_WORK/beagle_tinygpu_offline.log"   # the plugin's and the guard's TinyGPULog lines in these runs
out() { echo "$TINYGPU_TEST_WORK/run_device_$1.txt"; }   # the plugin's output ($W/<label>.txt: run_fake_device.sh's)
dev() { echo "$TINYGPU_TEST_WORK/fake_device_$1.log"; }
device() { grep -E "fake TinyGPU.app \((AD107|GA104|GB205) device\): " "$(dev $1)" | tail -1; }
counts() { grep -E "fake TinyGPU.app \((AD107|GA104|GB205) device\): client done: " "$(dev $1)" | tail -1; }
glog() { sed -n "/p4 run $1 starts/,\$p" "$TL"; }   # this run's TinyGPULog lines
status() { sed -n "s/^\[$1\] tinygputest exit=\([0-9]*\) .*/\1/p" "$W/$1.txt"; }   # the test's own exit status
run() {   # <label> [VAR=value ...] [-- test args]: one fake run, its TinyGPULog lines marked; the VARs reach the fake and the test
    local l=$1 envs=() args=(--state-count 4 --reps 3 --poison); shift
    while [ $# -gt 0 ] && [ "$1" != "--" ]; do envs+=("$1"); shift; done
    [ "$1" = "--" ] && { shift; args=("$@"); }
    echo "p4 run $l starts" >> "$TL"
    env "${envs[@]}" "$TG_TESTS/run_fake_device.sh" $l -- "${args[@]}" > "$W/$l.txt" 2>&1
}
READS_ONLY='"cmd 1": 1, "cmd 11": 1, "cmd 3": 2, "cmd 6": '   # RESIZE_BAR, MAP_BAR, the probe's two config reads, then BAR0 reads

# 1. recovered
for gsp in suspended halted; do
    run p4_$gsp FAKE_WPR2_UP=$gsp; r=$?
    check "AD107, WPR2 up with the GSP $gsp: NVIDIA's teardown at boot brings WPR2 down, then the boot and the whole run, and the teardown at exit (NO ERRORS)" \
        "[ $r -eq 0 ] && grep -q 'TinyGPU/NV: a warm GPU (WPR2_HI=0x[0-9a-f]*, the GSP $gsp: MAILBOX0=0x[0-9a-f]*, RISCV_CPUCTL=0x[0-9a-f]*): NVIDIA.s teardown first (BEAGLE_NV_RECOVER=0 refuses instead)' '$(out p4_$gsp)' \
         && grep -q 'TinyGPU/NV: the teardown at boot: done: Booter Unload lowered WPR2; WPR2 is down, so the boot goes on' '$(out p4_$gsp)' \
         && counts p4_$gsp | grep -q '\"FWSEC-FRTS\": 1, \"FWSEC-SB\": 2, .*\"booter_load\": 1, \"booter_unload\": 2' \
         && device p4_$gsp | grep -q 'NO ERRORS'"
done

# 1b. an Ampere, recovered as an Ada (plan step G1)
run p4_ga104 FAKE_NV_CHIP=ga104 FAKE_WPR2_UP=suspended; r=$?
check "GA104, WPR2 up with the GSP suspended: NVIDIA's teardown at boot (ga102's Booter Unload) brings WPR2 down, then the boot and the whole run, and the teardown at exit (NO ERRORS)" \
    "[ $r -eq 0 ] && grep -q 'TinyGPU/NV: a warm GPU (WPR2_HI=0x[0-9a-f]*, the GSP suspended: MAILBOX0=0x[0-9a-f]*, RISCV_CPUCTL=0x[0-9a-f]*): NVIDIA.s teardown first (BEAGLE_NV_RECOVER=0 refuses instead)' '$(out p4_ga104)' \
     && grep -q 'TinyGPU/NV: the teardown at boot: done: Booter Unload lowered WPR2; WPR2 is down, so the boot goes on' '$(out p4_ga104)' \
     && counts p4_ga104 | grep -q '\"FWSEC-FRTS\": 1, \"FWSEC-SB\": 2, .*\"booter_load\": 1, \"booter_unload\": 2' \
     && device p4_ga104 | grep -q 'NO ERRORS'"

# 2. refused before any write
run p4_running FAKE_WPR2_UP=1
check "AD107, WPR2 up with a GSP that may still run: refused after four reads (WPR2, BOOT_42, the GSP's MAILBOX0 and RISCV_CPUCTL), nothing written" \
    "[ \"\$(status p4_running)\" = 1 ] && grep -q 'beagleCreateInstance failed (error -1)' '$(out p4_running)' \
     && grep -q 'neither suspended nor halted (MAILBOX0=0x[0-9a-f]*, RISCV_CPUCTL=0x[0-9a-f]*): GSP-RM may still run, so BEAGLE_NV_RECOVER does not recover it' '$(out p4_running)' \
     && counts p4_running | grep -q '$READS_ONLY''4}' && device p4_running | grep -q 'NO ERRORS'"
run p4_teardown0 FAKE_WPR2_UP=suspended BEAGLE_NV_TEARDOWN=0
check "AD107, BEAGLE_NV_TEARDOWN=0: refused after one read, nothing written" \
    "[ \"\$(status p4_teardown0)\" = 1 ] && grep -q 'which BEAGLE_NV_TEARDOWN=0 turns off' '$(out p4_teardown0)' \
     && counts p4_teardown0 | grep -q '$READS_ONLY''1}' && device p4_teardown0 | grep -q 'NO ERRORS'"
run p4_gb205 FAKE_NV_CHIP=gb205 FAKE_WPR2_UP=1
check "GB205, WPR2 up: refused after two reads (Ampere and Ada only), nothing written" \
    "[ \"\$(status p4_gb205)\" = 1 ] && grep -q 'BEAGLE_NV_RECOVER recovers Ampere and Ada GPUs only (NV_PMC_BOOT_42 architecture 0x1b)' '$(out p4_gb205)' \
     && counts p4_gb205 | grep -q '$READS_ONLY''2}' && device p4_gb205 | grep -q 'NO ERRORS'"
run p4_off FAKE_WPR2_UP=suspended BEAGLE_NV_RECOVER=0
check "AD107, the GSP suspended but BEAGLE_NV_RECOVER=0: refused as before, after one read, nothing written" \
    "[ \"\$(status p4_off)\" = 1 ] && grep -q 'WPR2 is up (NV_PFB_PRI_MMU_WPR2_ADDR_HI=0x[0-9a-f]*), so the previous boot was not torn down' '$(out p4_off)' \
     && counts p4_off | grep -q '$READS_ONLY''1}' && device p4_off | grep -q 'NO ERRORS'"

# 3. a recovery that fails
run p4_unload FAKE_WPR2_UP=suspended FAKE_FALCON_FAIL=unload
check "AD107, a Booter Unload at boot that leaves WPR2 up: the boot stops before GSP-RM starts (no FWSEC-FRTS, no booter_load), and the guard closes (NO ERRORS)" \
    "[ \"\$(status p4_unload)\" = 1 ] && grep -q 'beagleCreateInstance failed (error -1)' '$(out p4_unload)' \
     && grep -q 'NVIDIA.s teardown at boot did not bring WPR2 down (failed: Booter Unload returned mailbox0=0x[0-9a-f]* and WPR2_HI=0x[0-9a-f]*)' '$(out p4_unload)' \
     && glog p4_unload | grep -q 'the plugin.s boot stopped before GSP-RM started: nothing to unload, closing is safe' \
     && counts p4_unload | grep -q '\"FWSEC-SB\": 1, .*\"booter_unload\": 1' && ! counts p4_unload | grep -q 'FWSEC-FRTS\|booter_load' \
     && device p4_unload | grep -q 'NO ERRORS'"

left=$(ps -axo command | grep -cE "Python .*(tgproxy|tgreplay|fake_nv_device)\.py|[b]eagle-tinygpu-guard")
check "no harness process or guard is left running" "[ $left -eq 0 ]"
echo; [ $fails -eq 0 ] && echo "test_p4: PASS" || echo "test_p4: $fails FAILED"
[ $fails -eq 0 ]
