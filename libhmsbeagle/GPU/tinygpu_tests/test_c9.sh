#!/bin/bash
# TODO.md plan step C9, end to end with no eGPU: BEAGLE_NV_CPP_LEVEL=flcn_hw, where the daemon's boot stops after both init_sw
# calls (the images, the GSP's boot structures, the prequeued RPCs) and the plugin runs NV_FLCN.init_hw (FWSEC-FRTS, then
# booter_load, which starts GSP-RM; nv_init_helper's FRTS checks around it), ported (TinyGPUHybridNVFalcon.h), then level
# gsp_hw's init_hw and golden image and level rm's NVDevice. On the fake AD107 the device must receive exactly the bytes it
# receives at level sysmem; two instances in two threads share the plugin's GPU; when FWSEC-FRTS leaves WPR2 down, or
# booter_load fails, GSP-RM never started and the daemon closes, sending nothing and not holding; when booter_load ran but the
# GSP's core is not active, GSP-RM may run, and the daemon holds; each L0 hardware recording must replay exactly at flcn_hw,
# under the guard, the plugin running FWSEC-FRTS, booter_load and the CPU sequencer; and run_l0.sh --level flcn_hw must work in
# its dry run, and replay. The golden is golden_flcn_hw.py, the daemon's half test_c9.py. One PASS or FAIL line per check;
# exit 0 only if all pass.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
require_no_launch_guard
W="$TINYGPU_TEST_WORK/c9"; rm -rf "$W"; mkdir -p "$W"
fails=0
pass() { echo "PASS $1"; }
fail() { echo "FAIL $1"; fails=$((fails + 1)); }
check() { if eval "$2"; then pass "$1"; else fail "$1"; fi; }
replay_line() { grep -E "^replay session 2" "$TINYGPU_TEST_WORK/replay_$1.log"; }
out() { echo "$TINYGPU_TEST_WORK/run_device_c9_$1.txt"; }
dlog() { echo "$TINYGPU_TEST_WORK/run_device_c9_$1_daemon.log"; }
flog() { echo "$TINYGPU_TEST_WORK/fake_device_c9_$1.log"; }
BUILT="C++ runtime: built the NVDevice after the falcons"
TL="$TINYGPU_TEST_WORK/beagle_tinygpu_offline.log"   # the plugin's transport log in these runs

# 1. the whole session at the fake device at levels sysmem and flcn_hw: identical bytes
for lv in sysmem flcn_hw; do
    FAKE_TG_RECORD="$W/dev_$lv.bin" "$TG_TESTS/run_fake_device.sh" c9_$lv BEAGLE_NV_CPP_LEVEL=$lv -- --state-count 4 --reps 3 > "$W/$lv.txt" 2>&1
    eval "r_$lv=$?"
done
check "fake AD107: both levels PASS, and the device received identical bytes from the daemon's falcon boot and the plugin's" \
    "[ $r_sysmem -eq 0 ] && [ $r_flcn_hw -eq 0 ] && cmp -s '$W/dev_sysmem.bin' '$W/dev_flcn_hw.bin'"
check "at flcn_hw: the daemon's boot stopped after both init_sw; the plugin ran the falcons, booted GSP-RM, built the NVDevice and unloaded the GPU at fini" \
    "grep -q 'daemon booted the NVDev (level flcn_hw: the images prepared)' '$(out flcn_hw)' && grep -q '(level flcn_hw: sm_89, QMD v3' '$(out flcn_hw)' \
     && ! grep -q '$BUILT' '$(out sysmem)' && grep -q 'the C++ GSP unload and teardown' '$(out flcn_hw)' \
     && grep -q 'rm export: GSP queues (seq 2), .*the C++ side runs the falcons and boots GSP-RM' '$(dlog flcn_hw)' \
     && ! grep -q 'before FWSEC-FRTS' '$(dlog flcn_hw)' && grep -q 'before FWSEC-FRTS' '$(dlog sysmem)' \
     && grep -q 'state page mapped (phase 4, frame_in_flight 1, seq 2)' '$(dlog flcn_hw)'"
# ... and two instances in one process sharing the plugin's GPU (plan step P5), in two threads
"$TG_TESTS/run_fake_device.sh" c9_p5 BEAGLE_NV_CPP_LEVEL=flcn_hw -- --state-count 4,64 --threads --reps 3 > "$W/p5.txt" 2>&1
check "fake AD107 at flcn_hw: two instances in two threads share the boot and the plugin's NVDevice" \
    "[ $? -eq 0 ] && [ \$(grep -c '$BUILT' '$(out p5)') -eq 1 ] && grep -q '^tips: every instance read back its own tip partials exactly' '$(out p5)'"

# 2. the falcons fail: before booter_load, or with booter_load failing, GSP-RM never started (the daemon closes); after booter_load
#    started it, the daemon holds
FAKE_FALCON_FAIL=frts "$TG_TESTS/run_fake_device.sh" c9_frts BEAGLE_NV_CPP_LEVEL=flcn_hw -- --state-count 4 --reps 3 > "$W/frts.txt" 2>&1
check "fake AD107 at flcn_hw, FWSEC-FRTS leaves WPR2 down: the plugin stops, and the daemon closes without holding, sending nothing" \
    "grep -q 'level flcn_hw: building the NVDevice: AssertionError: WPR2 is not initialized' '$(out frts)' \
     && grep -q 'C++ state page: phase 4, frame_in_flight 0, last_submitted 0, seq 2,' '$(dlog frts)' \
     && grep -q 'stopped before booter_load started GSP-RM: nothing to unload' '$(dlog frts)' && ! grep -q 'HOLDING' '$(dlog frts)' \
     && grep -q 'NO ERRORS' '$(flog frts)' && ! grep -q '\"booter_load\"' '$(flog frts)'"
FAKE_FALCON_FAIL=booter "$TG_TESTS/run_fake_device.sh" c9_booter BEAGLE_NV_CPP_LEVEL=flcn_hw -- --state-count 4 --reps 3 > "$W/booter.txt" 2>&1
check "fake AD107 at flcn_hw, booter_load returns MAILBOX0 0x29: GSP-RM never started, and the daemon closes without holding" \
    "grep -Eq 'level flcn_hw: building the NVDevice: AssertionError: Booter failed to execute, mailbox is 00000029, [0-9a-f]{8}$' '$(out booter)' \
     && grep -q 'C++ state page: phase 4,' '$(dlog booter)' && ! grep -q 'HOLDING' '$(dlog booter)' && grep -q 'NO ERRORS' '$(flog booter)' \
     && grep -q '\"booter_load\": 1' '$(flog booter)'"
FAKE_FALCON_FAIL=core "$TG_TESTS/run_fake_device.sh" c9_core BEAGLE_NV_CPP_LEVEL=flcn_hw -- --state-count 4 --reps 3 > "$W/core.txt" 2>&1
check "fake AD107 at flcn_hw, booter_load ran but the GSP core is not active: GSP-RM may run, and the daemon holds, sending nothing" \
    "grep -q 'level flcn_hw: building the NVDevice: AssertionError: GSP Core is not active' '$(out core)' \
     && grep -q 'C++ state page: phase 3,' '$(dlog core)' && grep -q 'HOLDING the TinyGPU.app connection' '$(dlog core)' \
     && grep -q 'held the fake connection; ending it' '$W/core.txt'"

# 3. the L0 recordings, replayed to the plugin at level flcn_hw: it runs FWSEC-FRTS, booter_load and the RTX 4060's CPU sequencer
L0=(); for r in $TG_L0; do [ -f "$BEAGLE_TINYGPU_DATA/recordings/$r/events.bin" ] && L0+=("$BEAGLE_TINYGPU_DATA/recordings/$r"); done
if [ ${#L0[@]} -eq 3 ]; then
    for R in "${L0[@]}"; do
        l=c9_flcn_${R##*_}
        n0=$(wc -c < "$TL" 2>/dev/null || echo 0)
        BEAGLE_NV_CPP_LEVEL=flcn_hw "$TG_TESTS/run_replay.sh" "$R" $l --guard > "$W/replay_$l.txt" 2>&1
        check "L0 ${R##*/} at flcn_hw: replays exactly with the plugin's falcon and GSP-RM boot, under the guard" \
            "replay_line $l | grep -q 'PASS' && grep -q '$BUILT' '$TINYGPU_TEST_WORK/run_replay_$l.txt' \
             && tail -c +$((n0 + 1)) '$TL' | grep -q 'after FWSEC-FRTS: ' && tail -c +$((n0 + 1)) '$TL' | grep -q 'CPU sequencer (boot): ops'"
    done
else
    echo "(no L0 recordings in $BEAGLE_TINYGPU_DATA/recordings: their checks skipped)"
fi

# 4. run_l0.sh --level flcn_hw --guard in its dry run, and the replay of what it recorded
rm -rf "$TINYGPU_TEST_WORK/l0_dry"
L0_DRY_RUN=1 "$TG_TESTS/run_l0.sh" c9dry 4 3 --level flcn_hw --guard > "$W/l0_dry.txt" 2>&1
R=$(ls -d "$TINYGPU_TEST_WORK"/l0_dry/recordings/*_c9dry 2>/dev/null | head -1)
check "run_l0.sh --level flcn_hw --guard dry run: PASS through the guard, recorded with the level in run.json" \
    "tail -1 '$W/l0_dry.txt' | grep -q '^OK: PASS, recorded' && grep -q 'BEAGLE_NV_CPP_LEVEL=flcn_hw' '$R/run.json' \
     && grep -q '(guard mode)' \$(ls \"$TINYGPU_TEST_WORK\"/l0_dry/runs/*_L0_c9dry_proxy.log | head -1)"
"$TG_TESTS/run_replay.sh" "$R" c9_l0dry --record --guard > "$W/replay_l0dry.txt" 2>&1
check "the dry run's recording replays exactly, markers equal, under the guard" \
    "replay_line c9_l0dry | grep -q 'PASS' && replay_line c9_l0dry | grep -q '\"markers\": \"equal\"'"

left=$(ps -axo command | grep -cE "Python .*(tgdaemon|tgproxy|tgreplay|fake_nv_device)\.py")
check "no harness process is left running" "[ $left -eq 0 ]"
echo; [ $fails -eq 0 ] && echo "test_c9: PASS" || echo "test_c9: $fails FAILED"
[ $fails -eq 0 ]
