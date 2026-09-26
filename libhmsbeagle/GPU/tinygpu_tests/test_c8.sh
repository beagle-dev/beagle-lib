#!/bin/bash
# TODO.md plan step C8, end to end with no eGPU: BEAGLE_NV_CPP_LEVEL=gsp_hw, where the daemon's boot stops once GSP-RM started
# (booter_load) and the plugin runs NV_GSP.init_hw (GSP-RM's INIT_DONE, with its CPU sequencer and nv_init_helper's 20 s sleep
# after SEC2's start) and init_golden_image, ported (TinyGPUHybridNVRM.h), then builds the NVDevice as at level rm. On the fake
# AD107 (whose GSP posts no CPU sequencer) the device must receive exactly the bytes it receives at level sysmem; two instances
# in two threads share the plugin's GPU; a GSP that never posts GSP_INIT_DONE leaves the daemon holding, with nothing sent to
# the GPU; an allocation the GSP refuses in the golden image, after INIT_DONE, is torn down by the daemon from the plugin's GSP
# count, with the status queue init_hw would have made; each L0 hardware recording must replay exactly at gsp_hw, under the
# guard, the plugin running the RTX 4060's CPU sequencer; and run_l0.sh --level gsp_hw must work in its dry run, and replay.
# The golden is golden_gsp_hw.py, the daemon's half test_c8.py. One PASS or FAIL line per check; exit 0 only if all pass.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
require_no_launch_guard
W="$TINYGPU_TEST_WORK/c8"; rm -rf "$W"; mkdir -p "$W"
fails=0
pass() { echo "PASS $1"; }
fail() { echo "FAIL $1"; fails=$((fails + 1)); }
check() { if eval "$2"; then pass "$1"; else fail "$1"; fi; }
replay_line() { grep -E "^replay session 2" "$TINYGPU_TEST_WORK/replay_$1.log"; }
out() { echo "$TINYGPU_TEST_WORK/run_device_c8_$1.txt"; }
dlog() { echo "$TINYGPU_TEST_WORK/run_device_c8_$1_daemon.log"; }
flog() { echo "$TINYGPU_TEST_WORK/fake_device_c8_$1.log"; }
BUILT="C++ runtime: built the NVDevice after booting GSP-RM"
TL="$TINYGPU_TEST_WORK/beagle_tinygpu_offline.log"   # the plugin's transport log in these runs (run_replay.sh, run_fake_device.sh)

# 1. the whole session at the fake device at levels sysmem and gsp_hw: identical bytes
for lv in sysmem gsp_hw; do
    FAKE_TG_RECORD="$W/dev_$lv.bin" "$TG_TESTS/run_fake_device.sh" c8_$lv BEAGLE_NV_CPP_LEVEL=$lv -- --state-count 4 --reps 3 > "$W/$lv.txt" 2>&1
    eval "r_$lv=$?"
done
check "fake AD107: both levels PASS, and the device received identical bytes from the daemon's GSP-RM boot and the plugin's" \
    "[ $r_sysmem -eq 0 ] && [ $r_gsp_hw -eq 0 ] && cmp -s '$W/dev_sysmem.bin' '$W/dev_gsp_hw.bin'"
check "at gsp_hw: the daemon's boot stopped once GSP-RM started and it exported init_sw's state; the plugin booted GSP-RM, built the NVDevice and unloaded the GPU at fini" \
    "grep -q 'daemon booted the NVDev (level gsp_hw: GSP-RM started)' '$(out gsp_hw)' && grep -q '(level gsp_hw: sm_89, QMD v3' '$(out gsp_hw)' \
     && ! grep -q '$BUILT' '$(out sysmem)' && grep -q 'the C++ GSP unload and teardown' '$(out gsp_hw)' \
     && grep -q 'rm export: GSP queues (seq 2), .*next handle 0xcf000000.*the C++ side boots GSP-RM' '$(dlog gsp_hw)'"
# ... and two instances in one process sharing the plugin's GPU (plan step P5), in two threads
"$TG_TESTS/run_fake_device.sh" c8_p5 BEAGLE_NV_CPP_LEVEL=gsp_hw -- --state-count 4,64 --threads --reps 3 > "$W/p5.txt" 2>&1
check "fake AD107 at gsp_hw: two instances in two threads share the boot and the plugin's NVDevice" \
    "[ $? -eq 0 ] && [ \$(grep -c '$BUILT' '$(out p5)') -eq 1 ] && grep -q '^tips: every instance read back its own tip partials exactly' '$(out p5)'"

# 2. failures: before GSP_INIT_DONE (a GSP that never posts it) the daemon holds; after it (the golden image's VA space refused)
#    the daemon unloads the GPU, continuing the plugin's GSP command queue
FAKE_NO_INIT_DONE=1 "$TG_TESTS/run_fake_device.sh" c8_noinit BEAGLE_NV_CPP_LEVEL=gsp_hw -- --state-count 4 --reps 3 > "$W/noinit.txt" 2>&1
check "fake AD107 at gsp_hw, no GSP_INIT_DONE: the plugin's init_hw times out, and the daemon holds, sending nothing to the GPU" \
    "grep -q 'level gsp_hw: building the NVDevice: RuntimeError: Timeout waiting for RPC response for command 4097' '$(out noinit)' \
     && grep -q 'C++ state page: phase 3, frame_in_flight 0, last_submitted 0, seq 2,' '$(dlog noinit)' \
     && grep -q 'the C++ side did not finish booting GSP-RM (init_hw, before GSP_INIT_DONE): sending nothing to the GPU' '$(dlog noinit)' \
     && grep -q 'HOLDING the TinyGPU.app connection' '$(dlog noinit)' && grep -q 'held the fake connection; ending it' '$W/noinit.txt'"
FAKE_RM_FAIL=0x90f1 "$TG_TESTS/run_fake_device.sh" c8_refused BEAGLE_NV_CPP_LEVEL=gsp_hw -- --state-count 4 --reps 3 > "$W/refused.txt" 2>&1
check "fake AD107 at gsp_hw, the golden image's VA space refused: the daemon tears down from the plugin's count (2, then 6) with init_hw's status queue, and the fake's GSP agrees" \
    "grep -q 'level gsp_hw: building the NVDevice: .*RPC call 103 failed with result 34' '$(out refused)' && ! grep -q '$BUILT' '$(out refused)' \
     && fini_verdict '$(out refused)' && grep -q 'NO ERRORS' '$(flog refused)' && grep -q '\"rm_alloc refused (FAKE_RM_FAIL)\": 1' '$(flog refused)' \
     && grep -q 'C++ state page: phase 1, frame_in_flight 0, last_submitted 0, seq 6,' '$(dlog refused)'"

# 3. the L0 recordings, replayed to the plugin at level gsp_hw: its init_hw runs the RTX 4060's CPU sequencer (with the 20 s sleep)
L0=(); for r in $TG_L0; do [ -f "$BEAGLE_TINYGPU_DATA/recordings/$r/events.bin" ] && L0+=("$BEAGLE_TINYGPU_DATA/recordings/$r"); done
if [ ${#L0[@]} -eq 3 ]; then
    for R in "${L0[@]}"; do
        l=c8_gsp_${R##*_}
        n0=$(wc -c < "$TL" 2>/dev/null || echo 0)
        BEAGLE_NV_CPP_LEVEL=gsp_hw "$TG_TESTS/run_replay.sh" "$R" $l --guard > "$W/replay_$l.txt" 2>&1
        check "L0 ${R##*/} at gsp_hw: replays exactly with the plugin's GSP-RM boot, its CPU sequencer run by the plugin, under the guard" \
            "replay_line $l | grep -q 'PASS' && grep -q '$BUILT' '$TINYGPU_TEST_WORK/run_replay_$l.txt' \
             && tail -c +$((n0 + 1)) '$TL' | grep -q 'CPU sequencer (boot): ops'"
    done
else
    echo "(no L0 recordings in $BEAGLE_TINYGPU_DATA/recordings: their checks skipped)"
fi

# 4. run_l0.sh --level gsp_hw --guard in its dry run (the fake AD107 in place of TinyGPU.app; the proxy in guard mode, as a rung's
#    first hardware run uses it), and the replay of what it recorded
rm -rf "$TINYGPU_TEST_WORK/l0_dry"
L0_DRY_RUN=1 "$TG_TESTS/run_l0.sh" c8dry 4 3 --level gsp_hw --guard > "$W/l0_dry.txt" 2>&1
R=$(ls -d "$TINYGPU_TEST_WORK"/l0_dry/recordings/*_c8dry 2>/dev/null | head -1)
check "run_l0.sh --level gsp_hw --guard dry run: PASS through the guard, recorded with the level in run.json" \
    "tail -1 '$W/l0_dry.txt' | grep -q '^OK: PASS, recorded' && grep -q 'BEAGLE_NV_CPP_LEVEL=gsp_hw' '$R/run.json' \
     && grep -q '(guard mode)' \$(ls \"$TINYGPU_TEST_WORK\"/l0_dry/runs/*_L0_c8dry_proxy.log | head -1)"
"$TG_TESTS/run_replay.sh" "$R" c8_l0dry --record --guard > "$W/replay_l0dry.txt" 2>&1
check "the dry run's recording replays exactly, markers equal, under the guard" \
    "replay_line c8_l0dry | grep -q 'PASS' && replay_line c8_l0dry | grep -q '\"markers\": \"equal\"'"

left=$(ps -axo command | grep -cE "Python .*(tgdaemon|tgproxy|tgreplay|fake_nv_device)\.py")
check "no harness process is left running" "[ $left -eq 0 ]"
echo; [ $fails -eq 0 ] && echo "test_c8: PASS" || echo "test_c8: $fails FAILED"
[ $fails -eq 0 ]
