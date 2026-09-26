#!/bin/bash
# TODO.md plan step C5, end to end with no eGPU: the plugin's own GSP unload and NVIDIA's teardown at fini
# (BEAGLE_NV_CPP_LEVEL=teardown; TinyGPUHybridNVGsp.h, TinyGPUHybridNVFalcon.h) in the real plugin and daemon. On the fake AD107
# (fake_nv_device.py, which the real daemon boots) it must send the device exactly the bytes the daemon's Python teardown
# sends; each L0 hardware recording (where $BEAGLE_TINYGPU_DATA has them) must replay exactly with it in place of the
# daemon's; and run_l0.sh --level teardown must work in its dry run, and replay. The C++ halves are golden_gsp.py, the
# daemon's half test_c5.py. One PASS or FAIL line per check; exit 0 only if all pass.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
require_no_launch_guard
W="$TINYGPU_TEST_WORK/c5"; rm -rf "$W"; mkdir -p "$W"
fails=0
pass() { echo "PASS $1"; }
fail() { echo "FAIL $1"; fails=$((fails + 1)); }
check() { if eval "$2"; then pass "$1"; else fail "$1"; fi; }
replay_line() { grep -E "^replay session 2" "$TINYGPU_TEST_WORK/replay_$1.log"; }

# 1. the whole session at the fake device, level runtime (the daemon's Python teardown) and level teardown (the plugin's)
FAKE_TG_RECORD="$W/dev_runtime.bin" "$TG_TESTS/run_fake_device.sh" c5_runtime BEAGLE_NV_CPP_LEVEL=runtime -- --state-count 4 --reps 3 > "$W/runtime.txt" 2>&1; r1=$?
FAKE_TG_RECORD="$W/dev_teardown.bin" "$TG_TESTS/run_fake_device.sh" c5_teardown -- --state-count 4 --reps 3 > "$W/teardown.txt" 2>&1; r2=$?   # the default level
check "fake AD107: both levels PASS, at the default one the plugin tore the GPU down itself, and the device received identical bytes" \
    "[ $r1 -eq 0 ] && [ $r2 -eq 0 ] && grep -q 'C++ teardown: the GSP unload and NVIDIA' '$TINYGPU_TEST_WORK/run_device_c5_teardown.txt' \
     && ! grep -q 'C++ teardown:' '$TINYGPU_TEST_WORK/run_device_c5_runtime.txt' && cmp -s '$W/dev_runtime.bin' '$W/dev_teardown.bin'"
FAKE_TG_RECORD="$W/dev_teardown_off.bin" "$TG_TESTS/run_fake_device.sh" c5_teardown_off BEAGLE_NV_CPP_LEVEL=teardown BEAGLE_NV_TEARDOWN=0 \
    -- --state-count 4 --reps 3 > "$W/teardown_off.txt" 2>&1
check "fake AD107, BEAGLE_NV_TEARDOWN=0 at level teardown: the plugin's unload only, reported as needing a power cycle" \
    "grep -q 'C++ teardown: the GSP unload run here' '$TINYGPU_TEST_WORK/run_device_c5_teardown_off.txt' \
     && grep -q 'no teardown result (WPR2 is still up)' '$TINYGPU_TEST_WORK/run_device_c5_teardown_off.txt'"

# 2. the L0 recordings, replayed to the plugin with its own teardown: every request as the daemon's teardown sent it on the
#    RTX 4060, under the guard, and the same fini report
L0=(); for r in $TG_L0; do [ -f "$BEAGLE_TINYGPU_DATA/recordings/$r/events.bin" ] && L0+=("$BEAGLE_TINYGPU_DATA/recordings/$r"); done
if [ ${#L0[@]} -eq 3 ]; then
    for R in "${L0[@]}"; do
        l=c5_l0_${R##*_}
        BEAGLE_NV_CPP_LEVEL=teardown "$TG_TESTS/run_replay.sh" "$R" $l --guard > "$W/replay_$l.txt" 2>&1
        check "L0 ${R##*/}: replays exactly with the plugin's teardown, under the guard" \
            "replay_line $l | grep -q 'PASS' && grep -q 'C++ teardown: the GSP unload and NVIDIA' '$TINYGPU_TEST_WORK/run_replay_$l.txt' \
             && grep -q 'teardown: done: Booter Unload lowered WPR2' '$TINYGPU_TEST_WORK/run_replay_$l.txt'"
    done
else
    echo "(no L0 recordings in $BEAGLE_TINYGPU_DATA/recordings: their checks skipped)"
fi

# 3. run_l0.sh --level teardown in its dry run (the fake AD107 in place of TinyGPU.app), and the replay of what it recorded
rm -rf "$TINYGPU_TEST_WORK/l0_dry"
L0_DRY_RUN=1 "$TG_TESTS/run_l0.sh" c5dry 4 3 --level teardown > "$W/l0_dry.txt" 2>&1
R=$(ls -d "$TINYGPU_TEST_WORK"/l0_dry/recordings/*_c5dry 2>/dev/null | head -1)
check "run_l0.sh --level teardown dry run: PASS, recorded with the level in run.json" \
    "tail -1 '$W/l0_dry.txt' | grep -q '^OK: PASS, recorded' && grep -q 'BEAGLE_NV_CPP_LEVEL=teardown' '$R/run.json'"
"$TG_TESTS/run_replay.sh" "$R" c5_l0dry --record --guard > "$W/replay_l0dry.txt" 2>&1
check "the dry run's recording replays exactly, markers equal, under the guard" \
    "replay_line c5_l0dry | grep -q 'PASS' && replay_line c5_l0dry | grep -q '\"markers\": \"equal\"'"

left=$(ps -axo command | grep -cE "Python .*(tgdaemon|tgproxy|tgreplay|fake_nv_device)\.py")
check "no harness process is left running" "[ $left -eq 0 ]"
echo; [ $fails -eq 0 ] && echo "test_c5: PASS" || echo "test_c5: $fails FAILED"
[ $fails -eq 0 ]
