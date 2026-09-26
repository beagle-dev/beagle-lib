#!/bin/bash
# TODO.md plan step C6, end to end with no eGPU: the plugin's own memory manager (tinygrad's, ported: TinyGPUMemory.h,
# TinyGPUHybridNVMemory.h) at BEAGLE_NV_CPP_LEVEL=vram (the VRAM pool) and sysmem (also the four buffers), in the real plugin
# and daemon. On the fake AD107 (fake_nv_device.py, which the real daemon boots) the device must receive exactly the bytes it
# receives at level teardown, where the daemon allocates everything; a pool reaching into GSP-RM's reserved region is refused
# by the plugin's WPR check and the GPU torn down; each L0 hardware recording (where $BEAGLE_TINYGPU_DATA has them) must
# replay exactly at both levels, under the guard; and run_l0.sh --level sysmem must work in its dry run, and replay. The C++
# halves are golden_mm.py, the daemon's test_c6.py. One PASS or FAIL line per check; exit 0 only if all pass.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
require_no_launch_guard
W="$TINYGPU_TEST_WORK/c6"; rm -rf "$W"; mkdir -p "$W"
fails=0
pass() { echo "PASS $1"; }
fail() { echo "FAIL $1"; fails=$((fails + 1)); }
check() { if eval "$2"; then pass "$1"; else fail "$1"; fi; }
replay_line() { grep -E "^replay session 2" "$TINYGPU_TEST_WORK/replay_$1.log"; }
MM="C++ memory manager:"

# 1. the whole session at the fake device at levels teardown, vram and sysmem: identical bytes
for lv in teardown vram sysmem; do
    FAKE_TG_RECORD="$W/dev_$lv.bin" "$TG_TESTS/run_fake_device.sh" c6_$lv BEAGLE_NV_CPP_LEVEL=$lv -- --state-count 4 --reps 3 > "$W/$lv.txt" 2>&1
    eval "r_$lv=$?"
done
out() { echo "$TINYGPU_TEST_WORK/run_device_c6_$1.txt"; }
check "fake AD107: all three levels PASS, and the device received identical bytes from the daemon's allocations and the plugin's" \
    "[ $r_teardown -eq 0 ] && [ $r_vram -eq 0 ] && [ $r_sysmem -eq 0 ] && cmp -s '$W/dev_teardown.bin' '$W/dev_vram.bin' \
     && cmp -s '$W/dev_teardown.bin' '$W/dev_sysmem.bin'"
check "the plugin allocated the pool at vram, the buffers and the pool at sysmem, and nothing at teardown" \
    "grep -q '$MM pool, VRAM pool 4094 MiB' '$(out vram)' && grep -q '$MM buffers and pool, VRAM pool 4094 MiB' '$(out sysmem)' \
     && ! grep -q '$MM' '$(out teardown)' && grep -q '(level sysmem)' '$TINYGPU_TEST_WORK/run_device_c6_sysmem_daemon.log'"
# ... and two instances in one process sharing one boot and the plugin's one memory manager (plan step P5), in two threads
"$TG_TESTS/run_fake_device.sh" c6_p5 BEAGLE_NV_CPP_LEVEL=sysmem -- --state-count 4,64 --threads --reps 3 > "$W/p5.txt" 2>&1
check "fake AD107 at sysmem: two instances in two threads share the boot and the plugin's memory manager" \
    "[ $? -eq 0 ] && [ \$(grep -c '$MM' '$(out p5)') -eq 1 ] && grep -q '^tips: every instance read back its own tip partials exactly' '$(out p5)'"
# 2. the plugin's WPR check: a pool reaching into GSP-RM's reserved region is refused after the handoff, and the daemon tears down
"$TG_TESTS/run_fake_device.sh" c6_wpr BEAGLE_NV_CPP_LEVEL=sysmem BEAGLE_NV_DATA_MB=7900 -- --state-count 4 --reps 3 > "$W/wpr.txt" 2>&1
check "fake AD107, BEAGLE_NV_DATA_MB=7900 at sysmem: the plugin's WPR check refuses the pool, and the GPU is torn down" \
    "grep -q 'handoff: VRAM allocations end at 0x[0-9a-f]*, above the WPR bound 0x1f3a00000' '$(out wpr)' && ! grep -q 'handed over' '$(out wpr)' \
     && fini_verdict '$(out wpr)'"

# 3. the L0 recordings, replayed to the plugin at both levels: every request as the daemon's allocations sent it on the RTX 4060
L0=(); for r in $TG_L0; do [ -f "$BEAGLE_TINYGPU_DATA/recordings/$r/events.bin" ] && L0+=("$BEAGLE_TINYGPU_DATA/recordings/$r"); done
if [ ${#L0[@]} -eq 3 ]; then
    for lv in vram sysmem; do
        for R in "${L0[@]}"; do
            l=c6_${lv}_${R##*_}
            BEAGLE_NV_CPP_LEVEL=$lv "$TG_TESTS/run_replay.sh" "$R" $l --guard > "$W/replay_$l.txt" 2>&1
            check "L0 ${R##*/} at $lv: replays exactly with the plugin's memory manager, under the guard" \
                "replay_line $l | grep -q 'PASS' && grep -q '$MM' '$TINYGPU_TEST_WORK/run_replay_$l.txt'"
        done
    done
else
    echo "(no L0 recordings in $BEAGLE_TINYGPU_DATA/recordings: their checks skipped)"
fi

# 4. run_l0.sh --level sysmem --guard in its dry run (the fake AD107 in place of TinyGPU.app; the proxy in guard mode, as the
#    rungs' first hardware runs use it), and the replay of what it recorded
rm -rf "$TINYGPU_TEST_WORK/l0_dry"
L0_DRY_RUN=1 "$TG_TESTS/run_l0.sh" c6dry 4 3 --level sysmem --guard > "$W/l0_dry.txt" 2>&1
R=$(ls -d "$TINYGPU_TEST_WORK"/l0_dry/recordings/*_c6dry 2>/dev/null | head -1)
check "run_l0.sh --level sysmem --guard dry run: PASS through the guard, recorded with the level in run.json" \
    "tail -1 '$W/l0_dry.txt' | grep -q '^OK: PASS, recorded' && grep -q 'BEAGLE_NV_CPP_LEVEL=sysmem' '$R/run.json' \
     && grep -q '(guard mode)' \$(ls \"$TINYGPU_TEST_WORK\"/l0_dry/runs/*_L0_c6dry_proxy.log | head -1)"
"$TG_TESTS/run_replay.sh" "$R" c6_l0dry --record --guard > "$W/replay_l0dry.txt" 2>&1
check "the dry run's recording replays exactly, markers equal, under the guard" \
    "replay_line c6_l0dry | grep -q 'PASS' && replay_line c6_l0dry | grep -q '\"markers\": \"equal\"'"

left=$(ps -axo command | grep -cE "Python .*(tgdaemon|tgproxy|tgreplay|fake_nv_device)\.py")
check "no harness process is left running" "[ $left -eq 0 ]"
echo; [ $fails -eq 0 ] && echo "test_c6: PASS" || echo "test_c6: $fails FAILED"
[ $fails -eq 0 ]
