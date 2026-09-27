#!/bin/bash
# TODO.md plan step B2, end to end with no eGPU: a GB205 in fake_nv_device.py (FAKE_NV_CHIP=gb205: the FSP/FMC COT boot, MMU
# v3, QMD v5), booted by the real daemon (tinygrad's NV_FLCN_COT) for the real plugin. Levels runtime, teardown, vram and sysmem
# pass (from teardown on, the plugin's own COT teardown: the unload and the RISC-V halt wait), and the device receives identical
# bytes at all four; at rm the plugin builds the NVDevice itself, at gsp_hw it also completes GSP-RM's boot (init_hw, the golden
# image) after the daemon's COT message, and at flcn_hw it sends the COT message itself (level sysmem's bytes again, all three). run_l0.sh records levels runtime, vram and sysmem in its
# dry run, the last two through the guard; each recording replays exactly under the guard (MMU v3 walks, the COT start and
# teardown), and vram's and sysmem's are equivalent to runtime's (tgcanon); the guard refuses a bad sysmem PTE and a bad RPC
# checksum; and the GB205's own recordings (TG_GB20X) replay exactly under the guard, each later one equivalent to the L0, all
# with the plugin's COT teardown (the L0 at level teardown), and all at levels rm, gsp_hw and flcn_hw. One PASS or FAIL line per
# check; exit 0 only if all pass.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
require_no_launch_guard
export FAKE_NV_CHIP=gb205
W="$TINYGPU_TEST_WORK/b2"; rm -rf "$W"; mkdir -p "$W"
fails=0
pass() { echo "PASS $1"; }
fail() { echo "FAIL $1"; fails=$((fails + 1)); }
check() { if eval "$2"; then pass "$1"; else fail "$1"; fi; }
replay_line() { grep -E "^replay session 2" "$TINYGPU_TEST_WORK/replay_$1.log"; }
out() { echo "$TINYGPU_TEST_WORK/run_device_b2_$1.txt"; }
device() { grep -E "fake TinyGPU.app \(GB205 device\): " "$TINYGPU_TEST_WORK/fake_device_b2_$1.log" | tail -1; }
rec() { ls -d "$TINYGPU_TEST_WORK"/l0_dry/recordings/*_b2$1 2>/dev/null | head -1; }

# 1. the levels that run on the COT boot: PASS, the same bytes at the device whether the daemon or the plugin allocated
for lv in runtime teardown vram sysmem; do
    FAKE_TG_RECORD="$W/dev_$lv.bin" "$TG_TESTS/run_fake_device.sh" b2_$lv BEAGLE_NV_CPP_LEVEL=$lv -- --state-count 4 --reps 3 > "$W/$lv.txt" 2>&1
    eval "r_$lv=$?"
done
check "fake GB205: levels runtime, teardown, vram and sysmem PASS (QMD v5), with identical bytes at the device" \
    "[ $r_runtime -eq 0 ] && [ $r_teardown -eq 0 ] && [ $r_vram -eq 0 ] && [ $r_sysmem -eq 0 ] && grep -q 'QMD v5' '$(out sysmem)' \
     && cmp -s '$W/dev_runtime.bin' '$W/dev_teardown.bin' && cmp -s '$W/dev_runtime.bin' '$W/dev_vram.bin' \
     && cmp -s '$W/dev_runtime.bin' '$W/dev_sysmem.bin'"
COT_TD="C++ teardown: the GSP unload and the RISC-V halt wait (COT) run here at fini"
check "from teardown on the plugin unloads the COT boot itself, and its RISC-V core halted; its MMU v3 memory manager allocates the pool at vram, and the buffers at sysmem" \
    "! grep -q 'C++ teardown' '$(out runtime)' && grep -q '$COT_TD (BEAGLE_NV_CPP_LEVEL=teardown)' '$(out teardown)' \
     && grep -q '$COT_TD (BEAGLE_NV_CPP_LEVEL=vram)' '$(out vram)' && grep -q '$COT_TD (BEAGLE_NV_CPP_LEVEL=sysmem)' '$(out sysmem)' \
     && grep -q 'teardown: done: GSP RISC-V halted after' '$(out teardown)' \
     && grep -q 'C++ memory manager: pool, VRAM pool 6114 MiB' '$(out vram)' && grep -q 'C++ memory manager: buffers and pool' '$(out sysmem)'"

# 2. rm: the plugin builds the NVDevice with its RM client after the daemon's full boot; gsp_hw: it also boots GSP-RM after the
#    daemon's COT message; flcn_hw: it sends the COT message itself, on the daemon's prepared images
FAKE_TG_RECORD="$W/dev_rm.bin" "$TG_TESTS/run_fake_device.sh" b2_rm BEAGLE_NV_CPP_LEVEL=rm -- --state-count 4 --reps 3 > "$W/rm.txt" 2>&1
check "level rm: the plugin builds the NVDevice (sm_120, QMD v5) and tears the COT boot down; the device receives level sysmem's bytes" \
    "[ $? -eq 0 ] && grep -q 'built the NVDevice after the NVDev.s boot (level rm: sm_120, QMD v5' '$(out rm)' \
     && grep -q '$COT_TD (BEAGLE_NV_CPP_LEVEL=rm)' '$(out rm)' && cmp -s '$W/dev_sysmem.bin' '$W/dev_rm.bin'"
FAKE_TG_RECORD="$W/dev_gsp_hw.bin" "$TG_TESTS/run_fake_device.sh" b2_gsp_hw BEAGLE_NV_CPP_LEVEL=gsp_hw -- --state-count 4 --reps 3 > "$W/gsp_hw.txt" 2>&1
check "level gsp_hw: the plugin boots GSP-RM (init_hw, the golden image) after the daemon's COT message, builds the NVDevice and tears down; level sysmem's bytes" \
    "[ $? -eq 0 ] && grep -q 'daemon booted the NVDev (level gsp_hw: GSP-RM started)' '$(out gsp_hw)' \
     && grep -q 'built the NVDevice after booting GSP-RM (init_hw, the golden image) (level gsp_hw: sm_120, QMD v5' '$(out gsp_hw)' \
     && grep -q '$COT_TD (BEAGLE_NV_CPP_LEVEL=gsp_hw)' '$(out gsp_hw)' && cmp -s '$W/dev_sysmem.bin' '$W/dev_gsp_hw.bin'"
FAKE_TG_RECORD="$W/dev_flcn_hw.bin" "$TG_TESTS/run_fake_device.sh" b2_flcn_hw BEAGLE_NV_CPP_LEVEL=flcn_hw -- --state-count 4 --reps 3 > "$W/flcn_hw.txt" 2>&1
check "level flcn_hw: the plugin sends the COT message (the FSP boots the FMC once), then boots GSP-RM and builds the NVDevice; level sysmem's bytes" \
    "[ $? -eq 0 ] && grep -q 'daemon booted the NVDev (level flcn_hw: the images prepared)' '$(out flcn_hw)' \
     && grep -q 'built the NVDevice after the falcons. boot and GSP-RM.s (both init_hw, the golden image) (level flcn_hw: sm_120' '$(out flcn_hw)' \
     && grep -q '$COT_TD (BEAGLE_NV_CPP_LEVEL=flcn_hw)' '$(out flcn_hw)' && grep -q '\"FMC boots\": 1' '$TINYGPU_TEST_WORK/fake_device_b2_flcn_hw.log' \
     && cmp -s '$W/dev_sysmem.bin' '$W/dev_flcn_hw.bin'"

# 3. run_l0.sh's dry runs at runtime, then vram and sysmem through the guard (as the rungs' first hardware runs use it); each
#    recording replays exactly under the guard, and the plugin's allocations send what the daemon's did
rm -rf "$TINYGPU_TEST_WORK/l0_dry"
for lv in runtime vram sysmem; do
    L0_DRY_RUN=1 "$TG_TESTS/run_l0.sh" b2$lv 4 3 --poison --level $lv $([ $lv = runtime ] || echo --guard) > "$W/l0_$lv.txt" 2>&1
    check "run_l0.sh --level $lv dry run$([ $lv = runtime ] || echo ' through the guard'): PASS, recorded" "tail -1 '$W/l0_$lv.txt' | grep -q '^OK: PASS, recorded'"
    "$TG_TESTS/run_replay.sh" "$(rec $lv)" b2_rp_$lv --record --guard > "$W/replay_$lv.txt" 2>&1
    check "the $lv recording replays exactly, markers equal, under the guard (MMU v3, the COT start and teardown)" \
        "replay_line b2_rp_$lv | grep -q 'PASS' && replay_line b2_rp_$lv | grep -q '\"markers\": \"equal\"' && replay_line b2_rp_$lv | grep -q '\"launches\": [1-9]'"
done
for lv in vram sysmem; do
    "$BEAGLE_PYTHON" "$TG_TESTS/replay/tgcanon.py" "$(rec runtime)" "$(rec $lv)" > "$W/canon_$lv.txt" 2>&1
    check "tgcanon: the $lv recording is equivalent to runtime's (markers aside)" "[ $? -eq 0 ]"
done

# 4. the guard, live behind the proxy on MMU v3: each defect is refused before it reaches the device
for m in "pte-sys-bad|the PTE for VA .* points at device address" "rpc-corrupt|bad checksum"; do
    IFS='|' read -r mut rx <<< "$m"
    BEAGLE_TG_MUTATE=$mut FAKE_TG_GUARD=1 FAKE_EXPECT_TRIP="$rx" FAKE_TG_PROXY="$W/guard_$mut" "$TG_TESTS/run_fake_device.sh" b2_guard_$mut \
        BEAGLE_NV_CPP_LEVEL=sysmem -- --state-count 4 --reps 3 > "$W/guard_$mut.txt" 2>&1
    check "the guard refuses the $mut defect" "tail -1 '$W/guard_$mut.txt' | grep -q 'PASS (the expected refusal)'"
done

# 5. the GB205's own recordings, where $BEAGLE_TINYGPU_DATA has them: the L0 replays exactly under the guard (the GSP's status
#    queue wraps before the first doorbell there), H1's and H2's are equivalent to it (the COT boot's GSP log buffer masked), and
#    all three replay exactly with the plugin's COT teardown (below; not marker for marker: they recorded the daemon's NVDev.fini)
GB=(); for r in $TG_GB20X; do [ -f "$BEAGLE_TINYGPU_DATA/recordings/$r/events.bin" ] && GB+=("$BEAGLE_TINYGPU_DATA/recordings/$r"); done
if [ ${#GB[@]} -eq 6 ]; then
    "$TG_TESTS/run_replay.sh" "${GB[0]}" b2_hw_l0 --record --guard > "$W/replay_b2_hw_l0.txt" 2>&1
    check "GB205 ${GB[0]##*/}: replays exactly, markers equal, under the guard" \
        "replay_line b2_hw_l0 | grep -q 'PASS' && replay_line b2_hw_l0 | grep -q '\"markers\": \"equal\"'"
    for R in "${GB[@]:1}"; do
        "$BEAGLE_PYTHON" "$TG_TESTS/replay/tgcanon.py" "${GB[0]}" "$R" > "$W/canon_hw_${R##*_}.txt" 2>&1
        check "tgcanon: GB205 ${R##*/} is equivalent to the L0 (the GSP log buffer masked)" \
            "[ $? -eq 0 ] && grep -q 'GSP log buffer pages differ' '$W/canon_hw_${R##*_}.txt'"
    done
    # made with the daemon's teardown (T's with the plugin's): the plugin's sends the same (the L0 replayed at level teardown)
    for R in "${GB[@]}"; do
        l=b2_hwtd_${R##*_}
        "$TG_TESTS/run_replay.sh" "$R" $l --record --guard $([ "$R" = "${GB[0]}" ] && echo --level teardown) > "$W/replay_$l.txt" 2>&1
        check "GB205 ${R##*/}: replays exactly with the plugin's COT teardown, under the guard" \
            "replay_line $l | grep -q 'PASS' && grep -q '$COT_TD' '$TINYGPU_TEST_WORK/run_replay_$l.txt' \
             && grep -q 'teardown: done: GSP RISC-V halted after' '$TINYGPU_TEST_WORK/run_replay_$l.txt'"
    done
    # ... and at levels rm, gsp_hw and flcn_hw: the plugin's RM client builds the NVDevice with the RM calls tinygrad made on the
    #     card, at gsp_hw its init_hw and golden image run on the card's own GSP-RM replies, and at flcn_hw its COT message is the
    #     one tinygrad sent the card's FSP, word for word
    for lv in rm gsp_hw flcn_hw; do
        for R in "${GB[@]}"; do
            l=b2_hw${lv}_${R##*_}
            "$TG_TESTS/run_replay.sh" "$R" $l --record --guard --level $lv > "$W/replay_$l.txt" 2>&1
            check "GB205 ${R##*/}: replays exactly at level $lv, the plugin's NVDevice and COT teardown, under the guard" \
                "replay_line $l | grep -q 'PASS' && grep -q 'built the NVDevice after .* (level $lv: sm_120' '$TINYGPU_TEST_WORK/run_replay_$l.txt' \
                 && grep -q '$COT_TD (BEAGLE_NV_CPP_LEVEL=$lv)' '$TINYGPU_TEST_WORK/run_replay_$l.txt'"
        done
    done
else
    echo "(no GB205 recordings in $BEAGLE_TINYGPU_DATA/recordings: their checks skipped)"
fi

left=$(ps -axo command | grep -cE "Python .*(tgdaemon|tgproxy|tgreplay|fake_nv_device)\.py")
check "no harness process is left running" "[ $left -eq 0 ]"
echo; [ $fails -eq 0 ] && echo "test_b2: PASS" || echo "test_b2: $fails FAILED"
[ $fails -eq 0 ]
