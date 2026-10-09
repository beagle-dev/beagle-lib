#!/bin/bash
# TODO.md plan step B2, end to end with no eGPU: a GB205 in fake_nv_device.py (FAKE_NV_CHIP=gb205: the FSP/FMC COT boot, MMU
# v3, QMD v5), which the plugin boots itself (plan steps C11-C13). A run passes (sm_120, QMD v5), sending the COT message once,
# with the plugin's MMU v3 memory manager and its COT teardown (the unload and the RISC-V halt wait). run_l0.sh records in its
# dry run, plain and through the guard; each recording replays exactly under the guard (MMU v3 walks, the COT start and
# teardown), and the two are equivalent (tgcanon). The GB205's own recordings (TG_GB20X), made through the daemon, are each
# equivalent to the L0 (tgcanon), and each replays exactly under the guard to the C++ boot and the plugin's COT teardown. (The
# guard refuses a PTE to memory it does not know, and a corrupt RPC, on MMU v3: each put into its inputs alone in the replay.)
# One PASS or FAIL line per check; exit 0 only if all pass.
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

# 1. a run: the C++ boot, the plugin's memory manager on MMU v3, QMD v5 and the COT teardown
FAKE_TG_RECORD="$W/dev_boot.bin" "$TG_TESTS/run_fake_device.sh" b2_boot -- --state-count 4 --reps 3 > "$W/boot.txt" 2>&1; r=$?
COT_TD="C++ teardown: the GSP unload and the RISC-V halt wait (COT) run here at exit"
check "fake GB205: level boot PASS (sm_120, QMD v5), the FSP booting the FMC once" \
    "[ $r -eq 0 ] && grep -q 'built the NVDevice after the C++ boot, with no daemon (level boot: sm_120, QMD v5' '$(out boot)' \
     && grep -q '\"FMC boots\": 1' '$TINYGPU_TEST_WORK/fake_device_b2_boot.log'"
check "the plugin's MMU v3 memory manager allocates the pool and the buffers, and its COT teardown halts the RISC-V core" \
    "grep -q 'C++ memory manager: buffers and pool, VRAM pool 6114 MiB' '$(out boot)' && grep -q '$COT_TD' '$(out boot)' \
     && grep -q 'teardown: done: GSP RISC-V halted after' '$(out boot)'"

# 2. run_l0.sh's dry runs, plain and through the guard (as a new client's first hardware run uses it): each recording replays
#    exactly under the guard, and the two are equivalent
rm -rf "$TINYGPU_TEST_WORK/l0_dry"
for m in plain guard; do
    L0_DRY_RUN=1 "$TG_TESTS/run_l0.sh" b2$m 4 3 --poison $([ $m = guard ] && echo --guard) > "$W/l0_$m.txt" 2>&1
    check "run_l0.sh dry run$([ $m = guard ] && echo ' through the guard'): PASS, recorded" "tail -1 '$W/l0_$m.txt' | grep -q '^OK: PASS, recorded'"
    "$TG_TESTS/run_replay.sh" "$(rec $m)" b2_rp_$m --record --guard > "$W/replay_$m.txt" 2>&1
    check "the $m recording replays exactly, markers equal, under the guard (MMU v3, the COT start and teardown)" \
        "replay_line b2_rp_$m | grep -q 'PASS' && replay_line b2_rp_$m | grep -q '\"markers\": \"equal\"' && replay_line b2_rp_$m | grep -q '\"launches\": [1-9]'"
done
"$BEAGLE_PYTHON" "$TG_TESTS/replay/tgcanon.py" "$(rec plain)" "$(rec guard)" > "$W/canon_guard.txt" 2>&1
check "tgcanon: the recording through the guard is equivalent to the plain one" "[ $? -eq 0 ]"
for m in "pte-sys-bad|the PTE for VA .* points at device address .*, not a live sysmem page" "rpc-corrupt|NV_PGSP_QUEUE_HEAD.*has a bad checksum"; do
    IFS='|' read -r d rx <<< "$m"
    "$TG_TESTS/run_replay.sh" "$(rec plain)" b2_gd_$d --guard --guard-defect $d > "$W/replay_gd_$d.txt" 2>&1
    check "the guard refuses the $d defect on MMU v3 (in the replay)" "replay_line b2_gd_$d | grep -qE 'FAIL: the guard refused .*$rx'"
done

# 3. the GB205's own recordings, where $BEAGLE_TINYGPU_DATA has them: H1's to H4's are equivalent to the L0 (the COT boot's GSP
#    log buffer masked), and each replays exactly to the C++ boot and the plugin's COT teardown under the guard: on the card's
#    own replies, the boot and the NVDevice make the requests tinygrad's made, the COT message word for word, and the teardown
#    sends what T's recorded (not marker for marker: the recordings hold the daemon's markers)
GB=(); for r in $TG_GB20X; do [ -f "$BEAGLE_TINYGPU_DATA/recordings/$r/events.bin" ] && GB+=("$BEAGLE_TINYGPU_DATA/recordings/$r"); done
if [ ${#GB[@]} -eq 6 ]; then
    for R in "${GB[@]:1}"; do
        "$BEAGLE_PYTHON" "$TG_TESTS/replay/tgcanon.py" "${GB[0]}" "$R" > "$W/canon_hw_${R##*_}.txt" 2>&1
        check "tgcanon: GB205 ${R##*/} is equivalent to the L0 (the GSP log buffer masked)" \
            "[ $? -eq 0 ] && grep -q 'GSP log buffer pages differ' '$W/canon_hw_${R##*_}.txt'"
    done
    for R in "${GB[@]}"; do
        l=b2_hw_${R##*_}
        "$TG_TESTS/run_replay.sh" "$R" $l --guard > "$W/replay_$l.txt" 2>&1
        check "GB205 ${R##*/}: replays exactly at level boot, with the plugin's COT teardown, under the guard" \
            "replay_line $l | grep -q 'PASS' && grep -q 'built the NVDevice after the C++ boot, with no daemon (level boot: sm_120' '$TINYGPU_TEST_WORK/run_replay_$l.txt' \
             && grep -q '$COT_TD' '$TINYGPU_TEST_WORK/run_replay_$l.txt' && grep -q 'teardown: done: GSP RISC-V halted after' '$TINYGPU_TEST_WORK/run_replay_$l.txt'"
    done
else
    echo "(no GB205 recordings in $BEAGLE_TINYGPU_DATA/recordings: their checks skipped)"
fi

left=$(ps -axo command | grep -cE "Python .*(tgproxy|tgreplay|fake_nv_device)\.py|[b]eagle-tinygpu-guard")
check "no harness process or guard is left running" "[ $left -eq 0 ]"
echo; [ $fails -eq 0 ] && echo "test_b2: PASS" || echo "test_b2: $fails FAILED"
[ $fails -eq 0 ]
