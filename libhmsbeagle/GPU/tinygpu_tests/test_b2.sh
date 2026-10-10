#!/bin/bash
# TODO.md plan step B2, end to end with no eGPU: a GB205 in fake_nv_device.py (FAKE_NV_CHIP=gb205: the FSP/FMC COT boot, MMU
# v3, QMD v5), which the plugin boots itself (plan steps C11-C13). A run passes (sm_120, QMD v5), sending the COT message once,
# with the plugin's MMU v3 memory manager and its COT teardown (the unload and the RISC-V halt wait). run_l0.sh records in its
# dry run, plain and through the guard; each recording replays exactly under the guard (MMU v3 walks, the COT start and
# teardown), and the two are equivalent (tgcanon). The GB205's own recordings (TG_GB20X), made through the daemon, are each
# equivalent to the L0 (tgcanon), and each replays exactly under the guard to the C++ boot and the plugin's COT teardown. (The
# guard refuses a PTE to memory it does not know, and a corrupt RPC, on MMU v3: each put into its inputs alone in the replay.)
# Every run is a job of its own (env.sh's pjob), started longest first, a run that needs a dry run's recording once that dry run
# has ended; the checks then read the runs' files in this order.
# One PASS or FAIL line per check; exit 0 only if all pass.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
require_no_launch_guard
W="$TINYGPU_TEST_WORK/b2"; rm -rf "$W"; mkdir -p "$W"
fails=0
pass() { echo "PASS $1"; }
fail() { echo "FAIL $1"; fails=$((fails + 1)); }
check() { if eval "$2"; then pass "$1"; else fail "$1"; fi; }
replay_line() { grep -E "^replay session 2" "$TINYGPU_TEST_WORK/replay_$1.log"; }
out() { echo "$TINYGPU_TEST_WORK/run_device_b2_$1.txt"; }
device() { grep -E "fake TinyGPU.app \(GB205 device\): " "$TINYGPU_TEST_WORK/fake_device_b2_$1.log" | tail -1; }
rec() { ls -d "$W"/l0_dry/recordings/*_b2$1 2>/dev/null | head -1; }
to() { local f=$1; shift; "$@" > "$f" 2>&1; }   # <output file> <command ...>: a job's command, its output to its own file

# the runs; every fake is the GB205 (FAKE_NV_CHIP=gb205 comes with each command that starts one, for that fake alone)
boot() {   # the C++ boot, the plugin's memory manager on MMU v3, QMD v5 and the COT teardown (the fake appends to its record)
    rm -f "$W/dev_boot.bin"
    FAKE_NV_CHIP=gb205 FAKE_TG_RECORD="$W/dev_boot.bin" "$TG_TESTS/run_fake_device.sh" b2_boot -- --state-count 4 --reps 3 > "$W/boot.txt" 2>&1
}
dry_run() {   # <plain|guard>: run_l0.sh's dry run (its own directory), plain or through the guard
    FAKE_NV_CHIP=gb205 L0_DRY_RUN=1 L0_DRY_DIR="$W/l0_dry" "$TG_TESTS/run_l0.sh" b2$1 4 3 --poison $([ $1 = guard ] && echo --guard) > "$W/l0_$1.txt" 2>&1
}
replay_dry() { "$TG_TESTS/run_replay.sh" "$(rec $1)" b2_rp_$1 --record --guard > "$W/replay_$1.txt" 2>&1; }   # <plain|guard>
dry_guard() { dry_run guard; replay_dry guard; }
GDS=("pte-sys-bad|the PTE for VA .* points at device address .*, not a live sysmem page" "rpc-corrupt|NV_PGSP_QUEUE_HEAD.*has a bad checksum")
GB=(); for r in $TG_GB20X; do [ -f "$BEAGLE_TINYGPU_DATA/recordings/$r/events.bin" ] && GB+=("$BEAGLE_TINYGPU_DATA/recordings/$r"); done

pjob_init "$W"; export TG_PIDS="$W/pids"   # every fake, proxy, replay server and guard the runs start (tracked_alive)
pjob dry_plain dry_run plain; DRY_PLAIN=$!   # its recording's guard-defect replays wait out a 10 s RPC timeout: the critical path
pjob dry_guard dry_guard; DRY_GUARD=$!
wait $DRY_PLAIN
for m in "${GDS[@]}"; do d=${m%%|*}; pjob gd_$d to "$W/replay_gd_$d.txt" "$TG_TESTS/run_replay.sh" "$(rec plain)" b2_gd_$d --guard --guard-defect $d; done
if [ ${#GB[@]} -eq 6 ]; then
    for R in "${GB[@]}"; do l=b2_hw_${R##*_}; pjob $l to "$W/replay_$l.txt" "$TG_TESTS/run_replay.sh" "$R" $l --guard; done
fi
pjob rp_plain replay_dry plain
pjob boot boot
if [ ${#GB[@]} -eq 6 ]; then
    for R in "${GB[@]:1}"; do pjob canon_hw_${R##*_} to "$W/canon_hw_${R##*_}.txt" "$BEAGLE_PYTHON" "$TG_TESTS/replay/tgcanon.py" "${GB[0]}" "$R"; done
fi
wait $DRY_GUARD
pjob canon_guard to "$W/canon_guard.txt" "$BEAGLE_PYTHON" "$TG_TESTS/replay/tgcanon.py" "$(rec plain)" "$(rec guard)"
pwait

# 1. a run: the C++ boot, the plugin's memory manager on MMU v3, QMD v5 and the COT teardown
r=$(pjob_rc boot)
COT_TD="C++ teardown: the GSP unload and the RISC-V halt wait (COT) run here at exit"
check "fake GB205: level boot PASS (sm_120, QMD v5), the FSP booting the FMC once" \
    "[ $r -eq 0 ] && grep -q 'built the NVDevice after the C++ boot, with no daemon (level boot: sm_120, QMD v5' '$(out boot)' \
     && grep -q '\"FMC boots\": 1' '$TINYGPU_TEST_WORK/fake_device_b2_boot.log'"
check "the plugin's MMU v3 memory manager allocates the pool and the buffers, and its COT teardown halts the RISC-V core" \
    "grep -q 'C++ memory manager: buffers and pool, VRAM pool 6114 MiB' '$(out boot)' && grep -q '$COT_TD' '$(out boot)' \
     && grep -q 'teardown: done: GSP RISC-V halted after' '$(out boot)'"

# 2. run_l0.sh's dry runs, plain and through the guard (as a new client's first hardware run uses it): each recording replays
#    exactly under the guard, and the two are equivalent
for m in plain guard; do
    check "run_l0.sh dry run$([ $m = guard ] && echo ' through the guard'): PASS, recorded" "tail -1 '$W/l0_$m.txt' | grep -q '^OK: PASS, recorded'"
    check "the $m recording replays exactly, markers equal, under the guard (MMU v3, the COT start and teardown)" \
        "replay_line b2_rp_$m | grep -q 'PASS' && replay_line b2_rp_$m | grep -q '\"markers\": \"equal\"' && replay_line b2_rp_$m | grep -q '\"launches\": [1-9]'"
done
check "tgcanon: the recording through the guard is equivalent to the plain one" "[ $(pjob_rc canon_guard) -eq 0 ]"
for m in "${GDS[@]}"; do
    IFS='|' read -r d rx <<< "$m"
    check "the guard refuses the $d defect on MMU v3 (in the replay)" "replay_line b2_gd_$d | grep -qE 'FAIL: the guard refused .*$rx'"
done

# 3. the GB205's own recordings, where $BEAGLE_TINYGPU_DATA has them: H1's to H4's are equivalent to the L0 (the COT boot's GSP
#    log buffer masked), and each replays exactly to the C++ boot and the plugin's COT teardown under the guard: on the card's
#    own replies, the boot and the NVDevice make the requests tinygrad's made, the COT message word for word, and the teardown
#    sends what T's recorded (not marker for marker: the recordings hold the daemon's markers)
if [ ${#GB[@]} -eq 6 ]; then
    for R in "${GB[@]:1}"; do
        check "tgcanon: GB205 ${R##*/} is equivalent to the L0 (the GSP log buffer masked)" \
            "[ $(pjob_rc canon_hw_${R##*_}) -eq 0 ] && grep -q 'GSP log buffer pages differ' '$W/canon_hw_${R##*_}.txt'"
    done
    for R in "${GB[@]}"; do
        l=b2_hw_${R##*_}
        check "GB205 ${R##*/}: replays exactly at level boot, with the plugin's COT teardown, under the guard" \
            "replay_line $l | grep -q 'PASS' && grep -q 'built the NVDevice after the C++ boot, with no daemon (level boot: sm_120' '$TINYGPU_TEST_WORK/run_replay_$l.txt' \
             && grep -q '$COT_TD' '$TINYGPU_TEST_WORK/run_replay_$l.txt' && grep -q 'teardown: done: GSP RISC-V halted after' '$TINYGPU_TEST_WORK/run_replay_$l.txt'"
    done
else
    echo "(no GB205 recordings in $BEAGLE_TINYGPU_DATA/recordings: their checks skipped)"
fi

left=$(tracked_alive)   # this stage's own fakes, proxies, replay servers and guards only: other stages run at the same time
check "no harness process or guard is left running" "[ -z \"\$left\" ]"
echo; [ $fails -eq 0 ] && echo "test_b2: PASS" || echo "test_b2: $fails FAILED"
[ $fails -eq 0 ]
