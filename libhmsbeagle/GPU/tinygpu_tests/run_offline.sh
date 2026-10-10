#!/bin/bash
# Everything that can be checked without the eGPU: the firmware staging, the goldens (each C++ port against the tinygrad code it
# follows, in tinygpu_tests/oracle), the C++ TinyGPU.app client, then the plugin end to end on fake_nv_device.py, which it boots
# itself as it boots the eGPU (plan steps C11-C13): plan step V1's record/replay tools and the hardware recordings, the fake GB205,
# the crash guard, the library's error returns, the uploads, D1's kernels, several instances in one process, the routing, the
# failures and the kills; the AMD runtime, boot, V1 tools and crash guard on fake_amd_device.py; then the no-launch guard
# (nothing listening => the plugin errors out and no TinyGPU.app is spawned). Build hmsbeagle-tinygpu, beagle-tinygpu-guard, tinygputest, synthetictest and hmctest first.
#
# The stages run at the same time, each in its own work directory ($TINYGPU_TEST_WORK/s/<key>) and TMPDIR, sharing one build
# cache (TINYGPU_TEST_CACHE); their runs share this computer's TINYGPU_TEST_JOBS slots (env.sh pjob), so together they keep
# about that many runs going. Each stage's output is $TINYGPU_TEST_WORK/<its log>, as before; the summary keeps the old order,
# and ends with each stage's wall time. The suite holds the hardware lock while it runs, so no hardware script starts beside
# it (C3's and V1's stub servers use TinyGPU.app's shm names, which C3 also unlinks).
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
require_no_launch_guard   # static check before anything below could reach a spawn path
unset FAKE_TEST_BIN       # every run below is tinygputest's unless it names another binary itself
BASE=$TINYGPU_TEST_WORK
# one suite at a time on this computer (a stage rerun by hand beside it would share its slots, not its files), and never
# beside a hardware script: the suite takes the hardware scripts' lock (hw_begin's, in the real TMPDIR) for as long as it runs,
# so none starts beside it either. The stages, which get private TMPDIRs, look for that lock in BEAGLE_HW_TMPDIR: an empty
# directory here, since the suite holds it.
LOCK="$BASE/.suite.lock"
mkdir "$LOCK" 2>/dev/null || { echo "another offline suite holds $LOCK (remove it if none is running); not running"; exit 2; }
HW_LOCK=${TMPDIR:-/tmp}; HW_LOCK="${HW_LOCK%/}/beagle_tinygpu_hw.lock"
mkdir "$HW_LOCK" 2>/dev/null || { rmdir "$LOCK"; echo "a hardware script holds $HW_LOCK; not running"; exit 2; }
echo "run_offline.sh, pid $$" > "$HW_LOCK/owner"
trap 'rm -rf "$HW_LOCK" "$LOCK"' EXIT
SDIR="$BASE/.stages"
rm -rf "$SDIR"; mkdir -p "$SDIR/hw" "$BASE/s"
export TINYGPU_TEST_CACHE=${TINYGPU_TEST_CACHE:-$BASE} BEAGLE_HW_TMPDIR="$SDIR/hw"
HARNESS_RE='Python .*(tgproxy|tgreplay|fake_nv_device|fake_amd_device)\.py|(^|/)beagle-tinygpu-guard$'
harness_pids() { ps -axo pid=,command= | grep -E "$HARNESS_RE" | grep -v grep | awk '{print $1}' | sort; }
before_pids=$(harness_pids)
T0=$SECONDS
results=()
# first, the firmware the C++ boots below read from BEAGLE's cache (downloads are off in every test)
"$BEAGLE_PYTHON" "$TG_TESTS/check_firmware.py" > "$BASE/check_firmware.log" 2>&1
rc=$?; cat "$BASE/check_firmware.log"
results+=("firmware staging: $([ $rc -eq 0 ] && echo PASS || echo "FAIL (see $BASE/check_firmware.log)")")

# run_stage <key> <log> <command> [args ...]: one stage, in the foreground, in its own work directory and TMPDIR; its exit
# status and wall time go to $SDIR/<key>. Keys stay short: socket paths under the work directory must fit in 104 bytes. A
# stage's work directory starts empty, but the goldens' (their perturbed builds' cache keys live next to the builds).
run_stage() {
    local key=$1 log=$2; shift 2
    (
        export TINYGPU_TEST_WORK="$BASE/s/$key"
        export TMPDIR; TMPDIR=$(mktemp -d /tmp/tgs.XXXXXX)
        [ "$key" = gold ] || rm -rf "$TINYGPU_TEST_WORK"; mkdir -p "$TINYGPU_TEST_WORK"
        s=$SECONDS
        "$@" > "$BASE/$log" 2>&1
        echo "$? $((SECONDS - s))" > "$SDIR/$key"
        rm -rf "$TMPDIR"
    )
}
# slot_stage: run_stage in the background once one of the slots is free, for a stage that is one Python process (the shell
# stages take a slot for each of their runs instead)
slot_stage() { ( slot_take; trap 'rm -f "$SLOT"' EXIT; run_stage "$@" ) & }
# the stages, longest first; the AMD ones on each card's fake (TODO.md plan step N13: FAKE_AMD_CHIP, the RX 7900 XT's gfx1100
# and the RX 9070 XT's gfx1201)
mkdir -p "$TINYGPU_TEST_SLOTS"
run_stage gold run_goldens.log "$TG_TESTS/run_goldens.sh" &
for chip in gfx1100 gfx1201; do
    c=${chip#gfx}; c=${c%0?}   # 11 or 12
    run_stage a1h-$c test_a1h_$chip.log env FAKE_AMD_CHIP=$chip "$TG_TESTS/test_a1h.sh" &
done
run_stage c13 test_c13_e2e.log "$TG_TESTS/test_c13.sh" &
run_stage c12 test_c12_e2e.log "$TG_TESTS/test_c12.sh" &
run_stage v1 test_v1.log "$TG_TESTS/test_v1.sh" &
run_stage b2 test_b2_e2e.log "$TG_TESTS/test_b2.sh" &
run_stage c3 test_c3_transport.log "$TG_TESTS/test_c3_transport.sh" &   # its stub waits for test_v1_proxy.py's (shm lock)
run_stage c11 test_c11_e2e.log "$TG_TESTS/test_c11.sh" &
run_stage n6 test_n6.log "$TG_TESTS/test_n6.sh" &
for chip in gfx1100 gfx1201; do
    c=${chip#gfx}; c=${c%0?}
    for t in a2k a3 a5 a2 a7 a4; do run_stage $t-$c test_${t}_$chip.log env FAKE_AMD_CHIP=$chip "$TG_TESTS/test_$t.sh" & done
    slot_stage a2i-$c test_a2i_$chip.log env FAKE_AMD_CHIP=$chip "$BEAGLE_PYTHON" "$TG_TESTS/test_a2i.py"
done
run_stage c16 test_c16.log "$TG_TESTS/test_c16.sh" &
run_stage c14 test_c14.log "$TG_TESTS/test_c14.sh" &
run_stage n9 test_n9.log "$BEAGLE_PYTHON" "$TG_TESTS/test_n9.py" $TG_AMD_L0_RDNA4 &   # its parts take slots
run_stage p4 test_p4.log "$TG_TESTS/test_p4.sh" &
run_stage n3 test_n3.log "$TG_TESTS/test_n3.sh" &
wait

# the results, in the order they always had
result() {   # <key> <text> <log>
    local rc=255 t=""
    { read -r rc t < "$SDIR/$1"; } 2>/dev/null
    results+=("$2: $([ "$rc" = 0 ] && echo PASS || echo "FAIL (see $BASE/$3)")")
    times+=("${t:-?} $2")
}
times=()
cat "$BASE/run_goldens.log"
result gold "goldens" run_goldens.log
result c3 "transport (C3)" test_c3_transport.log
result v1 "record/replay (V1)" test_v1.log
result b2 "GB205 and V1 tools (B2)" test_b2_e2e.log
result c11 "full C++ boot (C11)" test_c11_e2e.log
result c12 "default boot and error returns (C12)" test_c12_e2e.log
result c13 "fake-daemon checks on the C++ boot (C13)" test_c13_e2e.log
for chip in gfx1100 gfx1201; do
    c=${chip#gfx}; c=${c%0?}
    result a1h-$c "AMD C++ runtime end to end (A1h) on $chip" test_a1h_$chip.log
    result a2-$c "AMD C++ boot end to end (A2h, A2j) on $chip" test_a2_$chip.log
    result a2i-$c "AMD V1 tools: guard, record, replay (A2i) on $chip" test_a2i_$chip.log
    result a2k-$c "AMD crash guard (A2k) on $chip" test_a2k_$chip.log
    result a3-$c "AMD error returns (A3) on $chip" test_a3_$chip.log
    result a4-$c "D1 lines on the AMD card (A4) on $chip" test_a4_$chip.log
    result a5-$c "one AMD boot per process (A5) on $chip" test_a5_$chip.log
    result a7-$c "double precision on the AMD card (A7) on $chip" test_a7_$chip.log
done
result c16 "double precision on NV (C16)" test_c16.log
result c14 "FreeMemory on both pools (C14)" test_c14.log
result p4 "warm-GPU recovery at boot (P4)" test_p4.log
result n3 "AMD discovery capture, no boot (N3)" test_n3.log
result n6 "AMD boot prediction and safety tools on gfx11 and gfx12 (N6)" test_n6.log
result n9 "the oracle's rehearsal on the RDNA 4 fake (N9)" test_n9.log

# every stage checked its own processes; across the suite, nothing it started is left (a process that ran before it is not
# counted)
left=$(comm -13 <(echo "$before_pids") <(harness_pids) | tr '\n' ' ')
if [ -z "${left// /}" ]; then results+=("no harness process or guard left running (all stages): PASS")
else results+=("no harness process or guard left running (all stages): FAIL (pids $left: $(ps -o command= -p ${left// /,} | cut -c1-80 | tr '\n' ';'))"); fi

# no-launch guard: point the plugin at a socket nobody listens on
SOCKDIR=$(mktemp -d "${TMPDIR:-/tmp}/tg.XXXXXX")
before=$(pgrep -f "TinyGPU.app/Contents/MacOS/TinyGPU server" | wc -l)
env BEAGLE_TINYGPU_NO_LAUNCH=1 APL_REMOTE_SOCK="$SOCKDIR/none.sock" DYLD_LIBRARY_PATH="$TEST_LIBS" \
    "$TEST_BIN" --reps 1 > "$BASE/nolaunch.txt" 2>&1
sleep 0.5
after=$(pgrep -f "TinyGPU.app/Contents/MacOS/TinyGPU server" | wc -l)
rmdir "$SOCKDIR"
if grep -q "BEAGLE_TINYGPU_NO_LAUNCH is set; not starting TinyGPU.app" "$BASE/nolaunch.txt" && [ "$after" -le "$before" ]; then
    results+=("no-launch guard: PASS")
else
    results+=("no-launch guard: FAIL (see $BASE/nolaunch.txt)")
fi

echo; echo "=== summary"; printf '%s\n' "${results[@]}"
echo; echo "=== wall time per stage (s), $((SECONDS - T0)) s in all with $TINYGPU_TEST_JOBS slots"
printf '%s\n' "${times[@]}" | sort -rn | head -40
! printf '%s\n' "${results[@]}" | grep -q FAIL
