#!/bin/bash
# TODO.md plan step V1, offline: the recording proxy, the recording shim, the replay server, the guard and the comparator,
# checked end to end with no eGPU (fake_nv_device.py plays the AD107 at the register level, TinyGPU's real server.c runs on an
# IOKit stub), run_l0.sh itself in its dry run, and the L0 hardware recordings where they are present (replayed, compared).
# One PASS or FAIL line per check; exit 0 only if all pass.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
require_no_launch_guard
W="$TINYGPU_TEST_WORK/v1"; rm -rf "$W"; mkdir -p "$W"
P="$BEAGLE_PYTHON"; fails=0
# the fake AD107 runs below boot at level sysmem, where the daemon builds the whole NVDevice: BEAGLE_TG_MUTATE's defects and the
# shim's markers are in its Python (at the default level, gsp_hw, the plugin boots GSP-RM and builds the NVDevice itself)
REF=BEAGLE_NV_CPP_LEVEL=sysmem
pass() { echo "PASS $1"; }
fail() { echo "FAIL $1"; fails=$((fails + 1)); }
check() { if eval "$2"; then pass "$1"; else fail "$1"; fi; }
replay_line() { grep -E "^replay session 2" "$TINYGPU_TEST_WORK/replay_$1.log"; }

# 1. the proxy against TinyGPU's real server.c (the protocol, refusals, markers, diffs, snapshots, fail-stop)
"$P" "$TG_TESTS/replay/test_v1_proxy.py" > "$W/proxy.log" 2>&1
check "proxy vs TinyGPU's server.c (test_v1_proxy.py)" "tail -1 '$W/proxy.log' | grep -q 'test_v1_proxy: PASS'"

# 2. the plugin's fake flows through the proxy: the fake TinyGPU.app receives the same bytes with and without it
for mode in "runtime BEAGLE_NV_USE_DAEMON=0" "dispatch BEAGLE_NV_CPP_DISPATCH=1" "daemon BEAGLE_NV_CPP_DISPATCH=0"; do
    set -- $mode
    FAKE_TG_RECORD="$W/flow_$1_direct.bin" "$TG_TESTS/run_fake_runtime.sh" v1_$1_direct "$2" -- --state-count 4 --reps 5 > "$W/flow_$1_direct.txt" 2>&1; r1=$?
    FAKE_TG_RECORD="$W/flow_$1_proxy.bin" FAKE_TG_PROXY="$W/flow_$1_rec" "$TG_TESTS/run_fake_runtime.sh" v1_$1_proxy "$2" -- --state-count 4 --reps 5 > "$W/flow_$1_proxy.txt" 2>&1; r2=$?
    check "fake flow $1 through the proxy: both PASS, identical bytes at the fake TinyGPU.app" "[ $r1 -eq 0 ] && [ $r2 -eq 0 ] && cmp -s '$W/flow_$1_direct.bin' '$W/flow_$1_proxy.bin'"
done

# 3. tinygrad's real boot, the C++ runtime and the teardown on the fake AD107: plain, then through the proxy with the recording
#    shim and the plugin's markers, which must leave the device's stream unchanged
FAKE_TG_RECORD="$W/dev_plain.bin" "$TG_TESTS/run_fake_device.sh" v1_plain "$REF" -- --state-count 4 --reps 3 > "$W/dev_plain.txt" 2>&1; r1=$?
FAKE_TG_RECORD="$W/dev_rec.bin" FAKE_TG_PROXY="$W/rec" "$TG_TESTS/run_fake_device.sh" v1_rec "$REF" BEAGLE_TG_RECORD=1 BEAGLE_TG_MARKERS=1 \
    BEAGLE_TG_RECORD_LOG="$W/rec_shim.jsonl" -- --state-count 4 --reps 3 > "$W/dev_rec.txt" 2>&1; r2=$?
check "full boot on the fake AD107 (the real daemon, C++ runtime, teardown): PASS plain and recorded" "[ $r1 -eq 0 ] && [ $r2 -eq 0 ]"
check "the shim and markers, behind the proxy, leave the device's byte stream unchanged" "cmp -s '$W/dev_plain.bin' '$W/dev_rec.bin'"
check "the recording holds the shim's and the plugin's markers" \
    "'$P' -c 'import sys; sys.path.insert(0, \"$TG_TESTS/replay\"); import tgwire as w; ev, _ = w.read(\"$W/rec\"); ids = {e.f[\"id\"] for e in ev if e.kind == w.K_MARKER}; sys.exit(0 if {1, 100, 0x100, 0x101, 0x102} <= ids else 1)'"
"$P" "$TG_TESTS/replay/tgproxy.py" --help > /dev/null   # importable

# 4. self-replay: strict (every request byte for byte, client-written pages at the snapshots and at the end), with the shim and
#    markers (equal, in order), and under the guard
"$TG_TESTS/run_replay.sh" "$W/rec" v1_self > "$W/replay_self.txt" 2>&1
check "self-replay of the recorded boot: PASS" "replay_line v1_self | grep -q 'PASS'"
"$TG_TESTS/run_replay.sh" "$W/rec" v1_self_rec --record --guard > "$W/replay_self_rec.txt" 2>&1
check "self-replay with the shim and markers (all equal) and the guard (no refusal): PASS" \
    "replay_line v1_self_rec | grep -q 'PASS' && replay_line v1_self_rec | grep -q '\"markers\": \"equal\"'"

# 5. a replay reports each deliberate defect in the client (tgharness_py MUTATIONS), and the mutated oracle
for m in "pte-bit|MMIO_WRITE BAR1 .*the payload differs" "rpc-field|client-written page" "swap-writes|expected MMIO_WRITE BAR0 GSP.NV_PFALCON2_FALCON_BROM_ENGIDMASK" \
         "drop-zero|expected MMIO_WRITE BAR1 0x0 \+0x1000" "reset-short|an engine reset held" "boot42|got MMIO_READ BAR0 0x168"; do
    IFS='|' read -r mut rx <<< "$m"
    BEAGLE_TG_MUTATE=$mut "$TG_TESTS/run_replay.sh" "$W/rec" v1_mut_$mut > "$W/replay_mut_$mut.txt" 2>&1
    check "replay reports the $mut mutation" "replay_line v1_mut_$mut | grep -qE 'FAIL: .*$rx'"
done

# 6. differential replay: a mutated recording makes the reference fail the same way twice
for m in diff-late rpc-fail; do
    "$TG_TESTS/run_replay.sh" "$W/rec" v1_rm_${m}_1 --mutate $m > /dev/null 2>&1
    "$TG_TESTS/run_replay.sh" "$W/rec" v1_rm_${m}_2 --mutate $m > /dev/null 2>&1
    a=$(replay_line v1_rm_${m}_1 | sed -E 's/ \{.*//'); b=$(replay_line v1_rm_${m}_2 | sed -E 's/ \{.*//')
    check "the $m recording mutation fails the reference, identically twice ($a)" "[ -n \"$a\" ] && [ \"$a\" = \"$b\" ] && echo \"$a\" | grep -q FAIL"
done
# ... and the GSP's replies ahead of the requests that asked for them, as a C++ runtime's recording can hold them (STATUS.md
#     R41): the reference must pass all the same
"$TG_TESTS/run_replay.sh" "$W/rec" v1_diff_early --mutate diff-early > "$W/replay_diff_early.txt" 2>&1
check "the diff-early recording mutation (a channel's replies before its rm_alloc reaches the queue head) replays exactly" \
    "replay_line v1_diff_early | grep -q 'PASS'"

# 7. the guard, live behind the proxy: a clean boot passes, and each defect is refused before it reaches the device
FAKE_TG_GUARD=1 FAKE_TG_PROXY="$W/guard_clean" "$TG_TESTS/run_fake_device.sh" v1_guard "$REF" -- --state-count 4 --reps 3 > "$W/guard_clean.txt" 2>&1
check "the guard passes a clean boot and its teardown" "tail -1 '$W/guard_clean.txt' | grep -q PASS"
for m in "pte-sys-bad|the PTE for VA .* points at device address" "mailbox-bad|SEC2's mailboxes point at" "rpc-corrupt|bad checksum"; do
    IFS='|' read -r mut rx <<< "$m"
    BEAGLE_TG_MUTATE=$mut FAKE_TG_GUARD=1 FAKE_EXPECT_TRIP="$rx" FAKE_TG_PROXY="$W/guard_$mut" "$TG_TESTS/run_fake_device.sh" v1_guard_$mut "$REF" -- --state-count 4 --reps 3 > "$W/guard_$mut.txt" 2>&1
    check "the guard refuses the $mut defect" "tail -1 '$W/guard_$mut.txt' | grep -q 'PASS (the expected refusal)'"
done

# 8. the comparator: a recording is equivalent to itself and to the same boot recorded without the shim (markers aside), and not to
#    a boot whose client sent a different RPC field
FAKE_TG_PROXY="$W/rec_plain" "$TG_TESTS/run_fake_device.sh" v1_rec_plain "$REF" -- --state-count 4 --reps 3 > /dev/null 2>&1
BEAGLE_TG_MUTATE=rpc-field FAKE_TG_PROXY="$W/rec_rpc" "$TG_TESTS/run_fake_device.sh" v1_rec_rpc "$REF" -- --state-count 4 --reps 3 > /dev/null 2>&1
"$P" "$TG_TESTS/replay/tgcanon.py" "$W/rec" "$W/rec" > "$W/canon_self.txt" 2>&1; c1=$?
"$P" "$TG_TESTS/replay/tgcanon.py" "$W/rec_plain" "$W/rec" > "$W/canon_shim.txt" 2>&1; c2=$?
"$P" "$TG_TESTS/replay/tgcanon.py" "$W/rec_plain" "$W/rec_rpc" > "$W/canon_rpc.txt" 2>&1; c3=$?
check "tgcanon: a recording is equivalent to itself" "[ $c1 -eq 0 ]"
check "tgcanon: with and without the shim and markers, equivalent (markers aside)" "[ $c2 -eq 0 ] && grep -q 'markers differ' '$W/canon_shim.txt'"
check "tgcanon: a changed RPC field is a different GSP reply and different queue pages" "[ $c3 -ne 0 ] && grep -q 'GSP replies differ' '$W/canon_rpc.txt' && grep -q 'sysmem pages differ' '$W/canon_rpc.txt'"

# 9. run_l0.sh itself, in its dry run (the fake AD107 in place of TinyGPU.app), and the offline replay of what it recorded; at
#    level gsp_hw, where the daemon runs the shim (at the default level, boot, there is no daemon: plan step C12)
rm -rf "$TINYGPU_TEST_WORK/l0_dry"
L0_DRY_RUN=1 "$TG_TESTS/run_l0.sh" v1dry 4 3 --level gsp_hw > "$W/l0_dry.txt" 2>&1
R=$(ls -d "$TINYGPU_TEST_WORK"/l0_dry/recordings/*_v1dry 2>/dev/null | head -1)
check "run_l0.sh dry run: PASS, the recording with its run.json, shim log and test output" \
    "tail -1 '$W/l0_dry.txt' | grep -q '^OK: PASS, recorded' && [ -f '$R/run.json' ] && [ -f '$R/shim.jsonl' ] && [ -f '$R/test_output.txt' ]"
"$TG_TESTS/run_replay.sh" "$R" v1_l0dry --record --guard > "$W/replay_l0dry.txt" 2>&1
check "the dry run's recording replays exactly, markers equal, under the guard" \
    "replay_line v1_l0dry | grep -q 'PASS' && replay_line v1_l0dry | grep -q '\"markers\": \"equal\"'"

# 10. the L0 recordings, where $BEAGLE_TINYGPU_DATA has them: each replays exactly, markers equal, under the guard, and the cold
#     one also to a client that sends no markers; the guard refuses a corrupt RPC on real data; the three are one boot (through
#     the program load: the tests differ after it) and one teardown under tgcanon's masks
L0=(); for r in $TG_L0; do [ -f "$BEAGLE_TINYGPU_DATA/recordings/$r/events.bin" ] && L0+=("$BEAGLE_TINYGPU_DATA/recordings/$r"); done
if [ ${#L0[@]} -eq 3 ]; then
    for R in "${L0[@]}"; do
        l=l0_${R##*_}
        "$TG_TESTS/run_replay.sh" "$R" $l --record --guard > "$W/replay_$l.txt" 2>&1
        check "L0 ${R##*/}: replays exactly, markers equal, under the guard" "replay_line $l | grep -q 'PASS' && replay_line $l | grep -q '\"markers\": \"equal\"'"
    done
    "$TG_TESTS/run_replay.sh" "${L0[0]}" l0_cold_nomarkers --guard > "$W/replay_l0_cold_nomarkers.txt" 2>&1
    check "L0 cold: replays exactly to a client that sends no markers (as a port's), under the guard" "replay_line l0_cold_nomarkers | grep -q 'PASS'"
    BEAGLE_TG_MUTATE=rpc-corrupt "$TG_TESTS/run_replay.sh" "${L0[0]}" l0_rpc_corrupt --guard > "$W/replay_l0_rpc_corrupt.txt" 2>&1
    check "the guard refuses the rpc-corrupt defect in the L0 cold boot's replay" \
        "replay_line l0_rpc_corrupt | grep -q 'FAIL: the guard refused NV_PGSP_QUEUE_HEAD.*bad checksum'"
    for p in "0 1" "1 2" "0 2"; do
        set -- $p; a=${L0[$1]}; b=${L0[$2]}
        "$P" "$TG_TESTS/replay/tgcanon.py" "$a" "$b" --until-marker 0x101 > "$W/canon_l0_$1$2_boot.txt" 2>&1; c1=$?
        "$P" "$TG_TESTS/replay/tgcanon.py" "$a" "$b" --from-marker 0x102 > "$W/canon_l0_$1$2_fini.txt" 2>&1; c2=$?
        check "tgcanon: L0 ${a##*_} and ${b##*_} are one boot through the program load and one teardown, under the masks" "[ $c1 -eq 0 ] && [ $c2 -eq 0 ]"
    done
else
    echo "(no L0 recordings in $BEAGLE_TINYGPU_DATA/recordings: their checks skipped)"
fi

left=$(ps -axo command | grep -cE "Python .*(tgdaemon|tgproxy|tgreplay|fake_nv_device)\.py")
check "no harness process is left running" "[ $left -eq 0 ]"
echo; [ $fails -eq 0 ] && echo "test_v1: PASS" || echo "test_v1: $fails FAILED"
[ $fails -eq 0 ]
