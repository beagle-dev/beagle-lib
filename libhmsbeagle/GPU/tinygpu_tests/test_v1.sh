#!/bin/bash
# TODO.md plan step V1, offline: the recording proxy, the replay server, the guard and the comparator, checked end to end with
# no eGPU (fake_nv_device.py plays the AD107 at the register level, which the plugin boots itself, plan steps C11-C13; TinyGPU's
# real server.c runs on an IOKit stub), run_l0.sh itself in its dry run, and the L0 hardware recordings where they are present
# (compared here; test_c11.sh replays them). One PASS or FAIL line per check; exit 0 only if all pass.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
require_no_launch_guard
W="$TINYGPU_TEST_WORK/v1"; rm -rf "$W"; mkdir -p "$W"
P="$BEAGLE_PYTHON"; fails=0
pass() { echo "PASS $1"; }
fail() { echo "FAIL $1"; fails=$((fails + 1)); }
check() { if eval "$2"; then pass "$1"; else fail "$1"; fi; }
replay_line() { grep -E "^replay session 2" "$TINYGPU_TEST_WORK/replay_$1.log"; }

# 1. the proxy against TinyGPU's real server.c (the protocol, refusals, markers, diffs, snapshots, fail-stop)
"$P" "$TG_TESTS/replay/test_v1_proxy.py" > "$W/proxy.log" 2>&1
check "proxy vs TinyGPU's server.c (test_v1_proxy.py)" "tail -1 '$W/proxy.log' | grep -q 'test_v1_proxy: PASS'"

# 2. the plugin's boot, run and teardown on the fake AD107: plain, then through the proxy with the plugin's markers, which must
#    leave the device's stream unchanged
FAKE_TG_RECORD="$W/dev_plain.bin" "$TG_TESTS/run_fake_device.sh" v1_plain -- --state-count 4 --reps 3 > "$W/dev_plain.txt" 2>&1; r1=$?
FAKE_TG_RECORD="$W/dev_rec.bin" FAKE_TG_PROXY="$W/rec" "$TG_TESTS/run_fake_device.sh" v1_rec BEAGLE_TG_MARKERS=1 -- --state-count 4 --reps 3 \
    > "$W/dev_rec.txt" 2>&1; r2=$?
check "the plugin's boot, run and teardown on the fake AD107: PASS plain and recorded" "[ $r1 -eq 0 ] && [ $r2 -eq 0 ]"
check "the proxy and the plugin's markers leave the device's byte stream unchanged" "cmp -s '$W/dev_plain.bin' '$W/dev_rec.bin'"
check "the recording holds the plugin's markers (the boot's end, the programs loaded, fini)" \
    "'$P' -c 'import sys; sys.path.insert(0, \"$TG_TESTS/replay\"); import tgwire as w; ev, _ = w.read(\"$W/rec\"); ids = {e.f[\"id\"] for e in ev if e.kind == w.K_MARKER}; sys.exit(0 if {0x100, 0x101, 0x102} <= ids else 1)'"
"$P" "$TG_TESTS/replay/tgproxy.py" --help > /dev/null   # importable

# 3. self-replay: strict (every request byte for byte, client-written pages at the snapshots and at the end), with the markers
#    (equal, in order), and under the guard
"$TG_TESTS/run_replay.sh" "$W/rec" v1_self > "$W/replay_self.txt" 2>&1
check "self-replay of the recorded boot: PASS" "replay_line v1_self | grep -q 'PASS'"
"$TG_TESTS/run_replay.sh" "$W/rec" v1_self_rec --record --guard > "$W/replay_self_rec.txt" 2>&1
check "self-replay with the markers (all equal) and the guard (no refusal): PASS" \
    "replay_line v1_self_rec | grep -q 'PASS' && replay_line v1_self_rec | grep -q '\"markers\": \"equal\"'"

# 4. differential replay: a mutated recording makes the plugin fail the same way twice, on the fault the mutation made
for m in "diff-late|Timeout waiting for RPC response for command 103" "rpc-fail|RPC call 103 failed with result 31"; do
    IFS='|' read -r m why <<< "$m"
    "$TG_TESTS/run_replay.sh" "$W/rec" v1_rm_${m}_1 --mutate $m > /dev/null 2>&1
    "$TG_TESTS/run_replay.sh" "$W/rec" v1_rm_${m}_2 --mutate $m > /dev/null 2>&1
    a=$(replay_line v1_rm_${m}_1 | sed -E 's/ \{.*//'); b=$(replay_line v1_rm_${m}_2 | sed -E 's/ \{.*//')
    check "the $m recording mutation fails the plugin ($why), identically twice ($a)" \
        "[ -n \"$a\" ] && [ \"$a\" = \"$b\" ] && echo \"$a\" | grep -q FAIL && grep -q 'level boot: building the NVDevice: RuntimeError: $why' '$TINYGPU_TEST_WORK/run_replay_v1_rm_${m}_1.txt'"
done
# ... and the GSP's replies ahead of the requests that asked for them, as a C++ runtime's recording can hold them (STATUS.md
#     R41): the plugin must pass all the same
"$TG_TESTS/run_replay.sh" "$W/rec" v1_diff_early --mutate diff-early > "$W/replay_diff_early.txt" 2>&1
check "the diff-early recording mutation (a channel's replies before its rm_alloc reaches the queue head) replays exactly" \
    "replay_line v1_diff_early | grep -q 'PASS'"

# 5. the guard, live behind the proxy: a clean boot passes (its refusals were checked on BEAGLE_TG_MUTATE's defects, which were
#    in the daemon's Python: plan step C13c removed them with it)
FAKE_TG_GUARD=1 FAKE_TG_PROXY="$W/guard_clean" "$TG_TESTS/run_fake_device.sh" v1_guard -- --state-count 4 --reps 3 > "$W/guard_clean.txt" 2>&1
check "the guard passes a clean boot and its teardown" "tail -1 '$W/guard_clean.txt' | grep -q PASS"

# 6. the comparator: a recording is equivalent to itself and to the same boot recorded without the markers (markers aside), and
#    not to a run at another state count
FAKE_TG_PROXY="$W/rec_plain" "$TG_TESTS/run_fake_device.sh" v1_rec_plain -- --state-count 4 --reps 3 > /dev/null 2>&1
FAKE_TG_PROXY="$W/rec_64" "$TG_TESTS/run_fake_device.sh" v1_rec_64 -- --state-count 64 --reps 3 > /dev/null 2>&1
"$P" "$TG_TESTS/replay/tgcanon.py" "$W/rec" "$W/rec" > "$W/canon_self.txt" 2>&1; c1=$?
"$P" "$TG_TESTS/replay/tgcanon.py" "$W/rec_plain" "$W/rec" > "$W/canon_markers.txt" 2>&1; c2=$?
"$P" "$TG_TESTS/replay/tgcanon.py" "$W/rec_plain" "$W/rec_64" > "$W/canon_64.txt" 2>&1; c3=$?
check "tgcanon: a recording is equivalent to itself" "[ $c1 -eq 0 ]"
check "tgcanon: with and without the markers, equivalent (markers aside)" "[ $c2 -eq 0 ] && grep -q 'markers differ' '$W/canon_markers.txt'"
check "tgcanon: a run at 64 states is not equivalent to one at 4" "[ $c3 -ne 0 ] && grep -q '^session 2: DIFFERENT' '$W/canon_64.txt'"

# 7. run_l0.sh itself, in its dry run (the fake AD107 in place of TinyGPU.app), and the offline replay of what it recorded
rm -rf "$TINYGPU_TEST_WORK/l0_dry"
L0_DRY_RUN=1 "$TG_TESTS/run_l0.sh" v1dry 4 3 > "$W/l0_dry.txt" 2>&1
R=$(ls -d "$TINYGPU_TEST_WORK"/l0_dry/recordings/*_v1dry 2>/dev/null | head -1)
check "run_l0.sh dry run: PASS, the recording with its run.json and test output" \
    "tail -1 '$W/l0_dry.txt' | grep -q '^OK: PASS, recorded' && [ -f '$R/run.json' ] && [ -f '$R/test_output.txt' ]"
"$TG_TESTS/run_replay.sh" "$R" v1_l0dry --record --guard > "$W/replay_l0dry.txt" 2>&1
check "the dry run's recording replays exactly, markers equal, under the guard" \
    "replay_line v1_l0dry | grep -q 'PASS' && replay_line v1_l0dry | grep -q '\"markers\": \"equal\"'"

# 8. the L0 recordings, where $BEAGLE_TINYGPU_DATA has them: the three are one boot (through the program load: the tests differ
#    after it) and one teardown under tgcanon's masks (test_c11.sh replays each to the plugin)
L0=(); for r in $TG_L0; do [ -f "$BEAGLE_TINYGPU_DATA/recordings/$r/events.bin" ] && L0+=("$BEAGLE_TINYGPU_DATA/recordings/$r"); done
if [ ${#L0[@]} -eq 3 ]; then
    for p in "0 1" "1 2" "0 2"; do
        set -- $p; a=${L0[$1]}; b=${L0[$2]}
        "$P" "$TG_TESTS/replay/tgcanon.py" "$a" "$b" --until-marker 0x101 > "$W/canon_l0_$1$2_boot.txt" 2>&1; c1=$?
        "$P" "$TG_TESTS/replay/tgcanon.py" "$a" "$b" --from-marker 0x102 > "$W/canon_l0_$1$2_fini.txt" 2>&1; c2=$?
        check "tgcanon: L0 ${a##*_} and ${b##*_} are one boot through the program load and one teardown, under the masks" "[ $c1 -eq 0 ] && [ $c2 -eq 0 ]"
    done
else
    echo "(no L0 recordings in $BEAGLE_TINYGPU_DATA/recordings: their checks skipped)"
fi

left=$(ps -axo command | grep -cE "Python .*(tgproxy|tgreplay|fake_nv_device)\.py|[b]eagle-tinygpu-guard")
check "no harness process or guard is left running" "[ $left -eq 0 ]"
echo; [ $fails -eq 0 ] && echo "test_v1: PASS" || echo "test_v1: $fails FAILED"
[ $fails -eq 0 ]
