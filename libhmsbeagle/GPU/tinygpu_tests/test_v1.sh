#!/bin/bash
# TODO.md plan step V1, offline: the recording proxy, the replay server, the guard and the comparator, checked end to end with
# no eGPU (fake_nv_device.py plays the AD107 at the register level, which the plugin boots itself, plan steps C11-C13; TinyGPU's
# real server.c runs on an IOKit stub), run_l0.sh itself in its dry run, and the L0 hardware recordings where they are present
# (compared here; test_c11.sh replays them). Every run is a job of its own (env.sh's pjob), started longest first, a run that
# needs another's recording once that one has ended; the checks then read the runs' files in this order.
# One PASS or FAIL line per check; exit 0 only if all pass.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
require_no_launch_guard
W="$TINYGPU_TEST_WORK/v1"; rm -rf "$W"; mkdir -p "$W"
P="$BEAGLE_PYTHON"; fails=0
pass() { echo "PASS $1"; }
fail() { echo "FAIL $1"; fails=$((fails + 1)); }
check() { if eval "$2"; then pass "$1"; else fail "$1"; fi; }
replay_line() { grep -E "^replay session 2" "$TINYGPU_TEST_WORK/replay_$1.log"; }
to() { local f=$1; shift; "$@" > "$f" 2>&1; }   # <output file> <command ...>: a job's command, its output to its own file
dry_rec() { ls -d "$W"/l0_dry/recordings/*_v1dry 2>/dev/null | head -1; }

# the runs (the fakes' settings come with each command, for that fake alone; the fake appends to FAKE_TG_RECORD's file)
dev_run() {   # <plain|rec>: the plugin's boot, run and teardown on the fake AD107, plain or through the proxy with the markers
    rm -f "$W/dev_$1.bin"
    if [ $1 = plain ]; then FAKE_TG_RECORD="$W/dev_plain.bin" "$TG_TESTS/run_fake_device.sh" v1_plain -- --state-count 4 --reps 3 > "$W/dev_plain.txt" 2>&1
    else FAKE_TG_RECORD="$W/dev_rec.bin" FAKE_TG_PROXY="$W/rec" "$TG_TESTS/run_fake_device.sh" v1_rec BEAGLE_TG_MARKERS=1 -- --state-count 4 --reps 3 \
        > "$W/dev_rec.txt" 2>&1; fi
}
proxy_test() {   # the proxy against TinyGPU's real server.c; and tgproxy is importable
    "$P" "$TG_TESTS/replay/tgproxy.py" --help > /dev/null
    "$P" "$TG_TESTS/replay/test_v1_proxy.py" > "$W/proxy.log" 2>&1
}
markers() {   # the recording holds the plugin's markers
    "$P" -c 'import sys; sys.path.insert(0, sys.argv[1]); import tgwire as w; ev, _ = w.read(sys.argv[2]); ids = {e.f["id"] for e in ev if e.kind == w.K_MARKER}; sys.exit(0 if {0x100, 0x101, 0x102} <= ids else 1)' \
        "$TG_TESTS/replay" "$W/rec"
}
dry_run() {   # run_l0.sh in its dry run (its own directory), then the offline replay of what it recorded
    L0_DRY_RUN=1 L0_DRY_DIR="$W/l0_dry" "$TG_TESTS/run_l0.sh" v1dry 4 3 > "$W/l0_dry.txt" 2>&1
    "$TG_TESTS/run_replay.sh" "$(dry_rec)" v1_l0dry --record --guard > "$W/replay_l0dry.txt" 2>&1
}
MUT=("diff-late|Timeout waiting for RPC response for command 103" "rpc-fail|RPC call 103 failed with result 31")
GD=("pte-sys-bad|the PTE for VA .* points at device address .*, not a live sysmem page" "mailbox-bad|SEC2.s mailboxes point at 0x1000"
    "rpc-corrupt|NV_PGSP_QUEUE_HEAD.*has a bad checksum")
L0=(); for r in $TG_L0; do [ -f "$BEAGLE_TINYGPU_DATA/recordings/$r/events.bin" ] && L0+=("$BEAGLE_TINYGPU_DATA/recordings/$r"); done

pjob_init "$W"; export TG_PIDS="$W/pids"   # every fake, proxy, replay server and guard the runs start (tracked_alive)
pjob dev_rec dev_run rec; wait $!   # the recording the replays below replay, the longest of which are the critical path
# the mutated replays wait out the plugin's 10 s RPC timeouts (diff-late twice, the others once); each mutation twice, at once
for m in "${MUT[@]}"; do
    for n in 1 2; do pjob rm_${m%%|*}_$n to /dev/null "$TG_TESTS/run_replay.sh" "$W/rec" v1_rm_${m%%|*}_$n --mutate ${m%%|*}; done
    [ ${m%%|*} = diff-late ] && pjob dry dry_run
done
for m in "${GD[@]}"; do d=${m%%|*}; pjob gd_$d to "$W/replay_gd_$d.txt" "$TG_TESTS/run_replay.sh" "$W/rec" v1_gd_$d --guard --guard-defect $d; done
pjob proxy proxy_test
pjob guard_clean to "$W/guard_clean.txt" env FAKE_TG_GUARD=1 FAKE_TG_PROXY="$W/guard_clean" "$TG_TESTS/run_fake_device.sh" v1_guard -- --state-count 4 --reps 3
pjob rec_plain to /dev/null env FAKE_TG_PROXY="$W/rec_plain" "$TG_TESTS/run_fake_device.sh" v1_rec_plain -- --state-count 4 --reps 3; REC_PLAIN=$!
pjob rec_64 to /dev/null env FAKE_TG_PROXY="$W/rec_64" "$TG_TESTS/run_fake_device.sh" v1_rec_64 -- --state-count 64 --reps 3; REC_64=$!
pjob self_rec to "$W/replay_self_rec.txt" "$TG_TESTS/run_replay.sh" "$W/rec" v1_self_rec --record --guard
pjob dev_plain dev_run plain
pjob self to "$W/replay_self.txt" "$TG_TESTS/run_replay.sh" "$W/rec" v1_self
pjob diff_early to "$W/replay_diff_early.txt" "$TG_TESTS/run_replay.sh" "$W/rec" v1_diff_early --mutate diff-early
pjob markers markers
if [ ${#L0[@]} -eq 3 ]; then
    pjob l0_gd_rpc to "$W/replay_l0_gd_rpc.txt" "$TG_TESTS/run_replay.sh" "${L0[0]}" l0_gd_rpc --guard --guard-defect rpc-corrupt
    for p in "0 1" "1 2" "0 2"; do
        set -- $p
        pjob canon_l0_$1$2_boot to "$W/canon_l0_$1$2_boot.txt" "$P" "$TG_TESTS/replay/tgcanon.py" "${L0[$1]}" "${L0[$2]}" --until-marker 0x101
        pjob canon_l0_$1$2_fini to "$W/canon_l0_$1$2_fini.txt" "$P" "$TG_TESTS/replay/tgcanon.py" "${L0[$1]}" "${L0[$2]}" --from-marker 0x102
    done
fi
wait $REC_PLAIN $REC_64
pjob canon_self to "$W/canon_self.txt" "$P" "$TG_TESTS/replay/tgcanon.py" "$W/rec" "$W/rec"
pjob canon_markers to "$W/canon_markers.txt" "$P" "$TG_TESTS/replay/tgcanon.py" "$W/rec_plain" "$W/rec"
pjob canon_64 to "$W/canon_64.txt" "$P" "$TG_TESTS/replay/tgcanon.py" "$W/rec_plain" "$W/rec_64"
pwait

# 1. the proxy against TinyGPU's real server.c (the protocol, refusals, markers, diffs, snapshots, fail-stop)
check "proxy vs TinyGPU's server.c (test_v1_proxy.py)" "tail -1 '$W/proxy.log' | grep -q 'test_v1_proxy: PASS'"

# 2. the plugin's boot, run and teardown on the fake AD107: plain, then through the proxy with the plugin's markers, which must
#    leave the device's stream unchanged
r1=$(pjob_rc dev_plain); r2=$(pjob_rc dev_rec)
check "the plugin's boot, run and teardown on the fake AD107: PASS plain and recorded" "[ $r1 -eq 0 ] && [ $r2 -eq 0 ]"
check "the proxy and the plugin's markers leave the device's byte stream unchanged" "cmp -s '$W/dev_plain.bin' '$W/dev_rec.bin'"
check "the recording holds the plugin's markers (the boot's end, the programs loaded, fini)" "[ $(pjob_rc markers) -eq 0 ]"

# 3. self-replay: strict (every request byte for byte, client-written pages at the snapshots and at the end), with the markers
#    (equal, in order), and under the guard
check "self-replay of the recorded boot: PASS" "replay_line v1_self | grep -q 'PASS'"
check "self-replay with the markers (all equal) and the guard (no refusal): PASS" \
    "replay_line v1_self_rec | grep -q 'PASS' && replay_line v1_self_rec | grep -q '\"markers\": \"equal\"'"

# 4. differential replay: a mutated recording makes the plugin fail the same way twice, on the fault the mutation made
for m in "${MUT[@]}"; do
    IFS='|' read -r m why <<< "$m"
    a=$(replay_line v1_rm_${m}_1 | sed -E 's/ \{.*//'); b=$(replay_line v1_rm_${m}_2 | sed -E 's/ \{.*//')
    check "the $m recording mutation fails the plugin ($why), identically twice ($a)" \
        "[ -n \"$a\" ] && [ \"$a\" = \"$b\" ] && echo \"$a\" | grep -q FAIL && grep -q 'level boot: building the NVDevice: RuntimeError: $why' '$TINYGPU_TEST_WORK/run_replay_v1_rm_${m}_1.txt'"
done
# ... and the GSP's replies ahead of the requests that asked for them, as a C++ runtime's recording can hold them (STATUS.md
#     R41): the plugin must pass all the same
check "the diff-early recording mutation (a channel's replies before its rm_alloc reaches the queue head) replays exactly" \
    "replay_line v1_diff_early | grep -q 'PASS'"

# 5. the guard, live behind the proxy: a clean boot passes; and in the replay each defect put into its inputs alone, just before
#    the audit that must catch it, is refused there (tgreplay's GUARD_DEFECTS: what BEAGLE_TG_MUTATE's defects in the daemon's
#    Python were, before plan step C13c removed them with it)
check "the guard passes a clean boot and its teardown" "tail -1 '$W/guard_clean.txt' | grep -q PASS"
for m in "${GD[@]}"; do
    IFS='|' read -r d rx <<< "$m"
    check "the guard refuses the $d defect in the replay" "replay_line v1_gd_$d | grep -qE 'FAIL: the guard refused .*$rx'"
done

# 6. the comparator: a recording is equivalent to itself and to the same boot recorded without the markers (markers aside), and
#    not to a run at another state count
c1=$(pjob_rc canon_self); c2=$(pjob_rc canon_markers); c3=$(pjob_rc canon_64)
check "tgcanon: a recording is equivalent to itself" "[ $c1 -eq 0 ]"
check "tgcanon: with and without the markers, equivalent (markers aside)" "[ $c2 -eq 0 ] && grep -q 'markers differ' '$W/canon_markers.txt'"
check "tgcanon: a run at 64 states is not equivalent to one at 4" "[ $c3 -ne 0 ] && grep -q '^session 2: DIFFERENT' '$W/canon_64.txt'"

# 7. run_l0.sh itself, in its dry run (the fake AD107 in place of TinyGPU.app), and the offline replay of what it recorded
R=$(dry_rec)
check "run_l0.sh dry run: PASS, the recording with its run.json and test output" \
    "tail -1 '$W/l0_dry.txt' | grep -q '^OK: PASS, recorded' && [ -f '$R/run.json' ] && [ -f '$R/test_output.txt' ]"
check "the dry run's recording replays exactly, markers equal, under the guard" \
    "replay_line v1_l0dry | grep -q 'PASS' && replay_line v1_l0dry | grep -q '\"markers\": \"equal\"'"

# 8. the L0 recordings, where $BEAGLE_TINYGPU_DATA has them: the guard refuses a corrupt RPC in the cold boot's replay (on the
#    RTX 4060's own data), and the three are one boot (through the program load: the tests differ after it) and one teardown
#    under tgcanon's masks (test_c11.sh replays each to the plugin)
if [ ${#L0[@]} -eq 3 ]; then
    check "the guard refuses the rpc-corrupt defect in the L0 cold boot's replay" \
        "replay_line l0_gd_rpc | grep -q 'FAIL: the guard refused NV_PGSP_QUEUE_HEAD.*has a bad checksum'"
    for p in "0 1" "1 2" "0 2"; do
        set -- $p; a=${L0[$1]}; b=${L0[$2]}
        c1=$(pjob_rc canon_l0_$1$2_boot); c2=$(pjob_rc canon_l0_$1$2_fini)
        check "tgcanon: L0 ${a##*_} and ${b##*_} are one boot through the program load and one teardown, under the masks" "[ $c1 -eq 0 ] && [ $c2 -eq 0 ]"
    done
else
    echo "(no L0 recordings in $BEAGLE_TINYGPU_DATA/recordings: their checks skipped)"
fi

left=$(tracked_alive)   # this stage's own fakes, proxies, replay servers and guards only: other stages run at the same time
check "no harness process or guard is left running" "[ -z \"\$left\" ]"
echo; [ $fails -eq 0 ] && echo "test_v1: PASS" || echo "test_v1: $fails FAILED"
[ $fails -eq 0 ]
