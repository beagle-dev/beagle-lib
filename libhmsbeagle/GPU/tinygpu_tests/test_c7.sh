#!/bin/bash
# TODO.md plan step C7, end to end with no eGPU: BEAGLE_NV_CPP_LEVEL=rm, where the daemon boots only the NVDev and the plugin
# builds the NVDevice with tinygrad's RM client ported (TinyGPUHybridNVRM.h, TinyGPUHybridNVDevice.h), in the real plugin and
# daemon. On the fake AD107 (fake_nv_device.py, whose GSP checks that every command's sequence number follows the one before)
# the device must receive exactly the bytes it receives at level sysmem, where the daemon builds the NVDevice; two instances in
# two threads share the plugin's one NVDevice; a pool reaching into GSP-RM's reserved region, refused after the NVDevice is
# built, is torn down by the daemon, which continues the plugin's GSP command queue from the state page (a daemon that
# continues from its own count is caught by the fake's GSP), and so is an allocation the GSP refuses while the plugin builds the
# NVDevice, where the plugin stops before any submission; each L0 hardware recording (where $BEAGLE_TINYGPU_DATA has them)
# must replay exactly at level rm, under the guard; and run_l0.sh --level rm must work in its dry run, and replay. The C++ RM
# client's golden is golden_rm.py, the daemon's half test_c7.py. One PASS or FAIL line per check; exit 0 only if all pass.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
require_no_launch_guard
W="$TINYGPU_TEST_WORK/c7"; rm -rf "$W"; mkdir -p "$W"
fails=0
pass() { echo "PASS $1"; }
fail() { echo "FAIL $1"; fails=$((fails + 1)); }
check() { if eval "$2"; then pass "$1"; else fail "$1"; fi; }
replay_line() { grep -E "^replay session 2" "$TINYGPU_TEST_WORK/replay_$1.log"; }
out() { echo "$TINYGPU_TEST_WORK/run_device_c7_$1.txt"; }
dlog() { echo "$TINYGPU_TEST_WORK/run_device_c7_$1_daemon.log"; }
BUILT="C++ runtime: built the NVDevice"

# 1. the whole session at the fake device at levels sysmem and rm: identical bytes
for lv in sysmem rm; do
    FAKE_TG_RECORD="$W/dev_$lv.bin" "$TG_TESTS/run_fake_device.sh" c7_$lv BEAGLE_NV_CPP_LEVEL=$lv -- --state-count 4 --reps 3 > "$W/$lv.txt" 2>&1
    eval "r_$lv=$?"
done
check "fake AD107: both levels PASS, and the device received identical bytes from the daemon's NVDevice and the plugin's" \
    "[ $r_sysmem -eq 0 ] && [ $r_rm -eq 0 ] && cmp -s '$W/dev_sysmem.bin' '$W/dev_rm.bin'"
check "at rm: the daemon booted the NVDev only and exported the GSP; the state page came before the plugin's timeline; the plugin built the NVDevice and unloaded the GPU at fini" \
    "grep -q 'daemon booted the NVDev (level rm)' '$(out rm)' && grep -q '(level rm: sm_89, QMD v3' '$(out rm)' && ! grep -q '$BUILT' '$(out sysmem)' \
     && grep -q 'the C++ GSP unload and teardown' '$(out rm)' && grep -q 'rm export: GSP queues' '$(dlog rm)' \
     && awk '/state page mapped .*the C\+\+ side records the GSP/{s=NR} /C\+\+ timeline mapped/{t=NR} END{exit !(s && t && s < t)}' '$(dlog rm)' \
     && grep -q 'state page mapped (phase 1, frame_in_flight 1, seq 13)' '$(dlog rm)'"
# ... and two instances in one process sharing the plugin's one NVDevice (plan step P5), in two threads
"$TG_TESTS/run_fake_device.sh" c7_p5 BEAGLE_NV_CPP_LEVEL=rm -- --state-count 4,64 --threads --reps 3 > "$W/p5.txt" 2>&1
check "fake AD107 at rm: two instances in two threads share the boot and the plugin's NVDevice" \
    "[ $? -eq 0 ] && [ \$(grep -c '$BUILT' '$(out p5)') -eq 1 ] && grep -q '^tips: every instance read back its own tip partials exactly' '$(out p5)'"

# 2. a failure after the NVDevice is built: the plugin's WPR check refuses the pool, and the daemon unloads the GPU, continuing
#    the plugin's GSP command queue from the state page; then the same with a daemon that continues from its own count
"$TG_TESTS/run_fake_device.sh" c7_wpr BEAGLE_NV_CPP_LEVEL=rm BEAGLE_NV_DATA_MB=7900 -- --state-count 4 --reps 3 > "$W/wpr.txt" 2>&1
es=$(grep -o 'rm export: GSP queues (seq [0-9]*' "$(dlog wpr)" | grep -o '[0-9]*$'); ps=$(grep -o 'C++ state page: .* seq [0-9]*' "$(dlog wpr)" | grep -o '[0-9]*$')
check "fake AD107, BEAGLE_NV_DATA_MB=7900 at rm: refused after the NVDevice was built; the daemon tears down from the plugin's count ($es, then $ps), and the fake's GSP agrees" \
    "grep -q 'level rm: VRAM allocations end at 0x[0-9a-f]*, above the WPR bound 0x1f3a00000' '$(out wpr)' && ! grep -q '$BUILT' '$(out wpr)' \
     && fini_verdict '$(out wpr)' && grep -q 'NO ERRORS' '$W/wpr.txt' && [ -n '$es' ] && [ -n '$ps' ] && [ '$ps' -gt '$es' ]"
cat > "$W/stale_seq_daemon.py" <<EOF
# test_c7.sh's perturbation: the real daemon (through replay/tgdaemon.py), except that at fini it continues the GSP's command
# queue from the count it exported, not from the C++ side's state page
import sys, runpy
TG = "$TG_TESTS/replay/tgdaemon.py"
sys.path[:0] = ["$TG_TESTS/replay", "$GPU_DIR"]
import nv_dispatch_daemon as d
fini = d.Daemon._fini
def stale(self, hung):
    if self._state is not None and self.rm_exported:
        self._state = memoryview(bytearray(self._state.tobytes()[:24] + self.dev.iface.dev_impl.gsp.cmd_q.seq.to_bytes(8, "little"))).cast("Q")
    return fini(self, hung)
d.Daemon._fini = stale
sys.argv[0] = TG
runpy.run_path(TG, run_name="__main__")
EOF
"$TG_TESTS/run_fake_device.sh" c7_stale BEAGLE_NV_CPP_LEVEL=rm BEAGLE_NV_DATA_MB=7900 BEAGLE_NV_DISPATCH_DAEMON="$W/stale_seq_daemon.py" \
    -- --state-count 4 --reps 3 > "$W/stale.txt" 2>&1
check "a daemon that continues the GSP command queue from its own count is caught by the fake's GSP" \
    "grep -Eq 'RPC 0x2f: sequence number $es, not $ps' '$W/stale.txt'"
# ... and a GSP refusal while the plugin builds the NVDevice (the channel group's allocation, before any submission)
FAKE_RM_FAIL=0xa06c "$TG_TESTS/run_fake_device.sh" c7_refused BEAGLE_NV_CPP_LEVEL=rm -- --state-count 4 --reps 3 > "$W/refused.txt" 2>&1
FD="$TINYGPU_TEST_WORK/fake_device_c7_refused.log"
check "fake AD107 at rm, the channel group refused by the GSP: the plugin stops before any submission, and the daemon tears down from the plugin's count" \
    "grep -q 'level rm: building the NVDevice: .*RPC call 103 failed with result 34' '$(out refused)' && ! grep -q '$BUILT' '$(out refused)' \
     && fini_verdict '$(out refused)' && grep -q 'NO ERRORS' '$FD' && grep -q '\"rm_alloc refused (FAKE_RM_FAIL)\": 1' '$FD' \
     && ! grep -q '\"doorbells\"' '$FD' && ! grep -q 'the hung path\|HOLDING' '$(dlog refused)'"

# 3. the L0 recordings, replayed to the plugin at level rm: every request as the daemon's NVDevice sent it on the RTX 4060
L0=(); for r in $TG_L0; do [ -f "$BEAGLE_TINYGPU_DATA/recordings/$r/events.bin" ] && L0+=("$BEAGLE_TINYGPU_DATA/recordings/$r"); done
if [ ${#L0[@]} -eq 3 ]; then
    for R in "${L0[@]}"; do
        l=c7_rm_${R##*_}
        BEAGLE_NV_CPP_LEVEL=rm "$TG_TESTS/run_replay.sh" "$R" $l --guard > "$W/replay_$l.txt" 2>&1
        check "L0 ${R##*/} at rm: replays exactly with the plugin's NVDevice, under the guard" \
            "replay_line $l | grep -q 'PASS' && grep -q '$BUILT' '$TINYGPU_TEST_WORK/run_replay_$l.txt'"
    done
else
    echo "(no L0 recordings in $BEAGLE_TINYGPU_DATA/recordings: their checks skipped)"
fi

# 4. run_l0.sh --level rm --guard in its dry run (the fake AD107 in place of TinyGPU.app; the proxy in guard mode, as a rung's
#    first hardware run uses it), and the replay of what it recorded
rm -rf "$TINYGPU_TEST_WORK/l0_dry"
L0_DRY_RUN=1 "$TG_TESTS/run_l0.sh" c7dry 4 3 --level rm --guard > "$W/l0_dry.txt" 2>&1
R=$(ls -d "$TINYGPU_TEST_WORK"/l0_dry/recordings/*_c7dry 2>/dev/null | head -1)
check "run_l0.sh --level rm --guard dry run: PASS through the guard, recorded with the level in run.json" \
    "tail -1 '$W/l0_dry.txt' | grep -q '^OK: PASS, recorded' && grep -q 'BEAGLE_NV_CPP_LEVEL=rm' '$R/run.json' \
     && grep -q '(guard mode)' \$(ls \"$TINYGPU_TEST_WORK\"/l0_dry/runs/*_L0_c7dry_proxy.log | head -1)"
"$TG_TESTS/run_replay.sh" "$R" c7_l0dry --record --guard > "$W/replay_l0dry.txt" 2>&1
check "the dry run's recording replays exactly, markers equal, under the guard" \
    "replay_line c7_l0dry | grep -q 'PASS' && replay_line c7_l0dry | grep -q '\"markers\": \"equal\"'"

left=$(ps -axo command | grep -cE "Python .*(tgdaemon|tgproxy|tgreplay|fake_nv_device)\.py")
check "no harness process is left running" "[ $left -eq 0 ]"
echo; [ $fails -eq 0 ] && echo "test_c7: PASS" || echo "test_c7: $fails FAILED"
[ $fails -eq 0 ]
