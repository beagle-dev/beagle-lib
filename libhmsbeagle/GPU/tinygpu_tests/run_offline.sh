#!/bin/bash
# Everything that can be checked without the eGPU: the goldens, the firmware staging, then the plugin end to end
# against the fakes in the three NV modes at 4 and 64 states, then the hung path, then the no-launch guard (nothing listening => the
# plugin errors out and no TinyGPU.app is spawned). Build hmsbeagle-tinygpu-hybrid and tinygpuhybridtest first.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
require_no_launch_guard   # static check before anything below could reach a spawn path
results=()
"$TG_TESTS/run_goldens.sh"; results+=("goldens: $([ $? -eq 0 ] && echo PASS || echo FAIL)")
"$BEAGLE_PYTHON" "$TG_TESTS/check_firmware.py" > "$TINYGPU_TEST_WORK/check_firmware.log" 2>&1
rc=$?; cat "$TINYGPU_TEST_WORK/check_firmware.log"
results+=("firmware staging: $([ $rc -eq 0 ] && echo PASS || echo "FAIL (see $TINYGPU_TEST_WORK/check_firmware.log)")")

for states in 4 64; do
    for mode in "daemon BEAGLE_NV_CPP_DISPATCH=0" "dispatch BEAGLE_NV_CPP_DISPATCH=1" "runtime BEAGLE_NV_USE_DAEMON=0"; do
        set -- $mode
        label="$1_$states"
        "$TG_TESTS/run_fake_runtime.sh" "$label" "$2" -- --state-count $states --reps 5 > "$TINYGPU_TEST_WORK/fake_$label.summary" 2>&1
        rc=$?
        results+=("fake $label: $([ $rc -eq 0 ] && echo PASS || echo "FAIL (see $TINYGPU_TEST_WORK/fake_$label.summary)")")
    done
done

# the C++ side's cmdq ring wraps after 2 MiB of pushbuffers (about 4,400 evaluations): the wrap must wait for the frames
# before the one being submitted, not for that one (which never completes: a false hung GPU)
"$TG_TESTS/run_fake_runtime.sh" wrap BEAGLE_NV_USE_DAEMON=0 -- --reps 10000 > "$TINYGPU_TEST_WORK/fake_wrap.summary" 2>&1
results+=("fake wrap: $([ $? -eq 0 ] && echo PASS || echo "FAIL (see $TINYGPU_TEST_WORK/fake_wrap.summary)")")

# hung path (plan step P2): the fake GPU never writes a semaphore release, so the C++ runtime's 30 s timeline wait times
# out during setup; the plugin must send fini{hung} to the daemon, print its unload report, and leave no daemon behind
before=$(pgrep -f "fake_nv_daemon.py" | wc -l)
FAKE_NV_HANG=1 FAKE_RUN_TIMEOUT=120 "$TG_TESTS/run_fake_runtime.sh" hung BEAGLE_NV_USE_DAEMON=0 -- --reps 1 \
    > "$TINYGPU_TEST_WORK/fake_hung.summary" 2>&1
sleep 0.5
after=$(pgrep -f "fake_nv_daemon.py" | wc -l)
out="$TINYGPU_TEST_WORK/run_fake_hung.txt"
if grep -q "timeline wait timed out" "$out" && grep -q "GPU teardown: unload confirmed" "$out" \
   && ! grep -q "keeps the TinyGPU.app connection open" "$out" && [ "$after" -le "$before" ]; then
    results+=("fake hung: PASS")
else
    results+=("fake hung: FAIL (see $out)")
fi

# no-launch guard: point the plugin at a socket nobody listens on
SOCKDIR=$(mktemp -d "${TMPDIR:-/tmp}/tg.XXXXXX")
before=$(pgrep -f "TinyGPU.app/Contents/MacOS/TinyGPU server" | wc -l)
# (defence in depth: even if the guard regressed, no daemon or Python could start a real boot)
env BEAGLE_TINYGPU_NO_LAUNCH=1 APL_REMOTE_SOCK="$SOCKDIR/none.sock" DYLD_LIBRARY_PATH="$TEST_LIBS" \
    BEAGLE_NV_DISPATCH_DAEMON=/nonexistent/nv_dispatch_daemon.py BEAGLE_PYTHON=/usr/bin/false \
    "$TEST_BIN" --reps 1 > "$TINYGPU_TEST_WORK/nolaunch.txt" 2>&1
sleep 0.5
after=$(pgrep -f "TinyGPU.app/Contents/MacOS/TinyGPU server" | wc -l)
rmdir "$SOCKDIR"
if grep -q "BEAGLE_TINYGPU_NO_LAUNCH is set; not starting TinyGPU.app" "$TINYGPU_TEST_WORK/nolaunch.txt" && [ "$after" -le "$before" ]; then
    results+=("no-launch guard: PASS")
else
    results+=("no-launch guard: FAIL (see $TINYGPU_TEST_WORK/nolaunch.txt)")
fi

echo; echo "=== summary"; printf '%s\n' "${results[@]}"
! printf '%s\n' "${results[@]}" | grep -q FAIL
