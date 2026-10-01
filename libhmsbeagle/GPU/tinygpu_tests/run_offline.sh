#!/bin/bash
# Everything that can be checked without the eGPU: the goldens (each C++ port against the tinygrad code it follows, in
# tinygpu_tests/oracle), the C++ TinyGPU.app client, then the plugin end to end on fake_nv_device.py, which it boots itself as it
# boots the eGPU (plan steps C11-C13): plan step V1's record/replay tools and the hardware recordings, the fake GB205, the crash
# guard, the library's error returns, the uploads, D1's kernels, several instances in one process, the routing, the failures and
# the kills; the firmware staging; then the no-launch guard (nothing listening => the plugin errors out and no TinyGPU.app is
# spawned). Build hmsbeagle-tinygpu-hybrid, beagle-tinygpu-guard, tinygpuhybridtest, synthetictest and hmctest first.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
require_no_launch_guard   # static check before anything below could reach a spawn path
unset FAKE_TEST_BIN       # every run below is tinygpuhybridtest's unless it names another binary itself
results=()
"$TG_TESTS/run_goldens.sh"; results+=("goldens: $([ $? -eq 0 ] && echo PASS || echo FAIL)")
# plan step C3: the C++ TinyGPU.app client against TinyGPU's real server.c on an IOKit stub (its limits, error replies,
# sysmem, a lost server, the lock), the socket path and the TinyGPU.app check
"$TG_TESTS/test_c3_transport.sh" > "$TINYGPU_TEST_WORK/test_c3_transport.log" 2>&1
results+=("transport (C3): $([ $? -eq 0 ] && echo PASS || echo "FAIL (see $TINYGPU_TEST_WORK/test_c3_transport.log)")")
# plan step V1: the recording proxy, the replay server, the guard and the comparator, end to end on the fake AD107 and TinyGPU's
# server.c, and run_l0.sh's dry run
"$TG_TESTS/test_v1.sh" > "$TINYGPU_TEST_WORK/test_v1.log" 2>&1
results+=("record/replay (V1): $([ $? -eq 0 ] && echo PASS || echo "FAIL (see $TINYGPU_TEST_WORK/test_v1.log)")")
# plan step B2: a GB205 in fake_nv_device.py (the COT boot, MMU v3, QMD v5), run_l0.sh's dry runs, their replays under the guard
# and tgcanon, and the GB205's own recordings replayed to the C++ boot and the plugin's COT teardown
"$TG_TESTS/test_b2.sh" > "$TINYGPU_TEST_WORK/test_b2_e2e.log" 2>&1
results+=("GB205 and V1 tools (B2): $([ $? -eq 0 ] && echo PASS || echo "FAIL (see $TINYGPU_TEST_WORK/test_b2_e2e.log)")")
# plan step C11: the full C++ boot and the crash guard from before the first request (a warm GPU, kills in the boot and idle, no
# INIT_DONE, FWSEC-FRTS failing), and the RTX 4060's L0 recordings replayed to it
"$TG_TESTS/test_c11.sh" > "$TINYGPU_TEST_WORK/test_c11_e2e.log" 2>&1
results+=("full C++ boot (C11): $([ $? -eq 0 ] && echo PASS || echo "FAIL (see $TINYGPU_TEST_WORK/test_c11_e2e.log)")")
# plan step C12: level boot by default, and a library that returns errors instead of exiting its host (the finalize order, a
# silent GSP, exit() from another thread, SIGINT, TinyGPU.app gone, a warm GPU, a failed instance, a hang)
"$TG_TESTS/test_c12.sh" > "$TINYGPU_TEST_WORK/test_c12_e2e.log" 2>&1
results+=("default boot and error returns (C12): $([ $? -eq 0 ] && echo PASS || echo "FAIL (see $TINYGPU_TEST_WORK/test_c12_e2e.log)")")
# plan step C13c: the checks that ran on the fake daemon (the uploaded images, a GPU no cubin serves, D1's kernels, several
# instances in one process, the routing, the ring's wrap, no teardown, the falcons' and the GSP's failures, the kills)
"$TG_TESTS/test_c13.sh" > "$TINYGPU_TEST_WORK/test_c13_e2e.log" 2>&1
results+=("fake-daemon checks on the C++ boot (C13): $([ $? -eq 0 ] && echo PASS || echo "FAIL (see $TINYGPU_TEST_WORK/test_c13_e2e.log)")")
# plan step A1h: the AMD C++ runtime end to end on fake_amd_device.py and fake_amd_daemon.py (the DART audit, wraps, a fault)
"$TG_TESTS/test_a1h.sh" > "$TINYGPU_TEST_WORK/test_a1h.log" 2>&1
results+=("AMD C++ runtime end to end (A1h): $([ $? -eq 0 ] && echo PASS || echo "FAIL (see $TINYGPU_TEST_WORK/test_a1h.log)")")
"$BEAGLE_PYTHON" "$TG_TESTS/check_firmware.py" > "$TINYGPU_TEST_WORK/check_firmware.log" 2>&1
rc=$?; cat "$TINYGPU_TEST_WORK/check_firmware.log"
results+=("firmware staging: $([ $rc -eq 0 ] && echo PASS || echo "FAIL (see $TINYGPU_TEST_WORK/check_firmware.log)")")

# no-launch guard: point the plugin at a socket nobody listens on
SOCKDIR=$(mktemp -d "${TMPDIR:-/tmp}/tg.XXXXXX")
before=$(pgrep -f "TinyGPU.app/Contents/MacOS/TinyGPU server" | wc -l)
# (defence in depth: even if the guard regressed, the AMD side's Python could not start a boot)
env BEAGLE_TINYGPU_NO_LAUNCH=1 APL_REMOTE_SOCK="$SOCKDIR/none.sock" DYLD_LIBRARY_PATH="$TEST_LIBS" BEAGLE_PYTHON=/usr/bin/false \
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
