#!/bin/bash
# TODO.md plan step A2, end to end with no eGPU, on fake_amd_device.py's register-level card (fake_am_gpu.py), which runs every
# PM4 and SDMA packet through the GMC page tables and audits every system address the GPU could reach (the DART check):
#   - A2h: the plugin boots the card itself, cold (a full boot) and warm (a partial one), and runs tinygputest. Its instance
#     session, through the handoff, must send the requests golden_amd_boot's C++ session sends on a card in the same state, byte
#     for byte; that session is the oracle daemon's (golden_amd_boot.py, A2g). A dirty card is refused before the mode1 reset.
#   - A2j: the AMD L0 recordings (TG_AMD_L0, env.sh) replay exactly to the oracle's daemon and to the C++ boot (amd_l0_replay.py).
# Kernels are not run, so logL is wrong by design: a case passes when the plugin handed over, ran to the end with no runtime
# error, and the fake saw NO ERRORS.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
require_no_launch_guard
[ -x "$TEST_BIN" ] || { echo "no $TEST_BIN; build it first"; exit 2; }
GUARD_BIN="$BEAGLE_BUILD/libhmsbeagle/GPU/CMake_TinyGPU/beagle-tinygpu-guard"
POOL_MB=64
results=()

run_case() {   # <label> [VAR=value ...] -- [tinygputest args ...]
    local label=$1; shift
    local envs=(); while [ $# -gt 0 ] && [ "$1" != "--" ]; do envs+=("$1"); shift; done; [ "$1" = "--" ] && shift
    local sockdir; sockdir=$(mktemp -d /tmp/tga.XXXXXX)
    local sock="$sockdir/dev.sock" mem="$TINYGPU_TEST_WORK/fake_amd_$label" dlog="$TINYGPU_TEST_WORK/fake_amd_$label.log"
    local out="$TINYGPU_TEST_WORK/run_amd_$label.txt"
    rm -rf "$mem" "$dlog" "$out"; mkdir -p "$mem"
    env "${envs[@]}" "$BEAGLE_PYTHON" "$TG_TESTS/fake_amd_device.py" "$sock" "$mem" > "$dlog" 2>&1 &
    local srv=$!
    for i in $(seq 100); do grep -q listening "$dlog" 2>/dev/null && break; sleep 0.1; done
    if ! grep -q listening "$dlog"; then results+=("$label: FAIL (the fake device did not start; $dlog)"); kill $srv 2>/dev/null; return; fi
    env BEAGLE_TINYGPU_NO_LAUNCH=1 BEAGLE_TINYGPU_NO_DOWNLOAD=1 BEAGLE_TINYGPU_LOG="$TINYGPU_TEST_WORK/beagle_tinygpu_offline.log" \
        APL_REMOTE_SOCK="$sock" TMPDIR="$sockdir" BEAGLE_AMD_GUARD="$TG_TESTS/replay/crash_guard_wrap.sh" BEAGLE_TG_GUARD_BIN="$GUARD_BIN" \
        BEAGLE_TG_GUARD_PIDFILE="$sockdir/guard.pid" BEAGLE_AMD_DATA_MB=$POOL_MB DYLD_LIBRARY_PATH="$TEST_LIBS" "${envs[@]}" \
        "$TEST_BIN" "$@" > "$out" 2>&1 &
    local tst=$! rc=124
    for i in $(seq 1800); do kill -0 $tst 2>/dev/null || { wait $tst; rc=$?; break; }; sleep 0.1; done
    [ $rc -eq 124 ] && { kill -KILL $tst 2>/dev/null; wait $tst 2>/dev/null; }
    # the crash guard exits at the plugin's clean; one that holds keeps the fake's connection, so it is ended here (offline only)
    local gpid held=""; gpid=$(cat "$sockdir/guard.pid" 2>/dev/null)
    if [ -n "$gpid" ]; then
        for i in $(seq 100); do kill -0 "$gpid" 2>/dev/null || break; sleep 0.1; done
        if kill -0 "$gpid" 2>/dev/null; then held=1; kill -KILL "$gpid"; for i in $(seq 50); do kill -0 "$gpid" 2>/dev/null || break; sleep 0.1; done; fi
    fi
    # the last session's verdict: BEAGLE's resource listing makes a session of its own (one CFG_READ) before the run's
    for i in $(seq 100); do [ "$(grep -c 'client done' "$dlog")" -ge 2 ] && break; sleep 0.1; done
    kill $srv 2>/dev/null; wait $srv 2>/dev/null; rm -rf "$sockdir"
    local launches verdict; launches=$(grep -o '"launches": [0-9]*' "$dlog" | tail -1 | grep -o '[0-9]*$')
    verdict=$(grep -E "fake TinyGPU.app \(AMD device\): (NO ERRORS|[0-9]+ ERRORS)" "$dlog" | tail -1)
    local why=""
    [ $rc -eq 124 ] && why="the test hung (killed after 180 s)"
    [ -n "$held" ] && why="${why:+$why; }the crash guard held"
    [ "$(grep -c 'client done' "$dlog")" -ge 2 ] || why="${why:+$why; }the run's session never ended"
    echo "$verdict" | grep -q "NO ERRORS" || why="${why:+$why; }the fake saw errors: $(echo "$verdict" | cut -c1-300)"
    grep -q "TinyGPU/AMD: C++ runtime: handed over after the C++ boot" "$out" || why="${why:+$why; }no handoff"
    grep -q "^per evaluation:" "$out" || why="${why:+$why; }the test did not finish its evaluations"
    grep -qE "TinyGPU/AMD: .*(failed|refused|fault)" "$out" && why="${why:+$why; }$(grep -m1 -E "TinyGPU/AMD: .*(failed|refused|fault)" "$out" | cut -c1-200)"
    [ "${launches:-0}" -gt 0 ] || why="${why:+$why; }no launches reached the fake GPU"
    if [ -z "$why" ]; then results+=("$label: PASS ($launches launches; $(grep -oE '"(copied bytes|signals|psp cmd 0x6|tlb flushes|system PTEs audited|compute queues activated|sdma queues activated)": [0-9]*' "$dlog" | tail -7 | tr '\n' ' '))")
    else results+=("$label: FAIL ($why; $out, $dlog)"); fi
}

# A2h: the plugin's C++ boot, cold and warm. FAKE_AMD_RECORD: the second session is the instance's (after the resource listing's)
for st in cold warm; do
    rec="$TINYGPU_TEST_WORK/a2h_rec_$st" gold="$TINYGPU_TEST_WORK/a2h_golden_$st.bin"
    rm -f "$rec".* "$gold"
    run_case a2h_cpp_boot_$st FAKE_AMD_STATE=$st FAKE_AMD_RECORD="$rec" -- --state-count 4 --reps 3 --diag-compare-cpu
    # golden_amd_boot's C++ session on a fresh card in the same state, up to its handoff (--die-before-fini): the oracle's
    "$BEAGLE_PYTHON" - "$TG_TESTS" "$st" "$POOL_MB" "$gold" <<'EOF'
import sys, tempfile
sys.path.insert(0, sys.argv[1])
import tgpaths
tgpaths.setup()
import golden_amd_boot as gab
state, pool_mb, out = sys.argv[2], int(sys.argv[3]), sys.argv[4]
gab.WORK.mkdir(parents=True, exist_ok=True)
work = tempfile.mkdtemp(dir=gab.WORK)
exe = gab.WORK / "golden_amd_boot"
tgpaths.build_cpp(tgpaths.HERE / "golden_amd_boot.cpp", exe)
card = gab.Card(work, state)
env = {"APL_REMOTE_SOCK": card.path, "BEAGLE_TINYGPU_NO_LAUNCH": "1", "TMPDIR": work, "BEAGLE_TINYGPU_LOG": str(gab.WORK / "golden_amd_boot.log")}
lines, rec, _ = gab.run(card, [str(exe), gab.blobs_file(work), "--session", str(pool_mb << 20), "--die-before-fini"], env)
card.srv.close()
open(out, "wb").write(b"".join(rec))
print(lines[0][:60] if lines else "(no output)")
EOF
    n=$(wc -c < "$gold" 2>/dev/null | tr -d ' ')
    if [ -z "$n" ] || [ "$n" -eq 0 ]; then results+=("a2h_session_$st: FAIL (no golden session: $gold)")
    elif head -c "$n" "$rec.1" | cmp -s - "$gold"; then   # (macOS cmp -n still reports the shorter file's EOF)
        results+=("a2h_session_$st: PASS (the instance session's first $n bytes of requests, through the handoff, equal golden_amd_boot's C++ session's)")
    else results+=("a2h_session_$st: FAIL ($(head -c "$n" "$rec.1" | cmp - "$gold" 2>&1 | head -1))"); fi
done
# a card an earlier session left unclean: the C++ boot refuses before tinygrad's mode1 reset, the run fails cleanly
label=a2h_cpp_boot_dirty sockdir=$(mktemp -d /tmp/tga.XXXXXX)
dlog="$TINYGPU_TEST_WORK/fake_amd_$label.log" out="$TINYGPU_TEST_WORK/run_amd_$label.txt" mem="$TINYGPU_TEST_WORK/fake_amd_$label"
rm -rf "$mem"; mkdir -p "$mem"
FAKE_AMD_STATE=dirty "$BEAGLE_PYTHON" "$TG_TESTS/fake_amd_device.py" "$sockdir/dev.sock" "$mem" > "$dlog" 2>&1 & srv=$!
for i in $(seq 100); do grep -q listening "$dlog" 2>/dev/null && break; sleep 0.1; done
env BEAGLE_TINYGPU_NO_LAUNCH=1 BEAGLE_TINYGPU_NO_DOWNLOAD=1 BEAGLE_TINYGPU_LOG="$TINYGPU_TEST_WORK/beagle_tinygpu_offline.log" APL_REMOTE_SOCK="$sockdir/dev.sock" \
    TMPDIR="$sockdir" DYLD_LIBRARY_PATH="$TEST_LIBS" "$TEST_BIN" --state-count 4 --reps 1 > "$out" 2>&1
rc=$?
for i in $(seq 50); do [ "$(grep -c 'client done' "$dlog")" -ge 2 ] && break; sleep 0.1; done
kill $srv 2>/dev/null; wait $srv 2>/dev/null; rm -rf "$sockdir"
if [ $rc -ne 0 ] && grep -q "needs an SMU mode1 reset" "$out" && grep -q "NO ERRORS" "$dlog" && ! grep -q '"mode1 resets"' "$dlog" \
   && ! grep -q '"compute queues activated"' "$dlog"; then results+=("$label: PASS (refused before the mode1 reset; exit $rc, no queue set up)")
else results+=("$label: FAIL (exit $rc; $out, $dlog)"); fi
# A2j: the AMD L0 recordings of this card's chip, replayed under the guard to the oracle's daemon and to the C++ boot (TODO.md
# plan step N14: each card's in its own pass)
for r in $TG_AMD_L0; do
    R="$BEAGLE_TINYGPU_DATA/recordings/$r"
    if [ ! -f "$R/events.bin" ]; then results+=("$r: (not on this computer: its replay skipped)"); continue; fi
    [ "$("$BEAGLE_PYTHON" "$TG_TESTS/amd_l0_replay.py" --chip "$R")" = "${FAKE_AMD_CHIP:-gfx1100}" ] || continue
    if "$BEAGLE_PYTHON" "$TG_TESTS/amd_l0_replay.py" "$R" > "$TINYGPU_TEST_WORK/a2j_replay_$r.txt" 2>&1; then
        results+=("$r: PASS (replays exactly to the oracle's daemon and to the C++ boot)")
    else results+=("$r: FAIL ($(tail -1 "$TINYGPU_TEST_WORK/a2j_replay_$r.txt" | cut -c1-200); $TINYGPU_TEST_WORK/a2j_replay_$r.txt)"); fi
done
echo "=== A2"; printf '%s\n' "${results[@]}"
! printf '%s\n' "${results[@]}" | grep -q FAIL
