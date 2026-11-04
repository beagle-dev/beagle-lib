#!/bin/bash
# TODO.md plan step A2, end to end with no eGPU, on fake_amd_device.py's register-level card (fake_am_gpu.py), which runs every
# PM4 and SDMA packet through the GMC page tables and audits every system address the GPU could reach (the DART check):
#   - A2h: the plugin boots the card itself, cold (a full boot) and warm (a partial one), and runs tinygputest. Its instance
#     session, through the handoff, must send the requests golden_amd_boot's C++ session sends on a card in the same state, byte
#     for byte; that session is the oracle daemon's (golden_amd_boot.py, A2g). A dirty card is refused before the mode1 reset.
#   - A2j: the AMD L0 recordings (TG_AMD_L0, env.sh) replay exactly to the oracle's daemon and to the C++ boot (amd_l0_replay.py).
# Kernels are not run, so logL is wrong by design: a case passes when the plugin handed over, ran to the end with no runtime
# error, and the fake saw NO ERRORS.
# The runs go at the same time (env.sh's pjob), each with its own files, plugin log included, in $TINYGPU_TEST_WORK/a2_<chip>;
# the checks are then made in this order.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
require_no_launch_guard
[ -x "$TEST_BIN" ] || { echo "no $TEST_BIN; build it first"; exit 2; }
GUARD_BIN="$BEAGLE_BUILD/libhmsbeagle/GPU/CMake_TinyGPU/beagle-tinygpu-guard"
POOL_MB=64
CHIP=${FAKE_AMD_CHIP:-gfx1100}
W="$TINYGPU_TEST_WORK/a2_$CHIP"
rm -rf "$W"; mkdir -p "$W"
pjob_init "$W"
BUDGET=600   # seconds a plugin run may take before it counts as hung (about 1 s on an idle Mac)

run_case() {   # <label> [VAR=value ...] -- [tinygputest args ...]: its result line goes to $W/<label>.result
    local label=$1; shift
    local envs=(); while [ $# -gt 0 ] && [ "$1" != "--" ]; do envs+=("$1"); shift; done; [ "$1" = "--" ] && shift
    local sockdir; sockdir=$(mktemp -d /tmp/tga.XXXXXX)
    local sock="$sockdir/dev.sock" mem="$W/fake_amd_${label}_$CHIP" dlog="$W/fake_amd_${label}_$CHIP.log"
    local out="$W/run_amd_${label}_$CHIP.txt" tglog="$W/${label}_$CHIP.tglog" res="$W/$label.result"
    rm -rf "$mem" "$dlog" "$out" "$tglog" "$res"; mkdir -p "$mem"
    env "${envs[@]}" "$BEAGLE_PYTHON" "$TG_TESTS/fake_amd_device.py" "$sock" "$mem" > "$dlog" 2>&1 &
    local srv=$!; track_pid $srv fake_amd_device
    for i in $(seq 300); do grep -q listening "$dlog" 2>/dev/null && break; sleep 0.1; done
    if ! grep -q listening "$dlog"; then echo "$label: FAIL (the fake device did not start; $dlog)" > "$res"; kill $srv 2>/dev/null; return; fi
    env BEAGLE_TINYGPU_NO_LAUNCH=1 BEAGLE_TINYGPU_NO_DOWNLOAD=1 BEAGLE_TINYGPU_LOG="$tglog" \
        APL_REMOTE_SOCK="$sock" TMPDIR="$sockdir" BEAGLE_AMD_GUARD="$TG_TESTS/replay/crash_guard_wrap.sh" BEAGLE_TG_GUARD_BIN="$GUARD_BIN" \
        BEAGLE_TG_GUARD_PIDFILE="$sockdir/guard.pid" BEAGLE_AMD_DATA_MB=$POOL_MB DYLD_LIBRARY_PATH="$TEST_LIBS" "${envs[@]}" \
        "$TEST_BIN" "$@" > "$out" 2>&1 &
    local tst=$! rc=124
    for i in $(seq $((BUDGET * 10))); do kill -0 $tst 2>/dev/null || { wait $tst; rc=$?; break; }; sleep 0.1; done
    [ $rc -eq 124 ] && { kill -KILL $tst 2>/dev/null; wait $tst 2>/dev/null; }
    # the crash guard exits at the plugin's clean; one that holds keeps the fake's connection, so it is ended here (offline only)
    local gpid held=""; gpid=$(cat "$sockdir/guard.pid" 2>/dev/null); track_pid "$gpid" beagle-tinygpu-guard
    if [ -n "$gpid" ]; then
        for i in $(seq 300); do kill -0 "$gpid" 2>/dev/null || break; sleep 0.1; done
        if kill -0 "$gpid" 2>/dev/null; then held=1; kill -KILL "$gpid"; for i in $(seq 50); do kill -0 "$gpid" 2>/dev/null || break; sleep 0.1; done; fi
    fi
    # the last session's verdict: BEAGLE's resource listing makes a session of its own (one CFG_READ) before the run's
    for i in $(seq 300); do [ "$(grep -c 'client done' "$dlog")" -ge 2 ] && break; sleep 0.1; done
    kill $srv 2>/dev/null; wait $srv 2>/dev/null; rm -rf "$sockdir"
    local launches verdict; launches=$(grep -o '"launches": [0-9]*' "$dlog" | tail -1 | grep -o '[0-9]*$')
    verdict=$(grep -E "fake TinyGPU.app \(AMD device\): (NO ERRORS|[0-9]+ ERRORS)" "$dlog" | tail -1)
    local why=""
    [ $rc -eq 124 ] && why="the test hung (killed after $BUDGET s)"
    [ -n "$held" ] && why="${why:+$why; }the crash guard held"
    [ "$(grep -c 'client done' "$dlog")" -ge 2 ] || why="${why:+$why; }the run's session never ended"
    echo "$verdict" | grep -q "NO ERRORS" || why="${why:+$why; }the fake saw errors: $(echo "$verdict" | cut -c1-300)"
    grep -q "TinyGPU/AMD: C++ runtime: handed over after the C++ boot" "$out" || why="${why:+$why; }no handoff"
    grep -q "^per evaluation:" "$out" || why="${why:+$why; }the test did not finish its evaluations"
    grep -qE "TinyGPU/AMD: .*(failed|refused|fault)" "$out" && why="${why:+$why; }$(grep -m1 -E "TinyGPU/AMD: .*(failed|refused|fault)" "$out" | cut -c1-200)"
    [ "${launches:-0}" -gt 0 ] || why="${why:+$why; }no launches reached the fake GPU"
    if [ -z "$why" ]; then echo "$label: PASS ($launches launches; $(grep -oE '"(copied bytes|signals|psp cmd 0x6|tlb flushes|system PTEs audited|compute queues activated|sdma queues activated)": [0-9]*' "$dlog" | tail -7 | tr '\n' ' '))" > "$res"
    else echo "$label: FAIL ($why; $out, $dlog)" > "$res"; fi
}

# golden_amd_boot's C++ session on a fresh card in the same state, up to its handoff (--die-before-fini): the oracle's. Its
# requests go to $W/a2h_golden_<state>_<chip>.bin, its output to .txt; the card's socket, sysmem and the session's lock in a
# directory of its own under /tmp (socket paths are short)
golden() {   # <cold|warm>
    local st=$1 gold="$W/a2h_golden_${1}_$CHIP.bin"
    rm -f "$gold"
    "$BEAGLE_PYTHON" - "$TG_TESTS" "$st" "$POOL_MB" "$gold" "$W/a2h_golden_${st}_$CHIP.tglog" > "$W/a2h_golden_${st}_$CHIP.txt" 2>&1 <<'EOF'
import sys, shutil, tempfile
sys.path.insert(0, sys.argv[1])
import tgpaths
tgpaths.setup()
import golden_amd_boot as gab
state, pool_mb, out, log = sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5]
gab.WORK.mkdir(parents=True, exist_ok=True)
work = tempfile.mkdtemp(prefix="a2g.", dir="/tmp")
try:
    exe = gab.WORK / "golden_amd_boot"
    tgpaths.build_cpp(tgpaths.HERE / "golden_amd_boot.cpp", exe)
    card = gab.Card(work, state)
    env = {"APL_REMOTE_SOCK": card.path, "BEAGLE_TINYGPU_NO_LAUNCH": "1", "TMPDIR": work, "BEAGLE_TINYGPU_LOG": log}
    lines, rec, _ = gab.run(card, [str(exe), gab.blobs_file(work), "--session", str(pool_mb << 20), "--die-before-fini"], env)
    card.srv.close()
finally:
    shutil.rmtree(work, ignore_errors=True)
open(out, "wb").write(b"".join(rec))
print(lines[0][:60] if lines else "(no output)")
EOF
}

# a card an earlier session left unclean: the C++ boot refuses before tinygrad's mode1 reset, the run fails cleanly
dirty() {
    local label=a2h_cpp_boot_dirty sockdir; sockdir=$(mktemp -d /tmp/tga.XXXXXX)
    local dlog="$W/fake_amd_${label}_$CHIP.log" out="$W/run_amd_${label}_$CHIP.txt" mem="$W/fake_amd_${label}_$CHIP" res="$W/$label.result"
    rm -rf "$mem" "$dlog" "$out" "$res"; mkdir -p "$mem"
    FAKE_AMD_STATE=dirty "$BEAGLE_PYTHON" "$TG_TESTS/fake_amd_device.py" "$sockdir/dev.sock" "$mem" > "$dlog" 2>&1 &
    local srv=$! rc; track_pid $srv fake_amd_device
    for i in $(seq 300); do grep -q listening "$dlog" 2>/dev/null && break; sleep 0.1; done
    env BEAGLE_TINYGPU_NO_LAUNCH=1 BEAGLE_TINYGPU_NO_DOWNLOAD=1 BEAGLE_TINYGPU_LOG="$W/${label}_$CHIP.tglog" APL_REMOTE_SOCK="$sockdir/dev.sock" \
        TMPDIR="$sockdir" DYLD_LIBRARY_PATH="$TEST_LIBS" "$TEST_BIN" --state-count 4 --reps 1 > "$out" 2>&1
    rc=$?
    for i in $(seq 300); do [ "$(grep -c 'client done' "$dlog")" -ge 2 ] && break; sleep 0.1; done
    kill $srv 2>/dev/null; wait $srv 2>/dev/null; rm -rf "$sockdir"
    if [ $rc -ne 0 ] && grep -q "needs an SMU mode1 reset" "$out" && grep -q "NO ERRORS" "$dlog" && ! grep -q '"mode1 resets"' "$dlog" \
       && ! grep -q '"compute queues activated"' "$dlog"; then echo "$label: PASS (refused before the mode1 reset; exit $rc, no queue set up)" > "$res"
    else echo "$label: FAIL (exit $rc; $out, $dlog)" > "$res"; fi
}

# A2j: an AMD L0 recording, if it is this card's chip's (TODO.md plan step N14: each card's in its own pass), replayed under the
# guard to the oracle's daemon and to the C++ boot. Its result line goes to $W/a2j_<recording>.result, none for another chip's.
a2j() {   # <recording name>
    local r=$1 R="$BEAGLE_TINYGPU_DATA/recordings/$1" res="$W/a2j_$1.result" txt="$W/a2j_replay_${1}_$CHIP.txt"
    rm -f "$res" "$txt"
    [ "$("$BEAGLE_PYTHON" "$TG_TESTS/amd_l0_replay.py" --chip "$R")" = "$CHIP" ] || return 0
    if "$BEAGLE_PYTHON" "$TG_TESTS/amd_l0_replay.py" "$R" > "$txt" 2>&1; then
        echo "$r: PASS (replays exactly to the oracle's daemon and to the C++ boot)" > "$res"
    else echo "$r: FAIL ($(tail -1 "$txt" | cut -c1-200); $txt)" > "$res"; fi
}

# longest first: the replays, the goldens, then the plugin runs. FAKE_AMD_RECORD: the second session is the instance's (after
# the resource listing's)
for r in $TG_AMD_L0; do [ -f "$BEAGLE_TINYGPU_DATA/recordings/$r/events.bin" ] && pjob "a2j_$r" a2j "$r"; done
for st in cold warm; do pjob "golden_$st" golden $st; done
for st in cold warm; do
    rm -f "$W/a2h_rec_${st}_$CHIP".*
    pjob "a2h_cpp_boot_$st" run_case "a2h_cpp_boot_$st" FAKE_AMD_STATE=$st FAKE_AMD_RECORD="$W/a2h_rec_${st}_$CHIP" -- --state-count 4 --reps 3 --diag-compare-cpu
done
pjob dirty dirty
pwait

result_of() {   # <job> <label>: the job's result line
    cat "$W/$2.result" 2>/dev/null || echo "$2: FAIL (the run ended without a verdict, exit $(pjob_rc "$1"))"
}
results=()
for st in cold warm; do
    cat "$W/a2h_golden_${st}_$CHIP.txt"
    results+=("$(result_of "a2h_cpp_boot_$st" "a2h_cpp_boot_$st")")
    rec="$W/a2h_rec_${st}_$CHIP" gold="$W/a2h_golden_${st}_$CHIP.bin"
    n=$(wc -c < "$gold" 2>/dev/null | tr -d ' ')
    if [ -z "$n" ] || [ "$n" -eq 0 ]; then results+=("a2h_session_$st: FAIL (no golden session: $gold)")
    elif head -c "$n" "$rec.1" | cmp -s - "$gold"; then   # (macOS cmp -n still reports the shorter file's EOF)
        results+=("a2h_session_$st: PASS (the instance session's first $n bytes of requests, through the handoff, equal golden_amd_boot's C++ session's)")
    else results+=("a2h_session_$st: FAIL ($(head -c "$n" "$rec.1" | cmp - "$gold" 2>&1 | head -1))"); fi
done
results+=("$(result_of dirty a2h_cpp_boot_dirty)")
for r in $TG_AMD_L0; do
    if [ ! -f "$BEAGLE_TINYGPU_DATA/recordings/$r/events.bin" ]; then results+=("$r: (not on this computer: its replay skipped)"); continue; fi
    rc=$(pjob_rc "a2j_$r")
    if [ -f "$W/a2j_$r.result" ]; then results+=("$(cat "$W/a2j_$r.result")")
    elif [ "$rc" != 0 ]; then results+=("$r: FAIL (its replay job ended without a verdict, exit $rc)"); fi   # (exit 0, no line: another chip's)
done
echo "=== A2"; printf '%s\n' "${results[@]}"
! printf '%s\n' "${results[@]}" | grep -q FAIL
