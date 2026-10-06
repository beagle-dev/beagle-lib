#!/bin/bash
# TODO.md plan step A2j: one tinygputest run on the AMD eGPU with the plugin's C++ boot, through tgproxy.py --guard: the
# AMD guard (replay/tgguard_amd.py) audits every TLB flush, queue and doorbell before it is forwarded, refuses a mode1 reset,
# and holds the TinyGPU.app connection if the plugin goes away with a queue live instead of closing it under the GPU. The
# sessions are recorded into $BEAGLE_TINYGPU_DATA/recordings/<stamp>_amd_cpp_N<n>. At DEBUG=2 (the C++ boot prints tinygrad's
# boot lines), compared with the CPU. With amd_hw_begin's protections and amd_boot_check (env.sh); log stream watched; the Mac
# kept awake; never killed: a test that hangs, or a proxy that holds, is reported and left as it is.
# Plan step A2k: the plugin's crash guard (beagle-tinygpu-guard) keeps the GPU from before the boot; the proxy is ended only once
# the guard has exited, since its connection runs through the proxy. With --kill idle the plugin SIGKILLs itself at fini once the
# GPU is idle (BEAGLE_AMD_TEST_KILL=idle), as a crash would, and the guard finalizes the card: the run then passes if the test
# died of the SIGKILL and the guard's TinyGPULog lines say it saw every queue off and closed.
#   run_amd_cpp_point.sh <state-count> [reps] [--kill idle] [tinygputest args ...]
# Exits 0 only if the test passed (or died of --kill's SIGKILL), both sessions ended clean through the guard, the crash guard
# exited, log stream saw nothing from the eGPU and the boot reset nothing; 1 is a STOP (stop all hardware work), 2 a refusal
# before anything ran, 3 a clean run whose test failed.
source "$(dirname "$0")/env.sh"
USAGE="usage: $0 <state-count> [reps] [--kill idle] [tinygputest args ...]"
[ $# -ge 1 ] || { echo "$USAGE"; exit 2; }
N=$1; REPS=${2:-5}; shift; [ $# -gt 0 ] && shift
KILL=""; if [ "$1" = "--kill" ]; then [ "$2" = idle ] || { echo "$USAGE"; exit 2; }; KILL=$2; shift 2; fi
[ -x "$TEST_BIN" ] || { echo "no $TEST_BIN (BEAGLE_BUILD=$BEAGLE_BUILD); not running"; exit 2; }
amd_hw_begin
amd_boot_check
STAMP=$(date +%Y%m%d-%H%M%S)_$HW_HOST
OUT="$BEAGLE_TINYGPU_DATA/runs/${STAMP}_amd_cpp_N${N}${KILL:+_kill_$KILL}.txt"; LS="${OUT%.txt}_logstream.txt"; PLOG="${OUT%.txt}_proxy.txt"
TLOG="${OUT%.txt}_tinygpulog.txt"   # the plugin's and the crash guard's TinyGPULog lines
REC="$BEAGLE_TINYGPU_DATA/recordings/${STAMP}_amd_cpp_N${N}${KILL:+_kill_$KILL}"
UP="${TMPDIR%/}/tinygpu.sock"; PXD=$(mktemp -d /tmp/tgcp.XXXXXX); PX="$PXD/px.sock"
hw_logstream "$LS"
"$BEAGLE_PYTHON" "$TG_TESTS/replay/tgproxy.py" --listen "$PX" --upstream "$UP" --out "$REC" --guard --start-app --label "amd cpp N$N${KILL:+ kill $KILL}" > "$PLOG" 2>&1 &
PXP=$!
for i in $(seq 100); do grep -q "tgproxy listening" "$PLOG" 2>/dev/null && break; sleep 0.1; done
grep -q "tgproxy listening" "$PLOG" || { echo "the proxy did not start ($PLOG); nothing ran"; kill $PXP 2>/dev/null; exit 2; }
cd "$REPO"
caffeinate -ims env APL_REMOTE_SOCK="$PX" BEAGLE_AMD_PROFILE=1 DEBUG=2 \
    BEAGLE_TINYGPU_LOG="$TLOG" ${KILL:+BEAGLE_AMD_TEST_KILL=$KILL} \
    DYLD_LIBRARY_PATH="$TEST_LIBS" "$TEST_BIN" --state-count "$N" --reps "$REPS" --diag-compare-cpu "$@" > "$OUT" 2>&1 &
TP=$!
rc=""
for i in $(seq 3000); do   # up to 300 s; never killed
    kill -0 $TP 2>/dev/null || { wait $TP; rc=$?; break; }
    grep -q "FAIL-STOP" "$PLOG" && break
    sleep 0.1
done
if grep -q "FAIL-STOP" "$PLOG" || [ -z "$rc" ]; then
    for p in $PXP $TP; do kill -0 $p 2>/dev/null && nohup caffeinate -ims -w $p > /dev/null 2>&1 & done
    if grep -q "FAIL-STOP" "$PLOG"; then echo "STOP: tgproxy's guard fail-stopped and holds the TinyGPU.app connection ($PLOG): unplug the eGPU first, then kill -9 $PXP (and the test, $TP, if it still runs)"
    else echo "STOP: the test is still running after 300 s ($OUT; pid $TP, the proxy $PXP): not killed; unplug the eGPU first if it must be stopped"; fi
    grep -E "FAIL-STOP|ended:" "$PLOG" | cut -c1-300 | tail -3
    hw_hold_check
    amd_hw_end "$OUT"; exit 1
fi
# the crash guard: it exits once the plugin is done with the GPU, or once its own fini saw every queue off; one that holds keeps
# the connection, and so the proxy too, until the eGPU is unplugged
for i in $(seq 600); do pgrep -f "$GUARD_RE" > /dev/null || break; grep -q "HOLDING" "$TLOG" 2>/dev/null && break; sleep 0.1; done
if pgrep -f "$GUARD_RE" > /dev/null; then
    nohup caffeinate -ims -w $PXP > /dev/null 2>&1 &
    grep -E "guard|HOLDING" "$TLOG" | tail -4 | cut -c1-300
    hw_hold_check
    echo "STOP: the crash guard holds the TinyGPU.app connection through the proxy (pid $PXP): unplug the eGPU first, then kill the guard and the proxy"
    amd_hw_end "$OUT"; exit 1
fi
kill -TERM $PXP; for i in $(seq 100); do kill -0 $PXP 2>/dev/null || break; sleep 0.1; done
amd_hw_end "$OUT"; hw=$?
rmdir "$PXD" 2>/dev/null
echo "N=$N reps=$REPS${KILL:+ kill=$KILL} $* exit=$rc output=$OUT recording=$REC"
grep -E "Rsrc Name|TinyGPU/AMD: (C\+\+ boot|C\+\+ runtime|.*failed|.*crash guard)|^PASS|^FAIL|CPU-reference logL|per evaluation|repeats|rror" "$OUT" \
    | grep -v "^  \[" | cut -c1-160 | head -20
echo "boot: $(grep -oE "^am [^:]*: AM_[A-Z0-9]+ initialized" "$OUT" | sed -E 's/.*AM_([A-Z0-9]+) .*/\1/' | xargs)"
grep -E "guard [0-9]+: (the plugin|the GPU is|no queue)|BEAGLE_AMD_TEST_KILL" "$TLOG" | cut -c1-240
grep -E "ended:|session: the AMD card|recording ended" "$PLOG" | cut -c1-240
kill -0 $PXP 2>/dev/null && { echo "STOP: tgproxy did not end (pid $PXP): it may hold the GPU; unplug the eGPU first, then kill -9 $PXP"; exit 1; }
[ $hw -eq 0 ] || exit 1
[ "$(grep -c "ended: eof" "$PLOG")" -ge 2 ] || { echo "FAIL: the sessions did not both end clean through the guard ($PLOG)"; exit 3; }
if [ -n "$KILL" ]; then
    [ "$rc" -eq 137 ] || { echo "FAIL: the test did not die of the SIGKILL (exit $rc); log stream clean"; exit 3; }
    grep -q "the GPU is finalized, every queue off; closing" "$TLOG" || { echo "FAIL: the crash guard did not finalize the GPU ($TLOG); log stream clean"; exit 3; }
    echo "OK: killed $KILL; the crash guard finalized the GPU, every queue off, both sessions clean through the proxy's guard, log stream clean"
    exit 0
fi
grep -q "the plugin finalized the GPU itself; exiting" "$TLOG" || { echo "FAIL: the crash guard did not take the plugin's clean ($TLOG)"; exit 3; }
[ $rc -eq 0 ] || { echo "FAIL: the test failed (exit $rc); log stream clean"; exit 3; }
echo "OK: PASS with the C++ boot through the guard, the crash guard exited at the plugin's clean, log stream clean"
