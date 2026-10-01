#!/bin/bash
# TODO.md plan step A2j: an AMD L0 recording. amd_dispatch_daemon.py's boot-only session on the eGPU, driven as the plugin
# drives it (amd_daemon_session.py --hw: the PCI id read, boot, handoff with the daemon's default pool, fini; no kernels), through
# tgproxy.py --guard (the AMD guard, replay/tgguard_amd.py) into $BEAGLE_TINYGPU_DATA/recordings/<stamp>_amd_l0_<label>. The C++
# boot then replays against it offline (golden_amd_boot --session under tgreplay.py --guard: test_a2j_replay.sh). With
# amd_hw_begin's protections and amd_boot_check (env.sh), log stream watched, the Mac kept awake, never killed.
#   run_amd_l0.sh [label]
# Exits 0 only if the session went through the proxy and ended clean, the daemon exited, log stream saw nothing from the eGPU
# and tinygrad reset nothing; 1 is a STOP (stop all hardware work: the proxy may hold the GPU), 2 a refusal before anything ran.
source "$(dirname "$0")/env.sh"
LABEL=${1:-warm}
amd_hw_begin
amd_boot_check
STAMP=$(date +%Y%m%d-%H%M%S)_$HW_HOST
REC="$BEAGLE_TINYGPU_DATA/recordings/${STAMP}_amd_l0_$LABEL"; OUT="$BEAGLE_TINYGPU_DATA/runs/${STAMP}_amd_l0_$LABEL.txt"
LS="${OUT%.txt}_logstream.txt"; PLOG="${OUT%.txt}_proxy.txt"
UP="${TMPDIR%/}/tinygpu.sock"; PXD=$(mktemp -d /tmp/tgl0.XXXXXX); PX="$PXD/px.sock"
hw_logstream "$LS"
"$BEAGLE_PYTHON" "$TG_TESTS/replay/tgproxy.py" --listen "$PX" --upstream "$UP" --out "$REC" --guard --start-app --label "amd l0 $LABEL" > "$PLOG" 2>&1 &
PXP=$!
for i in $(seq 100); do grep -q "tgproxy listening" "$PLOG" 2>/dev/null && break; sleep 0.1; done
grep -q "tgproxy listening" "$PLOG" || { echo "the proxy did not start ($PLOG); nothing ran"; kill $PXP 2>/dev/null; exit 2; }
caffeinate -ims env DEBUG=2 "$BEAGLE_PYTHON" "$TG_TESTS/amd_daemon_session.py" --hw "$PX" > "$OUT" 2>&1
rc=$?
for i in $(seq 30); do pgrep -f "amd_dispatch_daemon.py" > /dev/null || break; sleep 1; done
if grep -q "FAIL-STOP" "$PLOG"; then   # the guard holds the GPU: nothing may close its connection
    nohup caffeinate -ims -w $PXP > /dev/null 2>&1 &
    echo "STOP: tgproxy fail-stopped and holds the TinyGPU.app connection ($PLOG): unplug the eGPU first, then kill -9 $PXP"
    grep -E "FAIL-STOP|session 1 ended" "$PLOG" | cut -c1-300
    amd_hw_end "$OUT"; exit 1
fi
kill -TERM $PXP; for i in $(seq 100); do kill -0 $PXP 2>/dev/null || break; sleep 0.1; done
amd_hw_end "$OUT"; hw=$?
cp ~/Library/Logs/amd_dispatch_daemon.log "${OUT%.txt}_daemon.log" 2>/dev/null
rmdir "$PXD" 2>/dev/null
echo "label=$LABEL exit=$rc output=$OUT recording=$REC"
grep -E "^session:|Traceback|Error" "$OUT" | cut -c1-200 | head -8
echo "boot: $(grep -oE "^am [^:]*: AM_[A-Z0-9]+ initialized" "$OUT" | sed -E 's/.*AM_([A-Z0-9]+) .*/\1/' | xargs)"
grep -E "session 1 ended|recording ended|session: the AMD card" "$PLOG" | cut -c1-300
pgrep -fl "amd_dispatch_daemon.py" && { echo "STOP: the AMD daemon is still running (it may hold the GPU): unplug the eGPU first, then kill it"; exit 1; }
kill -0 $PXP 2>/dev/null && { echo "STOP: tgproxy did not end (pid $PXP): it may hold the GPU; unplug the eGPU first, then kill -9 $PXP"; exit 1; }
[ $hw -eq 0 ] || exit 1
[ $rc -eq 0 ] && grep -q "session 1 ended: eof" "$PLOG" || { echo "FAIL: the session did not end clean (exit $rc); log stream clean"; exit 3; }
echo "OK: recorded, the session clean, log stream clean"
