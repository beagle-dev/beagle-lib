#!/bin/bash
# TODO.md plan steps A0 and N10c: stock tinygrad (the pin) on the AMD eGPU at DEBUG=2: tinygrad's own boot and kernels, no
# BEAGLE code. (Tensor([1,2,3])+1).tolist(), then an 8 x 32 x 64 sum of three host vectors, exact in float32, twice: with NOOPT
# (a 64 x 32 x 8 grid of one-lane workgroups, so the Y and Z workgroup ids, which gfx12 reads from ttmp7, as BEAGLE's launches
# on Y and Z need) and with tinygrad's default optimizations (local ids in X and Y). Since N10c through tgproxy.py --guard
# --card (APL_REMOTE_SOCK; the AMD guard from the card's captured table, armed before the first request), recorded into
# $BEAGLE_TINYGPU_DATA/recordings/<stamp>_amd_smoke_<label>. With amd_hw_begin's protections, amd_boot_check and
# amd_require_app_zip (env.sh); log stream watched; the Mac kept awake; never killed.
#   run_amd_smoke.sh [label]
# Exits 0 only if every result is exact, the session ended clean through the proxy, log stream saw nothing from the eGPU,
# tinygrad reset nothing and took the predicted boot; 1 is a STOP (stop all hardware work: the proxy may hold the GPU), 2 a
# refusal before anything ran, 3 a clean run with a wrong result.
source "$(dirname "$0")/env.sh"
LABEL=${1:-smoke}
amd_require_app_zip
amd_hw_begin
amd_boot_check
STAMP=$(date +%Y%m%d-%H%M%S)_$HW_HOST
REC="$BEAGLE_TINYGPU_DATA/recordings/${STAMP}_amd_smoke_$LABEL"; OUT="$BEAGLE_TINYGPU_DATA/runs/${STAMP}_amd_$LABEL.txt"
LS="${OUT%.txt}_logstream.txt"; PLOG="${OUT%.txt}_proxy.txt"
UP="${TMPDIR%/}/tinygpu.sock"; PXD=$(mktemp -d /tmp/tgsm.XXXXXX); PX="$PXD/px.sock"
hw_logstream "$LS"
"$BEAGLE_PYTHON" "$TG_TESTS/replay/tgproxy.py" --listen "$PX" --upstream "$UP" --out "$REC" --guard --start-app --label "amd smoke $LABEL" \
    --card "$AMD_PCI" > "$PLOG" 2>&1 &   # stock tinygrad never reads config dword 0: the proxy is told the card
PXP=$!
for i in $(seq 100); do grep -q "tgproxy listening" "$PLOG" 2>/dev/null && break; sleep 0.1; done
grep -q "tgproxy listening" "$PLOG" || { echo "the proxy did not start ($PLOG); nothing ran"; kill $PXP 2>/dev/null; exit 2; }
caffeinate -ims env DEV=AMD DEBUG=2 AM_DEBUG=1 APL_REMOTE_SOCK="$PX" PYTHONPATH="$TINYGRAD_PATH" TG_TESTS="$TG_TESTS" "$BEAGLE_PYTHON" -c '
import os, sys
sys.path.append(os.environ["TG_TESTS"])
import tgpaths
tgpaths.block_network()   # TODO.md plan step N6: the boot takes its firmware from tinygrad'"'"'s cache, never a download
import tinygrad, time
print("tinygrad:", tinygrad.__file__, flush=True)
from tinygrad import Tensor, Device
from tinygrad.helpers import Context
t0 = time.perf_counter()
d = Device["AMD"]
print("device:", d, "arch:", getattr(d, "arch", "?"), f"opened in {time.perf_counter()-t0:.1f} s", flush=True)
print("result:", (Tensor([1.0, 2.0, 3.0], device="AMD") + 1).tolist(), flush=True)
X, Y, Z = 8, 32, 64
want = [[[i * 10000.0 + j * 100.0 + k for k in range(Z)] for j in range(Y)] for i in range(X)]
def grid():
    v = lambda n: Tensor([float(i) for i in range(n)], device="AMD")
    return v(X).reshape(X, 1, 1) * 10000 + v(Y).reshape(1, Y, 1) * 100 + v(Z).reshape(1, 1, Z)
with Context(NOOPT=1): g = grid().tolist()
print("grid noopt:", "exact" if g == want else "WRONG", flush=True)
print("grid opt:", "exact" if grid().tolist() == want else "WRONG", flush=True)
' > "$OUT" 2>&1
rc=$?
if grep -q "FAIL-STOP" "$PLOG"; then   # the guard holds the GPU: nothing may close its connection
    nohup caffeinate -ims -w $PXP > /dev/null 2>&1 &
    echo "STOP: tgproxy fail-stopped and holds the TinyGPU.app connection ($PLOG): unplug the eGPU first, then kill -9 $PXP"
    grep -E "FAIL-STOP|session 1 ended" "$PLOG" | cut -c1-300
    amd_hw_end "$OUT"; exit 1
fi
kill -TERM $PXP; for i in $(seq 100); do kill -0 $PXP 2>/dev/null || break; sleep 0.1; done
amd_hw_end "$OUT"; hw=$?
rmdir "$PXD" 2>/dev/null
echo "exit=$rc output=$OUT recording=$REC"
grep -E "^tinygrad:|^device:|^result:|^grid |am .*(boot|Malformed|reset|initialized)|Traceback|rror" "$OUT" | cut -c1-200 | head -30
grep -E "session 1 ended|recording ended|session: the AMD card" "$PLOG" | cut -c1-300
kill -0 $PXP 2>/dev/null && { echo "STOP: tgproxy did not end (pid $PXP): it may hold the GPU; unplug the eGPU first, then kill -9 $PXP"; exit 1; }
[ $hw -eq 0 ] || exit 1
grep -q "session: the AMD card ($AMD_PCI, " "$PLOG" && grep -q "session 1 ended: eof" "$PLOG" \
    || { echo "FAIL: the session did not end clean through the proxy's AMD guard (exit $rc); log stream clean"; exit 3; }
grep -q "^result: \[2.0, 3.0, 4.0\]" "$OUT" && grep -q "^grid noopt: exact" "$OUT" && grep -q "^grid opt: exact" "$OUT" \
    || { echo "FAIL: not the expected results (exit $rc); log stream clean"; exit 3; }
echo "OK: PASS, the session clean through the proxy, log stream clean"
