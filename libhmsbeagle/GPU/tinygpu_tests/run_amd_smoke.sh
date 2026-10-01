#!/bin/bash
# TODO.md plan step A0: stock tinygrad (the pin) on the AMD eGPU, (Tensor([1,2,3])+1).tolist() at DEBUG=2: tinygrad's own boot
# and one kernel, no BEAGLE code. With amd_hw_begin's protections, amd_boot_check and amd_require_app_zip (env.sh); log stream
# watched; the Mac kept awake; never killed.
#   run_amd_smoke.sh [label]
# Exits 0 only if the result is right, log stream saw nothing from the eGPU and tinygrad reset nothing; 1 is a STOP, 2 a
# refusal before anything ran, 3 a clean run with the wrong result.
source "$(dirname "$0")/env.sh"
LABEL=${1:-smoke}
amd_require_app_zip
amd_hw_begin
amd_boot_check
STAMP=$(date +%Y%m%d-%H%M%S)_$HW_HOST; OUT="$BEAGLE_TINYGPU_DATA/runs/${STAMP}_amd_$LABEL.txt"; LS="${OUT%.txt}_logstream.txt"
hw_logstream "$LS"
caffeinate -ims env DEV=AMD DEBUG=2 AM_DEBUG=1 PYTHONPATH="$TINYGRAD_PATH" "$BEAGLE_PYTHON" -c '
import tinygrad, time
print("tinygrad:", tinygrad.__file__, flush=True)
from tinygrad import Tensor, Device
t0 = time.perf_counter()
d = Device["AMD"]
print("device:", d, "arch:", getattr(d, "arch", "?"), f"opened in {time.perf_counter()-t0:.1f} s", flush=True)
print("result:", (Tensor([1.0, 2.0, 3.0], device="AMD") + 1).tolist(), flush=True)
' > "$OUT" 2>&1
rc=$?
amd_hw_end "$OUT"; hw=$?
echo "exit=$rc output=$OUT"
grep -E "^tinygrad:|^device:|^result:|am .*(boot|Malformed|reset|initialized)|Traceback|rror" "$OUT" | cut -c1-200 | head -30
[ $hw -eq 0 ] || exit 1
grep -q "^result: \[2.0, 3.0, 4.0\]" "$OUT" || { echo "FAIL: not the expected result (exit $rc); log stream clean"; exit 3; }
echo "OK: PASS, log stream clean"
