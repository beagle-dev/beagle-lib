#!/bin/bash
# HARDWARE, TODO.md plan step N3 (run at N5): a new AMD card's IP discovery table into $BEAGLE_TINYGPU_DATA/discovery/, with only
# the requests the pin's own boot makes before it needs the table (amd_discovery_ro.py: RESIZE_BAR, the LNKCTL write, the BAR
# maps, the IOV and MEMSIZE reads, the indirect window's index writes and data reads, then three config reads; test_n3.sh
# checks it offline). No firmware, PSP, SMU or boot; no amd_boot_check (there is no table to predict from yet) and no proxy
# (its AMD guard is the RX 7900 XT's until plan step N6). With amd_hw_begin's protections, log stream and caffeinate.
#   run_amd_discovery_ro.sh [out dir, default $BEAGLE_TINYGPU_DATA/discovery]
# Exits 0 only if the table was captured and log stream saw nothing from the eGPU; 1 is a STOP (stop all hardware work: a
# log-stream event, or the capture's own stop on MEMSIZE, BAR0 or a failed parse, whose raw bytes it keeps); 2 a refusal
# before anything was sent; 3 a clean run without a capture.
source "$(dirname "$0")/env.sh"
amd_require_app_zip
amd_hw_begin
STAMP=$(date +%Y%m%d-%H%M%S)_$HW_HOST; OUT="$BEAGLE_TINYGPU_DATA/runs/${STAMP}_amd_discovery_ro.txt"; LS="${OUT%.txt}_logstream.txt"
hw_logstream "$LS"
caffeinate -ims "$BEAGLE_PYTHON" "$TG_TESTS/amd_discovery_ro.py" "${1:-$BEAGLE_TINYGPU_DATA/discovery}" > "$OUT" 2>&1
rc=$?
amd_hw_end "$OUT"; hw=$?
echo "exit=$rc output=$OUT"
grep -E "^captured:|^ip_ver:|^pci |^STOP|nothing sent|Traceback|rror" "$OUT" | cut -c1-260 | head -20
[ $hw -eq 0 ] || exit 1
[ $rc -eq 1 ] && { echo "STOP: the capture stopped (above); log stream clean"; exit 1; }
[ $rc -eq 2 ] && { echo "nothing was sent (above)"; exit 2; }
grep -q "^captured:" "$OUT" || { echo "FAIL: no capture (exit $rc); log stream clean"; exit 3; }
echo "OK: captured, log stream clean"
