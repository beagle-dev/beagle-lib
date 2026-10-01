#!/bin/bash
# TODO.md plan step A0: capture the AMD card's IP discovery table (amd_discovery.py, stock tinygrad at DEBUG=2) into
# $BEAGLE_TINYGPU_DATA/discovery/, for A2's AM mock and amd_boot_check. With run_amd_smoke.sh's protections; the first capture
# for a card has no table to predict its boot from.
#   run_amd_discovery.sh
# Exits 0 only if the table was captured, log stream saw nothing from the eGPU and tinygrad reset nothing; 1 is a STOP, 2 a
# refusal before anything ran, 3 a clean run without a capture.
source "$(dirname "$0")/env.sh"
amd_require_app_zip
amd_hw_begin
amd_boot_check
STAMP=$(date +%Y%m%d-%H%M%S)_$HW_HOST; OUT="$BEAGLE_TINYGPU_DATA/runs/${STAMP}_amd_discovery.txt"; LS="${OUT%.txt}_logstream.txt"
hw_logstream "$LS"
caffeinate -ims env DEV=AMD DEBUG=2 AM_DEBUG=1 PYTHONPATH="$TINYGRAD_PATH" "$BEAGLE_PYTHON" "$TG_TESTS/amd_discovery.py" \
    "$BEAGLE_TINYGPU_DATA/discovery" "$AMD_PCI" > "$OUT" 2>&1
rc=$?
amd_hw_end "$OUT"; hw=$?
echo "exit=$rc output=$OUT"
grep -E "^captured:|^ip_ver:|^vram_size:|am .*(boot|Malformed|reset|initialized|Finalizing)|Traceback|rror" "$OUT" | cut -c1-260 | head -30
[ $hw -eq 0 ] || exit 1
grep -q "^captured:" "$OUT" || { echo "FAIL: no capture (exit $rc); log stream clean"; exit 3; }
echo "OK: captured, log stream clean"
