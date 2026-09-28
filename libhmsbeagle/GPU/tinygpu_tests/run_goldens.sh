#!/bin/bash
# Offline golden tests: each C++ port against the tinygrad (hcq1) code it follows, byte for byte, plus the daemon's
# wire protocol. No GPU, no socket to TinyGPU.app. Needs the generated kernels header and embedded cubins (TinyGPUKernels
# and TinyGPUCubins build targets),
# the pinned tinygrad at $TINYGRAD_PATH, and cached cubins in $BEAGLE_TINYGPU_DATA/cubins (compiled once with
# ptxas through nv_compile_helper.compile_ptx if missing).
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
fail=0
for t in golden_encode golden_runtime golden_program golden_transport test_c1_cubins test_daemon_wire test_p1_diagnostics test_p2_teardown test_p3 test_b1_cot test_c2_tables golden_gsp test_c5 test_c4_firmware golden_mm test_c6 golden_rm test_c7 golden_gsp_hw test_c8 golden_flcn_hw test_c9 test_c10 golden_boot; do
    echo "== $t"
    "$BEAGLE_PYTHON" "$TG_TESTS/$t.py" > "$TINYGPU_TEST_WORK/$t.log" 2>&1
    rc=$?
    grep -vE "launch-dims fill|\[profile\]" "$TINYGPU_TEST_WORK/$t.log" | tail -8
    [ $rc -eq 0 ] || { echo "   FAILED (exit $rc; log: $TINYGPU_TEST_WORK/$t.log)"; fail=1; }
done
exit $fail
