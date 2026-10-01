#!/bin/bash
# Offline golden tests: each C++ port against the tinygrad (hcq1) code it follows, byte for byte, and the oracle's own
# checks (tinygpu_tests/oracle: tinygrad plus BEAGLE's patches). No GPU, no socket to TinyGPU.app. Needs the generated kernels header and embedded cubins (TinyGPUKernels
# and TinyGPUCubins build targets),
# the pinned tinygrad at $TINYGRAD_PATH, and cached cubins in $BEAGLE_TINYGPU_DATA/cubins (compiled once with
# ptxas through nv_compile_helper.compile_ptx if missing). The AMD goldens (plan step A1) need tinygrad's comgr build at
# /opt/homebrew/lib/libamd_comgr.dylib for golden_amd_program's offline HSACO compiles (STATUS.md R64).
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
export BEAGLE_TINYGPU_NO_DOWNLOAD=1   # no golden downloads firmware (test_c4_firmware.py turns it on against a file:// mirror)
fail=0
for t in golden_encode golden_runtime golden_program golden_transport test_c1_cubins test_p1_diagnostics test_p2_teardown test_b1_cot test_c2_tables golden_gsp test_c4_firmware golden_mm golden_rm golden_gsp_hw golden_flcn_hw golden_boot \
         golden_amd_encode golden_amd_copy golden_amd_program golden_amd_handoff golden_amd_hsaco test_a2b_tables golden_amd_boot; do
    echo "== $t"
    "$BEAGLE_PYTHON" "$TG_TESTS/$t.py" > "$TINYGPU_TEST_WORK/$t.log" 2>&1
    rc=$?
    grep -vE "launch-dims fill|\[profile\]" "$TINYGPU_TEST_WORK/$t.log" | tail -8
    [ $rc -eq 0 ] || { echo "   FAILED (exit $rc; log: $TINYGPU_TEST_WORK/$t.log)"; fail=1; }
done
exit $fail
