#!/bin/bash
# Offline golden tests: each C++ port against the tinygrad (hcq1) code it follows, byte for byte, and the oracle's own
# checks (tinygpu_tests/oracle: tinygrad plus BEAGLE's patches). No GPU, no socket to TinyGPU.app. Needs the generated kernels header and embedded cubins (TinyGPUKernels
# and TinyGPUCubins build targets),
# the pinned tinygrad at $TINYGRAD_PATH, and cached cubins in $BEAGLE_TINYGPU_DATA/cubins (compiled once with
# ptxas through nv_compile_helper.compile_ptx if missing). The AMD goldens (plan step A1) need tinygrad's comgr build at
# /opt/homebrew/lib/libamd_comgr.dylib for golden_amd_program's offline HSACO compiles (STATUS.md R64).
# The goldens run at the same time (env.sh pjob, longest first; golden_mm and golden_amd_hsaco in their parts), each with
# its output in $TINYGPU_TEST_WORK/<golden>.log; their summaries follow in the order they always had.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
export BEAGLE_TINYGPU_NO_DOWNLOAD=1   # no golden downloads firmware (test_c4_firmware.py turns it on against a file:// mirror)
W=$TINYGPU_TEST_WORK
golden() {   # <log> <golden> [args ...]: $TG_TESTS/<golden>.py's output in $W/<log>.log
    local log=$1 t=$2; shift 2
    "$BEAGLE_PYTHON" "$TG_TESTS/$t.py" "$@" > "$W/$log.log" 2>&1
}
# TODO.md plan step N12: the AM boot's goldens on the RX 9070 XT's fake too (gfx12)
amd_boot_gfx1201() { FAKE_AMD_CHIP=gfx1201 golden golden_amd_boot_gfx1201 golden_amd_boot; }
MM_PARTS=5   # golden_mm.py's PARTS
HS_PARTS=3   # golden_amd_hsaco.py's ARCHS
parts() {   # <golden> <K>: its K parts' output, in order, in $W/<golden>.log; prints their failures (a part exits with its count)
    local n=0 p
    for p in $(seq 1 $2); do cat "$W/$1.$p.log"; n=$((n + $(pjob_rc $1_$p))); done > "$W/$1.log"
    echo $n
}
pjob_init "$W"
pjob golden_boot golden golden_boot golden_boot
pjob golden_amd_boot golden golden_amd_boot golden_amd_boot
pjob golden_amd_boot_gfx1201 amd_boot_gfx1201
for t in golden_amd_program golden_gsp golden_gsp_hw; do pjob $t golden $t $t; done
for p in $(seq 1 $HS_PARTS); do pjob golden_amd_hsaco_$p golden golden_amd_hsaco.$p golden_amd_hsaco --part $p/$HS_PARTS; done
for p in $(seq 1 $MM_PARTS); do pjob golden_mm_$p golden golden_mm.$p golden_mm --part $p/$MM_PARTS; done
for t in test_a2b_tables test_b1_cot golden_flcn_hw golden_rm test_c2_tables golden_program test_c4_firmware test_p2_teardown \
         test_p1_diagnostics test_c1_cubins golden_amd_encode golden_encode golden_runtime golden_transport golden_amd_handoff golden_amd_copy; do
    pjob $t golden $t $t
done
pwait
# the parted goldens' logs: their parts' output, then the verdict each prints when run whole
mm=$(parts golden_mm $MM_PARTS)
echo "C6 memory manager vs tinygrad: $([ $mm -eq 0 ] && echo "all identical" || echo "$mm FAILED")" >> "$W/golden_mm.log"
hs=$(parts golden_amd_hsaco $HS_PARTS)
echo "A1j build-time HSACOs vs tinygrad's compile_hip: $([ $hs -eq 0 ] && echo "all identical" || echo MISMATCH)" >> "$W/golden_amd_hsaco.log"
fail=0
for t in golden_encode golden_runtime golden_program golden_transport test_c1_cubins test_p1_diagnostics test_p2_teardown test_b1_cot test_c2_tables golden_gsp test_c4_firmware golden_mm golden_rm golden_gsp_hw golden_flcn_hw golden_boot \
         golden_amd_encode golden_amd_copy golden_amd_program golden_amd_handoff golden_amd_hsaco test_a2b_tables golden_amd_boot; do
    echo "== $t"
    grep -vE "launch-dims fill|\[profile\]" "$W/$t.log" | tail -8
    case $t in golden_mm) rc=$((mm > 0)) ;; golden_amd_hsaco) rc=$((hs > 0)) ;; *) rc=$(pjob_rc $t) ;; esac
    [ $rc -eq 0 ] || { echo "   FAILED (exit $rc; log: $W/$t.log)"; fail=1; }
done
echo "== golden_amd_boot (FAKE_AMD_CHIP=gfx1201)"
tail -8 "$W/golden_amd_boot_gfx1201.log"
rc=$(pjob_rc golden_amd_boot_gfx1201)
[ $rc -eq 0 ] || { echo "   FAILED (exit $rc; log: $W/golden_amd_boot_gfx1201.log)"; fail=1; }
exit $fail
