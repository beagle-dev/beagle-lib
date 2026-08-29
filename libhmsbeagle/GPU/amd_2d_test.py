#!/usr/bin/env python3
"""
amd_2d_test.py — kernelMatrixMulADB (STATUS.md AMD §22-23) is the first
real BEAGLE kernel to use the Y work-item dimension (KW_LOCAL_ID_1/
KW_GROUP_ID_1/KW_NUM_GROUPS_1, via a 16x16 2D block) -- every kernel tested
on hardware so far (kernelSumSites1, the math-function test) only ever used
dimension 0. exp()/cos()/sin()/log() are confirmed correct (§23) and the
kernel's real inputs are confirmed valid (§23 debug-dump), so the next
untested variable is whether __ockl_get_local_id(1)/__ockl_get_group_id(1)/
__ockl_get_num_groups(1) -- and the KW_*_1 macros built on them -- actually
report correct per-thread values for a 2D block.

Launches a trivial kernel with a 2D block (block=(4,4,1), one group) that
writes each thread's (KW_LOCAL_ID_0, KW_LOCAL_ID_1, KW_GROUP_ID_0,
KW_NUM_GROUPS_1) into a unique output slot (index = ty*4+tx, matching
kernelMatrixMulADB's own C[PADDED_STATE_COUNT*ty+tx] indexing pattern
exactly) -- if every one of the 16 threads' reported (tx,ty) matches its
actual dispatch position, dimension-1 queries are cleared as a variable.

    python3 amd_2d_test.py
"""
import sys, os, pathlib, struct

_TINYGRAD_PATH = os.environ.get("TINYGRAD_PATH", str(pathlib.Path.home() / "Dropbox/Projects/tinygrad"))
sys.path.insert(0, _TINYGRAD_PATH)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tinygrad.helpers import DEV
from tinygrad.runtime.support.hcq import HCQBuffer

import amd_compile_helper as ach
from amd_dispatch_daemon import BeagleAMDProgram
import amd_hcq_patch

BLOCK = 4  # 4x4 block, matching kernelMatrixMulADB's real PADDED_STATE_COUNT=4 EDGE

KERNEL_SRC_TAIL = f'''
KW_GLOBAL_KERNEL void dims2d(KW_GLOBAL_VAR int* out) {{
    int tx = KW_LOCAL_ID_0;
    int ty = KW_LOCAL_ID_1;
    int gx = KW_GROUP_ID_0;
    int ngy = KW_NUM_GROUPS_1;
    int idx = ({BLOCK} * ty + tx) * 4;
    out[idx + 0] = tx;
    out[idx + 1] = ty;
    out[idx + 2] = gx;
    out[idx + 3] = ngy;
}}
'''


def log(msg):
    print(f"[amd_2d_test] {msg}", file=sys.stderr, flush=True)


def main():
    os.makedirs(os.path.expanduser("~/Library/Logs"), exist_ok=True)
    fd = os.open(os.path.expanduser("~/Library/Logs/amd_2d_test.log"),
                 os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_SYNC, 0o644)
    sys.stderr = os.fdopen(fd, 'w', buffering=1)

    log("starting")
    amd_hcq_patch.set_logger(log)
    amd_hcq_patch.apply()
    log("amd_hcq_patch applied (hidden-kernel-args fix)")
    DEV.value = "AMD"
    from tinygrad import Device
    dev = Device["AMD:0"]
    log(f"booted -- {dev}, arch={dev.arch}")

    impl_defs = pathlib.Path(__file__).with_name("GPUImplDefs.h").read_text()
    src = "#define STATE_COUNT 4\n" + impl_defs + KERNEL_SRC_TAIL
    hsaco = ach.compile_hip(src, dev.arch)
    log(f"compiled -- {len(hsaco)} byte HSACO")

    image, kernels = ach.parse_kernels(hsaco)
    log(f"kernels found: {list(kernels)}")
    assert "dims2d" in kernels, f"dims2d not found in {sorted(kernels)}"
    kd_addr, desc = kernels["dims2d"]
    log(f"kd_addr={kd_addr:#x} kernarg_size={desc.kernarg_size} "
        f"group_segment={desc.group_segment_fixed_size} kernel_code_properties={desc.kernel_code_properties:#x}")

    prg = BeagleAMDProgram(dev, "dims2d", hsaco, kd_addr, n_int_args=0)
    log("BeagleAMDProgram constructed")

    n_threads = BLOCK * BLOCK
    d_out = dev.allocator.alloc(n_threads * 4 * 4)  # 4 ints per thread
    log(f"buffer allocated: dOut={d_out.va_addr:#x}")

    # global_size = groups (1 group in each dim); local_size = the 2D block.
    prg(HCQBuffer(d_out.va_addr, n_threads * 4 * 4),
        global_size=(1, 1, 1), local_size=(BLOCK, BLOCK, 1), vals=(), wait=False)
    log("kernel launched, syncing...")
    dev.synchronize()
    log("synchronized -- no fault")

    out = memoryview(bytearray(n_threads * 4 * 4))
    dev.allocator._copyout(out, HCQBuffer(d_out.va_addr, n_threads * 4 * 4))
    vals = struct.unpack(f"<{n_threads * 4}i", bytes(out))

    all_ok = True
    for ty in range(BLOCK):
        for tx in range(BLOCK):
            idx = (BLOCK * ty + tx) * 4
            got_tx, got_ty, got_gx, got_ngy = vals[idx:idx + 4]
            ok = (got_tx == tx and got_ty == ty and got_gx == 0 and got_ngy == 1)
            line = f"expected (tx={tx},ty={ty}) -> got (tx={got_tx},ty={got_ty},gx={got_gx},ngy={got_ngy}) {'OK' if ok else 'MISMATCH'}"
            log(line)
            print(line, file=sys.stdout, flush=True)
            if not ok:
                all_ok = False

    print(f"RESULT: {'ALL_OK' if all_ok else 'MISMATCH'}", file=sys.stdout, flush=True)
    log("SUCCESS: all 16 threads report correct 2D indices" if all_ok else "FAILURE: mismatches found, see above")
    log("exiting cleanly")


if __name__ == "__main__":
    main()
