#!/usr/bin/env python3
"""
amd_real_kernel_test.py — dispatch one real, unmodified BEAGLE GPU kernel
(kernelSumSites1) compiled from the actual production kernel source
(kernelsAll.cu, via GPUImplDefs.h's new FW_TINYGPU_HYBRID_AMD branch),
compiled via comgr's HIP language (amd_compile_helper.compile_hip -- see
STATUS.md AMD §18 for why HIP instead of OpenCL), dispatched through the
same BeagleAMDProgram/HCQProgram path the daemon uses.

kernelSumSites1(dArray, dSum, dPatternWeights, patternCount): one local-
memory tree-reduction kernel, one workgroup of SUM_SITES_BLOCK_SIZE (128)
threads computes dSum[0] = sum(dArray[i] * dPatternWeights[i] for i in
range(patternCount)). Trivial to check: dArray[i]=1, dPatternWeights[i]=1,
patternCount=128 -> dSum[0] should be exactly 128.0.

Bridges the gap between the earlier hand-written trivial kernels (proven
safe) and the full ~80-kernel BEAGLE pipeline through the C++ daemon: same
dispatch code, but a real, multi-kernel-image compile and a kernel that
actually exercises KW_LOCAL_MEM/KW_LOCAL_FENCE (shared memory + barrier).

Requires libhmsbeagle/GPU/kernels/BeagleOpenCL_kernels.h to be up to date
(regenerate via `bash kernels/make_opencl_kernels.sh` from the kernels/
directory if GPUImplDefs.h or the kernel .cu files changed).

    python3 amd_real_kernel_test.py
"""
import sys, os, pathlib, struct, ctypes

_TINYGRAD_PATH = os.environ.get("TINYGRAD_PATH", str(pathlib.Path.home() / "Dropbox/Projects/tinygrad"))
sys.path.insert(0, _TINYGRAD_PATH)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tinygrad.helpers import DEV
from tinygrad.runtime.support.hcq import HCQBuffer

import amd_compile_helper as ach
from amd_dispatch_daemon import BeagleAMDProgram

KERNEL_NAME = "kernelSumSites1"
BLOCK_SIZE = 128  # SUM_SITES_BLOCK_SIZE (GPUImplDefs.h)


def log(msg):
    print(f"[amd_real_kernel_test] {msg}", file=sys.stderr, flush=True)


def extract_kernel_source():
    """Stringify KERNELS_STRING_SP_4 out of the generated header via a tiny
    C program -- the same thing the real C++ backend does at a different
    layer (BeagleOpenCL_kernels.h is a real C string; the host reads the
    already-preprocessed macro value, we just need the raw text)."""
    import subprocess, tempfile
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    src_c = '#include <stdio.h>\n#include "libhmsbeagle/GPU/kernels/BeagleOpenCL_kernels.h"\n' \
            'int main(){fputs(KERNELS_STRING_SP_4, stdout); return 0;}\n'
    with tempfile.TemporaryDirectory() as d:
        cpath = os.path.join(d, "extract.c")
        bpath = os.path.join(d, "extract")
        with open(cpath, "w") as f:
            f.write(src_c)
        subprocess.run(["clang", "-I", str(repo_root), cpath, "-o", bpath], check=True)
        out = subprocess.run([bpath], check=True, capture_output=True)
        return out.stdout.decode()


def main():
    os.makedirs(os.path.expanduser("~/Library/Logs"), exist_ok=True)
    fd = os.open(os.path.expanduser("~/Library/Logs/amd_real_kernel_test.log"),
                 os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_SYNC, 0o644)
    sys.stderr = os.fdopen(fd, 'w', buffering=1)

    log("starting")
    DEV.value = "AMD"
    from tinygrad import Device
    dev = Device["AMD:0"]
    log(f"booted -- {dev}, arch={dev.arch}")

    src = extract_kernel_source()
    log(f"extracted kernel source -- {len(src)} bytes")

    hsaco = ach.compile_hip(src, dev.arch)
    log(f"compiled -- {len(hsaco)} byte HSACO")

    image, kernels = ach.parse_kernels(hsaco)
    log(f"{len(kernels)} kernels found")
    assert KERNEL_NAME in kernels, f"{KERNEL_NAME} not found in {sorted(kernels)}"
    kd_addr, desc = kernels[KERNEL_NAME]
    log(f"{KERNEL_NAME}: kd_addr={kd_addr:#x} kernarg_size={desc.kernarg_size} "
        f"group_segment={desc.group_segment_fixed_size} private_segment={desc.private_segment_fixed_size} "
        f"kernel_code_properties={desc.kernel_code_properties:#x}")

    prg = BeagleAMDProgram(dev, KERNEL_NAME, hsaco, kd_addr, n_int_args=1)
    log("BeagleAMDProgram constructed")

    n = BLOCK_SIZE
    d_array = dev.allocator.alloc(n * 4)
    d_sum = dev.allocator.alloc(4)
    d_weights = dev.allocator.alloc(n * 4)
    log(f"buffers allocated: dArray={d_array.va_addr:#x} dSum={d_sum.va_addr:#x} dWeights={d_weights.va_addr:#x}")

    ones = struct.pack(f"<{n}f", *([1.0] * n))
    dev.allocator._copyin(HCQBuffer(d_array.va_addr, n * 4), memoryview(bytearray(ones)))
    dev.allocator._copyin(HCQBuffer(d_weights.va_addr, n * 4), memoryview(bytearray(ones)))
    dev.synchronize()
    log("input buffers uploaded")

    # global_size here is groups (blocks), not total work-items (matches
    # HCQProgram's own convention: exec() computes grid_size = global_size *
    # local_size) -- one group of BLOCK_SIZE threads, matching the kernel's
    # single-workgroup reduction.
    prg(HCQBuffer(d_array.va_addr, n * 4), HCQBuffer(d_sum.va_addr, 4), HCQBuffer(d_weights.va_addr, n * 4),
        global_size=(1, 1, 1), local_size=(BLOCK_SIZE, 1, 1), vals=(n,), wait=False)
    log("kernel launched, syncing...")
    dev.synchronize()
    log("synchronized -- no fault")

    out = memoryview(bytearray(4))
    dev.allocator._copyout(out, HCQBuffer(d_sum.va_addr, 4))
    val = struct.unpack("<f", bytes(out))[0]
    log(f"readback dSum[0] = {val}")
    print(f"RESULT: {val}", file=sys.stdout, flush=True)
    if val == float(n):
        log(f"SUCCESS: value matches expected {n}.0")
    else:
        log(f"MISMATCH: expected {n}.0, got {val}")
    log("exiting cleanly")


if __name__ == "__main__":
    main()
