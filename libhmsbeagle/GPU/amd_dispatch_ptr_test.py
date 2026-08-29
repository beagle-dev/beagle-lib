#!/usr/bin/env python3
"""
amd_dispatch_ptr_test.py — companion to amd_hip_compile_test.py /
amd_opencl_compile_test.py.

Both of those ran clean (STATUS.md AMD §13-14): dispatch mechanism,
BeagleAMDProgram, no-daemon invocation, and OpenCL-language compilation
itself are all now cleared as variables for a *trivial* kernel. But both
trivial kernels compiled to enable_dispatch_ptr=False, private_segment_
fixed_size=0 -- every real BEAGLE kernel calls get_global_id/get_local_id,
which flips enable_dispatch_ptr=True and (per an offline compile check)
also pulls in a nonzero private/scratch segment. That's BeagleAMDProgram's
additional_alloc_sz / extra kernarg-dispatch-packet branch and the
_ensure_has_local_memory(nonzero) / TMPRING_SIZE path -- neither exercised
by either prior test, both universal to every real BEAGLE kernel.

Same recipe as the two prior scripts (OpenCL-language compile via
amd_compile_helper, same BeagleAMDProgram dispatch, direct `python3`
invocation, no daemon) with one change: the kernel body calls
get_global_id(0) and launches over 4 work-items, so it must go through the
dispatch-ptr and scratch-segment branches for real.

    python3 amd_dispatch_ptr_test.py
"""
import sys, os, pathlib, struct, faulthandler

_TINYGRAD_PATH = os.environ.get("TINYGRAD_PATH", str(pathlib.Path.home() / "Dropbox/Projects/tinygrad"))
sys.path.insert(0, _TINYGRAD_PATH)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tinygrad.helpers import DEV
from tinygrad.runtime.support.hcq import HCQBuffer

import amd_compile_helper as ach
from amd_dispatch_daemon import BeagleAMDProgram
import amd_hcq_patch

OPENCL_SRC = '''
kernel void foo(global float* x) {
  x[get_global_id(0)] = 42.0f;
}
'''

N = 4  # work-items


def log(msg):
    print(f"[amd_dptr_test] {msg}", file=sys.stderr, flush=True)


def main():
    os.makedirs(os.path.expanduser("~/Library/Logs"), exist_ok=True)
    fd = os.open(os.path.expanduser("~/Library/Logs/amd_dispatch_ptr_test.log"),
                 os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_SYNC, 0o644)
    sys.stderr = os.fdopen(fd, 'w', buffering=1)
    faulthandler.enable(file=sys.stderr, all_threads=True)

    log("starting")
    amd_hcq_patch.set_logger(log)
    amd_hcq_patch.apply()
    log("amd_hcq_patch applied (AMDComputeQueue.exec dispatch-ptr fix)")
    DEV.value = "AMD"
    from tinygrad import Device
    dev = Device["AMD:0"]
    log(f"booted -- {dev}, arch={dev.arch}")

    comgr, C = ach._load_comgr()
    hsaco = ach.compile_opencl(comgr, C, OPENCL_SRC, dev.arch)
    log(f"compiled OpenCL kernel -- {len(hsaco)} bytes")

    image, kernels = ach.parse_kernels(hsaco)
    log(f"kernels found: {list(kernels)}")
    assert "foo" in kernels, f"kernel 'foo' not found in {list(kernels)}"
    kd_addr, desc = kernels["foo"]
    log(f"kd_addr={kd_addr:#x} kernarg_size={desc.kernarg_size} "
        f"group_segment={desc.group_segment_fixed_size} private_segment={desc.private_segment_fixed_size} "
        f"kernel_code_properties={desc.kernel_code_properties:#x}")
    from tinygrad.runtime.autogen import hsa
    log(f"enable_dispatch_ptr={bool(desc.kernel_code_properties & hsa.AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_DISPATCH_PTR)}")

    prg = BeagleAMDProgram(dev, "foo", hsaco, kd_addr, n_int_args=0)
    log("BeagleAMDProgram constructed")

    buf = dev.allocator.alloc(N * 4)
    log(f"buffer allocated at va={buf.va_addr:#x}")

    prg(HCQBuffer(buf.va_addr, N * 4), global_size=(N, 1, 1), local_size=(1, 1, 1), vals=(), wait=False)
    log("kernel launched, syncing...")
    dev.synchronize()
    log("synchronized -- no fault")

    out = memoryview(bytearray(N * 4))
    dev.allocator._copyout(out, HCQBuffer(buf.va_addr, N * 4))
    vals = struct.unpack(f"<{N}f", bytes(out))
    log(f"readback values = {vals}")
    print(f"RESULT: {vals}", file=sys.stdout, flush=True)
    if all(v == 42.0 for v in vals):
        log("SUCCESS: all values match expected 42.0")
    else:
        log(f"MISMATCH: expected all 42.0, got {vals}")
    log("exiting cleanly")


if __name__ == "__main__":
    main()
