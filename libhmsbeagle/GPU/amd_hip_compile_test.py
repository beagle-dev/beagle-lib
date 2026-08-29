#!/usr/bin/env python3
"""
amd_hip_compile_test.py — isolate compile *language* as a variable.

Every one of the five hardware crashes so far dispatched a kernel compiled
via comgr's OpenCL-C language mode (amd_compile_helper.compile_opencl,
AMD_COMGR_LANGUAGE_OPENCL_2_0). The only thing that has ever worked on this
hardware (STATUS.md AMD §8, Tensor([1,2,3])+1) used a kernel compiled via
comgr's HIP language mode (tinygrad's own HIPRenderer/HIPCompiler,
AMD_COMGR_LANGUAGE_HIP) -- a different clang frontend and device-library
linkage, not just a different dispatch mechanism. No OpenCL-compiled kernel
has ever run to completion here; no HIP-compiled kernel has ever failed
here. That's a variable this session never isolated.

This script compiles a trivial kernel via tinygrad's own compile_hip() (the
exact function HIPCompiler uses -- see
tinygrad/runtime/support/compiler_amd.py) and dispatches it through the same
BeagleAMDProgram/HCQProgram path amd_dispatch_daemon.py uses for BEAGLE's
real (OpenCL-compiled) kernels. Run directly with plain `python3` -- no C++,
no socketpair, no forked daemon -- matching exactly how the one known-good
reference test was invoked, so a crash here can only be attributed to
compile language + BeagleAMDProgram, not to the C++ spawn machinery.

Kernel source deliberately avoids every runtime-header-dependent construct
(no threadIdx/blockIdx, no __ockl_* calls) -- just the bare
extern "C" __attribute__((global)) spelling HIPRenderer itself emits
(cstyle.py's kernel_typedef), so this exercises only "HIP-language compile,
then dispatch," nothing else.

    python3 amd_hip_compile_test.py
"""
import sys, os, pathlib, struct

_TINYGRAD_PATH = os.environ.get("TINYGRAD_PATH", str(pathlib.Path.home() / "Dropbox/Projects/tinygrad"))
sys.path.insert(0, _TINYGRAD_PATH)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tinygrad.helpers import DEV
from tinygrad.runtime.support.compiler_amd import compile_hip
from tinygrad.runtime.support.hcq import HCQBuffer

import amd_compile_helper as ach          # reuse the already-verified parse_kernels()
from amd_dispatch_daemon import BeagleAMDProgram  # the exact class the daemon dispatches real BEAGLE kernels with

HIP_SRC = '''
extern "C" __attribute__((global)) void foo(float* x) {
  x[0] = 42.0f;
}
'''


def log(msg):
    print(f"[amd_hip_test] {msg}", file=sys.stderr, flush=True)


def main():
    os.makedirs(os.path.expanduser("~/Library/Logs"), exist_ok=True)
    fd = os.open(os.path.expanduser("~/Library/Logs/amd_hip_compile_test.log"),
                 os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_SYNC, 0o644)
    sys.stderr = os.fdopen(fd, 'w', buffering=1)

    log("starting")
    DEV.value = "AMD"
    from tinygrad import Device
    dev = Device["AMD:0"]
    log(f"booted -- {dev}, arch={dev.arch}")

    hsaco = compile_hip(HIP_SRC, dev.arch)
    log(f"compiled HIP kernel -- {len(hsaco)} bytes")

    image, kernels = ach.parse_kernels(hsaco)
    log(f"kernels found: {list(kernels)}")
    assert "foo" in kernels, f"kernel 'foo' not found in {list(kernels)}"
    kd_addr, desc = kernels["foo"]
    log(f"kd_addr={kd_addr:#x} kernarg_size={desc.kernarg_size} "
        f"group_segment={desc.group_segment_fixed_size} private_segment={desc.private_segment_fixed_size}")

    prg = BeagleAMDProgram(dev, "foo", hsaco, kd_addr, n_int_args=0)
    log("BeagleAMDProgram constructed")

    buf = dev.allocator.alloc(4)
    log(f"buffer allocated at va={buf.va_addr:#x}")

    prg(HCQBuffer(buf.va_addr, 4), global_size=(1, 1, 1), local_size=(1, 1, 1), vals=(), wait=False)
    log("kernel launched, syncing...")
    dev.synchronize()
    log("synchronized -- no fault")

    out = memoryview(bytearray(4))
    dev.allocator._copyout(out, HCQBuffer(buf.va_addr, 4))
    val = struct.unpack("<f", bytes(out))[0]
    log(f"readback value = {val}")
    print(f"RESULT: {val}", file=sys.stdout, flush=True)
    if val == 42.0:
        log("SUCCESS: value matches expected 42.0")
    else:
        log(f"MISMATCH: expected 42.0, got {val}")
    log("exiting cleanly")


if __name__ == "__main__":
    main()
