#!/usr/bin/env python3
"""
amd_opencl_compile_test.py — companion to amd_hip_compile_test.py.

That script (HIP-language compile, same BeagleAMDProgram/HCQProgram
dispatch, no C++/daemon spawn) ran clean: no fault, correct readback
(STATUS.md AMD §13). That clears the dispatch mechanism, BeagleAMDProgram,
and the no-daemon invocation style as variables -- what's left untested in
isolation is OpenCL-language comgr compilation itself (every one of the five
crashes used it; nothing that has ever worked did).

This script is amd_hip_compile_test.py with exactly one thing changed:
compile via amd_compile_helper.compile_opencl() (AMD_COMGR_LANGUAGE_OPENCL_2_0)
instead of tinygrad's compile_hip() (AMD_COMGR_LANGUAGE_HIP). Same trivial
one-store kernel semantics, same BeagleAMDProgram dispatch, same direct
`python3` invocation. If this crashes, it isolates the fault to OpenCL-mode
compilation specifically, independent of BEAGLE's kernel complexity, the
daemon, or the dispatch path -- all of which this variant still doesn't
exercise.

    python3 amd_opencl_compile_test.py
"""
import sys, os, pathlib, struct

_TINYGRAD_PATH = os.environ.get("TINYGRAD_PATH", str(pathlib.Path.home() / "Dropbox/Projects/tinygrad"))
sys.path.insert(0, _TINYGRAD_PATH)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tinygrad.helpers import DEV
from tinygrad.runtime.support.hcq import HCQBuffer

import amd_compile_helper as ach          # compile_opencl() + parse_kernels(), already verified this session
from amd_dispatch_daemon import BeagleAMDProgram  # the exact class the daemon dispatches real BEAGLE kernels with

OPENCL_SRC = '''
kernel void foo(global float* x) {
  x[0] = 42.0f;
}
'''


def log(msg):
    print(f"[amd_opencl_test] {msg}", file=sys.stderr, flush=True)


def main():
    os.makedirs(os.path.expanduser("~/Library/Logs"), exist_ok=True)
    fd = os.open(os.path.expanduser("~/Library/Logs/amd_opencl_compile_test.log"),
                 os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_SYNC, 0o644)
    sys.stderr = os.fdopen(fd, 'w', buffering=1)

    log("starting")
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
