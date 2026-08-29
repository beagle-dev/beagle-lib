#!/usr/bin/env python3
"""
amd_math_test.py — the pipeline's first real hardware run
(tinygpuhybridtest --diag-compare-cpu) got all the way through the new
architecture cleanly (no crash, no hang, no fault) and produced a
*correctness* bug instead: kernelMatrixMulADB's output is NaN for every
element (STATUS.md AMD §22). That kernel is the first one in the whole
pipeline to call exp() -- and GPUImplDefs.h's FW_TINYGPU_HYBRID_AMD branch's
exp()/cos()/sin()/log()/pow()/round() wrappers (routed to comgr's OCML
device-library functions, __ocml_<name>_f32/f64) have only ever been
offline-compile-checked, never runtime-verified: amd_real_kernel_test.py's
kernelSumSites1 (the one real kernel dispatched on hardware so far) calls
none of them.

This isolates that one variable directly and cheaply: a trivial kernel
that computes exp(x)/cos(x)/sin(x)/log(x) via GPUImplDefs.h's own
FW_TINYGPU_HYBRID_AMD wrappers (compiled the exact same way, via
amd_compile_helper.compile_hip()) on a handful of known values, checked
against Python's own math module. If these come back correct, exp() is
cleared and the bug is elsewhere (input upload, kernel indexing); if not,
this pins the OCML wrapper declarations as the cause -- much cheaper than
debugging it through the full BEAGLE pipeline.

    python3 amd_math_test.py
"""
import sys, os, pathlib, struct, math

_TINYGRAD_PATH = os.environ.get("TINYGRAD_PATH", str(pathlib.Path.home() / "Dropbox/Projects/tinygrad"))
sys.path.insert(0, _TINYGRAD_PATH)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tinygrad.helpers import DEV
from tinygrad.runtime.support.hcq import HCQBuffer

import amd_compile_helper as ach
from amd_dispatch_daemon import BeagleAMDProgram

# Uses GPUImplDefs.h's own FW_TINYGPU_HYBRID_AMD exp/cos/sin/log wrappers --
# compiled the same way (FW_TINYGPU_HYBRID_AMD + FW_OPENCL + OPENCL_KERNEL_BUILD
# defined, via amd_compile_helper.compile_hip()) as the real BEAGLE kernels.
# GPUImplDefs.h's content is embedded directly (not #include'd) -- comgr
# compiles an in-memory source string with no real filesystem/-I context,
# matching exactly how kernels/make_opencl_kernels.sh builds the real
# BeagleOpenCL_kernels.h (plain text concatenation, not a real #include).
KERNEL_SRC_TAIL = '''
KW_GLOBAL_KERNEL void mathTest(KW_GLOBAL_VAR float* x, KW_GLOBAL_VAR float* out) {
    int i = KW_GROUP_ID_0;
    float v = x[i];
    out[i * 4 + 0] = exp(v);
    out[i * 4 + 1] = cos(v);
    out[i * 4 + 2] = sin(v);
    out[i * 4 + 3] = log(v);
}
'''

TEST_VALUES = [0.0, 1.0, 2.0, 0.5]


def log(msg):
    print(f"[amd_math_test] {msg}", file=sys.stderr, flush=True)


def main():
    os.makedirs(os.path.expanduser("~/Library/Logs"), exist_ok=True)
    fd = os.open(os.path.expanduser("~/Library/Logs/amd_math_test.log"),
                 os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_SYNC, 0o644)
    sys.stderr = os.fdopen(fd, 'w', buffering=1)

    log("starting")
    DEV.value = "AMD"
    from tinygrad import Device
    dev = Device["AMD:0"]
    log(f"booted -- {dev}, arch={dev.arch}")

    impl_defs = pathlib.Path(__file__).with_name("GPUImplDefs.h").read_text()
    # STATE_COUNT defined first, matching how the real generated kernel
    # headers always precede GPUImplDefs.h's content with it.
    src = "#define STATE_COUNT 4\n" + impl_defs + KERNEL_SRC_TAIL
    hsaco = ach.compile_hip(src, dev.arch)  # amd_compile_helper.compile_hip prepends the #defines
    log(f"compiled -- {len(hsaco)} byte HSACO")

    image, kernels = ach.parse_kernels(hsaco)
    log(f"kernels found: {list(kernels)}")
    assert "mathTest" in kernels, f"mathTest not found in {sorted(kernels)}"
    kd_addr, desc = kernels["mathTest"]
    log(f"kd_addr={kd_addr:#x} kernarg_size={desc.kernarg_size} "
        f"group_segment={desc.group_segment_fixed_size} private_segment={desc.private_segment_fixed_size} "
        f"kernel_code_properties={desc.kernel_code_properties:#x}")

    prg = BeagleAMDProgram(dev, "mathTest", hsaco, kd_addr, n_int_args=0)
    log("BeagleAMDProgram constructed")

    n = len(TEST_VALUES)
    d_x = dev.allocator.alloc(n * 4)
    d_out = dev.allocator.alloc(n * 4 * 4)
    xs = struct.pack(f"<{n}f", *TEST_VALUES)
    dev.allocator._copyin(HCQBuffer(d_x.va_addr, n * 4), memoryview(bytearray(xs)))
    dev.synchronize()
    log(f"input uploaded: {TEST_VALUES}")

    prg(HCQBuffer(d_x.va_addr, n * 4), HCQBuffer(d_out.va_addr, n * 4 * 4),
        global_size=(n, 1, 1), local_size=(1, 1, 1), vals=(), wait=False)
    log("kernel launched, syncing...")
    dev.synchronize()
    log("synchronized -- no fault")

    out = memoryview(bytearray(n * 4 * 4))
    dev.allocator._copyout(out, HCQBuffer(d_out.va_addr, n * 4 * 4))
    vals = struct.unpack(f"<{n * 4}f", bytes(out))

    all_ok = True
    for i, v in enumerate(TEST_VALUES):
        gpu_exp, gpu_cos, gpu_sin, gpu_log = vals[i * 4:i * 4 + 4]
        cpu_exp, cpu_cos, cpu_sin = math.exp(v), math.cos(v), math.sin(v)
        cpu_log = math.log(v) if v > 0 else float('-inf')
        line = (f"x={v}: exp GPU={gpu_exp} CPU={cpu_exp} | cos GPU={gpu_cos} CPU={cpu_cos} | "
                f"sin GPU={gpu_sin} CPU={cpu_sin} | log GPU={gpu_log} CPU={cpu_log}")
        log(line)
        print(line, file=sys.stdout, flush=True)
        for gpu_v, cpu_v, name in [(gpu_exp, cpu_exp, "exp"), (gpu_cos, cpu_cos, "cos"),
                                    (gpu_sin, cpu_sin, "sin")] + ([(gpu_log, cpu_log, "log")] if v > 0 else []):
            if not math.isfinite(gpu_v) or abs(gpu_v - cpu_v) > 1e-4:
                all_ok = False
                log(f"  MISMATCH: {name}({v}) GPU={gpu_v} CPU={cpu_v}")

    print(f"RESULT: {'ALL_OK' if all_ok else 'MISMATCH'}", file=sys.stdout, flush=True)
    log("SUCCESS: all math functions correct" if all_ok else "FAILURE: mismatches found, see above")
    log("exiting cleanly")


if __name__ == "__main__":
    main()
