#!/usr/bin/env python3
"""
nv_grid_test.py — isolates whether grid=(16,1,1) block=(16,16,1) dispatch
itself (16 CTAs, independent of any of BEAGLE's own kernel logic) completes
correctly via the real NVProgram/NVComputeQueue/QMD path this daemon uses.

STATUS.md §82/TODO.md Phase 46: kernelMatrixMulADB's real launch parameters
(hardware-confirmed correct: grid.x=16, totalMatrix=16, listC/distanceQueue
both upload with full, correct, sensible content) still only produce output
for wMatrix 0-3 (BEAGLE's public matrix[0], the first 4 of 16 packed
per-(edge,category) computations) -- matrix[1..3] (wMatrix 4-15) come back
completely unwritten. That's consistent with either (a) something specific
to this kernel's own logic/shared-memory usage, or (b) a genuine dispatch-
level defect specific to this exact grid=(16,1,1) block=(16,16,1) shape
(notably, grid.x here numerically equals both block dimensions -- no other
kernel in this pipeline shares that coincidence, and none of the other
kernels' real hardware runs showed this failure mode).

This script tests (b) directly and cheaply: a trivial kernel, same grid/
block shape, same real compile+dispatch pipeline (nv_compile_helper's real
ptxas, real BeagleNVProgram/NVDevice, no BEAGLE C++/daemon involved at all),
writing only `blockIdx.x` into a dedicated 16-element output buffer. If all
16 slots come back correct, dispatch itself is cleared and the bug is
specific to BEAGLE's own kernel; if only slots 0-3 (or some other subset)
come back written, that's a direct, minimal reproduction independent of any
BEAGLE-specific complexity.

TODO.md "PICK UP HERE" -> NV Phase 53: also captures %smid (real CUDA PTX
ISA special register, same as the kernelMatrixMulADB ground-truth probe's
own Phase 50 addition) per block, into a second output buffer. This
kernel's 16-block launch already completes fully (every block writes
correctly) -- the open question this addition targets is whether those 16
*successful* blocks are spread across many distinct SMs, or -- like
kernelMatrixMulADB's own surviving blocks (Phase 47/48, SMID=0 for every
one) -- funneled onto the same one. All-SM0 here would mean "everything
funnels to SM 0" is a general property of this driver stack, not something
specific to kernelMatrixMulADB's heavier resource footprint; multiple
distinct SMs would mean the funneling is somehow tied to that heavier
footprint specifically (e.g. only manifesting once more than one wave is
needed, which this trivial kernel's low resource usage may never require).

    python3 nv_grid_test.py
"""
import sys, os, pathlib, struct

_TINYGRAD_PATH = os.environ.get("TINYGRAD_PATH", str(pathlib.Path.home() / "Dropbox/Projects/tinygrad"))
sys.path.insert(0, _TINYGRAD_PATH)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CUDA_SRC = '''
extern "C" __global__ void gridtest(unsigned int* out, unsigned int* smid_out) {
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        out[blockIdx.x] = blockIdx.x + 1000u;  // +1000: distinguishes "wrote 0" from "never wrote" (stays 0)
        unsigned int smid;
        asm("mov.u32 %0, %smid;" : "=r"(smid));
        smid_out[blockIdx.x] = smid;
    }
}
'''

GRID = (16, 1, 1)   # exact shape from the real kernelMatrixMulADB launch log
BLOCK = (16, 16, 1)


def log(msg):
    print(f"[nv_grid_test] {msg}", file=sys.stderr, flush=True)


def main():
    os.makedirs(os.path.expanduser("~/Library/Logs"), exist_ok=True)
    fd = os.open(os.path.expanduser("~/Library/Logs/nv_grid_test.log"),
                 os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_SYNC, 0o644)
    sys.stderr = os.fdopen(fd, 'w', buffering=1)

    log("starting")
    import nv_init_helper  # noqa: F401 -- GSP/RM boot safety patches (module-level side effects)
    from tinygrad.runtime.support.system import APLRemotePCIDevice
    def _safe_reset(self):
        log("PCIe FLR suppressed (macOS eGPU safety)")
    APLRemotePCIDevice.reset = _safe_reset

    from tinygrad.helpers import DEV
    DEV.value = "NV"
    from tinygrad import Device
    dev = Device["NV:0"]
    log(f"booted -- {dev}, arch={dev.arch}, renderer={type(dev.renderer).__name__}")

    import nv_compile_helper as nch
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".cu", delete=False, mode="w") as cf:
        cf.write(CUDA_SRC)
        cu_path = cf.name
    ptx_path = cu_path.replace(".cu", ".ptx")

    # Same nvcc used throughout this project (docker shim) to go from real
    # CUDA C++ source to PTX, then the same real ptxas nv_compile_helper.py
    # always uses to go from PTX to SASS/ELF.
    import subprocess
    nvcc = os.environ.get("TINYGPU_NVCC", os.path.expanduser("~/.local/bin/nvcc"))
    r = subprocess.run([nvcc, "-ptx", "-o", ptx_path, cu_path], capture_output=True)
    if r.returncode != 0:
        log(f"nvcc failed: {r.stderr.decode(errors='replace')}")
        sys.exit(1)
    log("nvcc -ptx OK")

    elf_bytes = nch.compile_ptx(ptx_path, dev.arch, kernel_name="gridtest")
    log(f"ptxas OK -- {len(elf_bytes)} byte ELF")

    from nv_dispatch_daemon import BeagleNVProgram
    from tinygrad.device import TinyELF, Target
    from tinygrad.runtime.support.hcq import HCQBuffer

    obj = TinyELF(lib=elf_bytes, name="gridtest", target=Target(), signature=())
    prg = BeagleNVProgram(dev, obj)
    log("BeagleNVProgram constructed")

    n = GRID[0] * GRID[1] * GRID[2]
    d_out = dev.allocator.alloc(n * 4)
    d_smid = dev.allocator.alloc(n * 4)
    # Zero both first (host zeros, h2d) so "never written" is unambiguous --
    # fresh VRAM isn't guaranteed zero-filled.
    dev.allocator._copyin(HCQBuffer(d_out.va_addr, n * 4), memoryview(bytearray(n * 4)))
    dev.allocator._copyin(HCQBuffer(d_smid.va_addr, n * 4), memoryview(bytearray(n * 4)))
    log(f"output buffer allocated + zeroed: {d_out.va_addr:#x}, smid buffer: {d_smid.va_addr:#x}")

    prg(HCQBuffer(d_out.va_addr, n * 4), HCQBuffer(d_smid.va_addr, n * 4), global_size=GRID, local_size=BLOCK, vals=(), wait=False)
    log("launched, syncing...")
    dev.synchronize()
    log("synchronized -- no fault")

    out = memoryview(bytearray(n * 4))
    dev.allocator._copyout(out, HCQBuffer(d_out.va_addr, n * 4))
    vals = struct.unpack(f"<{n}I", bytes(out))

    smid_out = memoryview(bytearray(n * 4))
    dev.allocator._copyout(smid_out, HCQBuffer(d_smid.va_addr, n * 4))
    smids = struct.unpack(f"<{n}I", bytes(smid_out))

    all_ok = True
    for i, (v, sm) in enumerate(zip(vals, smids)):
        expected = i + 1000
        ok = (v == expected)
        line = f"block[{i}]: expected={expected} got={v} {'OK' if ok else 'MISSING/WRONG' if v == 0 else 'UNEXPECTED'}  SMID={sm}"
        log(line)
        print(line, file=sys.stdout, flush=True)
        if not ok:
            all_ok = False

    distinct_sms = sorted(set(smids[i] for i, v in enumerate(vals) if v == i + 1000))
    sm_line = f"distinct SMs used by the {sum(1 for v,e in zip(vals,(i+1000 for i in range(n))) if v==e)} successful blocks: {distinct_sms}"
    log(sm_line)
    print(sm_line, file=sys.stdout, flush=True)

    print(f"RESULT: {'ALL_OK' if all_ok else 'MISMATCH'}", file=sys.stdout, flush=True)
    log("SUCCESS: all 16 blocks wrote correctly" if all_ok else "FAILURE: see per-block lines above")
    log("exiting cleanly")

    os.unlink(cu_path)
    if os.path.exists(ptx_path):
        os.unlink(ptx_path)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc(file=sys.stderr)
        print("RESULT: FAIL (exception, see log)", file=sys.stdout, flush=True)
        sys.exit(1)
