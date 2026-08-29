#!/usr/bin/env python3
"""
nv_footprint_sweep.py -- TODO.md "PICK UP HERE" -> NV Phase 55.

Phase 54 (nv_grid_test.py + %smid) proved dispatch/SM-distribution work
completely normally for a trivial kernel (grid=(16,1,1) block=(16,16,1),
all 16 blocks spread across 16 distinct SMs) -- ruling out "everything
funnels to SM 0" as a general property of this driver stack. Yet
kernelMatrixMulADB, launched with the *exact same* grid/block shape, has
every surviving block land on SM 0 only (Phase 47/48). The only remaining
difference between the two launches is the kernel's own compiled resource
footprint (registers/shared memory) and internal computation.

This script tests that directly: a synthetic kernel with a *tunable*
resource footprint (real register pressure via an unrolled, per-thread
accumulator array; real shared-memory allocation via a sized __shared__
array), same grid/block shape throughout, same %smid capture as Phase 54.
Sweeps registers and shared memory *independently* (one variable at a time,
this investigation's own established discipline) to find whether -- and at
what point -- SM distribution stops spreading and starts funneling.

Anti-dead-code-elimination: every thread (not just thread 0) computes the
padding sum and writes it to a `discard` buffer nothing ever reads back --
an observable global-memory side effect the compiler can't optimize away,
so the accumulator array and shared-memory reads/writes stay real rather
than being folded to nothing. The actual correctness marker (`out[]`) is
written unconditionally by thread (0,0) only, independent of the padding
computation, so correctness-checking stays simple and robust regardless of
what the padding does.

    python3 nv_footprint_sweep.py
"""
import sys, os, pathlib, struct, tempfile, subprocess

_TINYGRAD_PATH = os.environ.get("TINYGRAD_PATH", str(pathlib.Path.home() / "Dropbox/Projects/tinygrad"))
sys.path.insert(0, _TINYGRAD_PATH)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

GRID = (16, 1, 1)   # exact shape from the real kernelMatrixMulADB launch log
BLOCK = (16, 16, 1)

SEED_LEN = 256  # real, runtime-filled input buffer; see make_kernel_src's docstring below

KERNEL_TEMPLATE = '''
extern "C" __global__ void {kname}(unsigned int* out, unsigned int* smid_out, float* discard, const float* seed) {{
    __shared__ float buf[{shmem_decl}];
    float acc[{reg_decl}];
#pragma unroll
    for (int i = 0; i < {reg_decl}; i++)
        acc[i] = seed[(threadIdx.x + i * 7) % {seed_len}];
#if {shmem_floats} > 0
    if (threadIdx.x < {shmem_floats} && threadIdx.y == 0) buf[threadIdx.x] = seed[threadIdx.x % {seed_len}];
    __syncthreads();
#endif
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < {reg_decl}; i++)
        sum += acc[i];
#if {shmem_floats} > 0
    if (threadIdx.x < {shmem_floats}) sum += buf[threadIdx.x];
#endif
    discard[blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x] = sum;
    if (threadIdx.x == 0 && threadIdx.y == 0) {{
        out[blockIdx.x] = blockIdx.x + 1000u;
        unsigned int smid;
        asm("mov.u32 %0, %smid;" : "=r"(smid));
        smid_out[blockIdx.x] = smid;
    }}
}}
'''


def make_kernel_src(kname, reg_pad, shmem_floats):
    """
    Each acc[i] is a genuine load from a real, runtime-filled `seed` buffer
    (h2d-copied from the host, unknown to the compiler at compile time) --
    not a closed-form arithmetic expression. Two earlier approaches were
    tried and both failed, confirmed empirically each time before moving on:
      1. A plain closed-form `acc[i] = f(threadIdx.x, i, blockIdx.x)`
         formula: ptxas algebraically collapsed the whole unrolled
         accumulate-then-sum sequence into a single expression at compile
         time -- reported the identical ~14 registers for reg_decl=1, 8,
         and 56 alike.
      2. A closed-form formula plus a single combined `asm volatile("" :
         "+f"(acc[0]), ..., "+f"(acc[K-1]))` clobber listing every operand
         at once, intended to force simultaneous liveness: still collapsed
         to ~14 registers, 0 spill (checked `ptxas -v`'s stack-frame/spill
         line directly, not just the register count). Root cause, confirmed
         by reading the generated PTX: an *empty* inline-asm body compiles
         to zero real SASS instructions, so it's not a real scheduling
         barrier -- ptxas's instruction scheduler remains free to interleave
         "compute acc[i], sum += acc[i]" iteration by iteration (each has no
         real dependency on the others), needing only ~1-2 live registers
         at a time regardless of array size, same failure mode as (1) via a
         different route.
    A genuine memory load can't be constant-folded (defeats (1)) and is a
    real SASS instruction with real latency, so ptxas's own latency-hiding
    scheduler (issue many independent loads early, consume results later --
    the same reason real kernels like kernelMatrixMulADB create register
    pressure) has a real incentive to keep them all in flight -- defeats (2)
    too, without needing any inline-asm trick at all. Verified below.
    """
    reg_decl, shmem_decl = max(reg_pad, 1), max(shmem_floats, 1)
    return KERNEL_TEMPLATE.format(kname=kname, reg_decl=reg_decl, shmem_decl=shmem_decl,
                                   shmem_floats=shmem_floats, seed_len=SEED_LEN)

# One variable at a time: registers swept with shared memory pinned at 0,
# then shared memory swept with the register pad pinned at the minimum.
# kernelMatrixMulADB's own real (plain-build) footprint was 32 registers /
# 2112 bytes shared memory (Phase 47/48) -- both sweeps bracket that.
#
# NOTE on the register sweep's range: ptxas's own register allocator plateaus
# at 32 (real, occupancy-driven behavior, not an artifact of this probe --
# confirmed by trying reg_pad up to 96 and a much larger SEED_LEN, both
# still capped at exactly 32) regardless of how large reg_pad requests. 32
# happens to already match kernelMatrixMulADB's own real footprint exactly,
# so the achievable range still directly brackets the comparison that
# matters; reg_pad values are kept dense up to that point rather than
# wasting compiles past it.
REG_SWEEP   = [(r, 0) for r in (1, 4, 8, 12, 16, 20, 24, 28, 32)]
SHMEM_SWEEP = [(1, s) for s in (64, 128, 256, 512, 1024, 2048)]  # bytes = floats*4
VARIANTS = REG_SWEEP + SHMEM_SWEEP


def log(msg):
    print(f"[nv_footprint_sweep] {msg}", file=sys.stderr, flush=True)


def compile_variant(nch, dev, nvcc, reg_pad, shmem_floats):
    kname = f"footprint_r{reg_pad}_s{shmem_floats}"
    src = make_kernel_src(kname, reg_pad, shmem_floats)
    with tempfile.NamedTemporaryFile(suffix=".cu", delete=False, mode="w") as cf:
        cf.write(src)
        cu_path = cf.name
    ptx_path = cu_path.replace(".cu", ".ptx")
    r = subprocess.run([nvcc, "-ptx", "-o", ptx_path, cu_path], capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"nvcc failed for {kname}: {r.stderr.decode(errors='replace')}")
    elf_bytes = nch.compile_ptx(ptx_path, dev.arch, kernel_name=kname)
    os.unlink(cu_path)
    if os.path.exists(ptx_path):
        os.unlink(ptx_path)
    return kname, elf_bytes


def run_variant(dev, HCQBuffer, TinyELF, Target, BeagleNVProgram, kname, elf_bytes, discard_buf, n_threads_total, seed_buf):
    obj = TinyELF(lib=elf_bytes, name=kname, target=Target(), signature=())
    prg = BeagleNVProgram(dev, obj)

    n = GRID[0] * GRID[1] * GRID[2]
    d_out = dev.allocator.alloc(n * 4)
    d_smid = dev.allocator.alloc(n * 4)
    dev.allocator._copyin(HCQBuffer(d_out.va_addr, n * 4), memoryview(bytearray(n * 4)))
    dev.allocator._copyin(HCQBuffer(d_smid.va_addr, n * 4), memoryview(bytearray(n * 4)))

    prg(HCQBuffer(d_out.va_addr, n * 4), HCQBuffer(d_smid.va_addr, n * 4), HCQBuffer(discard_buf.va_addr, n_threads_total * 4),
        HCQBuffer(seed_buf.va_addr, SEED_LEN * 4), global_size=GRID, local_size=BLOCK, vals=(), wait=False)
    dev.synchronize()

    out = memoryview(bytearray(n * 4))
    dev.allocator._copyout(out, HCQBuffer(d_out.va_addr, n * 4))
    vals = struct.unpack(f"<{n}I", bytes(out))

    smid_out = memoryview(bytearray(n * 4))
    dev.allocator._copyout(smid_out, HCQBuffer(d_smid.va_addr, n * 4))
    smids = struct.unpack(f"<{n}I", bytes(smid_out))

    ok_count = sum(1 for i, v in enumerate(vals) if v == i + 1000)
    distinct_sms = sorted(set(smids[i] for i, v in enumerate(vals) if v == i + 1000))
    return prg, ok_count, n, distinct_sms, list(smids)


def main():
    os.makedirs(os.path.expanduser("~/Library/Logs"), exist_ok=True)
    fd = os.open(os.path.expanduser("~/Library/Logs/nv_footprint_sweep.log"),
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
    from nv_dispatch_daemon import BeagleNVProgram
    from tinygrad.device import TinyELF, Target
    from tinygrad.runtime.support.hcq import HCQBuffer

    nvcc = os.environ.get("TINYGPU_NVCC", os.path.expanduser("~/.local/bin/nvcc"))

    n_threads_total = GRID[0] * GRID[1] * GRID[2] * BLOCK[0] * BLOCK[1] * BLOCK[2]
    discard_buf = dev.allocator.alloc(n_threads_total * 4)
    log(f"discard buffer allocated: {discard_buf.va_addr:#x} ({n_threads_total} floats)")

    # Real, runtime-filled input every acc[i]/buf[] load reads from -- see
    # make_kernel_src's docstring for why this (not a closed-form formula,
    # not an inline-asm clobber) is what actually forces real register/
    # shared-memory pressure. Values themselves don't matter (never
    # correctness-checked), just that they're unknown to the compiler.
    import random
    seed_buf = dev.allocator.alloc(SEED_LEN * 4)
    seed_vals = [random.random() for _ in range(SEED_LEN)]
    dev.allocator._copyin(HCQBuffer(seed_buf.va_addr, SEED_LEN * 4), memoryview(struct.pack(f"<{SEED_LEN}f", *seed_vals)))
    log(f"seed buffer allocated + filled: {seed_buf.va_addr:#x} ({SEED_LEN} floats)")

    results = []
    for reg_pad, shmem_floats in VARIANTS:
        kname, elf_bytes = compile_variant(nch, dev, nvcc, reg_pad, shmem_floats)
        log(f"compiled {kname} -- {len(elf_bytes)} byte ELF")
        prg, ok_count, n, distinct_sms, smids = run_variant(
            dev, HCQBuffer, TinyELF, Target, BeagleNVProgram, kname, elf_bytes, discard_buf, n_threads_total, seed_buf)
        line = (f"{kname}: regs_usage={prg.regs_usage} shmem_usage={prg.shmem_usage}  "
                f"ok={ok_count}/{n}  distinct_sms={len(distinct_sms)} {distinct_sms}")
        log(line)
        print(line, file=sys.stdout, flush=True)
        results.append((reg_pad, shmem_floats, prg.regs_usage, prg.shmem_usage, ok_count, n, distinct_sms))

    print("\n=== summary ===", file=sys.stdout, flush=True)
    log("=== summary ===")
    for reg_pad, shmem_floats, regs_usage, shmem_usage, ok_count, n, distinct_sms in results:
        funneled = len(distinct_sms) <= 1 and ok_count > 1
        line = (f"reg_pad={reg_pad:3d} shmem_floats={shmem_floats:5d}  "
                f"regs_usage={regs_usage:3d} shmem_usage={shmem_usage:5d}  "
                f"ok={ok_count:2d}/{n}  #SMs={len(distinct_sms):2d}  {'FUNNELED' if funneled else 'spread'}")
        log(line)
        print(line, file=sys.stdout, flush=True)

    log("exiting cleanly")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc(file=sys.stderr)
        print("RESULT: FAIL (exception, see log)", file=sys.stdout, flush=True)
        sys.exit(1)
