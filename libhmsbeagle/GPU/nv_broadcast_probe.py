#!/usr/bin/env python3
"""
nv_broadcast_probe.py -- TODO.md "PICK UP HERE" -> NV Phase 56.

Phase 55 (nv_footprint_sweep.py) conclusively ruled out resource footprint
*size* (registers, shared memory, independently, up to and past
kernelMatrixMulADB's own real numbers) as the explanation for its blocks
funneling onto a single SM -- every sweep point spread cleanly across all
16 SMs. That redirects the investigation onto kernelMatrixMulADB's own
specific *computation*, not merely how many resources it declares.

This script isolates the one idiom flagged repeatedly across this whole
investigation but never yet tested on its own: the shared-memory
**broadcast+barrier** pattern kernelMatrixMulADB's "Last block" guard uses
--

    if (tx < EDGE && ty < EDGE) {
        if (ty == 0) Ds[tx] = <real value>;
        As[ty][tx] = <real value>;
        Bs[ty][tx] = <real value>;
    } else {
        if (ty == 0) Ds[tx] = 0;
        As[ty][tx] = 0;
        Bs[ty][tx] = 0;
    }
    __syncthreads();
    for (k = 0; k < EDGE; k++)
        Csub += As[ty][k] * Ds[k] * Bs[k][tx];

-- a *divergent branch* (every thread takes one of two paths) gating
whether real or zero data is written, then a barrier, then every thread
reads back values written by *other* threads (As[ty][k] for k != tx,
Bs[k][tx] for k != ty, and Ds[] written only by the ty==0 row but read by
all 256 threads). Phase 55's sweep kernel used barriers and shared memory
too, but every thread only ever wrote and read its *own* data -- it never
exercised this specific cross-thread hand-off shape. This shape is also
independently implicated by the still-open, purely-value-level finding
from Phase 47/48: wMatrix 0-2's As/Bs/Ds read back as exactly 0 right
after this exact barrier, on real hardware, for the real kernel.

Two variants, same grid=(16,1,1) block=(16,16,1) shape and %smid capture
as Phase 54-55, same shared-memory footprint as the real kernel (16x16 As
+ 16x16 Bs + 16 Ds floats = 2112 bytes, matching kernelMatrixMulADB's own
real plain-build usage exactly):
  - EDGE=4: faithful reproduction of the real kernel's exact parameters
    (this test's real stateCount=4 case) -- a genuine divergent branch,
    16 of 256 threads take the "real data" path.
  - EDGE=16: same idiom, but the branch condition (tx<16 && ty<16) is
    always true at runtime for a 16x16 block -- every thread takes the
    same ("real data") path uniformly, no per-thread divergence. (The
    branch *instruction* itself is still present in the compiled SASS
    either way -- block dimensions are a runtime launch parameter, not
    known to the compiler, so ptxas can't prove the condition statically
    and omit the check; verified via nvdisasm before relying on this.)
    Isolates whether genuine per-thread *runtime* divergence specifically
    matters, vs. just the cross-thread broadcast+barrier mechanism with a
    uniformly-taken branch.

Anti-dead-code-elimination / correctness marker discipline matches Phase
55: Csub (real, seed-buffer-sourced, unknown to the compiler) is written
unconditionally by every thread to a `discard` buffer nothing reads back;
`out[]`/`smid_out[]` are written by thread (0,0) only, independent of the
broadcast computation, so correctness-checking stays simple and robust.

    python3 nv_broadcast_probe.py
"""
import sys, os, pathlib, struct, tempfile, subprocess

_TINYGRAD_PATH = os.environ.get("TINYGRAD_PATH", str(pathlib.Path.home() / "Dropbox/Projects/tinygrad"))
sys.path.insert(0, _TINYGRAD_PATH)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

GRID = (16, 1, 1)   # exact shape from the real kernelMatrixMulADB launch log
BLOCK = (16, 16, 1)
MULTIPLY_BLOCK_SIZE = 16  # matches kernelsAll.cu's real As/Bs/Ds sizing
SEED_LEN = 256

KERNEL_TEMPLATE = '''
extern "C" __global__ void {kname}(unsigned int* out, unsigned int* smid_out, float* discard, const float* seed) {{
    __shared__ float As[{mbs}][{mbs}];
    __shared__ float Bs[{mbs}][{mbs}];
    __shared__ float Ds[{mbs}];

    int tx = threadIdx.x, ty = threadIdx.y;

    if (tx < {edge} && ty < {edge}) {{
        if (ty == 0)
            Ds[tx] = seed[(tx + blockIdx.x * 7) % {seed_len}];
        As[ty][tx] = seed[(ty * {mbs} + tx + blockIdx.x * 13) % {seed_len}];
        Bs[ty][tx] = seed[(ty * {mbs} + tx + blockIdx.x * 19) % {seed_len}];
    }} else {{
        if (ty == 0)
            Ds[tx] = 0.0f;
        As[ty][tx] = 0.0f;
        Bs[ty][tx] = 0.0f;
    }}
    __syncthreads();

    float Csub = 0.0f;
#pragma unroll
    for (int k = 0; k < {edge}; k++)
        Csub += As[ty][k] * Ds[k] * Bs[k][tx];

    discard[blockIdx.x * blockDim.x * blockDim.y + ty * blockDim.x + tx] = Csub;
    if (tx == 0 && ty == 0) {{
        out[blockIdx.x] = blockIdx.x + 1000u;
        unsigned int smid;
        asm("mov.u32 %0, %smid;" : "=r"(smid));
        smid_out[blockIdx.x] = smid;
    }}
}}
'''

# EDGE=4: this test's actual stateCount=4 case -- the real, exact
# divergent-branch shape. EDGE=16: same idiom, branch condition always
# true for this block size -- isolates whether branch divergence itself
# (vs. just the cross-thread broadcast+barrier) matters.
VARIANTS = [4, 16]


def log(msg):
    print(f"[nv_broadcast_probe] {msg}", file=sys.stderr, flush=True)


def make_kernel_src(kname, edge):
    return KERNEL_TEMPLATE.format(kname=kname, edge=edge, mbs=MULTIPLY_BLOCK_SIZE, seed_len=SEED_LEN)


def compile_variant(nch, dev, nvcc, edge):
    kname = f"broadcast_edge{edge}"
    src = make_kernel_src(kname, edge)
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
    return prg, ok_count, n, distinct_sms


def main():
    os.makedirs(os.path.expanduser("~/Library/Logs"), exist_ok=True)
    fd = os.open(os.path.expanduser("~/Library/Logs/nv_broadcast_probe.log"),
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

    import random
    seed_buf = dev.allocator.alloc(SEED_LEN * 4)
    seed_vals = [random.random() for _ in range(SEED_LEN)]
    dev.allocator._copyin(HCQBuffer(seed_buf.va_addr, SEED_LEN * 4), memoryview(struct.pack(f"<{SEED_LEN}f", *seed_vals)))
    log(f"seed buffer allocated + filled: {seed_buf.va_addr:#x} ({SEED_LEN} floats)")

    results = []
    for edge in VARIANTS:
        kname, elf_bytes = compile_variant(nch, dev, nvcc, edge)
        log(f"compiled {kname} -- {len(elf_bytes)} byte ELF")
        prg, ok_count, n, distinct_sms = run_variant(
            dev, HCQBuffer, TinyELF, Target, BeagleNVProgram, kname, elf_bytes, discard_buf, n_threads_total, seed_buf)
        line = (f"{kname}: regs_usage={prg.regs_usage} shmem_usage={prg.shmem_usage}  "
                f"ok={ok_count}/{n}  distinct_sms={len(distinct_sms)} {distinct_sms}")
        log(line)
        print(line, file=sys.stdout, flush=True)
        results.append((edge, prg.regs_usage, prg.shmem_usage, ok_count, n, distinct_sms))

    print("\n=== summary ===", file=sys.stdout, flush=True)
    log("=== summary ===")
    for edge, regs_usage, shmem_usage, ok_count, n, distinct_sms in results:
        funneled = len(distinct_sms) <= 1 and ok_count > 1
        line = (f"EDGE={edge:2d}  regs_usage={regs_usage:3d} shmem_usage={shmem_usage:5d}  "
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
