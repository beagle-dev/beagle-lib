#!/usr/bin/env python3
"""
nv_loop_probe.py -- TODO.md "PICK UP HERE" -> NV Phase 64.

Every probe so far (Phase 47-60) launched *multiple CTAs* (one per
wMatrix/block) to reproduce kernelMatrixMulADB's ground-truth-dump
failure -- and every synthetic reproduction attempt failed, even ones
matching the real kernel's resource footprint, branch shape, and
broadcast+barrier+burst structure exactly (Phase 60). User asked: what if
the bug has nothing to do with *multiple CTAs* at all -- does it show up
if a *single* CTA does the exact same sequence of work 16 times in a row,
internally, instead of 16 CTAs each doing it once?

This script tests that directly: launches `grid=(1,1,1)` -- exactly one
CTA, completely sidestepping CTA-scheduling/funneling (SMID funneling,
Phase 47-51's whole investigation thread) by construction, since there's
only ever one CTA to schedule -- and loops 16 times *inside* the kernel,
each iteration doing the identical broadcast+barrier+burst sequence
Phase 60's combo probe used (same divergent branch, same shared-memory
cross-thread hand-off, same 33-value sequential burst into a dedicated,
non-16-byte-aligned per-iteration region), using a loop-counter variable
in place of `blockIdx.x` for the per-"wMatrix" addressing.

If the same failure pattern (or any failure) appears here, that's a
striking result: it would mean the bug is NOT about multi-CTA scheduling
at all, but something about repeated execution of this instruction
sequence within a single thread's lifetime (e.g. a hardware resource --
the `desc[UR]` descriptor state, a cache line, a queue slot -- that
doesn't get refreshed correctly between iterations). If it stays clean,
that's equally informative: it would mean the bug genuinely requires
multiple *concurrent* CTAs, not just repetition, sharpening the
multi-CTA-specific candidates instead (real listC-style addressing,
concurrent-CTA resource contention, etc.).

One extra synchronization Phase 56/60's per-CTA versions never needed:
As/Bs/Ds are the *same* shared-memory arrays reused every iteration, so an
extra `__syncthreads()` is added at the end of each iteration (after the
debug-dump burst) to prevent iteration N+1's writes from racing
iteration N's still-in-flight reads on another warp.

Same two EDGE variants as Phase 56/60, same %smid capture, same
seed-buffer-sourced anti-DCE discipline.

    python3 nv_loop_probe.py
"""
import sys, os, pathlib, struct, tempfile, subprocess

_TINYGRAD_PATH = os.environ.get("TINYGRAD_PATH", str(pathlib.Path.home() / "Dropbox/Projects/tinygrad"))
sys.path.insert(0, _TINYGRAD_PATH)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

GRID = (1, 1, 1)    # exactly one CTA -- the whole point of this probe
BLOCK = (16, 16, 1)  # same block shape as every prior probe
MULTIPLY_BLOCK_SIZE = 16
SEED_LEN = 256
SLOT_REALS = 33  # deliberately not a multiple of 16 -- Phase 59's disambiguation technique
SENTINEL = -999.0
N_ITERS = 16  # matches totalMatrix in the real kernel

KERNEL_TEMPLATE = '''
extern "C" __global__ void {kname}(unsigned int* out, unsigned int* smid_out, float* discard, const float* seed, float* dbg) {{
    __shared__ float As[{mbs}][{mbs}];
    __shared__ float Bs[{mbs}][{mbs}];
    __shared__ float Ds[{mbs}];

    int tx = threadIdx.x, ty = threadIdx.y;

#pragma unroll 1
    for (int loopIdx = 0; loopIdx < {n_iters}; loopIdx++) {{
        if (tx < {edge} && ty < {edge}) {{
            if (ty == 0)
                Ds[tx] = seed[(tx + loopIdx * 7) % {seed_len}];
            As[ty][tx] = seed[(ty * {mbs} + tx + loopIdx * 13) % {seed_len}];
            Bs[ty][tx] = seed[(ty * {mbs} + tx + loopIdx * 19) % {seed_len}];
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

        discard[loopIdx * blockDim.x * blockDim.y + ty * blockDim.x + tx] = Csub;

        if (tx == 0 && ty == 0) {{
            float* slot = dbg + loopIdx * {slot_reals};
            slot[0]  = Csub;
            slot[1]  = As[0][0]; slot[2]  = As[0][1]; slot[3]  = As[0][2]; slot[4]  = As[0][3];
            slot[5]  = Bs[0][0]; slot[6]  = Bs[1][0]; slot[7]  = Bs[2][0]; slot[8]  = Bs[3][0];
            slot[9]  = Ds[0];    slot[10] = Ds[1];    slot[11] = Ds[2];    slot[12] = Ds[3];
            slot[13] = As[1][0]; slot[14] = As[1][1]; slot[15] = As[1][2]; slot[16] = As[1][3];
            slot[17] = Bs[0][1]; slot[18] = Bs[1][1]; slot[19] = Bs[2][1]; slot[20] = Bs[3][1];
            slot[21] = As[2][0]; slot[22] = As[2][1]; slot[23] = As[2][2]; slot[24] = As[2][3];
            slot[25] = Bs[0][2]; slot[26] = Bs[1][2]; slot[27] = Bs[2][2]; slot[28] = Bs[3][2];
            slot[29] = As[3][0]; slot[30] = As[3][1]; slot[31] = As[3][2]; slot[32] = As[3][3];

            out[loopIdx] = loopIdx + 1000u;
            unsigned int smid;
            asm("mov.u32 %0, %smid;" : "=r"(smid));
            smid_out[loopIdx] = smid;
        }}
        // Extra barrier vs. Phase 56/60's per-CTA versions: As/Bs/Ds are
        // the *same* shared memory reused every iteration here, so every
        // thread must wait for thread (0,0)'s debug-dump reads above to
        // finish before any thread starts iteration N+1's writes.
        __syncthreads();
    }}
}}
'''

VARIANTS = [4, 16]


def log(msg):
    print(f"[nv_loop_probe] {msg}", file=sys.stderr, flush=True)


def make_kernel_src(kname, edge):
    return KERNEL_TEMPLATE.format(kname=kname, edge=edge, mbs=MULTIPLY_BLOCK_SIZE, seed_len=SEED_LEN,
                                   slot_reals=SLOT_REALS, n_iters=N_ITERS)


def compile_variant(nch, dev, nvcc, edge):
    kname = f"loopcombo_edge{edge}"
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


def run_variant(dev, HCQBuffer, TinyELF, Target, BeagleNVProgram, kname, elf_bytes,
                 discard_buf, n_threads_total, seed_buf, dbg_buf, n_dbg):
    obj = TinyELF(lib=elf_bytes, name=kname, target=Target(), signature=())
    prg = BeagleNVProgram(dev, obj)

    n = N_ITERS
    d_out = dev.allocator.alloc(n * 4)
    d_smid = dev.allocator.alloc(n * 4)
    dev.allocator._copyin(HCQBuffer(d_out.va_addr, n * 4), memoryview(bytearray(n * 4)))
    dev.allocator._copyin(HCQBuffer(d_smid.va_addr, n * 4), memoryview(bytearray(n * 4)))
    dev.allocator._copyin(HCQBuffer(dbg_buf.va_addr, n_dbg * 4), memoryview(struct.pack(f"<{n_dbg}f", *([SENTINEL] * n_dbg))))

    prg(HCQBuffer(d_out.va_addr, n * 4), HCQBuffer(d_smid.va_addr, n * 4), HCQBuffer(discard_buf.va_addr, n_threads_total * 4),
        HCQBuffer(seed_buf.va_addr, SEED_LEN * 4), HCQBuffer(dbg_buf.va_addr, n_dbg * 4),
        global_size=GRID, local_size=BLOCK, vals=(), wait=False)
    dev.synchronize()

    out = memoryview(bytearray(n * 4))
    dev.allocator._copyout(out, HCQBuffer(d_out.va_addr, n * 4))
    vals = struct.unpack(f"<{n}I", bytes(out))

    smid_out = memoryview(bytearray(n * 4))
    dev.allocator._copyout(smid_out, HCQBuffer(d_smid.va_addr, n * 4))
    smids = struct.unpack(f"<{n}I", bytes(smid_out))

    dbg_out = memoryview(bytearray(n_dbg * 4))
    dev.allocator._copyout(dbg_out, HCQBuffer(dbg_buf.va_addr, n_dbg * 4))
    dbg_vals = struct.unpack(f"<{n_dbg}f", bytes(dbg_out))

    ok_count = sum(1 for i, v in enumerate(vals) if v == i + 1000)
    distinct_sms = sorted(set(smids[i] for i, v in enumerate(vals) if v == i + 1000))
    return prg, ok_count, n, distinct_sms, dbg_vals


def main():
    os.makedirs(os.path.expanduser("~/Library/Logs"), exist_ok=True)
    fd = os.open(os.path.expanduser("~/Library/Logs/nv_loop_probe.log"),
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

    n_threads_total = N_ITERS * BLOCK[0] * BLOCK[1] * BLOCK[2]
    discard_buf = dev.allocator.alloc(n_threads_total * 4)
    log(f"discard buffer allocated: {discard_buf.va_addr:#x} ({n_threads_total} floats)")

    import random
    seed_buf = dev.allocator.alloc(SEED_LEN * 4)
    seed_vals = [random.random() for _ in range(SEED_LEN)]
    dev.allocator._copyin(HCQBuffer(seed_buf.va_addr, SEED_LEN * 4), memoryview(struct.pack(f"<{SEED_LEN}f", *seed_vals)))
    log(f"seed buffer allocated + filled: {seed_buf.va_addr:#x} ({SEED_LEN} floats)")

    n_dbg = N_ITERS * SLOT_REALS
    dbg_buf = dev.allocator.alloc(n_dbg * 4)
    log(f"debug buffer allocated: {dbg_buf.va_addr:#x} ({n_dbg} floats, {SLOT_REALS} per iteration)")

    for edge in VARIANTS:
        kname, elf_bytes = compile_variant(nch, dev, nvcc, edge)
        log(f"compiled {kname} -- {len(elf_bytes)} byte ELF")
        prg, ok_count, n, distinct_sms, dbg_vals = run_variant(
            dev, HCQBuffer, TinyELF, Target, BeagleNVProgram, kname, elf_bytes,
            discard_buf, n_threads_total, seed_buf, dbg_buf, n_dbg)

        header = (f"\n=== {kname}: regs_usage={prg.regs_usage} shmem_usage={prg.shmem_usage}  "
                  f"ok={ok_count}/{n}  distinct_sms_across_iters={len(distinct_sms)} {distinct_sms} ===")
        log(header)
        print(header, file=sys.stdout, flush=True)

        all_local_fails = []
        for it in range(N_ITERS):
            fails = [i for i in range(SLOT_REALS) if dbg_vals[it * SLOT_REALS + i] == SENTINEL]
            all_local_fails.extend((it, i) for i in fails)
            local_byte_offsets = [i * 4 for i in fails]
            line = f"  iter {it:2d}: {len(fails)} failure(s)  local_byte_offsets={local_byte_offsets}"
            log(line)
            print(line, file=sys.stdout, flush=True)

        if not all_local_fails:
            line = f"  {kname}: no debug-burst failures across any of the {N_ITERS} internal iterations."
        else:
            distinct_local = sorted(set(i * 4 for _, i in all_local_fails))
            line = f"  {kname}: distinct LOCAL failure byte-offsets across iterations: {distinct_local}"
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
