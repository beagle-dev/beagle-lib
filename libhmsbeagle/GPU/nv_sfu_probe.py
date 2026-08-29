#!/usr/bin/env python3
"""
nv_sfu_probe.py -- TODO.md "PICK UP HERE" -> NV Phase 66.

Five real bisections on kernelMatrixMulADB have now each been tried; only
one had any effect at all: removing exp() from the "Last block" guard's
Ds[tx] computation (Phase 57) fixed wMatrix>=4's total non-execution
outright. Everything else tried since -- the main loop, the two-distinct-
pointer A/B split, data-dependent addressing (listC), repeated single-CTA
execution -- changed nothing about the residual Ds[2] failure. That
pattern raises a sharper question than "does exp() matter": is it *any*
SFU-class instruction (this GPU's transcendental/special-function unit --
MUFU.EX2 for exp2, MUFU.RCP for reciprocal, MUFU.RSQ for rsqrt -- a
different execution unit and latency/scheduling class than ordinary FFMA/
ALU ops) that this from-scratch driver stack mishandles under multi-CTA
occupancy, or something specific to exponentiation itself?

Starts from nv_combo_probe.py's exact proven-clean shape (Phase 60: 16
CTAs, broadcast+barrier+33-value sequential burst, all clean, 0 failures)
and changes exactly one thing per variant: what Ds[tx] is computed *from*
inside the same tx<EDGE branch the real kernel's guard uses, in the same
structural position exp() occupied in the real kernel.

Four variants, same GRID/BLOCK/EDGE=4 (real divergence) throughout:
  - control : Ds[tx] = x               (plain load -- Phase 60's own,
              already proven clean; included here as an in-run sanity
              check, not a new result)
  - exp     : Ds[tx] = expf(x)         (MUFU.EX2 -- exact same SASS
              instruction the real kernel's guard used before the Phase
              57 fix)
  - rcp     : Ds[tx] = __fdividef(1.0f, x)  (MUFU.RCP -- non-transcendental
              SFU op, verified via nvdisasm to compile to a single clean
              instruction, no slowpath/CALL -- __frcp_rn was tried first
              and rejected for this reason, see STATUS.md)
  - rsqrt   : Ds[tx] = rsqrtf(x)       (MUFU.RSQ -- another non-
              transcendental SFU op, single clean instruction)
x itself is a real, runtime seed-buffer read scaled into (0.1, 1.0] --
avoids reciprocal/rsqrt blowing up near 0 while keeping exp()'s argument
in a safe, non-overflowing range -- not foldable, not arbitrary.

If a variant starts failing where the plain-read control stays clean,
that's a strong, novel finding: this driver stack mishandles that specific
SFU-class instruction under multi-CTA occupancy, independent of the real
kernel's own address/loop/branch structure entirely. If all four stay
clean, that rules out "any SFU-class instruction in this position" and
narrows back toward something specific to the real kernel's own use of
exp() (its exact operand, e.g. distance*D[d+tx], or its interaction with
something else in the real "Last block" guard not reproduced here).

    python3 nv_sfu_probe.py
"""
import sys, os, pathlib, struct, tempfile, subprocess

_TINYGRAD_PATH = os.environ.get("TINYGRAD_PATH", str(pathlib.Path.home() / "Dropbox/Projects/tinygrad"))
sys.path.insert(0, _TINYGRAD_PATH)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

GRID = (16, 1, 1)   # exact shape from the real kernelMatrixMulADB launch log
BLOCK = (16, 16, 1)
MULTIPLY_BLOCK_SIZE = 16  # matches kernelsAll.cu's real As/Bs/Ds sizing
SEED_LEN = 256
SLOT_REALS = 33  # deliberately not a multiple of 16 -- Phase 59's disambiguation technique
SENTINEL = -999.0
EDGE = 4  # real divergence, matching the real kernel's "Last block" guard shape

KERNEL_TEMPLATE = '''
extern "C" __global__ void {kname}(unsigned int* out, unsigned int* smid_out, float* discard, const float* seed, float* dbg) {{
    __shared__ float As[{mbs}][{mbs}];
    __shared__ float Bs[{mbs}][{mbs}];
    __shared__ float Ds[{mbs}];

    int tx = threadIdx.x, ty = threadIdx.y;

    if (tx < {edge} && ty < {edge}) {{
        if (ty == 0) {{
            float x = 0.1f + 0.9f * seed[(tx + blockIdx.x * 7) % {seed_len}];
            Ds[tx] = {sfu_expr};
        }}
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
        // Same 33-value burst as Phase 60's combo probe -- only the Ds[]
        // computation above differs between variants.
        float* slot = dbg + blockIdx.x * {slot_reals};
        slot[0]  = Csub;
        slot[1]  = As[0][0]; slot[2]  = As[0][1]; slot[3]  = As[0][2]; slot[4]  = As[0][3];
        slot[5]  = Bs[0][0]; slot[6]  = Bs[1][0]; slot[7]  = Bs[2][0]; slot[8]  = Bs[3][0];
        slot[9]  = Ds[0];    slot[10] = Ds[1];    slot[11] = Ds[2];    slot[12] = Ds[3];
        slot[13] = As[1][0]; slot[14] = As[1][1]; slot[15] = As[1][2]; slot[16] = As[1][3];
        slot[17] = Bs[0][1]; slot[18] = Bs[1][1]; slot[19] = Bs[2][1]; slot[20] = Bs[3][1];
        slot[21] = As[2][0]; slot[22] = As[2][1]; slot[23] = As[2][2]; slot[24] = As[2][3];
        slot[25] = Bs[0][2]; slot[26] = Bs[1][2]; slot[27] = Bs[2][2]; slot[28] = Bs[3][2];
        slot[29] = As[3][0]; slot[30] = As[3][1]; slot[31] = As[3][2]; slot[32] = As[3][3];

        out[blockIdx.x] = blockIdx.x + 1000u;
        unsigned int smid;
        asm("mov.u32 %0, %smid;" : "=r"(smid));
        smid_out[blockIdx.x] = smid;
    }}
}}
'''

# label -> (kernel-name-safe suffix, SFU expression applied to real, scaled seed value x)
VARIANTS = [
    ("control", "x"),                          # plain load -- Phase 60's own, already proven clean
    ("exp",     "expf(x)"),                     # MUFU.EX2 -- exact match to the real kernel's pre-fix instruction
    ("rcp",     "__fdividef(1.0f, x)"),          # MUFU.RCP -- non-transcendental SFU op, verified single-instruction
    ("rsqrt",   "rsqrtf(x)"),                    # MUFU.RSQ -- non-transcendental SFU op, verified single-instruction
]


def log(msg):
    print(f"[nv_sfu_probe] {msg}", file=sys.stderr, flush=True)


def make_kernel_src(kname, sfu_expr):
    return KERNEL_TEMPLATE.format(kname=kname, edge=EDGE, mbs=MULTIPLY_BLOCK_SIZE,
                                   seed_len=SEED_LEN, slot_reals=SLOT_REALS, sfu_expr=sfu_expr)


def compile_variant(nch, dev, nvcc, label, sfu_expr):
    kname = f"sfu_{label}"
    src = make_kernel_src(kname, sfu_expr)
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

    n = GRID[0] * GRID[1] * GRID[2]
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
    fd = os.open(os.path.expanduser("~/Library/Logs/nv_sfu_probe.log"),
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

    n_dbg = GRID[0] * SLOT_REALS
    dbg_buf = dev.allocator.alloc(n_dbg * 4)
    log(f"debug buffer allocated: {dbg_buf.va_addr:#x} ({n_dbg} floats, {SLOT_REALS} per block)")

    for label, sfu_expr in VARIANTS:
        kname, elf_bytes = compile_variant(nch, dev, nvcc, label, sfu_expr)
        log(f"compiled {kname} ({sfu_expr}) -- {len(elf_bytes)} byte ELF")
        prg, ok_count, n, distinct_sms, dbg_vals = run_variant(
            dev, HCQBuffer, TinyELF, Target, BeagleNVProgram, kname, elf_bytes,
            discard_buf, n_threads_total, seed_buf, dbg_buf, n_dbg)

        header = (f"\n=== {kname} ({sfu_expr}): regs_usage={prg.regs_usage} shmem_usage={prg.shmem_usage}  "
                  f"ok={ok_count}/{n}  distinct_sms={len(distinct_sms)} {distinct_sms} ===")
        log(header)
        print(header, file=sys.stdout, flush=True)

        all_local_fails = []
        for b in range(GRID[0]):
            fails = [i for i in range(SLOT_REALS) if dbg_vals[b * SLOT_REALS + i] == SENTINEL]
            all_local_fails.extend((b, i) for i in fails)
            local_byte_offsets = [i * 4 for i in fails]
            global_byte_offsets = [(b * SLOT_REALS + i) * 4 for i in fails]
            line = (f"  block {b:2d}: {len(fails)} failure(s)  local_byte_offsets={local_byte_offsets}  "
                    f"global_byte_offsets={global_byte_offsets}")
            log(line)
            print(line, file=sys.stdout, flush=True)

        if not all_local_fails:
            line = f"  {kname}: no debug-burst failures."
        else:
            distinct_local = sorted(set(i * 4 for _, i in all_local_fails))
            line = f"  {kname}: distinct LOCAL failure byte-offsets across blocks: {distinct_local}"
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
