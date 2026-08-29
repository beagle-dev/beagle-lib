#!/usr/bin/env python3
"""
nv_distexp_probe.py -- TODO.md "PICK UP HERE" -> NV Phase 67.

Phase 66's SFU-class probe (nv_sfu_probe.py) added expf()/__fdividef()/
rsqrtf() -- each confirmed via nvdisasm to compile to a single, clean SFU
instruction -- into Phase 60's proven-clean combo probe, in the exact
branch position the real kernel's exp() occupied. All four variants,
*including* exp, came back completely clean: 0 failures, every block,
every burst slot. That rules out "any SFU-class instruction, including
exp() itself, in this generic structural position is sufficient" -- so
whatever made exp()'s removal matter in the real kernel (Phase 57) depends
on something this probe didn't include.

One real-kernel structural element has never appeared in any probe:
kernelMatrixMulADB's FW_TINYGPU path computes
    distance = distanceQueue[wMatrix];
*unconditionally*, outside any thread-0 guard -- all 256 threads in the
block redundantly load the same address from global memory, every launch,
regardless of whether TINYGPU_BISECT_NO_EXP is defined. That loaded value
then feeds directly into exp(): `Ds[tx] = exp(D[d + tx] * distance);`. No
probe's own addressing has ever gone beyond pure blockIdx.x/threadIdx.x
arithmetic -- Phase 65's listC bisection removed a data-dependent
*address* computation, but never an actual 256-way-simultaneous *load*.
A block-uniform value read identically by every thread in the block is
also exactly the shape of load Blackwell's uniform datapath (the
`desc[UR]` descriptor flagged as a suspect all the way back in Phase 5/6,
never directly readable via legal PTX -- Phase 61) is positioned to
special-case -- a second, independent reason this specific combination is
worth testing in isolation before it's tested nowhere else.

This is a proper 2x2 factorial on Phase 60's proven-clean shape, isolating
both factors and their interaction:
  - control  : load=NO  exp=NO   Ds[tx] = x
  - load_only: load=YES exp=NO   Ds[tx] = x * distance
  - exp_only : load=NO  exp=YES  Ds[tx] = expf(x)          (repeats Phase
               66's exp variant as an in-run freshness check, not a new
               result on its own)
  - combined : load=YES exp=YES  Ds[tx] = expf(x * distance)  -- this is
               the one cell of the 2x2 that has never been tested: the
               real kernel's exact `exp(D[d+tx] * distance)` shape, with
               distance loaded exactly the way the real kernel loads it
               (all 256 threads, unconditional, every launch).

"load" is a small per-block real value (`distq[blockIdx.x]`, one real
float per block, distinct across blocks, positive, moderate magnitude --
matching distanceQueue's real branch-length semantics) read by *every*
thread unconditionally, before the tx<EDGE branch -- not just in the
branch, so it can't be folded into "just another EDGE-only read."

If `combined` fails where `control`/`load_only`/`exp_only` stay clean,
that's the interaction Phase 66 couldn't see: the all-256-threads load and
exp() together, specifically. If `combined` stays clean too, the real
kernel's exp() effect must depend on something still not reproduced here
(the real D[]/distance multiply's exact register pressure, or something
about the real kernel's much larger surrounding body -- worth revisiting
via a more surgical real-kernel bisection at that point, not another
synthetic probe).

    python3 nv_distexp_probe.py
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

DISTANCE_LOAD = "    float distance = distq[blockIdx.x];  // all 256 threads, unconditional -- matches the real kernel's distanceQueue[wMatrix] read\n"

KERNEL_TEMPLATE = '''
extern "C" __global__ void {kname}(unsigned int* out, unsigned int* smid_out, float* discard, const float* seed, const float* distq, float* dbg) {{
    __shared__ float As[{mbs}][{mbs}];
    __shared__ float Bs[{mbs}][{mbs}];
    __shared__ float Ds[{mbs}];

    int tx = threadIdx.x, ty = threadIdx.y;

{distance_load}
    if (tx < {edge} && ty < {edge}) {{
        if (ty == 0) {{
            float x = 0.1f + 0.9f * seed[(tx + blockIdx.x * 7) % {seed_len}];
            Ds[tx] = {ds_expr};
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
        // Same 33-value burst as Phase 60/66's probes -- only the
        // distance-load presence and Ds[] computation above differ
        // between variants.
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

# label -> (has_distance_load, Ds[tx] expression)
VARIANTS = [
    ("control",   False, "x"),                    # neither factor -- Phase 60's own, already proven clean
    ("load_only", True,  "x * distance"),          # all-256-threads global load, no exp
    ("exp_only",  False, "expf(x)"),                # exp, no all-256-threads load (repeats Phase 66's exp variant)
    ("combined",  True,  "expf(x * distance)"),     # both together -- the real kernel's exact shape, untested until now
]


def log(msg):
    print(f"[nv_distexp_probe] {msg}", file=sys.stderr, flush=True)


def make_kernel_src(kname, has_load, ds_expr):
    return KERNEL_TEMPLATE.format(kname=kname, edge=EDGE, mbs=MULTIPLY_BLOCK_SIZE,
                                   seed_len=SEED_LEN, slot_reals=SLOT_REALS, ds_expr=ds_expr,
                                   distance_load=(DISTANCE_LOAD if has_load else ""))


def compile_variant(nch, dev, nvcc, label, has_load, ds_expr):
    kname = f"distexp_{label}"
    src = make_kernel_src(kname, has_load, ds_expr)
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
                 discard_buf, n_threads_total, seed_buf, distq_buf, dbg_buf, n_dbg):
    obj = TinyELF(lib=elf_bytes, name=kname, target=Target(), signature=())
    prg = BeagleNVProgram(dev, obj)

    n = GRID[0] * GRID[1] * GRID[2]
    d_out = dev.allocator.alloc(n * 4)
    d_smid = dev.allocator.alloc(n * 4)
    dev.allocator._copyin(HCQBuffer(d_out.va_addr, n * 4), memoryview(bytearray(n * 4)))
    dev.allocator._copyin(HCQBuffer(d_smid.va_addr, n * 4), memoryview(bytearray(n * 4)))
    dev.allocator._copyin(HCQBuffer(dbg_buf.va_addr, n_dbg * 4), memoryview(struct.pack(f"<{n_dbg}f", *([SENTINEL] * n_dbg))))

    prg(HCQBuffer(d_out.va_addr, n * 4), HCQBuffer(d_smid.va_addr, n * 4), HCQBuffer(discard_buf.va_addr, n_threads_total * 4),
        HCQBuffer(seed_buf.va_addr, SEED_LEN * 4), HCQBuffer(distq_buf.va_addr, GRID[0] * 4), HCQBuffer(dbg_buf.va_addr, n_dbg * 4),
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
    fd = os.open(os.path.expanduser("~/Library/Logs/nv_distexp_probe.log"),
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

    # One real, distinct, positive, moderate-magnitude value per block --
    # matching distanceQueue[wMatrix]'s real branch-length semantics.
    distq_buf = dev.allocator.alloc(GRID[0] * 4)
    distq_vals = [random.uniform(0.05, 2.0) for _ in range(GRID[0])]
    dev.allocator._copyin(HCQBuffer(distq_buf.va_addr, GRID[0] * 4), memoryview(struct.pack(f"<{GRID[0]}f", *distq_vals)))
    log(f"distq buffer allocated + filled: {distq_buf.va_addr:#x} ({GRID[0]} floats) -- {distq_vals}")

    n_dbg = GRID[0] * SLOT_REALS
    dbg_buf = dev.allocator.alloc(n_dbg * 4)
    log(f"debug buffer allocated: {dbg_buf.va_addr:#x} ({n_dbg} floats, {SLOT_REALS} per block)")

    for label, has_load, ds_expr in VARIANTS:
        kname, elf_bytes = compile_variant(nch, dev, nvcc, label, has_load, ds_expr)
        log(f"compiled {kname} (load={has_load}, Ds[tx]={ds_expr}) -- {len(elf_bytes)} byte ELF")
        prg, ok_count, n, distinct_sms, dbg_vals = run_variant(
            dev, HCQBuffer, TinyELF, Target, BeagleNVProgram, kname, elf_bytes,
            discard_buf, n_threads_total, seed_buf, distq_buf, dbg_buf, n_dbg)

        header = (f"\n=== {kname} (load={has_load}, Ds[tx]={ds_expr}): regs_usage={prg.regs_usage} "
                  f"shmem_usage={prg.shmem_usage}  ok={ok_count}/{n}  distinct_sms={len(distinct_sms)} {distinct_sms} ===")
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
