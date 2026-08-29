#!/usr/bin/env python3
"""
nv_trivial_kernel_probe.py -- TODO.md "PICK UP HERE" -> NV Phase 83.

Phase 82's result: the original, maximally-trivial version of this probe
(no shared memory, no barrier, no loop, one guarded global write per
block) came back **100% reliable on every one of 48 SMIDs**, including
every SMID kernelMatrixMulADB is unreliable on (e.g. SMID 12/14, 5-17%
for the real kernel). That's real evidence against dead/faulty silicon
-- a genuinely non-functional SM should fail on anything, not just a
specific, more complex kernel -- and points instead at something in the
real kernel's own resource footprint (regs_usage=40, shmem_usage=4224
bytes vs. this probe's ~8 registers/0 shared memory) or execution
characteristics (a real barrier, real instruction count, real exp())
interacting with this driver stack's handling of certain SMs.

This turns that binary trivial/non-trivial probe into a **dial**: two
independent, separately controllable knobs --

    --shmem N_FLOATS   allocates N_FLOATS of *real* shared memory, with
                        every one of the block's 256 threads writing into
                        it (matching kernelMatrixMulADB's own As[ty][tx]=
                        .../Bs[ty][tx]=... structure, not just a thread-0
                        no-op declaration a compiler could strip) followed
                        by a real __syncthreads() barrier -- matching the
                        real kernel's KW_LOCAL_FENCE before its own
                        ground-truth-dump write. 1056 floats = 4224 bytes
                        exactly matches the real kernel's own shmem_usage.

    --loop N_ITERS      has thread (0,0) additionally loop N_ITERS times
                        (reading back pad[] if --shmem is also set, or
                        doing throwaway arithmetic if not) before its
                        write -- a pure duration dial, independent of
                        shared-memory size.

    --sweep-shmem       automated: runs shmem in [0, 16, 64, 256, 1056]
                        floats (loop=0), one compile+sweep per value,
                        looking for the size at which failures start
                        appearing on the same SMIDs the real kernel fails
                        on (SMID 6/7/12/13/14/16/17/22/23 -- consistently
                        near-floor across every real-kernel run this
                        session, TODO.md Phase 79/80/§142/143).
    --sweep-loop        automated: same idea, loop in [0, 100, 10000,
                        1000000] iterations, shmem=0.

Same grid/block shape (block=(16,16,1), default grid=64 matching Phase
80's config) and per-block/per-SMID sweep-reporting methodology as
Phase 81/82's original version and nv_real_kernel_probe.py's --sweep, so
every config's result is directly comparable to the already-established
real-kernel tables for the same physical SMIDs.

    python3 nv_trivial_kernel_probe.py [--shmem N] [--loop N] [N_BLOCKS] [N_ITERS]
    python3 nv_trivial_kernel_probe.py --sweep-shmem [N_BLOCKS] [N_ITERS]
    python3 nv_trivial_kernel_probe.py --sweep-loop [N_BLOCKS] [N_ITERS]

Examples:
    python3 nv_trivial_kernel_probe.py                    # Phase 82's original: shmem=0 loop=0, 64 blocks, 20 iters -- known-clean baseline
    python3 nv_trivial_kernel_probe.py --shmem 1056        # real kernel's exact shared-memory footprint (4224 bytes), otherwise still trivial
    python3 nv_trivial_kernel_probe.py --sweep-shmem       # automated search across the shared-memory dial
    python3 nv_trivial_kernel_probe.py --sweep-loop        # automated search across the duration dial
"""
import sys, os, pathlib, struct, tempfile, subprocess
from collections import Counter, defaultdict

_TINYGRAD_PATH = os.environ.get("TINYGRAD_PATH", str(pathlib.Path.home() / "Dropbox/Projects/tinygrad"))
sys.path.insert(0, _TINYGRAD_PATH)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BLOCK = (16, 16, 1)   # matches kernelMatrixMulADB's real MULTIPLY_BLOCK_SIZE=16 shape/occupancy footprint
SENTINEL = -999.0

# Reference sets from this session's own mechanically-captured real-kernel
# data (TODO.md Phase 79/80, STATUS.md §142/143) -- SMIDs that have been
# consistently near-floor (<=20%) vs. consistently ~100% across every
# real-kernel --wide-grid run. Used only to print a quick, at-a-glance
# summary column in the sweep modes; the full per-SMID table (printed for
# every config regardless) is the actual data.
KNOWN_BAD_SMIDS = [6, 7, 12, 13, 14, 16, 17, 22, 23]
KNOWN_GOOD_SMIDS = [0, 1, 8, 9]

SHMEM_SWEEP_VALUES = [0, 16, 64, 256, 1056]   # 1056 floats = 4224 bytes = kernelMatrixMulADB's real shmem_usage
LOOP_SWEEP_VALUES = [0, 100, 10000, 1000000]


def log(msg):
    print(f"[nv_trivial_kernel_probe] {msg}", file=sys.stderr, flush=True)


def make_kernel_src(kname, shmem_floats, loop_iters):
    """Builds the trivial-kernel-with-dials source. shmem_floats=0,
    loop_iters=0 reproduces Phase 82's original kernel exactly (no shared
    memory, no barrier, no loop)."""
    if shmem_floats > 0:
        # `volatile` is load-bearing, not decorative: without it, when
        # loop_iters==0 there is no read of `pad` anywhere, so nvcc
        # correctly treats the write as dead code and strips it entirely
        # -- confirmed directly (compiler emits "variable 'pad' was set
        # but never used" and the array vanishes from the PTX) before
        # this fix was added. `volatile` forces every write/read to be a
        # real, non-eliminable memory operation regardless of whether
        # anything else in the kernel uses the result -- standard
        # practice for exactly this class of microbenchmark, not a
        # workaround specific to this probe.
        shmem_decl = f"    __shared__ volatile float pad[{shmem_floats}];\n"
        # Every one of the block's 256 threads writes -- matches
        # kernelsAll.cu's real As[ty][tx]=.../Bs[ty][tx]=... structure,
        # not a thread-0-only declaration -- then a real barrier,
        # matching the real kernel's KW_LOCAL_FENCE before its own
        # ground-truth-dump write.
        touch = (f"    int tx = threadIdx.x, ty = threadIdx.y;\n"
                 f"    pad[(ty * 16 + tx) % {shmem_floats}] = (float)(tx + ty);\n"
                 f"    __syncthreads();\n")
    else:
        shmem_decl = ""
        touch = ""

    if loop_iters > 0:
        if shmem_floats > 0:
            loop_body = (f"        float acc = 0.0f;\n"
                         f"        for (int i = 0; i < {loop_iters}; i++) acc += pad[i % {shmem_floats}];\n")
        else:
            loop_body = (f"        float acc = 0.0f;\n"
                         f"        for (int i = 0; i < {loop_iters}; i++) acc += (float) i * 1e-9f;\n")
        write_expr = "1.0f + acc * 0.0f"   # keeps slot[0]==1.0 numerically; acc's real work can't be dead-code-eliminated since it feeds the store
    elif shmem_floats > 0:
        # shmem requested but no duration loop: still need one real read
        # of `pad` so this path is structurally identical to the
        # loop_iters>0 case above (same write_expr shape), on top of
        # (not instead of) `volatile`'s own guarantee.
        loop_body = "        float acc = pad[0];\n"
        write_expr = "1.0f + acc * 0.0f"
    else:
        loop_body = ""
        write_expr = "1.0f"

    return f'''
extern "C" __global__ void {kname}(float* dbg) {{
{shmem_decl}{touch}    if (threadIdx.x == 0 && threadIdx.y == 0) {{
        unsigned int smid;
        asm("mov.u32 %0, %smid;" : "=r"(smid));
{loop_body}        float* slot = dbg + blockIdx.x * 2;
        slot[0] = {write_expr};
        slot[1] = (float) smid;
    }}
}}
'''


def compile_kernel(nch, dev, nvcc, kname, src):
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
    return elf_bytes


def run_config(dev, HCQBuffer, TinyELF, Target, BeagleNVProgram, nch, nvcc,
                shmem_floats, loop_iters, n_blocks, n_iters, verbose):
    kname = f"trivialProbe_s{shmem_floats}_l{loop_iters}"
    src = make_kernel_src(kname, shmem_floats, loop_iters)
    elf_bytes = compile_kernel(nch, dev, nvcc, kname, src)
    obj = TinyELF(lib=elf_bytes, name=kname, target=Target(), signature=())
    prg = BeagleNVProgram(dev, obj)
    log(f"[{kname}] compiled -- {len(elf_bytes)} byte ELF, "
        f"regs_usage={prg.regs_usage} shmem_usage={prg.shmem_usage} lcmem_usage={prg.lcmem_usage}")

    n_dbg = n_blocks * 2
    ran_count = [0] * n_blocks
    smid_by_block = [Counter() for _ in range(n_blocks)]

    header = f"\n=== {kname}: shmem={shmem_floats} floats ({shmem_floats*4} bytes) loop={loop_iters} -- {n_blocks} blocks x {n_iters} iterations ==="
    log(header)
    if verbose:
        print(header, file=sys.stdout, flush=True)

    for it in range(n_iters):
        dbg = dev.allocator.alloc(n_dbg * 4)
        dev.allocator._copyin(HCQBuffer(dbg.va_addr, n_dbg * 4), memoryview(struct.pack(f"<{n_dbg}f", *([SENTINEL] * n_dbg))))

        prg(HCQBuffer(dbg.va_addr, n_dbg * 4), global_size=(n_blocks, 1, 1), local_size=BLOCK, vals=(), wait=False)
        dev.synchronize()

        dbg_out = memoryview(bytearray(n_dbg * 4))
        dev.allocator._copyout(dbg_out, HCQBuffer(dbg.va_addr, n_dbg * 4))
        dbg_vals = struct.unpack(f"<{n_dbg}f", bytes(dbg_out))

        n_ran = 0
        for b in range(n_blocks):
            ran = dbg_vals[b * 2] != SENTINEL
            if ran:
                n_ran += 1
                ran_count[b] += 1
                smid = int(dbg_vals[b * 2 + 1]) if dbg_vals[b * 2 + 1] != SENTINEL else None
                if smid is not None:
                    smid_by_block[b][smid] += 1
        line = f"  iter {it:2d}: {n_ran:3d}/{n_blocks} ran"
        log(line)
        if verbose:
            print(line, file=sys.stdout, flush=True)

    if verbose:
        print(f"\n=== per-block ran rate ===", file=sys.stdout, flush=True)
    for b in range(n_blocks):
        line = f"  block {b:2d}: {ran_count[b]:2d}/{n_iters} ran ({100*ran_count[b]/n_iters:5.1f}%)"
        log(line)
        if verbose:
            print(line, file=sys.stdout, flush=True)

    if verbose:
        print(f"\n=== per-block observed SMID(s) ===", file=sys.stdout, flush=True)
    for b in range(n_blocks):
        fixed = len(smid_by_block[b]) <= 1
        smid_desc = ", ".join(f"smid={s}x{n}" for s, n in smid_by_block[b].most_common())
        line = (f"  block {b:2d}: ran {ran_count[b]:2d}/{n_iters}  "
                f"{'FIXED' if fixed else 'VARIES'} SMID  {smid_desc if smid_desc else '(never ran)'}")
        log(line)
        if verbose:
            print(line, file=sys.stdout, flush=True)

    # Aggregate by each block's dominant SMID -- every prior --sweep result has shown a
    # block's SMID is fixed per grid shape, so most_common()[0] is safe even for a block
    # that failed some iterations; a block that failed *every* iteration has no recorded
    # SMID at all and is excluded from the per-SMID table rather than guessed at.
    smid_ran = defaultdict(int)
    smid_total = defaultdict(int)
    never_ran_blocks = []
    for b in range(n_blocks):
        if smid_by_block[b]:
            smid = smid_by_block[b].most_common(1)[0][0]
            smid_ran[smid] += ran_count[b]
            smid_total[smid] += n_iters
        else:
            never_ran_blocks.append(b)

    if verbose:
        print(f"\n=== per-SMID ran rate (mechanical, directly comparable to nv_real_kernel_probe.py's --sweep tables) ===", file=sys.stdout, flush=True)
    for smid in sorted(smid_ran):
        ran, total = smid_ran[smid], smid_total[smid]
        line = f"  smid {smid:2d} (tpc {smid // 2}): {ran:3d}/{total:3d} ran ({100*ran/total:5.1f}%)"
        log(line)
        if verbose:
            print(line, file=sys.stdout, flush=True)
    if never_ran_blocks:
        line = f"  {len(never_ran_blocks)} block(s) never ran in any iteration, SMID unknown: {never_ran_blocks}"
        log(line)
        if verbose:
            print(line, file=sys.stdout, flush=True)

    total_ran = sum(ran_count)
    total_possible = n_blocks * n_iters
    bad_ran = sum(smid_ran.get(s, 0) for s in KNOWN_BAD_SMIDS)
    bad_total = sum(smid_total.get(s, 0) for s in KNOWN_BAD_SMIDS)
    good_ran = sum(smid_ran.get(s, 0) for s in KNOWN_GOOD_SMIDS)
    good_total = sum(smid_total.get(s, 0) for s in KNOWN_GOOD_SMIDS)
    return {
        "overall_pct": 100 * total_ran / total_possible,
        "known_bad_pct": (100 * bad_ran / bad_total) if bad_total else None,
        "known_good_pct": (100 * good_ran / good_total) if good_total else None,
        "regs_usage": prg.regs_usage,
        "shmem_usage": prg.shmem_usage,
    }


def main():
    argv = sys.argv[1:]
    shmem_floats = 0
    loop_iters = 0
    sweep_shmem = False
    sweep_loop = False
    while argv and argv[0] in ("--shmem", "--loop", "--sweep-shmem", "--sweep-loop"):
        if argv[0] == "--shmem":
            shmem_floats = int(argv[1])
            argv = argv[2:]
        elif argv[0] == "--loop":
            loop_iters = int(argv[1])
            argv = argv[2:]
        elif argv[0] == "--sweep-shmem":
            sweep_shmem = True
            argv = argv[1:]
        else:
            sweep_loop = True
            argv = argv[1:]
    n_blocks = int(argv[0]) if len(argv) >= 1 else 64
    n_iters = int(argv[1]) if len(argv) >= 2 else 20

    os.makedirs(os.path.expanduser("~/Library/Logs"), exist_ok=True)
    fd = os.open(os.path.expanduser("~/Library/Logs/nv_trivial_kernel_probe.log"),
                 os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_SYNC, 0o644)
    sys.stderr = os.fdopen(fd, 'w', buffering=1)

    log(f"starting -- shmem_floats={shmem_floats} loop_iters={loop_iters} "
        f"sweep_shmem={sweep_shmem} sweep_loop={sweep_loop} n_blocks={n_blocks} n_iters={n_iters}")
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

    if sweep_shmem or sweep_loop:
        values = SHMEM_SWEEP_VALUES if sweep_shmem else LOOP_SWEEP_VALUES
        dial_name = "shmem(floats)" if sweep_shmem else "loop(iters)"
        print(f"\n=== nv_trivial_kernel_probe --{'sweep-shmem' if sweep_shmem else 'sweep-loop'}: "
              f"{n_blocks} blocks x {n_iters} iterations per config ===", file=sys.stdout, flush=True)
        results = []
        for v in values:
            s, l = (v, 0) if sweep_shmem else (0, v)
            summary = run_config(dev, HCQBuffer, TinyELF, Target, BeagleNVProgram, nch, nvcc,
                                  s, l, n_blocks, n_iters, verbose=False)
            results.append((v, summary))
            # known_bad_pct/known_good_pct are None if n_blocks is small enough that none of
            # KNOWN_BAD_SMIDS/KNOWN_GOOD_SMIDS appear in this grid shape (they're all <=23,
            # so this only matters for n_blocks well below the default 64/32).
            bad_str = f"{summary['known_bad_pct']:.1f}%" if summary['known_bad_pct'] is not None else "N/A"
            good_str = f"{summary['known_good_pct']:.1f}%" if summary['known_good_pct'] is not None else "N/A"
            line = (f"  {dial_name}={v:>8}: overall={summary['overall_pct']:5.1f}%  "
                    f"known_bad_smids={bad_str}  known_good_smids={good_str}  "
                    f"(regs={summary['regs_usage']} shmem={summary['shmem_usage']}B)")
            log(line)
            print(line, file=sys.stdout, flush=True)
        print(f"\n(full per-iteration/per-block/per-SMID detail for every config above is in "
              f"~/Library/Logs/nv_trivial_kernel_probe.log)", file=sys.stdout, flush=True)
    else:
        run_config(dev, HCQBuffer, TinyELF, Target, BeagleNVProgram, nch, nvcc,
                   shmem_floats, loop_iters, n_blocks, n_iters, verbose=True)

    log("exiting cleanly")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc(file=sys.stderr)
        print("RESULT: FAIL (exception, see log)", file=sys.stdout, flush=True)
        sys.exit(1)
