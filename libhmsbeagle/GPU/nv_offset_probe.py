#!/usr/bin/env python3
"""
nv_offset_probe.py -- TODO.md "PICK UP HERE" -> NV Phase 59.

Phase 58 (the Ds[2]<->Ds[3] swap) settled that kernelMatrixMulADB's one
residual ground-truth failure (Phase 57, after the exp() bisection fix)
tracks the store's *target slot/address* (byte offset 0x2c within each
block's own 16-REAL debug region), not the value being written there --
reopening the undocumented Blackwell `desc[UR]` global-memory-descriptor
addressing mechanism (Phase 5/6) as the leading candidate, with real
evidence behind it for the first time.

But the data so far can't tell apart two different addressing hypotheses,
because every prior probe's per-block regions were all exactly 64 bytes
(16 REALs) -- a multiple of 16:
  (a) ABSOLUTE hypothesis: some fixed byte offset from the whole buffer's
      *base address* fails, periodically, independent of any per-block
      structure (consistent with a hardware/descriptor addressing defect
      tied to real memory addresses).
  (b) RELATIVE hypothesis: some fixed offset *within each block's own
      write sequence* fails, regardless of that block's absolute address
      (consistent with a per-block descriptor-setup or store-count/
      scheduling artifact that resets for every block).
  With 64-byte-aligned, 16-REAL slots, (a) and (b) predict the *identical*
  observable pattern (offset 0x2c in every slot, since 64 is itself a
  multiple of 16) -- indistinguishable from data collected so far.

This probe breaks that degeneracy on purpose: each block's write region is
33 REALs (132 bytes) -- deliberately *not* a multiple of 16 (132 mod 16 ==
4). If the failure tracks a fixed *global* byte offset from the buffer's
base, each block's own base shifts by 4 bytes (mod 16) relative to the
previous block's, so the *local* (within-block) offset where the failure
lands should shift by 4 bytes from block to block too. If the failure
tracks a fixed *local* offset instead, it should land at the exact same
local offset in every block regardless of that block's absolute address.

Also deliberately minimal compared to every real-kernel-adjacent probe so
far (Phase 47-58): no shared memory, no barrier, no branch -- just one
thread per block doing 33 sequential, unconditional global stores. If the
periodic failure still appears in this maximally simple context, that's
the strongest possible confirmation it's a pure addressing/store-sequence
phenomenon, independent of everything else this investigation has already
tried and ruled out (broadcast+barrier idiom, branch divergence, resource
footprint). If it *doesn't* appear here, that's equally informative --
means kernelMatrixMulADB's broader structure still matters somehow.

    python3 nv_offset_probe.py
"""
import sys, os, pathlib, struct

_TINYGRAD_PATH = os.environ.get("TINYGRAD_PATH", str(pathlib.Path.home() / "Dropbox/Projects/tinygrad"))
sys.path.insert(0, _TINYGRAD_PATH)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

GRID = (16, 1, 1)   # exact shape from the real kernelMatrixMulADB launch log
BLOCK = (16, 16, 1)
SLOT_REALS = 33  # deliberately not a multiple of 16 -- see module docstring
SENTINEL = -999.0

CUDA_SRC = f'''
extern "C" __global__ void offsetprobe(float* dbg) {{
    if (threadIdx.x == 0 && threadIdx.y == 0) {{
        float* slot = dbg + blockIdx.x * {SLOT_REALS};
#pragma unroll
        for (int i = 0; i < {SLOT_REALS}; i++)
            slot[i] = (float)(blockIdx.x * 1000 + i);
    }}
}}
'''


def log(msg):
    print(f"[nv_offset_probe] {msg}", file=sys.stderr, flush=True)


def main():
    os.makedirs(os.path.expanduser("~/Library/Logs"), exist_ok=True)
    fd = os.open(os.path.expanduser("~/Library/Logs/nv_offset_probe.log"),
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
    import tempfile, subprocess
    with tempfile.NamedTemporaryFile(suffix=".cu", delete=False, mode="w") as cf:
        cf.write(CUDA_SRC)
        cu_path = cf.name
    ptx_path = cu_path.replace(".cu", ".ptx")

    nvcc = os.environ.get("TINYGPU_NVCC", os.path.expanduser("~/.local/bin/nvcc"))
    r = subprocess.run([nvcc, "-ptx", "-o", ptx_path, cu_path], capture_output=True)
    if r.returncode != 0:
        log(f"nvcc failed: {r.stderr.decode(errors='replace')}")
        sys.exit(1)
    log("nvcc -ptx OK")

    elf_bytes = nch.compile_ptx(ptx_path, dev.arch, kernel_name="offsetprobe")
    log(f"ptxas OK -- {len(elf_bytes)} byte ELF")

    from nv_dispatch_daemon import BeagleNVProgram
    from tinygrad.device import TinyELF, Target
    from tinygrad.runtime.support.hcq import HCQBuffer

    obj = TinyELF(lib=elf_bytes, name="offsetprobe", target=Target(), signature=())
    prg = BeagleNVProgram(dev, obj)
    log(f"BeagleNVProgram constructed -- regs_usage={prg.regs_usage} shmem_usage={prg.shmem_usage}")

    n = GRID[0] * SLOT_REALS
    d_dbg = dev.allocator.alloc(n * 4)
    sentinel_buf = memoryview(struct.pack(f"<{n}f", *([SENTINEL] * n)))
    dev.allocator._copyin(HCQBuffer(d_dbg.va_addr, n * 4), sentinel_buf)
    log(f"debug buffer allocated + sentinel-seeded: {d_dbg.va_addr:#x} ({n} floats, {SLOT_REALS} per block)")

    prg(HCQBuffer(d_dbg.va_addr, n * 4), global_size=GRID, local_size=BLOCK, vals=(), wait=False)
    log("launched, syncing...")
    dev.synchronize()
    log("synchronized -- no fault")

    out = memoryview(bytearray(n * 4))
    dev.allocator._copyout(out, HCQBuffer(d_dbg.va_addr, n * 4))
    vals = struct.unpack(f"<{n}f", bytes(out))

    print(f"\n=== nv_offset_probe: {SLOT_REALS}-REAL ({SLOT_REALS*4}-byte) slots per block, "
          f"{GRID[0]} blocks ===", file=sys.stdout, flush=True)
    local_fail_offsets_by_block = {}
    all_local_fails = []
    for b in range(GRID[0]):
        fails, wrong = [], []
        for i in range(SLOT_REALS):
            v = vals[b * SLOT_REALS + i]
            expected = float(b * 1000 + i)
            if v == SENTINEL:
                fails.append(i)
            elif v != expected:
                wrong.append((i, v))  # neither sentinel nor expected -- worth flagging on its own
        local_fail_offsets_by_block[b] = fails
        all_local_fails.extend(fails)
        local_byte_offsets = [i * 4 for i in fails]
        global_byte_offsets = [(b * SLOT_REALS + i) * 4 for i in fails]
        line = (f"block {b:2d}: {len(fails)} failure(s)  local_byte_offsets={local_byte_offsets}  "
                f"global_byte_offsets={global_byte_offsets}")
        if wrong:
            line += f"  UNEXPECTED VALUES (neither sentinel nor expected): {wrong}"
        log(line)
        print(line, file=sys.stdout, flush=True)

    print("\n=== analysis ===", file=sys.stdout, flush=True)
    log("=== analysis ===")
    if not all_local_fails:
        line = "No failures at all -- this maximally-simple context (no shared mem/barrier/branch) is clean."
        log(line)
        print(line, file=sys.stdout, flush=True)
    else:
        distinct_local = sorted(set(local_fail_offsets_by_block[b][0] * 4 for b in range(GRID[0]) if local_fail_offsets_by_block[b]))
        line1 = f"distinct LOCAL (within-slot) failure byte-offsets seen across blocks: {distinct_local}"
        log(line1); print(line1, file=sys.stdout, flush=True)
        if len(distinct_local) == 1:
            line2 = ("SAME local offset in every failing block -> RELATIVE hypothesis: "
                     "failure tracks a fixed offset within each block's own write sequence.")
        else:
            line2 = ("Local offset SHIFTS from block to block -> check whether it shifts by "
                     f"exactly (block_index * {SLOT_REALS*4} bytes) mod 16 == 4*block_index mod 16 -- "
                     "if so, ABSOLUTE hypothesis: failure tracks a fixed global byte offset from the buffer base.")
        log(line2); print(line2, file=sys.stdout, flush=True)

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
