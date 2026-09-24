"""Device-free trace of tinygrad's GPU memory manager for BEAGLE's allocation sequence (the reference for porting
memory management to C++, TODO.md plan step C6). Runs tinygrad-hcq1's real NVMemoryManager, NVPageTableEntry and
PCIIfaceBase.alloc over a bytearray-backed fake BAR1 (no socket, no device), with BEAGLE's palloc zero patch, and
counts BAR1 reads/writes, BAR0 writes, sysmem allocations and page-table entries per phase.
    python mm_trace.py [--mmu 2|3]     # 2: Ada (8188 MiB, MMU v2); 3: Blackwell (16304 MiB, MMU v3)
The ctx-buffer sizes in the "golden" and "nvdevice" phases are approximations (recordings will pin them, plan V1).
trace(mmu, boot_images) is also the layout gate of plan step P2 (test_p2_teardown.py): boot_images are the sizes
of the falcon images palloc'd in VRAM during NV_FLCN.init_sw, before everything else here."""
import os, sys, collections, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tgpaths
tgpaths.setup()
import tinygrad.runtime.autogen.nv_regs.dev_mmu, tinygrad.runtime.autogen.nv_regs.dev_vm  # noqa: F401
from tinygrad.runtime.support.system import RemoteMMIOInterface, PCIIfaceBase
from tinygrad.runtime.support.nv.nvdev import NVDev, NVMemoryManager, NVPageTableEntry
from tinygrad.runtime.support.memory import TLSFAllocator, MemoryManager

MB = 1 << 20
BAR1 = 256 * MB
stats = collections.Counter()
levels = collections.Counter()
phase = ["boot"]

# BEAGLE's nv_init_helper.py patch 2: palloc zeroes only allocations of at most 64 KB
_orig_palloc = MemoryManager.palloc
def _palloc(self, size, align=0x1000, zero=True, boot=False, ptable=False):
    return _orig_palloc(self, size, align, zero=zero and size <= (64 << 10), boot=boot, ptable=ptable)
MemoryManager.palloc = _palloc

_orig_set_entry = NVPageTableEntry.set_entry
def _set_entry(self, i, paddr, table=False, **kw):
    levels[(self.lv, "table" if table else ("page" if kw.get("valid", True) else "invalidate"))] += 1
    return _orig_set_entry(self, i, paddr, table=table, **kw)
NVPageTableEntry.set_entry = _set_entry

class FakePCI:
    def __init__(self): self.vram = bytearray(BAR1); self.next_dma = 0x80000000
    def _bulk_read(self, cmd, idx, off, size):
        stats[(phase[0], "bar1_rd")] += 1
        assert off + size <= BAR1, f"read beyond BAR1 {off:#x}"
        return bytes(self.vram[off:off+size])
    def _bulk_write(self, cmd, idx, off, data):
        stats[(phase[0], "bar1_wr")] += 1; stats[(phase[0], "bar1_wr_bytes")] += len(data)
        if off + len(data) > BAR1: stats[(phase[0], "bar1_wr_dropped_beyond_256MiB")] += 1; return   # server.c drops these
        self.vram[off:off+len(data)] = data
    def bar_info(self, bar): return (0x4000000000, BAR1)
    def map_bar(self, bar, off=0, addr=0, size=None, fmt='B'): return RemoteMMIOInterface(self, bar, size or BAR1, fmt).view(off, size, fmt)
    def alloc_sysmem(self, size, vaddr=0, contiguous=False):
        stats[(phase[0], "sysmem_allocs")] += 1
        sz = max((size + 0xfff) & ~0xfff, 0x4000)
        base = self.next_dma; self.next_dma += (sz + 0x3fff) & ~0x3fff     # one DART segment
        return None, [base + i for i in range(0, sz, 0x1000)][:(size + 0xfff) // 0x1000]

def trace(MMU=2, boot_images=(), quiet=False):
    """Run tinygrad's memory manager for BEAGLE's allocation sequence; returns addresses of interest."""
    stats.clear(); levels.clear(); phase[0] = "boot"
    out = print if not quiet else (lambda *a, **k: None)
    pci = FakePCI()
    dev = NVDev.__new__(NVDev)
    dev.pci_dev, dev.devfmt, dev.smi_dev, dev.is_booting, dev.mmu_ver = pci, "usb4", False, True, MMU
    dev.wreg = lambda addr, val: stats.__setitem__((phase[0], "bar0_wr"), stats[(phase[0], "bar0_wr")] + 1)
    dev.include("dev_vm", "tu102"); dev.include("dev_mmu", "tu102" if MMU == 2 else "gh100")
    if MMU == 2: dev.pte_t, dev.pde_t, dev.dual_pde_t = dev.NV_MMU_VER2_PTE, dev.NV_MMU_VER2_PDE, dev.NV_MMU_VER2_DUAL_PDE
    else: dev.pte_t, dev.pde_t, dev.dual_pde_t = dev.NV_MMU_VER3_PTE, dev.NV_MMU_VER3_PDE, dev.NV_MMU_VER3_DUAL_PDE
    dev.vram_size = (8188 if MMU == 2 else 16304) * MB
    dev.vram = pci.map_bar(1)
    dev.large_bar = dev.vram.nbytes >= dev.vram_size
    NVMemoryManager.va_allocator = TLSFAllocator((1 << 44), base=0x1000000000)
    va_bits, va_shifts = (48, [12, 21, 29, 38, 47]) if MMU == 2 else (56, [12, 21, 29, 38, 47, 56])
    dev.mm = NVMemoryManager(dev, dev.vram_size - (64 << 20), boot_size=(2 << 20), pt_t=NVPageTableEntry, va_bits=va_bits,
                             va_shifts=va_shifts, va_base=0, palloc_ranges=[(x, x) for x in [512 << 20, 2 << 20, 4 << 10]],
                             reserve_ptable=not dev.large_bar)
    dev.is_booting = False
    out(f"MMU v{MMU}: ptable_allocator base/size {dev.mm.ptable_allocator.base:#x}/{dev.mm.ptable_allocator.size:#x}, "
          f"pa base {dev.mm.pa_allocator.base:#x}, root pt paddr {dev.mm.root_page_table.paddr:#x}")

    class Iface(PCIIfaceBase):
        def __init__(self): self.pci_dev, self.dev_impl, self.vram_bar, self.dev = pci, dev, 1, object()
    ifa, mm = Iface(), dev.mm

    phase[0] = "flcn"
    for size in boot_images: mm.palloc((size + 0xfff) & ~0xfff)   # NV_FLCN.init_sw's _alloc_boot_mem(sysmem=False)
    phase[0] = "golden"
    res_va = mm.alloc_vaddr(512 << 20); mm.page_tables(res_va, 512 << 20)
    mm.valloc(4 << 10, contiguous=True)                      # golden gpfifo_area
    mm.valloc(0x1000, contiguous=True); mm.palloc(0x5000)    # ramfc + method buffer
    for sz in [0x400000, 0x100000, 0x100000, 0x200000, 0x80000, 0x80000, 0x200000, 0x40000, 0x40000, 0x40000]:
        mm.valloc(sz, contiguous=True)                       # ctx buffers (sizes approximate)
    phase[0] = "nvdevice"
    gp = ifa.alloc(0x300000, contiguous=True, cpu_access=True, force_devmem=True)
    for _ in range(2):
        ifa.alloc(48 << 20, uncached=True); mm.valloc(0x1000, contiguous=True); mm.palloc(0x5000)
    for sz in [0x400000, 0x100000, 0x100000]: mm.valloc(sz, contiguous=True)   # compute promote_ctx (approximate)
    ifa.alloc(0x200000, cpu_access=True)                     # cmdq_page
    for _ in range(32): ifa.alloc(2 << 20, host=True)        # HCQAllocatorBase copy buffers
    ifa.alloc(0x1000, host=True, uncached=True, cpu_access=True)   # signal page
    ifa.alloc(16 << 20, cpu_access=True)                     # kernargs_buf
    phase[0] = "handoff"
    for sz in [2 << 20, 16 << 20, 16 << 20]: ifa.alloc(sz, cpu_access=True)
    ifa.alloc(0x1000, host=True, uncached=True, cpu_access=True)
    pool = ifa.alloc(dev.vram_size // 2)
    out(f"gpfifo_area paddr {gp.meta.mapping.paddrs[0][0]:#x}; pool va {pool.va_addr:#x}, {pool.size >> 20} MiB in "
          f"{len(pool.meta.mapping.paddrs)} segments, first {pool.meta.mapping.paddrs[0][0]:#x}, "
          f"end {pool.meta.mapping.paddrs[-1][0] + pool.meta.mapping.paddrs[-1][1]:#x}")
    result = {"vram_size": dev.vram_size, "gpfifo_paddr": gp.meta.mapping.paddrs[0][0], "gpfifo_size": 0x300000,
              "pool_paddr_end": max(p + sz for p, sz in pool.meta.mapping.paddrs), "stats": dict(stats)}
    phase[0] = "free_pool"
    mm.vfree(pool.meta.mapping)
    for k in sorted(stats): out(k, stats[k])
    out("page-table bytes used", dev.mm.ptable_allocator.size - sum(b[0] for s, b in dev.mm.ptable_allocator.blocks.items() if b[3]))
    out("set_entry by (level, kind):", dict(sorted(levels.items())))
    return result

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mmu", type=int, choices=(2, 3), default=2)
    trace(ap.parse_args().mmu)
