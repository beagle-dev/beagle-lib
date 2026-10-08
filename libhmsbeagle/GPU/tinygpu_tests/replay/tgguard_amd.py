"""Guard mode for the AMD card (TODO.md plan step A2i): the interlock tgguard.py is for NV, for the first C++ AMD boots on
hardware. The recording proxy (tgproxy.py --guard) and the replay server (tgreplay.py --guard) feed it everything they see
once a session's first config read names vendor 0x1002; before a trigger write is forwarded, check_trigger() audits what
the GPU is about to use, and a reason (not None) means: do not forward it, hold (the proxy's fail-stop: both connections
stay open, and the user unplugs the eGPU before killing anything). It is a second layer beside the port's own checks (the
IOVA fence, refuse_mode1) and sees only the wire. Its shadow of the card: VRAM from the client's BAR0 writes and the replies
to its BAR0 reads; the BAR5 registers it needs from the client's writes and read replies, the HQD block banked by
GRBM_GFX_CNTL as on the GPU; the sysmem allocations from MAP_SYSMEM_FD. Registers are at the card's discovered bases
(fake_am_gpu.regs_for: tinygrad's tables for its IP versions and its captured discovery table, which the proxy and the replay
server pick by the session's device ID since TODO.md plan step N6; the fake's own card by default), the PSP's named as
AMDev names them (MPASP from MP0 14), and a page-table leaf is gfx12's PDE_PTE bit on GC 12. What it checks:
  - a TLB invalidation (GCVM or MMVM INVALIDATE_ENG17_REQ): the page tables written since the last, from GCVM_CONTEXT0's
    root (4 levels, PDE_PTE huge pages), and any table they newly point to: a system PTE must point into a live
    MAP_SYSMEM_FD allocation (a stray device address faults the Mac's DART), a VRAM page or table below the VRAM size, and
    no page table may be in sysmem; before the client has set a root, there is nothing to audit;
  - a queue going live (CP_HQD_ACTIVE 1, SDMA0_QUEUE0_RB_CNTL's rb_enable) and each doorbell (BAR2): the doorbell must be
    a live queue's, and its ring, read and write pointers must translate through the page tables to known memory;
  - the SMU's mode1 reset (the debug message 2 on mmMP1_SMN_C2PMSG_75): refused, as plan step A0 aborts on one.
A client that leaves with a queue still live is not let go (clean_exit false): both queues read system memory while they
live (the compute HQD polls its write pointer there, SDMA has wptr polling on), so TinyGPU.app must not unwire it under
them. tinygrad's fini ends both: AM_SDMA.fini_hw's rb_enable 0, and AM_GFX's dequeue read back inactive."""
import os, sys, struct, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import fake_am_gpu as amg
from tinygrad.runtime.autogen.am import am

PAGE = 0x1000
VENDOR = 0x1002

def trigger_addrs(meta=None):
    """{(bar, dword or byte offset): name}: the writes after which the GPU acts on what the client prepared, on the card of
    a captured table (meta: fake_am_gpu.table_for's; the fake's own card by default)."""
    meta = meta or amg.card()[1]
    R, psp = amg.regs_for(meta), amg.psp_pref(meta)
    a = lambda n: R[n].addr[0] * 4
    t = {(5, a(n)): n for n in ("regGCVM_INVALIDATE_ENG17_REQ", "regMMVM_INVALIDATE_ENG17_REQ", "regCP_HQD_ACTIVE", "regSDMA0_QUEUE0_RB_CNTL",
                                "mmMP1_SMN_C2PMSG_75", "mmMP1_SMN_C2PMSG_66", f"{psp}_35", f"{psp}_64", f"{psp}_67")}
    return t

class AMDGuard:
    def __init__(self, meta=None, log=print):
        self.log = log
        meta = meta or amg.card()[1]
        self.R = amg.regs_for(meta)
        self.pde_pte = amg.pde_pte(meta)
        self.A = lambda n: self.R[n].addr[0]
        self.hqd_lo, self.hqd_hi = self.A("regCP_MQD_BASE_ADDR") - 9, self.A("regCP_HQD_PQ_WPTR_HI") + 0x20   # fake_am_gpu's banked block
        self.vram, self.regs, self.hqd, self.sel = {}, {}, {}, (0, 0, 0)
        self.allocs, self.iova_pages = {}, set()
        self.vram_size, self.pt_pages, self.pt_dirty = None, {}, set()
        self.queues = {}               # doorbell byte offset -> (kind, ring va, ring size, rptr va, wptr va, HQD selection)
        self.ever_live = False
        self.stats = dict(flushes=0, audits=0, ptes=0, doorbells=0, queues=0)

    # ── what the proxy or the replay server sees ─────────────────────────────────────────────────────────────────────
    def on_sysmem(self, alloc, segs, size, mm):
        self.allocs[alloc] = (segs, size, mm)
        self.iova_pages |= {p + i for p, sz in segs for i in range(0, sz, PAGE)}

    def vram_write(self, off, data):
        for page in range(off // PAGE, (off + len(data) + PAGE - 1) // PAGE):
            lo, hi = max(off, page * PAGE), min(off + len(data), (page + 1) * PAGE)
            p = self.vram.setdefault(page, bytearray(PAGE))
            p[lo - page * PAGE:hi - page * PAGE] = data[lo - off:hi - off]
            if page in self.pt_pages: self.pt_dirty.add(page)
    def u64(self, off):
        p = self.vram.get(off // PAGE)
        return struct.unpack_from("<Q", p, off % PAGE)[0] if p is not None else 0

    def reg_set(self, dw, v):
        if self.hqd_lo <= dw < self.hqd_hi: self.hqd.setdefault(self.sel, {})[dw] = v
        else: self.regs[dw] = v
        if dw == self.A("regGRBM_GFX_CNTL"):
            f = self.R["regGRBM_GFX_CNTL"].decode(v)
            self.sel = (f["meid"], f["pipeid"], f["queueid"])
    def reg(self, dw):
        return self.hqd.get(self.sel, {}).get(dw, 0) if self.hqd_lo <= dw < self.hqd_hi else self.regs.get(dw, 0)

    def on_write(self, bar, off, data):
        if bar == 0: self.vram_write(off, data); return
        if bar != 5 or len(data) % 4: return
        for i in range(0, len(data), 4):
            dw, v = off // 4 + i // 4, struct.unpack_from("<I", data, i)[0]
            self.reg_set(dw, v)
            if dw == self.A("regSDMA0_QUEUE0_RB_CNTL") and not self.R["regSDMA0_QUEUE0_RB_CNTL"].decode(v)["rb_enable"]:
                self.queues = {k: q for k, q in self.queues.items() if q[0] != "sdma"}

    def on_read(self, bar, off, data):
        if bar == 0: self.vram_write(off, data); return   # what VRAM holds there
        if bar != 5 or len(data) != 4: return
        dw, v = off // 4, struct.unpack("<I", data)[0]
        if dw == amg.MEMSIZE: self.vram_size = v << 20
        if dw == self.A("regCP_HQD_ACTIVE") and not v & 1:   # a dequeue read back inactive: the selected queue is gone
            self.queues = {k: q for k, q in self.queues.items() if not (q[0] == "compute" and q[5] == self.sel)}
        self.reg_set(dw, v)

    def clean_exit(self): return not self.queues
    def state(self): return f"live queues at BAR2 {', '.join(hex(k) for k in sorted(self.queues)) or 'none'}; any queue this session: {self.ever_live}"

    # ── the checks ───────────────────────────────────────────────────────────────────────────────────────────────────
    def known(self, iova, n=1): return all(p in self.iova_pages for p in range(iova & ~(PAGE - 1), iova + max(n, 1), PAGE))
    def pair(self, lo, hi): return self.reg(self.A(lo)) | (self.reg(self.A(hi)) << 32)
    def root(self):
        b = self.pair("regGCVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_LO32", "regGCVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_HI32")
        return b & 0x0000FFFFFFFFF000 if b & 1 else None
    def translate(self, va):
        root = self.root()
        if root is None: return None
        off, paddr = va - (self.pair("regGCVM_CONTEXT0_PAGE_TABLE_START_ADDR_LO32", "regGCVM_CONTEXT0_PAGE_TABLE_START_ADDR_HI32") << 12), root
        if off < 0: return None
        for lv, shift in enumerate((39, 30, 21, 12)):
            pte = self.u64(paddr + ((off >> shift) & 0x1ff) * 8)
            if not pte & am.AMDGPU_PTE_VALID: return None
            addr = pte & 0x0000FFFFFFFFF000
            if lv == 3 or pte & self.pde_pte: return bool(pte & am.AMDGPU_PTE_SYSTEM), addr + (off & ((1 << shift) - 1))
            paddr = addr
        return None
    def mapped_known(self, va, what):
        t = self.translate(va)
        if t is None: return f"{what} at {va:#x} is not mapped in the GPU's page tables"
        is_sys, addr = t
        if is_sys and not self.known(addr): return f"{what} at {va:#x} is system address {addr:#x}, in no live MAP_SYSMEM_FD allocation"
        if not is_sys and self.vram_size is not None and addr >= self.vram_size: return f"{what} at {va:#x} is VRAM {addr:#x}, past the VRAM size"
        return None

    def audit_page_tables(self):
        root = self.root()
        if root is None: return None
        self.stats["audits"] += 1
        if root // PAGE not in self.pt_pages: self.pt_pages, self.pt_dirty = {root // PAGE: 0}, {root // PAGE}
        todo, self.pt_dirty = sorted(self.pt_dirty), set()
        while todo:
            page = todo.pop()
            lv = self.pt_pages[page]
            for i in range(512):
                pte = self.u64(page * PAGE + i * 8)
                if not pte & am.AMDGPU_PTE_VALID: continue
                addr, leaf = pte & 0x0000FFFFFFFFF000, lv == 3 or bool(pte & self.pde_pte)
                self.stats["ptes"] += 1
                if pte & am.AMDGPU_PTE_SYSTEM:
                    if not leaf: return f"a page table entry {pte:#x} at VRAM {page * PAGE + i * 8:#x} points at a table in system memory"
                    if not self.known(addr, 1 << (12 + 9 * (3 - lv))):
                        return f"a system PTE {pte:#x} (level {lv}) at VRAM {page * PAGE + i * 8:#x} points at {addr:#x}, in no live MAP_SYSMEM_FD allocation"
                elif self.vram_size is not None and addr >= self.vram_size:
                    return f"a VRAM {'page' if leaf else 'table'} at {addr:#x} past the VRAM size {self.vram_size:#x}"
                elif not leaf and addr // PAGE not in self.pt_pages:
                    self.pt_pages[addr // PAGE] = lv + 1
                    todo.append(addr // PAGE)
        return None

    def check_queue(self, kind):
        if kind == "compute":
            g = lambda n: self.reg(self.A(n))
            ring = (g("regCP_HQD_PQ_BASE") | (g("regCP_HQD_PQ_BASE_HI") << 32)) << 8
            size = 4 << (self.R["regCP_HQD_PQ_CONTROL"].decode(g("regCP_HQD_PQ_CONTROL"))["queue_size"] + 1)
            db = self.R["regCP_HQD_PQ_DOORBELL_CONTROL"].decode(g("regCP_HQD_PQ_DOORBELL_CONTROL"))["doorbell_offset"] * 4
            rptr = g("regCP_HQD_PQ_RPTR_REPORT_ADDR") | (g("regCP_HQD_PQ_RPTR_REPORT_ADDR_HI") << 32)
            wptr = g("regCP_HQD_PQ_WPTR_POLL_ADDR") | (g("regCP_HQD_PQ_WPTR_POLL_ADDR_HI") << 32)
        else:
            ring = self.pair("regSDMA0_QUEUE0_RB_BASE", "regSDMA0_QUEUE0_RB_BASE_HI") << 8
            size = 4 << self.R["regSDMA0_QUEUE0_RB_CNTL"].decode(self.reg(self.A("regSDMA0_QUEUE0_RB_CNTL")))["rb_size"]
            db = self.R["regSDMA0_QUEUE0_DOORBELL_OFFSET"].decode(self.reg(self.A("regSDMA0_QUEUE0_DOORBELL_OFFSET")))["offset"] * 4
            rptr = self.pair("regSDMA0_QUEUE0_RB_RPTR_ADDR_LO", "regSDMA0_QUEUE0_RB_RPTR_ADDR_HI")
            wptr = self.pair("regSDMA0_QUEUE0_RB_WPTR_POLL_ADDR_LO", "regSDMA0_QUEUE0_RB_WPTR_POLL_ADDR_HI")
        for va, what in ((ring, f"the {kind} ring"), (ring + size - 1, f"the {kind} ring's end"), (rptr, f"the {kind} read pointer"), (wptr, f"the {kind} write pointer")):
            if (why := self.mapped_known(va, what)): return why
        self.queues[db] = (kind, ring, size, rptr, wptr, self.sel)
        self.ever_live = True
        self.stats["queues"] += 1
        return None

    def check_trigger(self, bar, off, data):
        """Before a trigger write goes out: a reason to hold, or None."""
        if bar == 2:
            self.stats["doorbells"] += 1
            q = self.queues.get(off)
            if q is None: return f"a doorbell at BAR2+{off:#x}, which no live queue has"
            for va, what in ((q[1], "its ring"), (q[3], "its read pointer"), (q[4], "its write pointer")):
                if (why := self.mapped_known(va, f"the doorbell's queue: {what}")): return why
            return None
        if bar != 5 or len(data) != 4: return None
        dw, v = off // 4, struct.unpack("<I", data)[0]
        if dw in (self.A("regGCVM_INVALIDATE_ENG17_REQ"), self.A("regMMVM_INVALIDATE_ENG17_REQ")):
            self.stats["flushes"] += 1
            return self.audit_page_tables()
        if dw == self.A("mmMP1_SMN_C2PMSG_75") and v == 2: return "the SMU's mode1 reset (debug message 2), which BEAGLE never sends over TinyGPU"
        if dw == self.A("regCP_HQD_ACTIVE") and v & 1:
            self.reg_set(dw, v)
            return self.check_queue("compute")
        if dw == self.A("regSDMA0_QUEUE0_RB_CNTL") and self.R["regSDMA0_QUEUE0_RB_CNTL"].decode(v)["rb_enable"]:
            self.reg_set(dw, v)
            return self.check_queue("sdma")
        return None
