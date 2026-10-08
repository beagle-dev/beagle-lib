"""A register-level RX 7900 XT (1002:744c, gfx1100) for the AMD boot (TODO.md plan step A2a): tinygrad's AM mock
(test/mockgpu/am/amgpu.py at the pin, written for GC 12.0.0) retargeted to this card, so tinygrad's own AMDev boots on it,
full and partial, and so does BEAGLE's C++ port. fake_amd_device.py serves it as TinyGPU.app would.

The card: its captured IP discovery table (STATUS.md R65) at the end of its 20464 MiB of VRAM, read through the indirect
window (MM_INDEX, MM_INDEX_HI, MM_DATA); the register bases from that table; tinygrad's register tables for its IP versions
(gc 11.0.0, mmhub 3.0.0, osssys 6.0.0, nbio 4.3.0, hdp 6.0.0, mp 13.0.0, and mp 11.0.0 at MP1's bases, as AMDev builds
them); a 256 MiB BAR0 (not resized), 2 MiB of doorbells, 1 MiB of registers. Registers past BAR5 go through the RSMU
window (BIF_BX_PF0_RSMU_INDEX/DATA). Every register reads what was last written, except:
  - the mock's: the PSP's bootloader is always ready, its SOS comes alive once LOAD_SOSDRV is written, its KM ring is
    created and destroyed, and each frame its write pointer passes is answered (status 0, the fence written; a LOAD_TOC
    gets a TMR size); the SMU answers every message (and GetDpmFreqByIndex from a DPM table, GetSmuVersion with a version);
    CP_STAT and RLC_SAFE_MODE's cmd bit read 0; the TLB invalidations are acknowledged at once (and the MMHUB's semaphore
    taken); the FB location registers hold the card's aperture; the HDP remap register points at a flush register;
  - the HQD registers (CP_MQD_*, CP_HQD_*) are banked by GRBM_GFX_CNTL's me, pipe and queue, as on the GPU, and a dequeue
    request deactivates the selected queue (not with FAKE_AMD_WEDGED=1: a wave that survives RESET_WAVES);
  - an SMU mode1 reset (the debug message) puts the card back to its power-on state.
Starting states (FAKE_AMD_STATE): cold (power-on: SCRATCH_REG7 0, the SOS not alive, no PSP ring), warm (an AM boot
finalized: SCRATCH_REG7 = AMDev.Version, SCRATCH_REG6 0; a partial boot) or dirty (SCRATCH_REG6 1: a full boot with a
mode1 reset first). The state lasts across client sessions, as the GPU's does. TODO.md plan step N1's checks:
FAKE_AMD_DEVICE_ID=<hex> puts another device ID in the config space (the card staying an RX 7900 XT), and FAKE_AMD_BAR0_MB=<n>
serves a BAR0 of n MiB; plan step N3's: FAKE_AMD_MEMSIZE=<hex> is what RCC_CONFIG_MEMSIZE reads.
Plan step N6: FAKE_AMD_CHIP=gfx1201 plays the RDNA 4 card whose table run_amd_discovery_ro.sh captured (1002:7550, Navi 48):
its table, config-space identity, register modules for its IP versions (nbif from GC 12, the PSP's MPASP registers from MP0
14) and gfx12's page-table leaf bit, with the gfx11 card's PSP, SMU and GPU behaviour otherwise (a boot on it waits for
its firmware, plan step N9). The default, gfx1100, is the RX 7900 XT.

The GPU: a queue goes live when its registers say so (CP_HQD_ACTIVE for the compute queue, SDMA0_QUEUE0_RB_CNTL's rb_enable
for SDMA), and a doorbell then runs it (fake_amd_device.py's PM4 and SDMA 6 executor), every address translated through
the GMC page tables in this VRAM: GCVM_CONTEXT0's base, 4 levels (39, 30, 21, 12), huge pages by PDE_PTE. A system PTE
must point into a MAP_SYSMEM_FD allocation's DMA segments: at every TLB invalidation the page tables written since the
last are audited (the DART check: a stray system address is a host panic on the Mac), and so is every GPU access."""
import os, sys, json, glob, struct, functools, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tgpaths
tgpaths.setup()
from tinygrad.runtime.support.amd import AMDReg, import_asic_regs, import_module
from tinygrad.runtime.autogen.am import am

PAGE = 0x1000
WEDGED = os.environ.get("FAKE_AMD_WEDGED", "") == "1"   # a dequeue request leaves the compute queue active (plan step A2k)
VERSION = 0xA0000008                       # AMDev.Version (amdev.py:147)
BARS = {0: (0x2e_4000_0000, int(os.environ.get("FAKE_AMD_BAR0_MB", "256")) << 20), 2: (0x2e_5000_0000, 2 << 20),
        5: (0x2e_0030_0000, 1 << 20)}   # the card's (STATUS.md R64)
BAR5_DWORDS = BARS[5][1] // 4
CHIP = os.environ.get("FAKE_AMD_CHIP", "gfx1100")   # plan step N6: whose captured table the fake plays
CHIP_IDS = {"gfx1100": "744c", "gfx1201": "7550"}
MM_INDEX, MM_DATA, MM_INDEX_HI = 0x0, 0x1, 0x6   # AMDev._read_vram's window (amdev.py:341-348)
MEMSIZE = 0xde3                            # mmRCC_CONFIG_MEMSIZE (amdev.py:353)
HDP_FLUSH_DW = 0x7f000 // 4                # where the remap register points (nbio 4.3's MMIO_REG_HOLE_OFFSET page)
FB_BASE = 0x80_0000_0000                   # MMMC_VM_FB_LOCATION_BASE << 24
SOS_VERSION, SMU_VERSION, TMR_SIZE = 0x00290043, 0x004e3b00, 0x00b00000
DPM = {0: [500, 2394], 1: [500, 960, 1200], 2: [96, 456, 772, 1250], 3: [400, 1100, 1600, 1900]}   # PPCLK GFX, SOC, U, F (MHz)
DPM_FINE = {0}                             # fine-grained DPM: the count's bit 31 set (tinygrad masks it)
CFG = {0x00: int(CHIP_IDS[CHIP], 16) << 16 | 0x1002, 0x04: 0x00100006, 0x08: 0x030000cc, 0x2c: 0x0e3b1002, 0x34: 0x48,
       0x48: 0x5009, 0x50: 0x6401, 0x64: 0xa010, 0x74: 0x0042, 0xa0: 0x0005}   # caps: vendor 0x48 -> PM 0x50 -> PCIe 0x64 -> MSI 0xa0

def card():
    """The captured discovery table (its binary_size bytes) and what was recorded with it."""
    meta_path = sorted(glob.glob(str(tgpaths.DATA / f"discovery/1002_{CHIP_IDS[CHIP]}_*.json")))[0]
    meta = json.load(open(meta_path))
    return open(meta_path[:-5] + ".bin", "rb").read(), meta
_meta = card()[1]   # the identity the capture recorded (amd_discovery_ro.py's tables; the RX 7900 XT's predates it)
if "revision" in _meta: CFG[0x08] = _meta["class"] << 8 | _meta["revision"]
if "subsystem" in _meta: CFG[0x2c] = int(_meta["subsystem"][5:], 16) << 16 | int(_meta["subsystem"][:4], 16)
if os.environ.get("FAKE_AMD_DEVICE_ID"): CFG[0x00] = int(os.environ["FAKE_AMD_DEVICE_ID"], 16) << 16 | 0x1002
def ipv(meta): return {k: tuple(v) for k, v in meta["ip_ver"].items()}
def psp_pref(meta): return "regMP0_SMN_C2PMSG" if ipv(meta)["MP0_HWIP"] < (14, 0, 0) else "regMPASP_SMN_C2PMSG"   # AM_PSP.reg_pref (ip.py:590)
def pde_pte(meta): return am.AMDGPU_PDE_PTE_GFX12 if ipv(meta)["GC_HWIP"] >= (12, 0, 0) else am.AMDGPU_PDE_PTE   # AM_GMC's leaf test (ip.py:192)
IPV, PSP_PREF, PDE_PTE = ipv(_meta), psp_pref(_meta), pde_pte(_meta)

def table_for(device):
    """TODO.md plan step N6: the captured table of the AMD card with this PCI device ID, (bin, meta), for the guard of a
    real session (tgproxy, tgreplay); None when there is none, or more than one to choose from."""
    paths = sorted(glob.glob(str(tgpaths.DATA / f"discovery/1002_{device:04x}_*.json")))
    if len(paths) != 1: return None
    return open(paths[0][:-5] + ".bin", "rb").read(), json.load(open(paths[0]))

def regs_for(meta):
    """name -> AMDReg at a captured table's bases, as AMDev._build_regs builds them for its IP versions (amdev.py:398-409)."""
    bases = {ip: {int(i): tuple(b) for i, b in v.items()} for ip, v in meta["regs_offset"].items()}
    v, R = ipv(meta), {}
    nbio = "nbio" if v["GC_HWIP"] < (12, 0, 0) else "nbif"   # amdev.py:400
    for prefix, ver, ip in (("mp", v["MP0_HWIP"], "MP0_HWIP"), ("hdp", v["HDP_HWIP"], "HDP_HWIP"), ("gc", v["GC_HWIP"], "GC_HWIP"),
                            ("mmhub", v["MMHUB_HWIP"], "MMHUB_HWIP"), ("osssys", v["OSSSYS_HWIP"], "OSSSYS_HWIP"),
                            (nbio, v["NBIO_HWIP"], "NBIO_HWIP"), ("mp", (11, 0, 0), "MP1_HWIP")):
        R.update(import_asic_regs(prefix, ver, cls=functools.partial(AMDReg, bases=bases[ip])))
    return R

def props_for(device):
    """PCIIface._compute_props (ops_amd.py:910-925) and AMDDevice.__init__'s counts (:1003-1012) on the captured table of the
    card with this PCI device ID, one XCC: its target, arch, xccs, cu_cnt, se_cnt, max_slots_scratch_cu and lds_size_in_kb."""
    import ctypes, types
    from tinygrad.runtime.ops_amd import PCIIface
    table, meta = table_for(device)
    off = am.struct_binary_header.from_buffer_copy(table).table_list[am.GC].offset
    h = am.struct_gc_info_v1_0.from_buffer_copy(table[off:off + ctypes.sizeof(am.struct_gc_info_v1_0)]).header
    T = getattr(am, f"struct_gc_info_v{h.version_major}_{h.version_minor}")
    gc = ipv(meta)["GC_HWIP"]
    it = types.SimpleNamespace(dev_impl=types.SimpleNamespace(ip_ver={am.GC_HWIP: gc}, gc_info=T.from_buffer_copy(table[off:off + ctypes.sizeof(T)]),
                                                              gfx=types.SimpleNamespace(xccs=1)))
    PCIIface._compute_props(it)
    p = it.props
    return dict(target=gc, arch="gfx%d%x%x" % gc, xccs=1, cu_cnt=p["simd_count"] // p["simd_per_cu"], se_cnt=p["array_count"] // p["simd_arrays_per_engine"],
                max_slots_scratch_cu=p["max_slots_scratch_cu"], lds_size_in_kb=p["lds_size_in_kb"])

@functools.cache
def card_regs():
    """name -> AMDReg at this card's bases, as AMDev._build_regs builds them (amdev.py:398-409)."""
    return regs_for(card()[1])

class AMGpu:
    def __init__(self, state=None, log=print):
        self.log = log
        self.table, self.meta = card()
        self.vram_size = self.meta["vram_size"]
        self.R = card_regs()
        self.name = {}   # dword -> register name
        for n, r in self.R.items(): self.name.setdefault(r.addr[0], n)
        a = lambda n: self.R[n].addr[0]
        self.A = a
        self.psp = {k: a(f"{PSP_PREF}_{k}") for k in (35, 36, 64, 67, 69, 70, 71, 81)}
        self.smu = {k: a(f"mmMP1_SMN_C2PMSG_{k}") for k in (53, 54, 66, 75, 82, 90)}
        self.smu_mod = import_module("smu", IPV["MP1_HWIP"])
        self.hqd_lo, self.hqd_hi = a("regCP_MQD_BASE_ADDR") - 9, a("regCP_HQD_PQ_WPTR_HI") + 0x20   # the CP_MQD_*/CP_HQD_* block
        self.errors, self.counts, self.touched = [], collections.Counter(), collections.Counter()
        self.reset(state or os.environ.get("FAKE_AMD_STATE", "warm"))

    def err(self, msg):
        if len(self.errors) < 50: self.log(f"fake AMD GPU: ERROR {msg}")
        self.errors.append(msg)

    # ── state ──────────────────────────────────────────────────────────────────────────────────────────────────────────
    def reset(self, state):
        """Power-on, plus what an earlier AM session left (warm, dirty). VRAM keeps the discovery table at its end."""
        assert state in ("cold", "warm", "dirty"), state
        self.state, self.r, self.vram, self.hqd, self.sel = state, {}, {}, collections.defaultdict(dict), (0, 0, 0, 0)
        self.cfg = bytearray(256)
        for off, v in CFG.items(): self.cfg[off:off + 4] = struct.pack("<I", v)
        self.sos_alive, self.smu_pending = state != "cold", None
        self.queues, self.pt_pages, self.pt_dirty, self.sysmem_segs = {}, {}, set(), []
        self.vram_write(self.vram_size - (64 << 10), self.table)
        self.r[self.A("regMMMC_VM_FB_LOCATION_BASE")] = FB_BASE >> 24
        self.r[self.A("regMMMC_VM_FB_LOCATION_TOP")] = (FB_BASE + self.vram_size - 1) >> 24
        if state != "cold":
            self.r[self.A("regSCRATCH_REG7")] = VERSION
            self.r[self.A("regSCRATCH_REG6")] = 1 if state == "dirty" else 0
            self.r[self.psp[71]] = 0x10000   # the PSP ring an earlier session created
            self.r[self.psp[81]] = SOS_VERSION

    def mode1_reset(self):
        self.counts["mode1 resets"] += 1
        segs = self.sysmem_segs
        self.reset("cold")
        self.sysmem_segs = segs

    # ── PCI config space ───────────────────────────────────────────────────────────────────────────────────────────────
    def cfg_read(self, off, size):
        if off + size > 256: return 0xffffffff
        return int.from_bytes(self.cfg[off:off + size], "little")
    def cfg_write(self, off, size, val):
        if off + size <= 256: self.cfg[off:off + size] = (val & ((1 << (8 * size)) - 1)).to_bytes(size, "little")

    # ── VRAM (BAR0's window and the indirect one) ──────────────────────────────────────────────────────────────────────
    def vram_read(self, off, n):
        out = bytearray()
        for page in range(off // PAGE, (off + n + PAGE - 1) // PAGE):
            lo, hi = max(off, page * PAGE), min(off + n, (page + 1) * PAGE)
            p = self.vram.get(page)
            out += p[lo - page * PAGE:hi - page * PAGE] if p is not None else bytes(hi - lo)
        return bytes(out)
    def vram_write(self, off, data):
        for page in range(off // PAGE, (off + len(data) + PAGE - 1) // PAGE):
            lo, hi = max(off, page * PAGE), min(off + len(data), (page + 1) * PAGE)
            p = self.vram.setdefault(page, bytearray(PAGE))
            p[lo - page * PAGE:hi - page * PAGE] = data[lo - off:hi - off]
            if page in self.pt_pages: self.pt_dirty.add(page)
    def u32(self, off): return struct.unpack("<I", self.vram_read(off, 4))[0]
    def u64(self, off): return struct.unpack("<Q", self.vram_read(off, 8))[0]

    # ── registers ──────────────────────────────────────────────────────────────────────────────────────────────────────
    def decode(self, name, v=None): return self.R[name].decode(self.r.get(self.A(name), 0) if v is None else v)
    def pair(self, lo, hi): return self.r.get(self.A(lo), 0) | (self.r.get(self.A(hi), 0) << 32)
    def banked(self, dw): return self.hqd_lo <= dw < self.hqd_hi

    def rd(self, dw):
        self.touched[("r", self.name.get(dw, hex(dw)))] += 1
        if dw == MEMSIZE: return int(os.environ["FAKE_AMD_MEMSIZE"], 16) if os.environ.get("FAKE_AMD_MEMSIZE") else self.vram_size >> 20
        if dw == MM_DATA:
            addr = (self.r.get(MM_INDEX_HI, 0) << 31) | (self.r.get(MM_INDEX, 0) & 0x7fffffff)
            return self.u32(addr) if addr + 4 <= self.vram_size else 0
        if dw == self.A("regBIF_BX_PF0_RSMU_DATA"): return self.rd(self.r.get(self.A("regBIF_BX_PF0_RSMU_INDEX"), 0) // 4)
        if dw == self.A("regBIF_BX0_REMAP_HDP_MEM_FLUSH_CNTL"): return HDP_FLUSH_DW * 4
        if dw == self.psp[35]: return 0x80000000                                  # the bootloader is ready
        if dw == self.psp[81]: return SOS_VERSION if self.sos_alive else 0
        if dw == self.psp[64]: return 0x80000000 if self.sos_alive else 0         # sOS ready, the last ring command's status 0
        if dw in (self.A("regCP_STAT"),): return 0
        if dw == self.A("regRLC_SAFE_MODE"): return self.r.get(dw, 0) & ~1       # the RLC acknowledged the command
        if dw in (self.A("regGCVM_INVALIDATE_ENG17_ACK"), self.A("regMMVM_INVALIDATE_ENG17_ACK")): return 0xffff
        if dw == self.A("regMMVM_INVALIDATE_ENG17_SEM"): return 1
        if self.banked(dw): return self.hqd[self.sel[:3]].get(dw, 0)
        return self.r.get(dw, 0)

    def wr(self, dw, v):
        self.touched[("w", self.name.get(dw, hex(dw)))] += 1
        if dw == self.A("regBIF_BX_PF0_RSMU_DATA"): return self.wr(self.r.get(self.A("regBIF_BX_PF0_RSMU_INDEX"), 0) // 4, v)
        if dw == HDP_FLUSH_DW:
            self.hdp_flushed = True
            self.counts["hdp flushes"] += 1
            return
        if self.banked(dw):
            q = self.hqd[self.sel[:3]]
            q[dw] = v
            if dw == self.A("regCP_HQD_ACTIVE") and v & 1: self.activate_compute()
            if dw == self.A("regCP_HQD_DEQUEUE_REQUEST") and v:   # (setup_ring writes the MQD's 0: no request)
                self.counts["hqd dequeue requests"] += 1
                if WEDGED: return   # a wave the reset does not clear: the queue stays active
                q[self.A("regCP_HQD_ACTIVE")] = 0
                self.queues = {k: x for k, x in self.queues.items() if x.get("hqd") != self.sel[:3]}
            return
        self.r[dw] = v
        if dw == self.A("regGRBM_GFX_CNTL"):
            f = self.decode("regGRBM_GFX_CNTL", v)
            self.sel = (f["meid"], f["pipeid"], f["queueid"], f["vmid"])
        elif dw == self.psp[35]:
            if v == am.PSP_BL__LOAD_SOSDRV: self.sos_alive = True
            self.counts["psp bootloader loads"] += 1
        elif dw == self.psp[64]:
            if v == am.GFX_CTRL_CMD_ID_DESTROY_RINGS: self.r[self.psp[71]] = 0
            elif v == am.PSP_RING_TYPE__KM << 16: self.r[self.psp[67]] = 0       # a fresh ring starts at 0
        elif dw == self.psp[67]: self.psp_frames(v)
        elif dw == self.smu[66]: self.smu_msg(v, self.r.get(self.smu[82], 0), 90, 82)
        elif dw == self.smu[75]: self.smu_msg(v, self.r.get(self.smu[53], 0), 54, 53, debug=True)
        elif dw in (self.A("regGCVM_INVALIDATE_ENG17_REQ"), self.A("regMMVM_INVALIDATE_ENG17_REQ")): self.tlb_flush()
        elif dw == self.A("regSDMA0_QUEUE0_RB_CNTL"):
            if self.decode("regSDMA0_QUEUE0_RB_CNTL", v)["rb_enable"]: self.activate_sdma()
            else: self.queues = {k: x for k, x in self.queues.items() if x["kind"] != "sdma"}

    # ── PSP and SMU ────────────────────────────────────────────────────────────────────────────────────────────────────
    def mc_base(self): return (self.r.get(self.A("regMMMC_VM_FB_LOCATION_BASE"), 0) & 0xFFFFFF) << 24
    def psp_frames(self, new_wptr):
        old = self.psp_wptr if hasattr(self, "psp_wptr") and self.psp_wptr <= new_wptr else 0
        ring = (self.r.get(self.psp[69], 0) | (self.r.get(self.psp[70], 0) << 32)) - self.mc_base()
        for w in range(old, new_wptr, 16):
            fr = self.vram_read(ring + w * 4, 64)
            cmd_lo, cmd_hi, _, fence_lo, fence_hi, fence_value = struct.unpack_from("<IIIIII", fr)
            cmd = (cmd_lo | (cmd_hi << 32)) - self.mc_base()
            cmd_id = self.u32(cmd + 8)
            self.counts[f"psp cmd {cmd_id:#x}"] += 1
            self.vram_write(cmd + 864, struct.pack("<I", 0))                       # resp.status
            if cmd_id == am.GFX_CMD_ID_LOAD_TOC: self.vram_write(cmd + 864 + 16, struct.pack("<I", TMR_SIZE))
            self.vram_write((fence_lo | (fence_hi << 32)) - self.mc_base(), struct.pack("<I", fence_value))
        self.psp_wptr = new_wptr
    def smu_msg(self, msg, param, resp_reg, arg_reg, debug=False):
        s = self.smu_mod
        self.counts[f"smu{' debug' if debug else ''} msg {msg}"] += 1
        self.r[self.smu[resp_reg]] = 1
        if debug:
            if msg == 2: self.mode1_reset(); self.r[self.smu[resp_reg]] = 1   # __DEBUGSMC_MSG_Mode1Reset (ip.py:211)
            return
        if msg == s.PPSMC_MSG_GetSmuVersion: self.r[self.smu[arg_reg]] = SMU_VERSION
        elif msg == s.PPSMC_MSG_GetDpmFreqByIndex:
            clk, idx = param >> 16, param & 0xffff
            levels = DPM.get(clk, [])
            if idx == 0xff: self.r[self.smu[arg_reg]] = len(levels) | ((1 << 31) if clk in DPM_FINE else 0)
            else: self.r[self.smu[arg_reg]] = levels[idx] if idx < len(levels) else 0

    # ── queues ─────────────────────────────────────────────────────────────────────────────────────────────────────────
    def activate_compute(self):
        q, a = self.hqd[self.sel[:3]], self.A
        g = lambda n: q.get(a(n), 0)
        ring = ((g("regCP_HQD_PQ_BASE") | (g("regCP_HQD_PQ_BASE_HI") << 32)) << 8)
        size = 4 << (self.R["regCP_HQD_PQ_CONTROL"].decode(g("regCP_HQD_PQ_CONTROL"))["queue_size"] + 1)
        db = self.R["regCP_HQD_PQ_DOORBELL_CONTROL"].decode(g("regCP_HQD_PQ_DOORBELL_CONTROL"))["doorbell_offset"] * 4
        self.queues[db] = dict(kind="compute", hqd=self.sel[:3], ring=ring, size=size, unit=4, done=0, doorbell=db,
                               rptr=g("regCP_HQD_PQ_RPTR_REPORT_ADDR") | (g("regCP_HQD_PQ_RPTR_REPORT_ADDR_HI") << 32),
                               wptr=g("regCP_HQD_PQ_WPTR_POLL_ADDR") | (g("regCP_HQD_PQ_WPTR_POLL_ADDR_HI") << 32))
        self.counts["compute queues activated"] += 1
    def activate_sdma(self):
        p = lambda lo, hi: self.pair(lo, hi)
        ring = p("regSDMA0_QUEUE0_RB_BASE", "regSDMA0_QUEUE0_RB_BASE_HI") << 8
        size = 4 << self.decode("regSDMA0_QUEUE0_RB_CNTL")["rb_size"]
        db = self.decode("regSDMA0_QUEUE0_DOORBELL_OFFSET")["offset"] * 4
        self.queues[db] = dict(kind="sdma", ring=ring, size=size, unit=1, done=0, doorbell=db,
                               rptr=p("regSDMA0_QUEUE0_RB_RPTR_ADDR_LO", "regSDMA0_QUEUE0_RB_RPTR_ADDR_HI"),
                               wptr=p("regSDMA0_QUEUE0_RB_WPTR_POLL_ADDR_LO", "regSDMA0_QUEUE0_RB_WPTR_POLL_ADDR_HI"))
        self.counts["sdma queues activated"] += 1

    # ── the GMC page tables ────────────────────────────────────────────────────────────────────────────────────────────
    def pt_root(self):
        base = self.pair("regGCVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_LO32", "regGCVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_HI32")
        return None if not base & 1 else base & 0x0000FFFFFFFFF000
    def vm_start(self): return self.pair("regGCVM_CONTEXT0_PAGE_TABLE_START_ADDR_LO32", "regGCVM_CONTEXT0_PAGE_TABLE_START_ADDR_HI32") << 12
    def add_sysmem(self, segs): self.sysmem_segs.append(segs)
    def in_sysmem(self, iova, n):
        return any(p <= iova and iova + n <= p + s for segs in self.sysmem_segs for p, s in segs)
    def translate(self, va, n):
        """(is_sys, address) of [va, va+n), which must not cross a page of the mapping, or None if unmapped."""
        root = self.pt_root()
        if root is None: return None
        off, paddr = va - self.vm_start(), root
        if off < 0: return None
        for lv, shift in enumerate((39, 30, 21, 12)):
            pte = self.u64(paddr + ((off >> shift) & 0x1ff) * 8)
            if not pte & am.AMDGPU_PTE_VALID: return None
            addr = pte & 0x0000FFFFFFFFF000
            if lv == 3 or pte & PDE_PTE:
                inpage = off & ((1 << shift) - 1)
                if inpage + n > (1 << shift): return None
                return bool(pte & am.AMDGPU_PTE_SYSTEM), addr + inpage
            paddr = addr
        return None
    def tlb_flush(self):
        """The DART audit: every page-table page written since the last flush, and any table it newly points to."""
        self.counts["tlb flushes"] += 1
        root = self.pt_root()
        if root is None: return
        if root // PAGE not in self.pt_pages:
            self.pt_pages = {root // PAGE: 0}
            self.pt_dirty = {root // PAGE}
        todo = sorted(self.pt_dirty)
        self.pt_dirty = set()
        while todo:
            page = todo.pop()
            lv = self.pt_pages[page]
            for i in range(512):
                pte = self.u64(page * PAGE + i * 8)
                if not pte & am.AMDGPU_PTE_VALID: continue
                addr = pte & 0x0000FFFFFFFFF000
                leaf = lv == 3 or pte & PDE_PTE
                if pte & am.AMDGPU_PTE_SYSTEM:
                    if not leaf: self.err(f"a page directory entry {pte:#x} at VRAM {page * PAGE + i * 8:#x} marked system")
                    elif not self.in_sysmem(addr, 1 << (12 + 9 * (3 - lv))):
                        self.err(f"a system PTE {pte:#x} (level {lv}) points at {addr:#x}, which no MAP_SYSMEM_FD allocation holds (a DART panic)")
                    self.counts["system PTEs audited"] += 1
                elif not leaf and addr // PAGE not in self.pt_pages:
                    self.pt_pages[addr // PAGE] = lv + 1
                    todo.append(addr // PAGE)
                elif leaf: self.counts["VRAM PTEs audited"] += 1
