"""Guard mode: an independent interlock for the first C++ hardware runs (TODO.md plan step V1, inv:verification#15).

The recording proxy (tgproxy.py --guard) and the replay server (tgreplay.py --guard) feed it everything they see; before a
trigger write is forwarded, check_trigger() audits what the GPU is about to use, and a reason (not None) means: do not
forward it, hold (the proxy's fail-stop: the connection stays open and the user unplugs the eGPU before killing anything).
It is a second layer beside the ports' own checks and cannot see inside the GPU or the GSP. What it checks:
  - MMU invalidate: every valid PTE of the page tables (a shadow of VRAM built from the client's BAR1 writes and the
    replies to its BAR1 reads), walked from the page-directory root with tinygrad's MMU v2 field layouts (nvdev.py:33-67):
    a sysmem PTE must point at a page of a live MAP_SYSMEM_FD allocation, a VRAM PTE or page table below the VRAM size, and
    no page table may be in sysmem. The root comes from the client's COPY_SERVER_RESERVED_PDES or SET_PAGE_DIRECTORY RPC
    (tinygrad's golden image and NVDevice); an invalidate before either is refused;
  - falcon start (CPUCTL.startcpu or CPUCTL_ALIAS at the GSP or SEC2 base): the DMA source (DMATRFBASE, DMATRFBASE1) and
    every FB offset written since, below the VRAM size. On SEC2, unless its mailboxes hold Booter Unload's 0xff: the WPR meta
    they point to is a known device address, and so is everything reachable from it and from the libos arguments the GSP
    mailboxes point to: the radix3 image's page lists, the bootloader, the signature, each libos region, the RM arguments'
    queue page list (ip.py:364-455);
  - command-queue head: each element queued since the last one has a valid checksum (NVRpcQueue._checksum, ip.py:33-37);
  - doorbell: each GPFIFO entry from the channel's last position to its GPPut points at a pushbuffer mapped by the page
    tables to known memory (channels from the client's GPFIFO rm_alloc, tokens from the GSP's replies)."""
import struct, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import tgwire as w
import tggpu
from tinygrad.runtime.autogen import nv, nv_570 as nv_gpu

PAGE = 0x1000

class Guard:
    def __init__(self, log=print):
        self.log = log
        self.R = tggpu.regs("ada")
        names = w.reg_names()
        self.addr = {n: a for a, n in names.items()}
        self.vram = tggpu.Vram()
        self.allocs = {}          # allocation number -> (device page set, size, mapping)
        self.iova_pages = set()   # every 4 KiB device page of a live allocation
        self.regs = {}            # the last value written to each BAR0 register
        self.fb_offsets = {}      # falcon base -> FB offsets written since its last DMA base
        self.vram_size, self.root = None, None
        self.channels = tggpu.Channels()
        self.memory = tggpu.Memory(self.vram, self.sys_rw, self.R)
        self.cmdq = self.statq = None
        self.gsp_started = self.torn_down = False
        self.stats = dict(audits=0, ptes=0, structures=0, rpcs=0, pushbuffers=0)

    # ── what the proxy or the replay server sees ─────────────────────────────────────────────────────────────────────
    def on_sysmem(self, alloc, segs, size, mm):
        pages = {p + i for p, sz in segs for i in range(0, sz, PAGE)}
        self.allocs[alloc] = (segs, size, mm)
        self.iova_pages |= pages
        if size == w.QUEUES_SIZE and self.cmdq is None:
            self.cmdq, self.statq = tggpu.QueueReader(mm, w.CMD_QUEUE), tggpu.QueueReader(mm, w.STATUS_REGION[0])

    def on_write(self, bar, off, data):
        if bar == 1: self.vram.write(off, data); return
        if bar != 0 or len(data) != 4: return
        v = struct.unpack("<I", data)[0]
        self.regs[off] = v
        for base in w.FALCONS:
            if off == base + self.reg_off("DMATRFBASE"): self.fb_offsets[base] = []
            elif off == base + self.reg_off("DMATRFFBOFFS"): self.fb_offsets.setdefault(base, []).append(v)

    def on_read(self, bar, off, data):
        if bar == 1: self.vram.write(off, data); return   # what VRAM holds there (e.g. zeroed page tables)
        if bar == 0 and len(data) >= 4:
            v = struct.unpack_from("<I", data)[0]
            if off == self.addr["NV_PGC6_AON_SECURE_SCRATCH_GROUP_42"]: self.vram_size = v << 20   # nvdev.py:131
            if off == self.addr["NV_PFB_PRI_MMU_WPR2_ADDR_HI"] and v == 0 and self.gsp_started and self.unload_ran: self.torn_down = True

    def reg_off(self, name): r = getattr(self.R, f"NV_PFALCON_FALCON_{name}"); return r.base + r.off

    def sys_rw(self, iova, n, data=None):
        for segs, size, mm in self.allocs.values():
            off = 0
            for p, sz in segs:
                if p <= iova < p + sz and off + (iova - p) + n <= size:
                    o = off + iova - p
                    if data is None: return bytes(mm[o:o + n])
                    raise RuntimeError("the guard never writes")
                off += sz
        raise RuntimeError(f"{iova:#x} (+{n}) is not a device address of a live allocation")

    def known(self, iova, n=1): return all(p in self.iova_pages for p in range(iova & ~(PAGE - 1), iova + max(n, 1), PAGE))
    def clean_exit(self): return not self.gsp_started or self.torn_down
    def state(self): return f"GSP started: {self.gsp_started}, torn down: {self.torn_down}"

    # ── the audits ───────────────────────────────────────────────────────────────────────────────────────────────────
    def check_trigger(self, off, data):
        """None to forward the trigger, or why not."""
        try:
            v = struct.unpack("<I", data)[0] if len(data) == 4 else 0
            name = w.reg_name(off)
            self.stats["audits"] += 1
            if name == "NV_VIRTUAL_FUNCTION_PRIV_MMU_INVALIDATE": return self.audit_page_tables()
            if name == "NV_PGSP_QUEUE_HEAD[0]": return self.audit_rpcs()
            if name == "NV_VIRTUAL_FUNCTION_DOORBELL": return self.audit_doorbell(v)
            for base, fal in w.FALCONS.items():
                if name == f"{fal}.NV_PFALCON_FALCON_CPUCTL" and self.R.NV_PFALCON_FALCON_CPUCTL.decode(v)["startcpu"] or \
                   name == f"{fal}.NV_PFALCON_FALCON_CPUCTL_ALIAS" and v & 0x2:
                    return self.audit_falcon(base, fal)
            return None
        except Exception as e:   # the guard could not audit: never forward what it could not check
            return f"the audit failed: {type(e).__name__}: {e}"

    def audit_page_tables(self):
        if self.root is None: return "an MMU invalidate before any page-directory root is known (no COPY_SERVER_RESERVED_PDES or SET_PAGE_DIRECTORY yet)"
        if self.vram_size is None: return "an MMU invalidate before the VRAM size was read"
        pte, pde, dual = self.R.NV_MMU_VER2_PTE, self.R.NV_MMU_VER2_PDE, self.R.NV_MMU_VER2_DUAL_PDE
        stack, seen = [(self.root, 0, 0)], set()
        while stack:
            pt, lv, va = stack.pop()
            if (pt, lv) in seen: continue
            seen.add((pt, lv))
            if pt + PAGE > self.vram_size: return f"a level-{lv} page table at VRAM {pt:#x}, past the VRAM size"
            shift, cnt = tggpu.LEVELS[lv]
            is_dual = lv == len(tggpu.LEVELS) - 2
            raw = self.vram.read(pt, PAGE)
            for i in range(cnt):
                e = (struct.unpack_from("<Q", raw, 16 * i + 8)[0] << 64 | struct.unpack_from("<Q", raw, 16 * i)[0]) if is_dual else struct.unpack_from("<Q", raw, 8 * i)[0]
                if e == 0: continue
                eva = va | (i << shift)
                if lv == len(tggpu.LEVELS) - 1 or e & 1:   # a PTE (NVPageTableEntry.is_page)
                    f = pte.decode(e)
                    if not f["valid"]: continue
                    self.stats["ptes"] += 1
                    target, size = f["address_sys"] << 12, 1 << shift
                    if f["aperture"] == 2:
                        if not self.known(target, min(size, PAGE)): return f"the PTE for VA {eva:#x} points at device address {target:#x}, not a live sysmem page"
                    elif f["aperture"] == 0:
                        if target + size > self.vram_size: return f"the PTE for VA {eva:#x} points at VRAM {target:#x}+{size:#x}, past the VRAM size"
                    else: return f"the PTE for VA {eva:#x} has aperture {f['aperture']}"
                else:
                    f = (dual if is_dual else pde).decode(e)
                    ap = f["aperture_small" if is_dual else "aperture"]
                    if ap == 0: continue
                    if ap != 1: return f"a level-{lv} PDE for VA {eva:#x} puts its page table in aperture {ap} (only VRAM page tables are expected)"
                    stack.append((f["address_small_sys" if is_dual else "address_sys"] << 12, lv + 1, eva))
        return None

    def audit_rpcs(self):
        if self.cmdq is None: return None   # no queue mapping yet: nothing the GSP could read
        for fn, payload, elem, ok in self.cmdq.new():
            self.stats["rpcs"] += 1
            if not ok: return f"the queued RPC {nv.rpc_fns.get(fn, hex(fn))} has a bad checksum"
            self.channels.observe_cmd(fn, payload, self.memory)
            if fn == nv.NV_VGPU_MSG_FUNCTION_GSP_RM_CONTROL:
                c = nv.rpc_gsp_rm_control_v.from_buffer_copy(payload[:24])
                if c.cmd == nv_gpu.NV90F1_CTRL_CMD_VASPACE_COPY_SERVER_RESERVED_PDES:   # the golden image's page tables (ip.py:477-485)
                    p = nv_gpu.struct_NV90F1_CTRL_VASPACE_COPY_SERVER_RESERVED_PDES_PARAMS.from_buffer_copy(payload[24:24 + c.paramsSize])
                    if self.root is None: self.root = p.levels[0].physAddress
                    for i in range(p.numLevelsToCopy):
                        if self.vram_size and p.levels[i].physAddress >= self.vram_size: return f"COPY_SERVER_RESERVED_PDES level {i} at VRAM {p.levels[i].physAddress:#x}, past the VRAM size"
            if fn == nv.NV_VGPU_MSG_FUNCTION_SET_PAGE_DIRECTORY:
                pa = nv.rpc_set_page_directory_v.from_buffer_copy(payload[:48]).params.physAddress
                if self.vram_size and pa >= self.vram_size: return f"SET_PAGE_DIRECTORY at VRAM {pa:#x}, past the VRAM size"
                self.root = pa
        return None

    def audit_falcon(self, base, fal):
        if self.vram_size is None: return f"{fal} started before the VRAM size was read"
        dma = ((self.regs.get(base + self.reg_off("DMATRFBASE1"), 0) & 0x1ff) << 32 | self.regs.get(base + self.reg_off("DMATRFBASE"), 0)) << 8
        for fb in self.fb_offsets.get(base, [0]):
            if dma + fb + 256 > self.vram_size: return f"{fal}'s DMA reads VRAM {dma + fb:#x}, past the VRAM size"
        if fal != "SEC2": return None
        m0, m1 = self.regs.get(base + self.reg_off("MAILBOX0"), 0), self.regs.get(base + self.reg_off("MAILBOX1"), 0)
        if (m0, m1) == (0xff, 0xff):   # Booter Unload (nv_init_helper's teardown): its mailboxes are no address
            self.unload_ran = True
            return None
        why = self.audit_boot_structures(m0 | m1 << 32)
        if why is None: self.gsp_started = True   # booter_load starts GSP-RM, which runs from sysmem from here on
        return why

    unload_ran = False

    def audit_boot_structures(self, wpr_meta):
        """booter_load's WPR meta and the libos arguments, and every device address they lead to (NV_GSP.init_sw)."""
        if not self.known(wpr_meta, nv.GspFwWprMeta.SIZE): return f"SEC2's mailboxes point at {wpr_meta:#x}, not a live sysmem page (the WPR meta)"
        meta = nv.GspFwWprMeta.from_buffer_copy(self.sys_rw(wpr_meta, nv.GspFwWprMeta.SIZE))
        if meta.magic != nv.GSP_FW_WPR_META_MAGIC: return f"the WPR meta at {wpr_meta:#x} has magic {meta.magic:#x}"
        for what, a, n in (("bootloader", meta.sysmemAddrOfBootloader, meta.sizeOfBootloader), ("signature", meta.sysmemAddrOfSignature, meta.sizeOfSignature)):
            if not self.known(a, n): return f"the WPR meta's {what} at {a:#x} (+{n:#x}) is not live sysmem"
        # the radix3 image: three levels of page lists, then the image's pages, each level's entry count from the image size as
        # tinygrad computes it (NV_GSP.init_gsp_image, ip.py:393-411); a level's pages hold nothing meaningful past its entries
        # (there, whatever the allocation held: TinyGPU.app's segment list, for the first page)
        if not self.known(meta.sysmemAddrOfRadix3Elf): return f"the WPR meta's radix3 root {meta.sysmemAddrOfRadix3Elf:#x} is not live sysmem"
        npages = [0, 0, 0, (meta.sizeOfRadix3Elf + PAGE - 1) // PAGE]
        for i in range(3, 0, -1): npages[i - 1] = ((npages[i] - 1) >> (nv.LIBOS_MEMORY_REGION_RADIX_PAGE_LOG2 - 3)) + 1
        level = [meta.sysmemAddrOfRadix3Elf]
        for depth in range(3):
            want, nxt = npages[depth + 1], []
            for page in level:
                take = min(PAGE // 8, want - len(nxt))
                for (a,) in struct.iter_unpack("<Q", self.sys_rw(page, 8 * take)):
                    if not self.known(a): return f"radix3 level {depth} lists {a:#x}, not a live sysmem page"
                    nxt.append(a)
            level = nxt
            self.stats["structures"] += len(nxt)
        libos = self.regs.get(self.addr["NV_PGSP_FALCON_MAILBOX0"], 0) | self.regs.get(self.addr["NV_PGSP_FALCON_MAILBOX1"], 0) << 32
        if not self.known(libos, 6 * 32): return f"the GSP's mailboxes point at {libos:#x}, not a live sysmem page (the libos arguments)"
        for i in range(6):
            r = nv.LibosMemoryRegionInitArgument.from_buffer_copy(self.sys_rw(libos + 32 * i, 32))
            if not self.known(r.pa, r.size): return f"libos region {i} ({r.id8:#x}) at {r.pa:#x} (+{r.size:#x}) is not live sysmem"
            if r.id8 == int.from_bytes(b"RMARGS", "big"):
                q = nv.GSP_ARGUMENTS_CACHED.from_buffer_copy(self.sys_rw(r.pa, nv.GSP_ARGUMENTS_CACHED.SIZE)).messageQueueInitArguments
                for (a,) in struct.iter_unpack("<Q", self.sys_rw(q.sharedMemPhysAddr, 8 * q.pageTableEntryCount)):
                    if not self.known(a): return f"the RM arguments' queue page list has {a:#x}, not a live sysmem page"
        return None

    def audit_doorbell(self, value):
        if self.statq is not None:
            for fn, payload, elem, ok in self.statq.new(): self.channels.observe_reply(fn, payload)
        ch = self.channels.for_doorbell(value)
        if ch is None: return f"a doorbell with token {value:#x}, which no channel was given"
        userd = ch["ring"] + ch["entries"] * 8
        put = struct.unpack("<I", self.memory.read(userd + getattr(nv_gpu.AmpereAControlGPFifo, "GPPut").offset, 4))[0]
        get = ch.setdefault("guard_get", 0)
        while get % ch["entries"] != put:
            entry = struct.unpack("<Q", self.memory.read(ch["ring"] + (get % ch["entries"]) * 8, 8))[0]
            va, n = entry & ((1 << 40) - 1), ((entry >> 42) & 0x1fffff) * 4
            sysm, a = self.memory.translate(va)   # raises for an unmapped VA
            if sysm and not self.known(a, n): return f"a GPFIFO entry's pushbuffer at VA {va:#x} maps to {a:#x}, not live sysmem"
            if not sysm and a + n > self.vram_size: return f"a GPFIFO entry's pushbuffer at VA {va:#x} maps past the VRAM size"
            get += 1; self.stats["pushbuffers"] += 1
        ch["guard_get"] = get
        return None
