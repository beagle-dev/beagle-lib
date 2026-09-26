"""A fake TinyGPU.app that plays an RTX 4060 (AD107) at the register level, for tinygrad's real boot (TODO.md plan step
V1: the offline stand-in for the eGPU behind the recording proxy and the replay tools, and later for the C++ boot ports).
The real daemon (nv_dispatch_daemon.py: tinygrad's NVDev, NV_FLCN, NV_GSP and NVDevice, with nv_init_helper's patches and
the P2 teardown) boots it, the C++ runtime dispatches on it, and the daemon unloads it, over TinyGPU.app's protocol.

What it models, and nothing more:
  - BAR0: the chip ids, the VRAM size, the VBIOS window (a captured AD107 VBIOS, $BEAGLE_TINYGPU_DATA/vbios), falcons that
    halt once started, a WPR2 that FWSEC-FRTS raises and Booter Unload lowers, and the GSP mailbox that reports the suspend
    after the unload RPC; every other register reads what was last written (0 before);
  - BAR1: VRAM, the 256 MiB window TinyGPU.app maps (sparse, and sparse VRAM above it for the GPU's own copies);
  - MAP_SYSMEM_FD: files in a private directory, handed out by fd, each with a DMA segment list at made-up device
    addresses (one segment up to 2 MiB, 2 MiB segments with gaps between them above, at most 32, below 2^40), cleaned up at
    the end of a session as server.c does;
  - a GSP that starts when SEC2 runs booter_load: it finds its queues from the libos arguments in the GSP mailboxes (checking
    the page list tinygrad wrote), sets up its status queue, answers every RPC (rm_alloc, rm_control with the values
    tinygrad reads, set_page_directory, the unload), and posts GSP_INIT_DONE, never an unrequested event; every command's
    sequence number must follow the one before (a client that continues the queue, such as the C++ side, must keep its count);
  - a GPU front end: on a doorbell it runs the channel's GPFIFO through the client's own page tables (tinygrad's MMU v2
    decoders): semaphore acquires and releases, QMD launches and their releases (kernels are not run), copy-engine DMA.
Values tinygrad reads that the real GPU would compute (the GR topology, context-buffer sizes, runlists) are plausible
constants, not the RTX 4060's: plan step V1's L0 recordings are the reference for those.
A session that ends while the GSP is live is an error: on the eGPU that unwires memory the GSP still uses (DART).
FAKE_RM_FAIL=<class>: the GSP refuses every rm_alloc of that class (rpc_result NV_ERR_INVALID_CLASS), as GSP-RM refuses a bad
request; the client's stop is then the test's (plan step C7: the C++ side must stop before any submission).
    <tinygrad venv>/python fake_nv_device.py <socket path> <memory dir>
It prints "fake TinyGPU.app (AD107 device) listening", and after each session its counts and NO ERRORS or the errors."""
import os, sys, json, mmap, glob, socket, struct, ctypes, types, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tgpaths
tgpaths.setup()
from tinygrad.runtime.autogen import nv, nv_570 as nv_gpu
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "replay"))
import tggpu   # VRAM, the MMU walk, the GPFIFO front end and the GSP queues, shared with the replay server

sock_path, MEM = sys.argv[1], sys.argv[2]
REQ, RESP = struct.Struct("<BIIQQQ"), struct.Struct("<BQQ")
MAP_BAR, MAP_SYSMEM_FD, CFG_READ, CFG_WRITE, MMIO_READ, MMIO_WRITE, RESIZE_BAR = 1, 2, 3, 4, 6, 7, 11
PAGE, MB = 0x1000, 1 << 20
VRAM_MB = 8188                                  # the RTX 4060's NV_PGC6_AON_SECURE_SCRATCH_GROUP_42
BARS = {0: (0x1c_0000_0000, 16 * MB), 1: (0x1d_0000_0000, 256 * MB), 3: (0x1e_0000_0000, 32 * MB)}
BOOT_0, BOOT_42 = 0x197000a1, 0x19700000        # AD107 (architecture 0x19, implementation 7; as test_b1_cot.py)
CFG = {0: 0x288210de, 4: 0x00100006, 8: 0x030000a1, 0x2c: 0x88861458}   # 10de:2882, command/status, class+revision, subsystem
GSP_BASE, SEC2_BASE = 0x110000, 0x840000
IOVA_BASE, IOVA_STRIDE = 0x40_0000_0000, 0x4000_0000
GR_INFO = {"num_gpcs": 3, "num_tpc_per_gpc": 4, "num_sm_per_tpc": 2, "max_warps_per_sm": 48, "sm_version": 0x809}   # as fake_nv_daemon
CTX_BUF = (0x20000, 0x1000)                     # every GR context buffer's (size, alignment)
errors, counts = [], collections.Counter()
# FAKE_TG_RECORD=<file>: every byte a client sends is appended to it (as fake_tinygpu_server.py; plan step V1's proxy check)
RECORD = open(os.environ["FAKE_TG_RECORD"], "ab") if os.environ.get("FAKE_TG_RECORD") else None
RM_FAIL = int(os.environ.get("FAKE_RM_FAIL", "0"), 0)

def err(msg):
    errors.append(msg)
    print(f"fake device: ERROR: {msg}", flush=True)

# ── tinygrad's register tables (Ada's include sequence) ─────────────────────────────────────────────────────────────
R = tggpu.regs("ada")
def addr(reg, base=0, idx=None): return base + reg.base + (reg.off(idx) if idx is not None else reg.off)
A = types.SimpleNamespace(
    WPR2_LO=addr(R.NV_PFB_PRI_MMU_WPR2_ADDR_LO), WPR2_HI=addr(R.NV_PFB_PRI_MMU_WPR2_ADDR_HI), BOOT_0=addr(R.NV_PMC_BOOT_0),
    BOOT_42=addr(R.NV_PMC_BOOT_42), VRAM=addr(R.NV_PGC6_AON_SECURE_SCRATCH_GROUP_42),
    PLM=addr(R.NV_PGC6_AON_SECURE_SCRATCH_GROUP_05_PRIV_LEVEL_MASK), GFW=addr(R.NV_PGC6_AON_SECURE_SCRATCH_GROUP_05, idx=0),
    BSI14=addr(R.NV_PGC6_BSI_SECURE_SCRATCH_14), QUEUE_HEAD=addr(R.NV_PGSP_QUEUE_HEAD, idx=0),
    GSP_ENGINE=addr(R.NV_PGSP_FALCON_ENGINE), SEC2_ENGINE=addr(R.NV_PSEC_FALCON_ENGINE), DOORBELL=0xbb0090)
FALCON_REGS = {n: getattr(R, f"NV_PFALCON_FALCON_{n}") for n in ("CPUCTL", "HWCFG2", "DMATRFCMD", "DMATRFBASE", "DMATRFBASE1", "MAILBOX0", "MAILBOX1")}
CPUCTL_ALIAS = R.NV_PFALCON_FALCON_CPUCTL_ALIAS   # a plain offset in tinygrad's tables (ip.py:230)
BCR, RISCV_CPUCTL = R.NV_PRISCV_RISCV_BCR_CTRL, R.NV_PRISCV_RISCV_CPUCTL
VBIOS_BASE, VBIOS = 0x300000, open(sorted(glob.glob(str(tgpaths.DATA / "vbios" / "AD107_*.rom")))[0], "rb").read()

# ── memory ────────────────────────────────────────────────────────────────────────────────────────────────────────────
class Sysmem:
    """One MAP_SYSMEM_FD allocation: a file, and its made-up DMA segments."""
    def __init__(self, n, size, contiguous):
        self.n, self.size = n, max((size + 0xfff) & ~0xfff, 0x4000)   # server.c: page-aligned, at least 16 KiB
        self.path = os.path.join(MEM, f"sysmem_{n}.bin")
        self.fd = os.open(self.path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o600)
        os.ftruncate(self.fd, self.size)
        self.mm = mmap.mmap(self.fd, self.size)
        # one segment up to 2 MiB, 2 MiB segments above (at most 32, the dext's DMA list): tinygrad hands the GSP one device address
        # for its bootloader and log buffer (ip.py:388-392, 419-432), so those must be contiguous; nothing asks for it (server.c
        # ignores the contiguous flag), and plan V1's L0 recordings show the real lists
        seg = self.size if contiguous or self.size <= (2 << 20) else max(2 << 20, (-(-self.size // 32) + (2 << 20) - 1) // (2 << 20) * (2 << 20))
        base = IOVA_BASE + n * IOVA_STRIDE
        self.segs = [(base + i * (seg + 0x10000), min(seg, self.size - off)) for i, off in enumerate(range(0, self.size, seg))]
        self.mm[:16 * len(self.segs) + 16] = b"".join(struct.pack("<QQ", p, s) for p, s in self.segs) + bytes(16)
    def offset(self, iova):
        off = 0
        for p, s in self.segs:
            if p <= iova < p + s: return off + iova - p
            off += s
        return None
    def close(self):
        self.mm.close(); os.close(self.fd); os.unlink(self.path)

# ── the device ────────────────────────────────────────────────────────────────────────────────────────────────────────
class Device:
    def __init__(self):
        self.vram, self.regs, self.sysmem, self.n_alloc = tggpu.Vram(), {}, [], 0
        self.wpr2 = False
        self.falcon = {GSP_BASE: dict(halted=False, riscv_active=False), SEC2_BASE: dict(halted=False, riscv_active=False)}
        self.gsp = None          # set up when booter_load starts GSP-RM
        self.unloaded = False    # after the unload RPC: the next GSP falcon run is FWSEC-SB, the next SEC2 run Booter Unload
        self.memory = tggpu.Memory(self.vram, self.sys_rw, R)
        self.channels = tggpu.Channels()
        self.frontend = tggpu.Frontend(self.memory, self.channels, counts, err)

    # sysmem by device address
    def sys_rw(self, iova, n, data=None):
        for s in self.sysmem:
            off = s.offset(iova)
            if off is not None:
                if off + n > s.size: break
                if data is None: return bytes(s.mm[off:off + n])
                s.mm[off:off + n] = data; return
        raise RuntimeError(f"device access to {iova:#x} (+{n}): not a device address of any live sysmem allocation")

    # ── BAR0 ─────────────────────────────────────────────────────────────────────────────────────────────────────────
    def rd32(self, a):
        if VBIOS_BASE <= a < VBIOS_BASE + len(VBIOS): return struct.unpack_from("<I", VBIOS, a - VBIOS_BASE)[0]
        if a == A.BOOT_0: return BOOT_0
        if a == A.BOOT_42: return BOOT_42
        if a == A.VRAM: return VRAM_MB
        if a == A.PLM: return R.NV_PGC6_AON_SECURE_SCRATCH_GROUP_05_PRIV_LEVEL_MASK.encode(read_protection_level0=1)
        if a == A.GFW: return 0xff
        if a == A.BSI14: return R.NV_PGC6_BSI_SECURE_SCRATCH_14.encode(boot_stage_3_handoff=1)
        if a == A.WPR2_LO: return R.NV_PFB_PRI_MMU_WPR2_ADDR_LO.encode(val=((VRAM_MB * MB - 2 * MB) >> 12)) if self.wpr2 else 0
        if a == A.WPR2_HI: return R.NV_PFB_PRI_MMU_WPR2_ADDR_HI.encode(val=((VRAM_MB * MB - 1 * MB) >> 12)) if self.wpr2 else 0
        for base, st in self.falcon.items():
            if a == addr(FALCON_REGS["CPUCTL"], base): return FALCON_REGS["CPUCTL"].encode(halted=int(st["halted"]), alias_en=0)
            if a == addr(FALCON_REGS["HWCFG2"], base): return FALCON_REGS["HWCFG2"].encode(mem_scrubbing=0, riscv=1)
            if a == addr(FALCON_REGS["DMATRFCMD"], base): return FALCON_REGS["DMATRFCMD"].encode(full=0, idle=1)
            if a == addr(BCR, base): return self.regs.get(a, 0) | BCR.encode(valid=1)
            if a == addr(RISCV_CPUCTL, base): return RISCV_CPUCTL.encode(active_stat=int(st["riscv_active"]), halted=int(not st["riscv_active"]))
        return self.regs.get(a, 0)

    def wr32(self, a, v):
        self.regs[a] = v
        if a == A.QUEUE_HEAD:
            if self.gsp: self.gsp.run()
            return
        if a == A.DOORBELL: self.frontend.doorbell(v); return
        for base, st in self.falcon.items():
            start = (a == addr(FALCON_REGS["CPUCTL"], base) and FALCON_REGS["CPUCTL"].decode(v)["startcpu"]) or \
                    (a == base + CPUCTL_ALIAS and v & 0x2)
            if start: self.falcon_run(base); return
        if a in (A.GSP_ENGINE, A.SEC2_ENGINE) and R.NV_PGSP_FALCON_ENGINE.decode(v)["reset"]:
            st = self.falcon[GSP_BASE if a == A.GSP_ENGINE else SEC2_BASE]
            st["halted"] = False
            if a == A.GSP_ENGINE: st["riscv_active"] = False

    def falcon_run(self, base):
        """What the falcon's ucode does, decided by which falcon and the boot's phase; then it halts."""
        mbx = lambda n: self.regs.get(addr(FALCON_REGS[n], base), 0)
        st = self.falcon[base]
        if base == GSP_BASE:
            counts["FWSEC-SB" if self.unloaded else "FWSEC-FRTS"] += 1
            if not self.unloaded: self.wpr2 = True   # FRTS sets up WPR2 (the scratch error codes read 0: none)
        elif not self.unloaded:   # booter_load, handed the WPR meta's device address: GSP-RM starts
            counts["booter_load"] += 1
            if not self.wpr2: err("booter_load ran with WPR2 down")
            wpr_meta = mbx("MAILBOX0") | mbx("MAILBOX1") << 32
            self.regs[addr(FALCON_REGS["MAILBOX0"], base)] = 0
            self.falcon[GSP_BASE]["riscv_active"] = True
            self.gsp = Gsp(self, wpr_meta)
        else:   # Booter Unload (mailboxes 0xff): WPR2 comes down
            counts["booter_unload"] += 1
            if (mbx("MAILBOX0"), mbx("MAILBOX1")) != (0xff, 0xff): err(f"Booter Unload's mailboxes are {mbx('MAILBOX0'):#x}, {mbx('MAILBOX1'):#x}")
            self.wpr2 = False
            self.regs[addr(FALCON_REGS["MAILBOX0"], base)] = 0
        st["halted"] = True

# ── the GSP ───────────────────────────────────────────────────────────────────────────────────────────────────────────
class Gsp:
    """GSP-RM as far as tinygrad and BEAGLE see it: the message queues, a reply to every RPC, GSP_INIT_DONE."""
    def __init__(self, dev, wpr_meta_iova):
        self.dev = dev
        meta = nv.GspFwWprMeta.from_buffer_copy(dev.sys_rw(wpr_meta_iova, nv.GspFwWprMeta.SIZE))
        if meta.magic != nv.GSP_FW_WPR_META_MAGIC: err(f"booter_load's WPR meta has magic {meta.magic:#x}")
        libos = dev.regs.get(addr(FALCON_REGS["MAILBOX0"], GSP_BASE), 0) | dev.regs.get(addr(FALCON_REGS["MAILBOX1"], GSP_BASE), 0) << 32
        args = [nv.LibosMemoryRegionInitArgument.from_buffer_copy(dev.sys_rw(libos + 32 * i, 32)) for i in range(6)]
        rm = next((a for a in args if a.id8 == int.from_bytes(b"RMARGS", "big")), None)
        if rm is None: raise RuntimeError("no RMARGS region in the libos arguments the GSP mailboxes point to")
        q = nv.GSP_ARGUMENTS_CACHED.from_buffer_copy(dev.sys_rw(rm.pa, nv.GSP_ARGUMENTS_CACHED.SIZE)).messageQueueInitArguments
        s = next(s for s in dev.sysmem if s.offset(q.sharedMemPhysAddr) == 0)
        pages = [p + i for p, sz in s.segs for i in range(0, sz, PAGE)]
        ptes = struct.unpack_from(f"<{q.pageTableEntryCount}Q", s.mm, 0)
        if list(ptes) != pages[:q.pageTableEntryCount]: err("the queue page list does not match the queue memory's device addresses")
        self.mm, self.cmd, self.stat = s.mm, q.cmdQueueOffset, q.statQueueOffset
        cmd_tx = nv.msgqTxHeader.from_buffer_copy(self.mm[self.cmd:self.cmd + 32])
        self.msg_size, self.msg_count = cmd_tx.msgSize, cmd_tx.msgCount
        self.cmd_rx = self.cmd + cmd_tx.rxHdrOff          # where the CPU keeps its status-queue read pointer
        self.stat_rx = self.stat + 32                     # where this GSP keeps its command-queue read pointer
        self.seq, self.cmd_seq, self.cmdq = 0, 0, tggpu.QueueReader(self.mm, self.cmd)
        stat_tx = nv.msgqTxHeader(version=0, size=cmd_tx.size, entryOff=0x1000, msgSize=self.msg_size, msgCount=self.msg_count,
                                  writePtr=0, flags=0, rxHdrOff=32)
        self.mm[self.stat:self.stat + 32] = bytes(stat_tx)
        self.mm[self.stat_rx:self.stat_rx + 4] = struct.pack("<I", 0)
        self.run()                                        # the prequeued SET_SYSTEM_INFO and SET_REGISTRY
        self.post(nv.NV_VGPU_MSG_EVENT_GSP_INIT_DONE, b"\x00" * 8)
        counts["gsp boots"] += 1

    def u32(self, off): return struct.unpack_from("<I", self.mm, off)[0]

    def run(self):
        """Every command the CPU has queued since the last run, in order."""
        for fn, msg, elem, ok in self.cmdq.new():
            if not ok: err(f"RPC {fn:#x}: bad checksum")
            if elem.seqNum != self.cmd_seq: err(f"RPC {fn:#x}: sequence number {elem.seqNum}, not {self.cmd_seq} (NVRpcQueue.seq counts every command)")
            self.cmd_seq = elem.seqNum + 1
            self.mm[self.stat_rx:self.stat_rx + 4] = struct.pack("<I", self.cmdq.rp)   # this GSP's command-queue read pointer
            counts[f"rpc {nv.rpc_fns.get(fn, hex(fn))}"] += 1
            self.dev.channels.observe_cmd(fn, msg, self.dev.memory)
            self.handle(fn, msg)

    def handle(self, fn, msg):
        if fn in (nv.NV_VGPU_MSG_FUNCTION_GSP_SET_SYSTEM_INFO, nv.NV_VGPU_MSG_FUNCTION_SET_REGISTRY): return   # prequeued, no reply
        if fn == nv.NV_VGPU_MSG_FUNCTION_GSP_RM_ALLOC:
            if RM_FAIL and struct.unpack_from("<I", msg, nv.rpc_gsp_rm_alloc_v.hClass.offset)[0] == RM_FAIL:
                counts["rm_alloc refused (FAKE_RM_FAIL)"] += 1
                return self.post(fn, msg, rpc_result=nv_gpu.NV_ERR_INVALID_CLASS)
            return self.post(fn, msg)   # its GPFIFO channel is in self.dev.channels
        if fn == nv.NV_VGPU_MSG_FUNCTION_GSP_RM_CONTROL:
            c = nv.rpc_gsp_rm_control_v.from_buffer_copy(msg[:24])
            return self.post(fn, msg[:24] + self.control(c, msg[24:24 + c.paramsSize]))
        if fn == nv.NV_VGPU_MSG_FUNCTION_SET_PAGE_DIRECTORY: return self.post(fn, msg)   # the root is in self.dev.memory
        if fn == nv.NV_VGPU_MSG_FUNCTION_UNLOADING_GUEST_DRIVER:
            self.post(fn, msg)
            self.dev.regs[addr(FALCON_REGS["MAILBOX0"], GSP_BASE)] = 0x80000000   # suspended (kernel_gsp_tu102.c:1116-1139)
            self.dev.falcon[GSP_BASE]["riscv_active"] = False
            self.dev.unloaded, self.dev.gsp = True, None
            return
        return self.post(fn, msg)   # anything else: acknowledged as sent

    def control(self, c, params):
        """rm_control's reply parameters: tinygrad's request back, with what the GSP fills in for the controls it reads."""
        if c.cmd == nv_gpu.NV2080_CTRL_CMD_INTERNAL_STATIC_KGR_GET_CONTEXT_BUFFERS_INFO:
            p = nv_gpu.NV2080_CTRL_INTERNAL_STATIC_KGR_GET_CONTEXT_BUFFERS_INFO_PARAMS.from_buffer_copy(params)
            for e in p.engineContextBuffersInfo[0].engine: e.size, e.alignment = CTX_BUF
            return bytes(p)
        if c.cmd == nv_gpu.NV2080_CTRL_CMD_INTERNAL_STATIC_KGR_GET_INFO:
            p = nv_gpu.NV2080_CTRL_INTERNAL_STATIC_GR_GET_INFO_PARAMS.from_buffer_copy(params)
            for name, v in GR_INFO.items():
                idx = getattr(nv_gpu, "NV2080_CTRL_GR_INFO_INDEX_" + name.upper(), getattr(nv_gpu, "NV2080_CTRL_GR_INFO_INDEX_LITTER_" + name.upper(), None))
                p.engineInfo[0].infoList[idx].data = v
            return bytes(p)
        if c.cmd == nv_gpu.NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN:
            if c.hObject not in self.dev.channels.by_handle: err(f"a work-submit token for {c.hObject:#x}, which is not a channel"); return params
            token = list(self.dev.channels.by_handle).index(c.hObject) + 1   # the channel id; tinygrad adds the runlist (0 here)
            self.dev.channels.token(c.hObject, token)
            return struct.pack("<I", token)
        return params

    def post(self, fn, payload, rpc_result=0):
        """One status-queue message, in one element (tinygrad advances its read pointer by the RPC length, ip.py:79)."""
        wp = self.u32(self.stat + 16)
        length = 0x20 + len(payload)
        if 0x30 + length > self.msg_size: raise RuntimeError(f"reply to {fn:#x} does not fit one element ({length} bytes)")
        if (wp + 1) % self.msg_count == self.u32(self.cmd_rx): raise RuntimeError("status queue full: the CPU is not reading it")
        hdr = nv.rpc_message_header_v(signature=nv.NV_VGPU_MSG_SIGNATURE_VALID, header_version=3 << 24, rpc_result=rpc_result, rpc_result_private=rpc_result,
                                      function=fn, length=length, sequence=self.seq)
        elem = nv.GSP_MSG_QUEUE_ELEMENT(elemCount=1, seqNum=self.seq)
        elem.checkSum = tggpu.checksum(bytes(elem) + bytes(hdr) + payload)
        off = self.stat + 0x1000 + wp * self.msg_size
        self.mm[off:off + 0x30 + length] = bytes(elem) + bytes(hdr) + payload
        self.mm[self.stat + 16:self.stat + 20] = struct.pack("<I", (wp + 1) % self.msg_count)
        self.seq += 1

# ── the protocol ──────────────────────────────────────────────────────────────────────────────────────────────────────
def recv_exact(conn, n):
    b = bytearray()
    while len(b) < n:
        c = conn.recv(min(n - len(b), 8 << 20))
        if not c: return None
        b += c
    return bytes(b)

def bar_ok(bar, off, n): return bar in (0, 1) and off + n <= BARS[bar][1] and n <= (64 << 20)

def serve(conn, dev):
    """One client, as server.c's handle_client (:185-257), over the device."""
    while (hdr := recv_exact(conn, 33)) is not None:
        cmd, _, bar, a0, a1, a2 = REQ.unpack(hdr)
        counts[f"cmd {cmd}"] += 1
        if RECORD: RECORD.write(hdr)
        try:
            if cmd == MMIO_WRITE:
                data = recv_exact(conn, a1)
                if data is None: break
                if RECORD: RECORD.write(data)
                if not bar_ok(bar, a0, a1): err(f"MMIO_WRITE outside BAR{bar} ({a0:#x}+{a1:#x}): server.c drops it"); continue
                if bar == 1: dev.vram.write(a0, data)
                else:
                    for o in range(0, len(data) - 3, 4): dev.wr32(a0 + o, struct.unpack_from("<I", data, o)[0])
                continue
            if cmd == MMIO_READ:
                if not bar_ok(bar, a0, a1): conn.sendall(RESP.pack(1, 0, 0)); continue
                if bar == 1: data = dev.vram.read(a0, a1)
                elif VBIOS_BASE <= a0 and a0 + a1 <= VBIOS_BASE + len(VBIOS): data = VBIOS[a0 - VBIOS_BASE:a0 - VBIOS_BASE + a1]
                else: data = b"".join(struct.pack("<I", dev.rd32(a0 + o)) for o in range(0, a1, 4))[:a1]
                conn.sendall(RESP.pack(0, a1, 0) + data); continue
            if cmd == MAP_BAR:
                conn.sendall(RESP.pack(0, *BARS[bar]) if bar in BARS else RESP.pack(1, 0, 0)); continue
            if cmd == MAP_SYSMEM_FD:
                if len(dev.sysmem) >= 128: conn.sendall(RESP.pack(1, 0, 0)); continue
                s = Sysmem(dev.n_alloc, a0, a1); dev.n_alloc += 1; dev.sysmem.append(s)
                conn.sendmsg([RESP.pack(0, s.size, len(dev.sysmem) - 1)], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, struct.pack("i", s.fd))])
                continue
            if cmd == CFG_READ:
                word = CFG.get(a0 & ~3, dev.regs.get(("cfg", a0 & ~3), 0))
                conn.sendall(RESP.pack(0, (word >> (8 * (a0 & 3))) & ((1 << (8 * a1)) - 1), 0)); continue
            if cmd == CFG_WRITE:
                dev.regs[("cfg", a0 & ~3)] = a2; conn.sendall(RESP.pack(0, 0, 0)); continue
            if cmd == RESIZE_BAR: conn.sendall(RESP.pack(0, 0, 0)); continue
            err(f"command {cmd}, which BEAGLE never sends"); conn.sendall(RESP.pack(1, 0, 0))
        except Exception as e:
            import traceback
            err(f"{type(e).__name__}: {e}\n{traceback.format_exc()}")
            if cmd not in (MMIO_WRITE,): conn.sendall(RESP.pack(1, 0, 0))
    conn.close()
    if RECORD: RECORD.flush()
    if dev.gsp is not None: err("the session ended while the GSP was live: on the eGPU that unwires memory it still uses (DART)")
    for s in dev.sysmem: s.close()   # server.c's cleanup (:171-183)
    dev.sysmem, dev.gsp = [], None
    print("fake TinyGPU.app (AD107 device): client done: " + json.dumps(dict(sorted(counts.items()))), flush=True)
    print("fake TinyGPU.app (AD107 device): " + ("NO ERRORS" if not errors else f"{len(errors)} ERRORS, first: " + "; ".join(errors[:3])), flush=True)

def main():
    os.makedirs(MEM, exist_ok=True)
    if os.path.exists(sock_path): os.unlink(sock_path)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(sock_path); srv.listen(1)
    dev = Device()   # the GPU keeps its state across sessions; each session's sysmem goes with it
    print("fake TinyGPU.app (AD107 device) listening", flush=True)
    while True: serve(srv.accept()[0], dev)

if __name__ == "__main__":
    main()
