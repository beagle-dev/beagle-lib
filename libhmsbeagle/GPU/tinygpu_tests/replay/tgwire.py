"""The TinyGPU.app wire protocol and the V1 recording format (TODO.md plan step V1), shared by the recording proxy
(tgproxy.py), the replay server (tgreplay.py), the comparator (tgcanon.py) and the guard (tgguard.py).

Protocol, as TinyGPU.app's server.c at the tinygrad pin serves it (extra/usbgpu/tbgpu/installer/Shared/server.c) and
tinygrad's client speaks it (tinygrad/runtime/support/system.py:311-447): a 33-byte request '<BIIQQQ' (cmd, dev_id, bar,
arg0, arg1, arg2), which the server reads with one plain recv (server.c:201), so a header must arrive whole. MMIO_WRITE
is followed by arg1 payload bytes and gets no reply. Every other command gets a 17-byte reply '<BQQ' (status, resp0,
resp1): MAP_SYSMEM_FD's carries one fd by SCM_RIGHTS when it succeeds, a successful MMIO_READ is followed by resp0 (the
length asked) data bytes, and a failure is followed by resp0 bytes of message (server.c:84-88; resp0 is 0 on every other
failure, so nothing follows).

Recording (a directory):
  events.bin   records of '<BIQ' (kind, body length, t_ns since the recording started) + body, in stream order
  blobs.bin    content-addressed data: (sha256, u64 length, data) entries, each content once: sysmem pages, big payloads
  meta.json    what was recorded, where, with what, and how it ended
Bodies (u32 seq numbers every client request, markers and refusals included, in arrival order):
  REQ      seq, the 33-byte header, a data reference (an MMIO_WRITE's payload; empty otherwise)
  REPLY    seq, the 17-byte reply, has_fd u8, a data reference (MMIO_READ data or an error message)
  MARKER   seq, id u32, arg u64: a step marker the client sent (plan V1: a CFG_READ with dev_id MARKER_DEV), answered locally
  SYSMEM   seq, allocation number u32, size u64, contiguous u8, mapped size u64, server index u64, the DMA segment list
           as the server wrote it at the start of the mapping ((paddr, size) u64 pairs up to and including (0, 0))
  DIFF     seq, allocation u32, offset u64, n u32, n bytes before, n bytes after: a watched region changed since the
           previous request, seen when request seq arrived (before it was forwarded)
  PAGES    seq, allocation u32, count u32, count x (page u32, sha256): pages changed since the allocation's previous page
           snapshot, taken before trigger request seq was forwarded; contents in blobs.bin
  REFUSED  seq, the 33-byte header, why (utf-8): a command not forwarded
  END      how the session ended (JSON)
  NOTE     JSON
A data reference is u8 0 + the bytes inline, or u8 1 + sha256 + u64 length (the bytes in blobs.bin), for over 64 KiB.
"""
import os, sys, json, mmap, struct, hashlib, pathlib, collections

REQ, RESP = struct.Struct("<BIIQQQ"), struct.Struct("<BQQ")
(PROBE, MAP_BAR, MAP_SYSMEM_FD, CFG_READ, CFG_WRITE, RESET, MMIO_READ, MMIO_WRITE, MAP_SYSMEM, SYSMEM_READ, SYSMEM_WRITE,
 RESIZE_BAR, PING) = range(13)   # RemoteCmd (system.py:311-312)
CMD_NAMES = ["PROBE", "MAP_BAR", "MAP_SYSMEM_FD", "CFG_READ", "CFG_WRITE", "RESET", "MMIO_READ", "MMIO_WRITE", "MAP_SYSMEM",
             "SYSMEM_READ", "SYSMEM_WRITE", "RESIZE_BAR", "PING"]
ALLOWED = frozenset((MAP_BAR, MAP_SYSMEM_FD, CFG_READ, CFG_WRITE, MMIO_READ, MMIO_WRITE, RESIZE_BAR))   # what BEAGLE sends
PAYLOAD_CMDS = frozenset((PROBE, SYSMEM_WRITE))   # tinygrad sends arg1 payload bytes after these (system.py:367, :403)
MAX_MESSAGE = 64 << 20        # server.c BULK_BUF_SIZE
QUEUES_SIZE, CMD_QUEUE, STATUS_REGION = 0x81000, 0x1000, (0x41000, 0x81000)   # tinygrad's GSP queue mapping (init_rm_args,
                              # ip.py:364-387: a page list, then the command and status queues of 0x40000 each); only the GSP
                              # writes the status half
MARKER_DEV = 0x42454147       # 'BEAG': a marker's dev_id, which server.c never reads (plan V1, inv:verification#1)
PAGE = 0x1000

# marker ids (the bar field of a marker; its arg2 is the argument): 1-99 the record shim's tinygrad functions (argument as
# listed, | MARKER_EXIT on exit), 100-199 its other events, 0x100 and up the C++ plugin's (TinyGPUTransport.h marker())
MARKER_EXIT = 1 << 63
MARKERS = {1: "NVDev.__init__", 2: "NVDev._early_ip_init", 3: "NVDev._early_mmu_init", 4: "NV_FLCN.init_sw", 5: "NV_FLCN.init_hw",
           6: "NV_FLCN.execute_hs (base)", 7: "NV_FLCN.execute_dma (base)", 8: "NV_FLCN.reset (base)", 9: "NV_FLCN.fini_hw",
           10: "NV_GSP.init_sw", 11: "NV_GSP.init_hw", 12: "NV_GSP.init_golden_image", 13: "NV_GSP.fini_hw",
           14: "NV_GSP.rpc_rm_alloc (hClass)", 15: "NV_GSP.rpc_rm_control (cmd)", 16: "NV_GSP.rpc_set_page_directory",
           17: "NV_GSP.rpc_unloading_guest_driver", 18: "NV_GSP.run_cpu_seq (words)", 19: "MemoryManager.map_range (size)",
           20: "NVDevice.__init__", 21: "NVDevice._setup_gpfifos", 22: "NVDev.fini", 23: "Daemon.cmd_boot", 24: "Daemon.cmd_handoff",
           25: "Daemon.cmd_fini", 26: "NV_FLCN_COT.init_hw", 27: "NV_FLCN_COT.fini_hw",
           100: "status-queue message consumed (function << 32 | read pointer after it)", 101: "time.sleep (microseconds)",
           0x100: "C++ runtime: handoff received", 0x101: "C++ runtime: programs loaded", 0x102: "C++: fini begins"}

def marker_name(mid, arg):
    base = MARKERS.get(mid, f"marker {mid}")
    return (base + " exit" if mid < 100 and arg & MARKER_EXIT else base) + (f" {arg & ~MARKER_EXIT:#x}" if arg & ~MARKER_EXIT else "")

def cmd_name(cmd): return CMD_NAMES[cmd] if cmd < len(CMD_NAMES) else f"cmd{cmd}"

def recv_exact(sock, n):
    """n bytes, or what arrived before EOF (fewer)."""
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(min(n - len(buf), 8 << 20))
        if not chunk: break
        buf += chunk
    return bytes(buf)

def segments(buf):
    """The (paddr, size) pairs TinyGPU.app writes at the start of a MAP_SYSMEM_FD mapping, up to the (x, 0) terminator
    (APLRemotePCIDevice.alloc_sysmem's takewhile, system.py:444-446), and the byte length including the terminator."""
    out = []
    for off in range(0, min(len(buf), 8192) - 15, 16):
        p, sz = struct.unpack_from("<QQ", buf, off)
        if sz == 0: return out, off + 16
        out.append((p, sz))
    return out, min(len(buf), 8192)

def page_iovas(segs, size):
    """The page device addresses alloc_sysmem hands tinygrad: each segment expanded to 4 KiB pages, cut at the size."""
    pages = [p + i for p, sz in segs for i in range(0, sz, PAGE)]
    return pages[:(size + PAGE - 1) // PAGE]

# ── recording ─────────────────────────────────────────────────────────────────────────────────────────────────────────
K_REQ, K_REPLY, K_MARKER, K_SYSMEM, K_DIFF, K_PAGES, K_REFUSED, K_END, K_NOTE = range(1, 10)
KIND_NAMES = {K_REQ: "REQ", K_REPLY: "REPLY", K_MARKER: "MARKER", K_SYSMEM: "SYSMEM", K_DIFF: "DIFF", K_PAGES: "PAGES",
              K_REFUSED: "REFUSED", K_END: "END", K_NOTE: "NOTE"}
REC_HDR = struct.Struct("<BIQ")
INLINE_MAX = 64 << 10

class Blobs:
    """blobs.bin: each content once, by sha256."""
    def __init__(self, path, mode):
        self.path, self.index = pathlib.Path(path), {}
        if mode == "w":
            self.f = open(self.path, "wb")
        else:
            self.f = None
            self.mm = mmap.mmap(os.open(self.path, os.O_RDONLY), 0, prot=mmap.PROT_READ) if self.path.stat().st_size else b""
            off = 0
            while off < len(self.mm):
                sha, n = bytes(self.mm[off:off + 32]), struct.unpack_from("<Q", self.mm, off + 32)[0]
                self.index[sha] = (off + 40, n)
                off += 40 + n

    def put(self, data):
        sha = hashlib.sha256(data).digest()
        if sha not in self.index:
            self.f.write(sha + struct.pack("<Q", len(data)))
            self.f.write(data)
            self.index[sha] = len(data)
        return sha

    def get(self, sha):
        off, n = self.index[sha]
        return bytes(self.mm[off:off + n])

    def close(self):
        if self.f: self.f.close()

class Writer:
    """Appends records; flushed at every record, so a recording that ends in a fail-stop or a kill -9 keeps what it saw."""
    def __init__(self, out_dir, t0_ns):
        self.dir = pathlib.Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=False)   # never overwrite a recording
        self.events = open(self.dir / "events.bin", "wb")
        self.blobs = Blobs(self.dir / "blobs.bin", "w")
        self.t0 = t0_ns
        self.counts = collections.Counter()

    def _rec(self, kind, body, t_ns):
        self.events.write(REC_HDR.pack(kind, len(body), t_ns - self.t0))
        self.events.write(body)
        self.counts[kind] += 1

    def _ref(self, data):
        if len(data) <= INLINE_MAX: return b"\x00" + data
        return b"\x01" + self.blobs.put(data) + struct.pack("<Q", len(data))

    def req(self, t, seq, hdr, payload=b""): self._rec(K_REQ, struct.pack("<I", seq) + hdr + self._ref(payload), t)
    def reply(self, t, seq, resp, has_fd, data=b""): self._rec(K_REPLY, struct.pack("<I", seq) + resp + bytes([has_fd]) + self._ref(data), t)
    def marker(self, t, seq, mid, arg): self._rec(K_MARKER, struct.pack("<IIQ", seq, mid, arg), t)
    def sysmem(self, t, seq, alloc, size, contiguous, mapped, idx, seglist):
        self._rec(K_SYSMEM, struct.pack("<IIQBQQ", seq, alloc, size, contiguous, mapped, idx) + seglist, t)
    def diff(self, t, seq, alloc, off, before, after):
        self._rec(K_DIFF, struct.pack("<IIQI", seq, alloc, off, len(before)) + before + after, t)
    def pages(self, t, seq, alloc, changed):   # changed: [(page, data)]
        body = struct.pack("<III", seq, alloc, len(changed)) + b"".join(struct.pack("<I", p) + self.blobs.put(d) for p, d in changed)
        self._rec(K_PAGES, body, t)
    def refused(self, t, seq, hdr, why): self._rec(K_REFUSED, struct.pack("<I", seq) + hdr + why.encode(), t)
    def end(self, t, info): self._rec(K_END, json.dumps(info).encode(), t)
    def note(self, t, info): self._rec(K_NOTE, json.dumps(info).encode(), t)

    def flush(self):
        self.events.flush()
        self.blobs.f.flush()

    def close(self, meta):
        self.flush()
        self.events.close()
        self.blobs.close()
        meta = dict(meta, counts={KIND_NAMES[k]: v for k, v in sorted(self.counts.items())})
        (self.dir / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")

Event = collections.namedtuple("Event", "kind t seq f")   # f: a dict of the kind's fields

def _deref(body, off, blobs):
    if body[off] == 0: return body[off + 1:]
    sha, n = body[off + 1:off + 33], struct.unpack_from("<Q", body, off + 33)[0]
    data = blobs.get(sha)
    assert len(data) == n
    return data

def read(rec_dir):
    """A recording's events (a list of Event) and its blobs."""
    rec_dir = pathlib.Path(rec_dir)
    blobs = Blobs(rec_dir / "blobs.bin", "r")
    raw, out, off = (rec_dir / "events.bin").read_bytes(), [], 0
    while off + REC_HDR.size <= len(raw):
        kind, n, t = REC_HDR.unpack_from(raw, off)
        body = raw[off + REC_HDR.size:off + REC_HDR.size + n]
        if len(body) < n: break   # a record cut by a kill: everything before it stands
        off += REC_HDR.size + n
        if kind == K_REQ:
            seq, hdr = struct.unpack_from("<I", body)[0], body[4:37]
            out.append(Event(kind, t, seq, dict(hdr=hdr, req=REQ.unpack(hdr), payload=_deref(body, 37, blobs))))
        elif kind == K_REPLY:
            seq, resp = struct.unpack_from("<I", body)[0], body[4:21]
            out.append(Event(kind, t, seq, dict(resp=resp, reply=RESP.unpack(resp), has_fd=body[21], data=_deref(body, 22, blobs))))
        elif kind == K_MARKER:
            seq, mid, arg = struct.unpack("<IIQ", body)
            out.append(Event(kind, t, seq, dict(id=mid, arg=arg)))
        elif kind == K_SYSMEM:
            seq, alloc, size, contiguous, mapped, idx = struct.unpack_from("<IIQBQQ", body)
            seglist = body[struct.calcsize("<IIQBQQ"):]
            out.append(Event(kind, t, seq, dict(alloc=alloc, size=size, contiguous=contiguous, mapped=mapped, idx=idx, seglist=seglist,
                                                segs=segments(seglist)[0])))
        elif kind == K_DIFF:
            seq, alloc, doff, m = struct.unpack_from("<IIQI", body)
            h = struct.calcsize("<IIQI")
            out.append(Event(kind, t, seq, dict(alloc=alloc, off=doff, before=body[h:h + m], after=body[h + m:h + 2 * m])))
        elif kind == K_PAGES:
            seq, alloc, count = struct.unpack_from("<III", body)
            pages = [(struct.unpack_from("<I", body, 12 + 36 * i)[0], body[16 + 36 * i:48 + 36 * i]) for i in range(count)]
            out.append(Event(kind, t, seq, dict(alloc=alloc, pages=pages)))
        elif kind == K_REFUSED:
            seq = struct.unpack_from("<I", body)[0]
            out.append(Event(kind, t, seq, dict(hdr=body[4:37], req=REQ.unpack(body[4:37]), why=body[37:].decode())))
        elif kind in (K_END, K_NOTE):
            out.append(Event(kind, t, None, json.loads(body)))
        else: raise ValueError(f"{rec_dir}: unknown record kind {kind} at byte {off}")
    return out, blobs

def meta(rec_dir):
    p = pathlib.Path(rec_dir) / "meta.json"
    return json.loads(p.read_text()) if p.exists() else {}

# ── registers: names and triggers from tinygrad's own tables ──────────────────────────────────────────────────────────
FALCONS = {0x110000: "GSP", 0x840000: "SEC2"}   # NV_FLCN.init_hw's falcon and sec2 bases (ip.py:187)
DOORBELL = 0xbb0090   # PCIIface.setup_usermode's window at BAR0 0xbb0000 (ops_nv.py:570), token written at +0x90 (:125)
INCLUDES = {   # tinygrad's include() calls in boot order; later ones overwrite names (nvdev.py:101-128, ip.py:98-105, 285-300)
    "ada": [("nv_ref", ""), ("dev_fb", "tu102"), ("dev_gc6_island", "ga102"), ("dev_vm", "tu102"), ("dev_mmu", "tu102"),
            ("dev_gsp", "ga102"), ("dev_falcon_v4", "ga102"), ("dev_riscv_pri", "ga102"), ("dev_fbif_v4", "ga102"),
            ("dev_falcon_second_pri", "ga102"), ("dev_sec_pri", "ga102"), ("dev_bus", "tu102")],
    "gb20x": [("nv_ref", ""), ("dev_fb", "tu102"), ("dev_gc6_island", "ga102"), ("dev_therm", "gb202"), ("dev_vm", "tu102"),
              ("dev_mmu", "gh100"), ("dev_riscv_pri", "ga102"),   # BEAGLE's, first in NV_FLCN_COT.init_sw (nv_init_helper.py:520)
              ("dev_gsp", "ga102"), ("dev_falcon_v4", "gh100"), ("dev_vm", "gh100"), ("dev_fsp_pri", "gh100"), ("dev_bus", "tu102")],
}

def _nv_regs():
    """{BAR0 byte address: name} for the registers of tinygrad's per-chip include() sequences, with falcon-relative
    registers at both falcon bases; plain registers take precedence over indexed ones, named for their first 8 indices."""
    here = pathlib.Path(__file__).resolve().parent
    if str(here.parent) not in sys.path: sys.path.insert(0, str(here.parent))
    import tgpaths
    tgpaths.setup()
    from tinygrad.runtime.support.nv.nvdev import NVDev, NVReg
    names = {}
    for indexed in (False, True):
        for includes in INCLUDES.values():
            d = type("FakeNVDev", (), {})()
            for name, arch in includes: NVDev.include(d, name, arch)
            for k, v in vars(d).items():
                if not isinstance(v, NVReg) or v.base is None or callable(v.off) != indexed: continue
                rel = k.startswith(("NV_PFALCON_", "NV_PFALCON2_", "NV_PRISCV_"))
                for label, off in ([(f"{k}[{i}]", v.off(i)) for i in range(8)] if indexed else [(k, v.off)]):
                    for base, fal in (FALCONS.items() if rel else [(0, None)]):
                        names.setdefault(base + v.base + off, f"{fal}.{label}" if fal else label)
            cpuctl_alias = getattr(d, "NV_PFALCON_FALCON_CPUCTL_ALIAS", None)   # a plain offset in tinygrad's tables (ip.py:230)
            if isinstance(cpuctl_alias, int):
                for base, fal in FALCONS.items(): names.setdefault(base + cpuctl_alias, f"{fal}.NV_PFALCON_FALCON_CPUCTL_ALIAS")
    names.setdefault(DOORBELL, "DOORBELL")
    return names

_NAMES = None
def reg_names():
    global _NAMES
    if _NAMES is None: _NAMES = _nv_regs()
    return _NAMES

def reg_name(addr):
    n = reg_names().get(addr)
    return n if n else f"{addr:#x}"

TRIGGER_SUFFIXES = ("NV_PFALCON_FALCON_CPUCTL", "NV_PFALCON_FALCON_CPUCTL_ALIAS", "NV_PFALCON_FALCON_BOOTVEC",
                    "NV_PFALCON_FALCON_MAILBOX0", "NV_PFALCON_FALCON_MAILBOX1", "NV_PRISCV_RISCV_CPUCTL")
TRIGGER_NAMES = ("NV_PGSP_QUEUE_HEAD[0]", "NV_PGSP_FALCON_MAILBOX0", "NV_PGSP_FALCON_MAILBOX1", "NV_VIRTUAL_FUNCTION_PRIV_MMU_INVALIDATE",
                 "NV_PFSP_QUEUE_HEAD[0]", "NV_VIRTUAL_FUNCTION_DOORBELL")

def trigger_addrs():
    """BAR0 writes after which the GPU consumes what the CPU prepared (inv:verification#0): the GSP command-queue head,
    the falcons' CPUCTL/CPUCTL_ALIAS/BOOTVEC/MAILBOX0/1 and the RISC-V CPUCTL at both bases, the GSP mailboxes, the MMU
    invalidate (NVMemoryManager.on_range_mapped, nvdev.py:72), the FSP queue head (the COT message) and the doorbell
    (NV_VIRTUAL_FUNCTION_DOORBELL = BAR0 0xbb0090, where PCIIface writes a work token, ops_nv.py:125,570)."""
    t = {a: n for a, n in reg_names().items() if n in TRIGGER_NAMES or n.split(".", 1)[-1] in TRIGGER_SUFFIXES}
    assert t.get(DOORBELL) == "NV_VIRTUAL_FUNCTION_DOORBELL" and "NV_VIRTUAL_FUNCTION_PRIV_MMU_INVALIDATE" in t.values(), t
    return t
