"""The GPU as the V1 harness models it (TODO.md plan step V1): VRAM, the GPU's MMU through the client's own page tables
(tinygrad's MMU v2 decoders, nvdev.py:33-67), and the front end that runs GPFIFOs: semaphore acquires and releases, QMD
launches (kernels are not run) and their releases, copy-engine DMA. Shared by fake_nv_device.py, which plays the whole
GPU, and tgreplay.py, which replays what the GSP wrote but runs what the GPU does itself, so a client's pushbuffers and
page tables are exercised, not assumed (inv:verification#5).
Also the GSP message queues as both see them: the elements tinygrad queues (NVRpcQueue._send_rpc_record, ip.py:38-54) and
the replies it reads (read_resp, :63-80), and the channel facts a front end learns from them."""
import os, sys, struct, types, pathlib, collections
HERE = pathlib.Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path: sys.path.insert(0, str(HERE.parent))
import tgpaths
tgpaths.setup()
from tinygrad.runtime.autogen import nv, nv_570 as nv_gpu
from tinygrad.runtime.support.nv.nvdev import NVDev, NVPageTableEntry
from tinygrad.runtime import ops_nv
sys.path.insert(0, str(HERE))
import tgwire

PAGE = 0x1000
LEVELS = tuple(zip((47, 38, 29, 21, 12), (4, 512, 512, 256, 512)))   # MMU v2: MemoryManager's pte_covers and pte_cnt (memory.py:185-187)

def regs(chip="ada"):
    """tinygrad's register namespace after the chip's include() sequence (tgwire.INCLUDES)."""
    d = types.SimpleNamespace()
    for name, arch in tgwire.INCLUDES[chip]: NVDev.include(d, name, arch)
    return d

class Vram:
    """Sparse VRAM: 4 KiB pages, zero until written."""
    def __init__(self): self.pages = {}
    def read(self, pa, n):
        out, end = bytearray(), pa + n
        while pa < end:
            p, o = divmod(pa, PAGE); k = min(PAGE - o, end - pa)
            out += self.pages[p][o:o + k] if p in self.pages else bytes(k)
            pa += k
        return bytes(out)
    def write(self, pa, data):
        i = 0
        while i < len(data):
            p, o = divmod(pa + i, PAGE); k = min(PAGE - o, len(data) - i)
            self.pages.setdefault(p, bytearray(PAGE))[o:o + k] = data[i:i + k]
            i += k
    def view(self, off, size, fmt='Q'):   # what NVPageTableEntry reads entries through (nvdev.vram.view, nvdev.py:34)
        vram = self
        class V:
            def __getitem__(self, i): return struct.unpack("<Q", vram.read(off + i * 8, 8))[0]
            def __setitem__(self, i, v): vram.write(off + i * 8, struct.pack("<Q", v))
        return V()

class Memory:
    """GPU virtual addresses, through the client's page tables to VRAM or to sysmem by device address (sys_rw(iova, n) reads,
    sys_rw(iova, n, data) writes)."""
    def __init__(self, vram, sys_rw, r):
        self.vram, self.sys_rw, self.root = vram, sys_rw, None
        self.mmu = types.SimpleNamespace(mm=types.SimpleNamespace(level_cnt=len(LEVELS), pte_covers=[1 << s for s, _ in LEVELS]),
                                         mmu_ver=2, vram=vram, pte_t=r.NV_MMU_VER2_PTE, pde_t=r.NV_MMU_VER2_PDE, dual_pde_t=r.NV_MMU_VER2_DUAL_PDE)

    def translate(self, va):
        """(is_sysmem, address), as the GPU's MMU would find it."""
        if self.root is None: raise RuntimeError(f"GPU access to VA {va:#x} before any SET_PAGE_DIRECTORY")
        pt = self.root
        for lv, (shift, cnt) in enumerate(LEVELS):
            e = NVPageTableEntry(self.mmu, pt, lv)
            idx = (va >> shift) & (cnt - 1)
            if not e.valid(idx): raise RuntimeError(f"GPU access to unmapped VA {va:#x} (level {lv})")
            if e.is_page(idx):   # a PTE at any level: tinygrad writes its address as address_sys, whatever the aperture (nvdev.py:39)
                f = e.read_fields(idx)
                return f["aperture"] == 2, (f["address_sys"] << 12) + (va & ((1 << shift) - 1))
            pt = e.address(idx)
        raise RuntimeError(f"GPU page walk for {va:#x} ran past the last level")

    def read(self, va, n):
        out = bytearray()
        while n:
            k = min(n, PAGE - (va & (PAGE - 1)))
            sysm, a = self.translate(va)
            out += self.sys_rw(a, k) if sysm else self.vram.read(a, k)
            va += k; n -= k
        return bytes(out)

    def write(self, va, data):
        i = 0
        while i < len(data):
            k = min(len(data) - i, PAGE - ((va + i) & (PAGE - 1)))
            sysm, a = self.translate(va + i)
            if sysm: self.sys_rw(a, k, data[i:i + k])
            else: self.vram.write(a, data[i:i + k])
            i += k

class Channels:
    """GPFIFO channels (from the GPFIFO rm_alloc parameters) and their work-submit tokens (the GSP's channel ids; tinygrad
    adds the runlist in bits 16 and up, ip.py:589-591, so a doorbell is matched on the low 16 bits)."""
    def __init__(self): self.by_handle, self.by_id = {}, {}
    def add(self, handle, ring_va, entries): self.by_handle[handle] = dict(ring=ring_va, entries=entries, get=0)
    def token(self, handle, token):
        if handle in self.by_handle: self.by_id[token & 0xffff] = self.by_handle[handle]
    def for_doorbell(self, value): return self.by_id.get(value & 0xffff)

    def observe_cmd(self, fn, payload, memory=None):
        """What a client's RPC tells the front end: a GPFIFO channel, or the page-directory root."""
        if fn == nv.NV_VGPU_MSG_FUNCTION_GSP_RM_ALLOC:
            a = nv.rpc_gsp_rm_alloc_v.from_buffer_copy(payload[:32])
            if a.hClass in (nv_gpu.AMPERE_CHANNEL_GPFIFO_A, nv_gpu.BLACKWELL_CHANNEL_GPFIFO_A):
                p = nv_gpu.NV_CHANNELGPFIFO_ALLOCATION_PARAMETERS.from_buffer_copy(payload[32:32 + a.paramsSize])
                self.add(a.hObject, p.gpFifoOffset, p.gpFifoEntries)
        elif fn == nv.NV_VGPU_MSG_FUNCTION_SET_PAGE_DIRECTORY and memory is not None:
            memory.root = nv.rpc_set_page_directory_v.from_buffer_copy(payload[:48]).params.physAddress

    def observe_reply(self, fn, payload):
        """What a GSP reply tells it: the work-submit token of a channel."""
        if fn == nv.NV_VGPU_MSG_FUNCTION_GSP_RM_CONTROL and len(payload) >= 28:
            c = nv.rpc_gsp_rm_control_v.from_buffer_copy(payload[:24])
            if c.cmd == nv_gpu.NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN: self.token(c.hObject, struct.unpack_from("<I", payload, 24)[0])

class Frontend:
    """On a doorbell, the channel's GPFIFO entries from GPGet to GPPut, as ops_nv.py submits them (NVCommandQueue._submit_to_gpfifo)."""
    def __init__(self, memory, channels, counts, err):
        self.mem, self.channels, self.counts, self.err = memory, channels, counts, err
        f = ops_nv.nv_flags
        self.m = types.SimpleNamespace(   # the words build_handoff takes from tinygrad too (nv_dispatch_daemon.py)
            sem_addr_lo=nv_gpu.NVC56F_SEM_ADDR_LO,
            sem_acquire=f("NVC56F_SEM_EXECUTE", operation="acq_circ_geq", payload_size="64bit"),
            sem_release=f("NVC56F_SEM_EXECUTE", operation="release", release_wfi="en", payload_size="64bit", release_timestamp="en"),
            pcas_a=nv_gpu.NVC6C0_SEND_PCAS_A, pcas2_b=nv_gpu.NVC6C0_SEND_SIGNALING_PCAS2_B,
            dma_offset_in_upper=nv_gpu.NVC6B5_OFFSET_IN_UPPER, dma_line_length_in=nv_gpu.NVC6B5_LINE_LENGTH_IN,
            dma_launch=nv_gpu.NVC6B5_LAUNCH_DMA, dma_sem_a=nv_gpu.NVC6B5_SET_SEMAPHORE_A,
            dma_copy=f("NVC6B5_LAUNCH_DMA", data_transfer_type="non_pipelined", src_memory_layout="pitch", dst_memory_layout="pitch"),
            dma_sem=f("NVC6B5_LAUNCH_DMA", flush_enable="true", semaphore_type="release_four_word_semaphore"))
        self.qmd = ops_nv.QMD(types.SimpleNamespace(iface=types.SimpleNamespace(compute_class=nv_gpu.ADA_COMPUTE_A)))
        self.ctl = nv_gpu.AmpereAControlGPFifo

    def doorbell(self, value):
        ch = self.channels.for_doorbell(value)
        if ch is None: self.err(f"doorbell with token {value:#x}, which no channel was given"); return
        userd = ch["ring"] + ch["entries"] * 8   # NVDevice._new_gpu_fifo's USERD, right after the ring (ops_nv.py:647)
        put = struct.unpack("<I", self.mem.read(userd + getattr(self.ctl, "GPPut").offset, 4))[0]
        while ch["get"] % ch["entries"] != put:
            entry = struct.unpack("<Q", self.mem.read(ch["ring"] + (ch["get"] % ch["entries"]) * 8, 8))[0]
            if not entry >> 41 & 1: self.err(f"GPFIFO entry {entry:#x} lacks bit 41")
            self.pushbuffer(entry & ((1 << 40) - 1), (entry >> 42) & 0x1fffff)
            ch["get"] += 1
        self.mem.write(userd + getattr(self.ctl, "GPGet").offset, struct.pack("<I", ch["get"] % ch["entries"]))
        self.counts["doorbells"] += 1

    def release(self, va, value):
        self.mem.write(va, struct.pack("<Q", value))
        self.counts["releases"] += 1

    def pushbuffer(self, va, n):
        m, mem, words, i, st = self.m, self.mem, struct.unpack(f"<{n}I", self.mem.read(va, n * 4)), 0, {}
        self.counts["pushbuffers"] += 1
        while i < n:
            hdr = words[i]
            typ, subc, mthd = hdr >> 28, (hdr >> 13) & 7, (hdr & 0x1fff) << 2
            if typ == 4: args, i = ((hdr >> 16) & 0x1fff,), i + 1   # immediate data
            else:
                count = (hdr >> 16) & 0x1fff
                args, i = words[i + 1:i + 1 + count], i + 1 + count
                if typ not in (1, 2, 3, 5): self.err(f"method header {hdr:#x} of an unknown type")
            if subc == 0 and mthd == m.sem_addr_lo:
                a, v = args[0] | args[1] << 32, args[2] | args[3] << 32
                if args[4] == m.sem_release: self.release(a, v)
                elif args[4] == m.sem_acquire:
                    if struct.unpack("<Q", mem.read(a, 8))[0] < v: self.err(f"acquire of {v} but the semaphore at {a:#x} is lower: work out of order")
                else: self.err(f"unexpected SEM_EXECUTE {args[4]:#x}")
            elif subc == 1 and mthd == m.pcas_a: st["qmd"] = args[0] << 8
            elif subc == 1 and mthd == m.pcas2_b: self.qmds(st.pop("qmd"))
            elif subc == 4 and mthd == m.dma_offset_in_upper: st["src"], st["dst"] = args[0] << 32 | args[1], args[2] << 32 | args[3]
            elif subc == 4 and mthd == m.dma_line_length_in: st["n"] = args[0]
            elif subc == 4 and mthd == m.dma_sem_a: st["sem"], st["val"] = args[0] << 32 | args[1], args[2]
            elif subc == 4 and mthd == m.dma_launch and args[0] == m.dma_copy:
                mem.write(st["dst"], mem.read(st["src"], st["n"])); self.counts["copies"] += 1
            elif subc == 4 and mthd == m.dma_launch and args[0] == m.dma_sem: self.release(st["sem"], st["val"])
            else: self.counts[f"method {subc}:{mthd:#x}"] += 1   # setup methods (objects, windows, local memory, invalidates): no effect here

    def qmds(self, va):
        q = self.qmd
        while True:
            q.mv[:] = self.mem.read(va, q.sz * 4)
            self.counts["launches"] += 1
            if q.read("release0_enable"):
                self.release(q.read("release0_address_upper") << 32 | q.read("release0_address_lower"),
                             q.read("release0_payload_upper") << 32 | q.read("release0_payload_lower"))
            if not q.read("dependent_qmd0_enable"): return
            va = q.read("dependent_qmd0_pointer") << 8

# ── the GSP message queues (tinygrad's 0x81000-byte queue mapping, init_rm_args, ip.py:364-387) ─────────────────────────
def checksum(data):   # NVRpcQueue._checksum (ip.py:33-37)
    if (pad := (-len(data)) % 8): data += b"\x00" * pad
    x = 0
    for o in range(0, len(data), 8): x ^= struct.unpack_from("<Q", data, o)[0]
    return (x >> 32) ^ (x & 0xffffffff)

class QueueReader:
    """Reads a message queue's elements in order, from a read position of its own, in a mapping (mm) at queue offset q:
    the msgqTxHeader at q, elements from q + entryOff. Each element: GSP_MSG_QUEUE_ELEMENT (0x30 bytes), then the RPC
    header and payload, wrapping at the queue's end as tinygrad writes them (ip.py:48-51)."""
    def __init__(self, mm, q):
        self.mm, self.q, self.rp = mm, q, 0

    def header(self): return nv.msgqTxHeader.from_buffer_copy(self.mm[self.q:self.q + 32])

    def ring(self, tx, slot, n):
        base, size, off = self.q + tx.entryOff, tx.msgSize * tx.msgCount, slot * tx.msgSize
        first = min(n, size - off)
        return bytes(self.mm[base + off:base + off + first]) + bytes(self.mm[base:base + n - first])

    def new(self):
        """[(function, payload, element, checksum ok)] for every element written since the last call."""
        tx, out = self.header(), []
        if tx.entryOff == 0 or tx.msgSize == 0: return out   # not set up yet (the GSP writes its status queue's header)
        while self.rp != tx.writePtr:
            elem = nv.GSP_MSG_QUEUE_ELEMENT.from_buffer_copy(self.ring(tx, self.rp, 0x30))
            hdr = nv.rpc_message_header_v.from_buffer_copy(self.ring(tx, self.rp, 0x50)[0x30:])
            data = bytearray(self.ring(tx, self.rp, 0x30 + hdr.length))
            payload = bytes(data[0x50:]); data[32:36] = b"\x00" * 4
            out.append((hdr.function, payload, elem, checksum(bytes(data)) == elem.checkSum))
            self.rp = (self.rp + max(1, elem.elemCount)) % tx.msgCount
        return out
