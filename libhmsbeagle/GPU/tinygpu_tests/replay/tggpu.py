"""The GPU as the V1 harness models it (TODO.md plan step V1): VRAM, the GPU's MMU through the client's own page tables
(tinygrad's MMU v2 and v3 decoders, nvdev.py:33-67), and the front end that runs GPFIFOs: semaphore acquires and releases, QMD
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

def levels(mmu_ver):
    """((shift, entries), ...) from the top level down: MemoryManager's pte_covers and pte_cnt (memory.py:186-187) for
    nvdev.py's va_bits and va_shifts (:143). MMU v2 (Ada) has 5 levels, v3 (GB20x) 6."""
    bits, shifts = (56, [12, 21, 29, 38, 47, 56]) if mmu_ver == 3 else (48, [12, 21, 29, 38, 47])
    msb = shifts + [bits + 1]
    return tuple(zip(shifts[::-1], [1 << (msb[i + 1] - msb[i]) for i in range(len(shifts))][::-1]))
LEVELS = levels(2)   # MMU v2: (47, 4), (38, 512), (29, 512), (21, 256), (12, 512)

def regs(chip="ada"):
    """tinygrad's register namespace after the chip's include() sequence (tgwire.INCLUDES)."""
    d = types.SimpleNamespace()
    for name, arch in tgwire.INCLUDES[chip]: NVDev.include(d, name, arch)
    return d

def chip(boot42):
    """(regs() chip, MMU version, compute class) for a value of NV_PMC_BOOT_42, as tinygrad decides: MMU v3 and the COT boot
    from architecture 0x1a (nvdev.py:112-116), the Blackwell classes on a GB chip, QMD v5 with them (ip.py:357-362)."""
    arch = regs().NV_PMC_BOOT_42.decode(boot42)["architecture"]
    return ("gb20x", 3, nv_gpu.BLACKWELL_COMPUTE_B) if arch >= 0x1a else ("ada", 2, nv_gpu.ADA_COMPUTE_A)

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
    def __init__(self, vram, sys_rw, r, mmu_ver=2):   # r: regs() of the chip, which has its MMU version's entry types
        self.vram, self.sys_rw, self.root, self.levels = vram, sys_rw, None, levels(mmu_ver)
        v = f"NV_MMU_VER{mmu_ver}"
        self.mmu = types.SimpleNamespace(mm=types.SimpleNamespace(level_cnt=len(self.levels), pte_covers=[1 << s for s, _ in self.levels]),
                                         mmu_ver=mmu_ver, vram=vram, pte_t=getattr(r, f"{v}_PTE"), pde_t=getattr(r, f"{v}_PDE"),
                                         dual_pde_t=getattr(r, f"{v}_DUAL_PDE"))

    def translate(self, va):
        """(is_sysmem, address), as the GPU's MMU would find it."""
        if self.root is None: raise RuntimeError(f"GPU access to VA {va:#x} before any SET_PAGE_DIRECTORY")
        pt = self.root
        for lv, (shift, cnt) in enumerate(self.levels):
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
    adds the runlist in bits 16 and up, ip.py:589-591, so a doorbell is matched on the low 16 bits). A token may come before
    its channel: a recording can hold the GSP's replies ahead of the requests that asked for them (the proxy takes its diffs
    as it reads each request, and an MMIO write has no reply, so a fast client, the C++ runtime, runs ahead of it)."""
    def __init__(self): self.by_handle, self.by_id, self.tokens = {}, {}, {}
    def add(self, handle, ring_va, entries):
        self.by_handle[handle] = dict(ring=ring_va, entries=entries, get=0)
        if handle in self.tokens: self.token(handle, self.tokens[handle])
    def token(self, handle, token):
        self.tokens[handle] = token
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
    def __init__(self, memory, channels, counts, err, compute_class=nv_gpu.ADA_COMPUTE_A):   # BLACKWELL_COMPUTE_B: QMD v5
        self.mem, self.channels, self.counts, self.err = memory, channels, counts, err
        self.on_copy = None   # (destination VA, the bytes copied), for each copy-engine copy (fake_nv_device.py's FAKE_COPY_LOG)
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
        self.qmd = ops_nv.QMD(types.SimpleNamespace(iface=types.SimpleNamespace(compute_class=compute_class)))
        self.ctl = nv_gpu.AmpereAControlGPFifo

    def gpput(self, value):   # the doorbell's channel's GPPut, in its USERD right after the ring (NVDevice._new_gpu_fifo, ops_nv.py:647)
        ch = self.channels.for_doorbell(value)
        return None if ch is None else struct.unpack("<I", self.mem.read(ch["ring"] + ch["entries"] * 8 + getattr(self.ctl, "GPPut").offset, 4))[0]

    def doorbell(self, value, put=None):   # put: run the ring only that far (a lagged doorbell's own entries), not to GPPut now
        ch = self.channels.for_doorbell(value)
        if ch is None: self.err(f"doorbell with token {value:#x}, which no channel was given"); return
        userd = ch["ring"] + ch["entries"] * 8
        if put is None: put = self.gpput(value)
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
                data = mem.read(st["src"], st["n"]); mem.write(st["dst"], data); self.counts["copies"] += 1
                if self.on_copy: self.on_copy(st["dst"], data)
            elif subc == 4 and mthd == m.dma_launch and args[0] == m.dma_sem: self.release(st["sem"], st["val"])
            else: self.counts[f"method {subc}:{mthd:#x}"] += 1   # setup methods (objects, windows, local memory, invalidates): no effect here

    def qmds(self, va):
        q = self.qmd
        # NVComputeQueue.signal's release 0 (ops_nv.py:159-170): QMD v5 names its address and payload differently (its address's
        # upper field is 25 bits wide; bind_sints_to_mem's mask=0xf clears 4 of them and ORs the whole upper word in)
        addr, pay = ("release_semaphore0_addr", "release_semaphore0_payload") if q.ver >= 4 else ("release0_address", "release0_payload")
        while True:
            q.mv[:] = self.mem.read(va, q.sz * 4)
            self.counts["launches"] += 1
            if q.read("release0_enable"):
                self.release(q.read(f"{addr}_upper") << 32 | q.read(f"{addr}_lower"),
                             q.read(f"{pay}_upper") << 32 | q.read(f"{pay}_lower"))
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
