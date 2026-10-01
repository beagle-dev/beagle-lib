"""A fake TinyGPU.app serving a fake RX 7900 (1002:744c), for the AMD C++ runtime (TODO.md plan step A1h) and the AMD boot
(A2): the offline end to end before the plugin's own PM4, SDMA or boot reach the card. The card is fake_am_gpu.py's
register-level model, on which tinygrad's AMDev and BEAGLE's C++ port boot. A1h's fake_amd_daemon.py boots nothing: it
allocates the queues' memory here, tells this fake where they are (FAKE_MAP, FAKE_QUEUE), compiles the real HSACO and hands
off.

What it models:
  - the protocol of TinyGPU.app's server.c: CFG_READ and CFG_WRITE (the card's config space), MAP_BAR, RESIZE_BAR (BAR0
    stays 256 MiB), MMIO_READ and MMIO_WRITE on BAR0 (VRAM's window), BAR2 (doorbells) and BAR5 (registers), and
    MAP_SYSMEM_FD (files with a DMA segment list);
  - the registers and VRAM: fake_am_gpu.py (FAKE_AMD_STATE cold, warm or dirty; the state lasts across sessions, as the
    GPU's does, and a session's sysmem goes with it);
  - a GPU front end: a doorbell runs its queue from where it last stopped to the doorbell's value, in order, before the next
    request: gfx11 PM4 (WAIT_REG_MEM, ACQUIRE_MEM, SET_SH_REG, DISPATCH_DIRECT, EVENT_WRITE, RELEASE_MEM) and SDMA 6 (NOP,
    COPY_LINEAR, FENCE, POLL_REGMEM), every packet decoded and checked; the read pointer then reports the queue's progress.
    Kernels are not run (results are wrong by design). With FAKE_HSACO each dispatch must name a kernel of the uploaded
    image with that kernel's rsrc registers and its kernargs inside the kernargs buffer; scratch must be in VRAM.
Every GPU access (a packet's address, a copy's source and destination, a kernel's pointers, its program and scratch) must
lie inside memory mapped for the GPU: after a boot, through the GMC page tables, every system address inside a
MAP_SYSMEM_FD allocation of the session (fake_am_gpu's DART check, also at every TLB flush); in A1h's runs, inside the
fake daemon's FAKE_MAP ranges. A session that ends with a queue live is an error too (TinyGPU.app unwires the sysmem it
polls). Test-only requests (the C++ side never sends them; the fake daemon does, on the connection
it shares): FAKE_MAP (a GPU address range and its backing), FAKE_QUEUE (a queue's ring, read and write pointers and
doorbell) and FAKE_HSACO.
FAKE_AMD_FAULT=1: from the 3rd dispatch on, the GPU stops (signals nothing more) and posts an SQ MEMVIOL in the IH ring.
FAKE_AMD_RECORD=<path>: each session's requests, headers and payloads, to <path>.<n> (n from 0).
    python3 fake_amd_device.py <socket path> <work dir>
It prints "fake TinyGPU.app (AMD device) listening", and after each session its counts and NO ERRORS or the errors."""
import os, sys, json, mmap, struct, socket, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fake_am_gpu as amg

REQ, RESP = struct.Struct("<BIIQQQ"), struct.Struct("<BQQ")
MAP_BAR, MAP_SYSMEM_FD, CFG_READ, CFG_WRITE, MMIO_READ, MMIO_WRITE, RESIZE_BAR = 1, 2, 3, 4, 6, 7, 11
FAKE_MAP, FAKE_QUEUE, FAKE_HSACO = 0x40, 0x41, 0x42
BARS = amg.BARS
IH_RING_PADDR, IH_RING_SIZE = 0x100000, 256 << 10   # the IH ring A1h's handoff names (a boot programs its own)
_a = lambda n: amg.card_regs()[n].addr[0]
# the registers A1h's handoff names, at the card's addresses (cmd_handoff's reg_*)
REG = dict(hdp_remap=_a("regBIF_BX0_REMAP_HDP_MEM_FLUSH_CNTL"), ih_wptr=_a("regIH_RB_WPTR"), ih_rptr=_a("regIH_RB_RPTR"),
           ih_cntl=_a("regIH_RB_CNTL"), fault_status=_a("regGCVM_L2_PROTECTION_FAULT_STATUS"),
           fault_addr_lo=_a("regGCVM_L2_PROTECTION_FAULT_ADDR_LO32"), fault_addr_hi=_a("regGCVM_L2_PROTECTION_FAULT_ADDR_HI32"),
           fault_cntl=_a("regGCVM_L2_PROTECTION_FAULT_CNTL"))
IOVA_BASE, IOVA_STRIDE = 0x80_0000_0000, 0x1_0000_0000

class Sysmem:
    """One MAP_SYSMEM_FD allocation: a file and made-up DMA segments, listed at the mapping's start as server.c does."""
    def __init__(self, work, n, size):
        self.n, self.size = n, max((size + 0x3fff) & ~0x3fff, 0x4000)
        self.path = os.path.join(work, f"amd_sysmem_{n}.bin")
        self.fd = os.open(self.path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o600)
        os.ftruncate(self.fd, self.size)
        self.mm = mmap.mmap(self.fd, self.size)
        self.base = IOVA_BASE + n * IOVA_STRIDE
        self.mm[:32] = struct.pack("<QQ", self.base, self.size) + bytes(16)
    def close(self):
        self.mm.close(); os.close(self.fd); os.unlink(self.path)

class Gpu:
    def __init__(self, say=None):
        self.say = say or (lambda m: print(m, flush=True))   # where its lines go (a harness running two fakes collects them)
        self.errors, self.counts = [], collections.Counter()
        self.am = amg.AMGpu(log=lambda m: self.say(m))
        self.am.errors, self.am.counts = self.errors, self.counts   # one list and one tally for the card and the front end
        self.reset()
    def reset(self):
        """A session's end: TinyGPU.app unwires its sysmem; the card keeps its registers, VRAM and queues."""
        self.sysmem, self.maps, self.fake_queues = [], [], {}
        self.kernels, self.image, self.lib_va, self.stopped, self.kargs_map = {}, b"", None, False, None
        self.am.sysmem_segs, self.am.hdp_flushed = [], False
    def err(self, msg): self.am.err(msg)

    # ── GPU memory: through the page tables once a boot set them, else inside a FAKE_MAP range ──
    def find(self, va, n):
        for m in self.maps:
            if m["va"] <= va and va + n <= m["va"] + m["size"]: return m
        return None
    def sys_at(self, iova, n):
        for s in self.sysmem:
            if s.base <= iova and iova + n <= s.base + s.size: return s, iova - s.base
        return None, None
    def mapped(self, va, n=1):
        if self.am.pt_root() is not None and not self.maps: return self.am.translate(va, min(n, 0x1000 - (va & 0xfff))) is not None
        return self.find(va, n) is not None
    def rw(self, va, n, data=None, what="access"):
        if self.am.pt_root() is not None and not self.maps: return self.rw_pt(va, n, data, what)
        m = self.find(va, n)
        if m is None:
            self.err(f"{what} of {n} bytes at {va:#x} is outside every mapping (a GPU page fault; in sysmem, a DART panic)")
            return bytes(n) if data is None else None
        off = va - m["va"] + m["off"]
        if m["sysmem"] >= 0:
            mm = self.sysmem[m["sysmem"]].mm
            if data is None: return bytes(mm[off:off + n])
            mm[off:off + n] = data
            return None
        if data is None: return self.am.vram_read(off, n)
        self.am.vram_write(off, data)
        return None
    def rw_pt(self, va, n, data, what):
        out, pos = bytearray(), 0
        while pos < n:
            k = min(n - pos, 0x1000 - ((va + pos) & 0xfff))
            t = self.am.translate(va + pos, k)
            if t is None:
                self.err(f"{what} of {n} bytes at {va:#x}: {va + pos:#x} is not mapped in the GPU's page tables (a GPU page fault)")
                return bytes(n) if data is None else None
            is_sys, addr = t
            if is_sys:
                s, off = self.sys_at(addr, k)
                if s is None:
                    self.err(f"{what} at {va + pos:#x}: system address {addr:#x} is in no MAP_SYSMEM_FD allocation of this session (a DART panic)")
                    return bytes(n) if data is None else None
                if data is None: out += s.mm[off:off + k]
                else: s.mm[off:off + k] = data[pos:pos + k]
            elif data is None: out += self.am.vram_read(addr, k)
            else: self.am.vram_write(addr, data[pos:pos + k])
            pos += k
        return bytes(out) if data is None else None
    def u32(self, va, what): return struct.unpack("<I", self.rw(va, 4, what=what))[0]

    # ── doorbells ──
    def queue_at(self, off):
        q = self.fake_queues.get(off)
        if q is not None: return q
        return self.am.queues.get(off)
    def doorbell(self, off, value):
        q = self.queue_at(off)
        if q is None: return self.err(f"a doorbell at BAR2+{off:#x} that no queue has")
        self.counts[f"{q['kind']} doorbells"] += 1
        if "ring_map" in q:   # A1h: the fake daemon's FAKE_QUEUE
            ring = self.sysmem[q["ring_map"]].mm
            wptr = struct.unpack("<Q", self.sysmem[q["wptr_map"]].mm[q["wptr_off"]:q["wptr_off"] + 8])[0]
            def dword(i):
                o = q["ring_off"] + (i * 4) % q["ring_size"]
                return struct.unpack("<I", ring[o:o + 4])[0]
            def report(v):
                rp = self.sysmem[q["rptr_map"]].mm
                rp[q["rptr_off"]:q["rptr_off"] + 8] = struct.pack("<Q", v)
            nbytes, unit = q["ring_size"], 4 if q["kind"] == "compute" else 1
        else:                 # a queue the boot set up in the registers
            wptr = struct.unpack("<Q", self.rw(q["wptr"], 8, what="the write pointer"))[0]
            dword = lambda i: struct.unpack("<I", self.rw(q["ring"] + (i * 4) % q["size"], 4, what="the ring"))[0]
            report = lambda v: self.rw(q["rptr"], 8, struct.pack("<Q", v), "the read pointer report")
            nbytes, unit = q["size"], q["unit"]
        # the doorbell is posted, so the client may already have stored a later submit's write pointer: never an earlier one
        if wptr < value: self.err(f"{q['kind']} doorbell {value} but the write pointer says {wptr}: wptr must be stored before the doorbell")
        if not self.am.hdp_flushed: self.err(f"{q['kind']} doorbell without an HDP flush since the last one (signal_doorbell's order)")
        self.am.hdp_flushed = False
        if value < q["done"]: return self.err(f"{q['kind']} doorbell went backwards ({value} after {q['done']})")
        start, end = q["done"] * unit // 4, value * unit // 4
        if start // (nbytes // 4) != end // (nbytes // 4) and end % (nbytes // 4): self.counts[f"{q['kind']} ring wraps"] += 1
        if end - start > nbytes // 4: self.err(f"{q['kind']}: {end - start} dwords submitted at once into a {nbytes // 4}-dword ring (overwritten unread)")
        pos = start
        while pos < end and not self.stopped:
            n = (self.pm4 if q["kind"] == "compute" else self.sdma)(dword, pos, end)
            if n is None: break
            pos += n
        if pos > end: self.err(f"{q['kind']}: a packet runs past the doorbell's write pointer")
        q["done"] = value if not self.stopped else q["done"]
        report(q["done"])   # the engine reports its read pointer

    def pm4(self, dw, pos, end):
        h = dw(pos)
        if h >> 30 != 3: return self.err(f"PM4 dword {h:#x} at {pos} is not a type-3 packet")
        op, n = (h >> 8) & 0xff, ((h >> 16) & 0x3fff) + 1
        if pos + 1 + n > end: return self.err(f"PM4 opcode {op:#x} at {pos} runs past the write pointer")
        v = [dw(pos + 1 + i) for i in range(n)]
        self.counts[f"pm4 {op:#x}"] += 1
        if op == 0x3c:   # WAIT_REG_MEM: memory (the timeline) or the HDP flush request/done registers
            if n != 6: return self.err(f"WAIT_REG_MEM with {n} dwords")
            if v[0] & (1 << 4):
                cur = self.u32(v[1] | (v[2] << 32), "WAIT_REG_MEM")
                if (v[0] & 7) != 5 or (cur & v[4]) < v[3]: self.err(f"WAIT_REG_MEM on {v[1] | (v[2] << 32):#x} for >= {v[3]} sees {cur}: the GPU would wait forever")
            elif (v[1], v[2]) != (0xe26, 0xe27): self.err(f"WAIT_REG_MEM on registers {v[1]:#x}/{v[2]:#x}, not the HDP flush request/done")
        elif op == 0x58:   # ACQUIRE_MEM
            if n != 7: return self.err(f"ACQUIRE_MEM with {n} dwords")
        elif op == 0x76:   # SET_SH_REG
            for i, x in enumerate(v[1:]): self.am.r[("sh", 0x2c00 + v[0] + i)] = x
        elif op == 0x15:   # DISPATCH_DIRECT
            if n != 4: return self.err(f"DISPATCH_DIRECT with {n} dwords")
            self.dispatch(v)
        elif op == 0x46:   # EVENT_WRITE (CS_PARTIAL_FLUSH)
            if v != [0x407]: self.err(f"EVENT_WRITE {v}")
        elif op == 0x49:   # RELEASE_MEM: the timeline's value
            if n != 7: return self.err(f"RELEASE_MEM with {n} dwords")
            if (v[1] >> 29) & 7 != 1: return self.err(f"RELEASE_MEM data_sel {(v[1] >> 29) & 7}")
            self.rw(v[2] | (v[3] << 32), 4, struct.pack("<I", v[4]), "RELEASE_MEM")
            self.counts["signals"] += 1
        else: return self.err(f"PM4 opcode {op:#x}, which the C++ encoder never emits")
        return 1 + n

    def fault(self):
        """The wave faulted: nothing more runs, and the IH ring gets an SQ MEMVIOL (client GFX, SQ_INTERRUPT_ID)."""
        self.stopped = True
        am, a = self.am, self.am.A
        base = (am.pair("regIH_RB_BASE", "regIH_RB_BASE_HI") << 8) - am.mc_base() if am.r.get(a("regIH_RB_BASE")) else IH_RING_PADDR
        am.vram_write(base, struct.pack("<8I", 10 | (239 << 8), 0, 0, 0, (2 << 21), (2 << 6), 0, 0))
        am.r[a("regIH_RB_WPTR")] = 8 << 2

    def dispatch(self, v):
        sh = lambda r: self.am.r.get(("sh", r), 0)
        self.counts["launches"] += 1
        if FAULT and self.counts["launches"] >= 3: return self.fault()
        prog = (sh(0x2e0c) | (sh(0x2e0d) << 32)) << 8
        kargs = sh(0x2e40) | (sh(0x2e41) << 32)
        scratch = (sh(0x2e10) | (sh(0x2e11) << 32)) << 8
        if not self.kernels:   # a booted run's: no FAKE_HSACO, so only where things are
            if not self.mapped(prog) or (self.am.translate(prog, 1) or (True,))[0]: self.err(f"a dispatch of {prog:#x}, which is not in mapped VRAM")
            if not self.mapped(kargs): self.err(f"kernargs at {kargs:#x} are not mapped")
            if not self.mapped(scratch) or (self.am.translate(scratch, 1) or (True,))[0]: self.err(f"scratch at {scratch:#x} is not in VRAM")
            return
        match = None
        for name, k in self.kernels.items():
            lib = prog - k["entry"] - k["kd_off"]
            if self.lib_va in (None, lib) and self.mapped(lib, len(self.image)) and self.rw(lib + k["kd_off"], 64, what="the kernel descriptor") == self.image[k["kd_off"]:k["kd_off"] + 64]:
                match = name
                if self.lib_va is None:
                    self.lib_va = lib
                    if self.rw(lib, len(self.image), what="the program image") != self.image: self.err("the uploaded program image differs from the HSACO's")
                break
        if match is None: return self.err(f"a dispatch of {prog:#x}, which is no kernel of the uploaded image")
        k = self.kernels[match]
        self.counts[f"kernel {match}"] += 1
        for reg, want in ((0x2e12, k["rsrc1"]), (0x2e13, k["rsrc2"]), (0x2e28, k["rsrc3"])):
            if sh(reg) != want: self.err(f"{match}: register {reg:#x} is {sh(reg):#x}, not {want:#x}")
        if self.maps and (self.find(kargs, k["kernarg_size"]) is None or self.find(kargs, 1)["sysmem"] != self.kargs_map):
            self.err(f"{match}: kernargs at {kargs:#x} are not inside the kernargs buffer")
        if k["kernarg_size"]:
            args = self.rw(kargs, k["kernarg_size"], what="the kernargs")
            for o in range(0, k["kernarg_size"] - 7, 8):   # 64-bit words in the VRAM or sysmem windows must be mapped pointers
                p = struct.unpack_from("<Q", args, o)[0]
                if (0x10_0000_0000 <= p < 0x20_0000_0000 or 0x7f_0000_0000 <= p < 0x80_0000_0000) and not self.mapped(p):
                    self.err(f"{match}: kernarg word {o} is {p:#x}, in the GPU's windows but unmapped")
        if not self.mapped(scratch) or (self.maps and self.find(scratch, 1)["sysmem"] >= 0): self.err(f"{match}: scratch at {scratch:#x} is not in VRAM")

    def sdma(self, dw, pos, end):
        h = dw(pos)
        op, sub = h & 0xff, (h >> 8) & 0xff
        sizes = {0: 1, 1: 7, 5: 4, 8: 6}
        if op not in sizes or (op == 1 and sub != 0): return self.err(f"SDMA opcode {op}/{sub}, which the C++ encoder never emits")
        n = sizes[op]
        if pos + n > end: return self.err(f"SDMA opcode {op} at {pos} runs past the write pointer")
        v = [dw(pos + i) for i in range(n)]
        self.counts[f"sdma {op}"] += 1
        if op == 1:   # COPY_LINEAR
            count = (v[1] & 0x3fffffff) + 1
            src, dst = v[3] | (v[4] << 32), v[5] | (v[6] << 32)
            data = self.rw(src, count, what="SDMA copy source")
            self.rw(dst, count, data, "SDMA copy destination")
            self.counts["copied bytes"] += count
        elif op == 5:   # FENCE
            if (h >> 16) & 7 != 3: self.err(f"FENCE with MTYPE {(h >> 16) & 7}")
            self.rw(v[1] | (v[2] << 32), 4, struct.pack("<I", v[3]), "FENCE")
            self.counts["signals"] += 1
        elif op == 8:   # POLL_REGMEM (memory, >=)
            cur = self.u32(v[1] | (v[2] << 32), "POLL_REGMEM")
            if (cur & v[4]) < v[3]: self.err(f"POLL_REGMEM for >= {v[3]} sees {cur}: the SDMA engine would wait forever")
        return n

def recv_exact(conn, n):
    b = bytearray()
    while len(b) < n:
        c = conn.recv(min(n - len(b), 8 << 20))
        if not c: return None
        b += c
    return bytes(b)

def serve(conn, gpu, work):
    """One client session. gpu.record, a list if set, gets every request: its header and an MMIO_WRITE's payload."""
    am = gpu.am
    while (hdr := recv_exact(conn, 33)) is not None:
        cmd, _, bar, a0, a1, a2 = REQ.unpack(hdr)
        gpu.counts[f"cmd {cmd}"] += 1
        if getattr(gpu, "record", None) is not None: gpu.record.append(hdr)
        if cmd == MMIO_WRITE:
            data = recv_exact(conn, a1)
            if data is None: break
            if getattr(gpu, "record", None) is not None: gpu.record.append(data)
            if bar not in BARS or a0 + a1 > BARS[bar][1]: gpu.err(f"MMIO_WRITE outside BAR{bar} ({a0:#x}+{a1:#x})"); continue
            if bar == 2:
                if a1 != 8 or a0 % 8: gpu.err(f"a {a1}-byte doorbell write at {a0:#x}"); continue
                gpu.doorbell(a0, struct.unpack("<Q", data)[0])
            elif bar == 5:
                if a1 % 4 or a0 % 4: gpu.err(f"a {a1}-byte register write at {a0:#x}"); continue
                for o in range(0, a1, 4): am.wr((a0 + o) // 4, struct.unpack_from("<I", data, o)[0])
            elif gpu.maps: gpu.err("an MMIO write to BAR0 in an A1h run, which the C++ runtime never makes")
            else: am.vram_write(a0, data)
            continue
        if cmd == MMIO_READ:
            if bar not in BARS or a0 + a1 > BARS[bar][1]: conn.sendall(RESP.pack(1, 0, 0)); gpu.err(f"MMIO_READ outside BAR{bar}"); continue
            if bar == 5: data = b"".join(struct.pack("<I", am.rd((a0 + o) // 4)) for o in range(0, a1, 4))
            elif bar == 0: data = am.vram_read(a0, a1)
            else: data = bytes(a1)
            conn.sendall(RESP.pack(0, a1, 0) + data); continue
        if cmd == MAP_BAR: conn.sendall(RESP.pack(0, *BARS[bar]) if bar in BARS else RESP.pack(1, 0, 0)); continue
        if cmd == RESIZE_BAR: conn.sendall(RESP.pack(0, 0, 0)); continue
        if cmd == CFG_READ: conn.sendall(RESP.pack(0, am.cfg_read(a0, a1), 0)); continue
        if cmd == CFG_WRITE: am.cfg_write(a0, a1, a2); conn.sendall(RESP.pack(0, 0, 0)); continue
        if cmd == MAP_SYSMEM_FD:
            s = Sysmem(work, len(gpu.sysmem), a0)
            gpu.sysmem.append(s)
            am.add_sysmem([(s.base, s.size)])
            conn.sendmsg([RESP.pack(0, s.size, s.n)], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, struct.pack("i", s.fd))]); continue
        if cmd in (FAKE_MAP, FAKE_QUEUE, FAKE_HSACO):
            payload = recv_exact(conn, a0)
            if cmd == FAKE_MAP:
                m = json.loads(payload)
                gpu.maps.append(m)
                if m.get("kargs"): gpu.kargs_map = m["sysmem"]
            elif cmd == FAKE_QUEUE:
                q = json.loads(payload)
                gpu.fake_queues[q["doorbell"]] = dict(q, done=q["put"])
            else:
                info = json.loads(payload[:a1])
                gpu.kernels, gpu.image = info["kernels"], payload[a1:]
            conn.sendall(RESP.pack(0, 0, 0)); continue
        gpu.err(f"command {cmd}, which BEAGLE never sends"); conn.sendall(RESP.pack(1, 0, 0))
    conn.close()
    for s in gpu.sysmem: s.close()
    if am.queues:   # plan step A2k: TinyGPU.app unwires the session's sysmem now, while these queues may still read it
        gpu.err(f"the session ended with {len(am.queues)} queue(s) live ({', '.join(sorted(q['kind'] for q in am.queues.values()))}): "
                "TinyGPU.app unwires the sysmem they poll (on the Mac, a DART fault)")
    gpu.say("fake TinyGPU.app (AMD device): client done: " + json.dumps(dict(sorted((k, v) for k, v in gpu.counts.items() if not k.startswith("kernel ")))))
    kernels = {k[7:]: v for k, v in gpu.counts.items() if k.startswith("kernel ")}
    gpu.say(f"fake TinyGPU.app (AMD device): kernels launched: {json.dumps(dict(sorted(kernels.items())))}")
    gpu.say("fake TinyGPU.app (AMD device): " + ("NO ERRORS" if not gpu.errors else f"{len(gpu.errors)} ERRORS, first: " + "; ".join(gpu.errors[:3])))
    if os.environ.get("FAKE_AMD_TOUCHED"):   # the registers the session read and wrote, merged into a JSON list (A2b's coverage)
        path = os.environ["FAKE_AMD_TOUCHED"]
        seen = {tuple(x) for x in json.load(open(path))} if os.path.exists(path) else set()
        json.dump(sorted(seen | set(am.touched)), open(path, "w"))
    am.touched.clear()
    gpu.errors.clear(); gpu.counts.clear()
    gpu.reset()

FAULT = os.environ.get("FAKE_AMD_FAULT", "") == "1"

def main():
    sock_path, work = sys.argv[1], sys.argv[2]
    os.makedirs(work, exist_ok=True)
    if os.path.exists(sock_path): os.unlink(sock_path)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(sock_path); srv.listen(1)
    gpu = Gpu()
    record, n = os.environ.get("FAKE_AMD_RECORD"), 0   # each session's requests (headers and payloads) to <path>.<n>
    print("fake TinyGPU.app (AMD device) listening", flush=True)
    while True:
        conn = srv.accept()[0]
        gpu.record = [] if record else None
        serve(conn, gpu, work)
        if record: open(f"{record}.{n}", "wb").write(b"".join(gpu.record))
        n += 1

if __name__ == "__main__":
    main()
