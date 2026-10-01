"""A fake TinyGPU.app serving a fake RX 7900 (1002:744c) to the AMD C++ runtime (TODO.md plan step A1h): the offline end to end
before the plugin's own PM4 and SDMA reach the card. fake_amd_daemon.py plays amd_dispatch_daemon.py (it boots nothing:
it allocates the queues' memory here, compiles the real HSACO and hands off), and the plugin then drives this fake through
TinyGPUHybridAMDRuntime.h, as it would drive the eGPU.

What it models:
  - the protocol of TinyGPU.app's server.c: CFG_READ (the vendor and device id), MAP_BAR, MMIO_READ and MMIO_WRITE on BAR0
    (VRAM's window: the IH ring), BAR2 (doorbells) and BAR5 (registers), and MAP_SYSMEM_FD (files with a DMA segment list);
  - registers: the HDP remap register points at a flush register; the IH ring has no entries (FAKE_AMD_FAULT=1: an SQ
    MEMVIOL entry once the GPU stops, see below); every other register reads what was last written;
  - a GPU front end: a doorbell runs its queue from where it last stopped to the doorbell's value, in order, before the next
    request: gfx11 PM4 (WAIT_REG_MEM, ACQUIRE_MEM, SET_SH_REG, DISPATCH_DIRECT, EVENT_WRITE, RELEASE_MEM) and SDMA 6 (NOP,
    COPY_LINEAR, FENCE, POLL_REGMEM), every packet decoded and checked; the read pointer then reports the queue's progress.
    Kernels are not run (results are wrong by design), but each dispatch must name a kernel of the uploaded image with
    that kernel's rsrc registers, kernargs inside the kernargs buffer and scratch inside VRAM.
Every GPU access (a packet's address, a copy's source and destination, a kernel's pointers, its program and scratch) must
lie inside memory the fake daemon mapped for the GPU: a sysmem access outside it is what DART turns into a host panic.
Test-only requests (the C++ side never sends them; the fake daemon does, on the connection it shares): FAKE_MAP (a GPU
address range and its backing), FAKE_QUEUE (a queue's ring, read and write pointers and doorbell) and FAKE_HSACO.
FAKE_AMD_FAULT=1: from the 3rd dispatch on, the GPU stops (signals nothing more) and posts an SQ MEMVIOL in the IH ring.
    python3 fake_amd_device.py <socket path> <work dir>
It prints "fake TinyGPU.app (AMD device) listening", and after each session its counts and NO ERRORS or the errors."""
import os, sys, json, mmap, struct, socket, collections

REQ, RESP = struct.Struct("<BIIQQQ"), struct.Struct("<BQQ")
MAP_BAR, MAP_SYSMEM_FD, CFG_READ, MMIO_READ, MMIO_WRITE = 1, 2, 3, 6, 7
FAKE_MAP, FAKE_QUEUE, FAKE_HSACO = 0x40, 0x41, 0x42
BARS = {0: (0x2e_4000_0000, 256 << 20), 2: (0x2e_5000_0000, 2 << 20), 5: (0x2e_0030_0000, 1 << 20)}   # the card's (STATUS.md R64)
CFG = {0: 0x744c1002}
# register dwords in BAR5 that fake_amd_daemon.py's handoff names (the real ones come from the card's discovery bases)
REG = dict(hdp_remap=0x5480, ih_wptr=0x4e00, ih_rptr=0x4e01, ih_cntl=0x4e02, fault_status=0x2828, fault_addr_lo=0x2829, fault_addr_hi=0x282a,
           fault_cntl=0x282b)
HDP_FLUSH = 0x3f000   # the dword the remap register points at
IH_RING_PADDR, IH_RING_SIZE = 0x100000, 256 << 10
IOVA_BASE, IOVA_STRIDE = 0x80_0000_0000, 0x1_0000_0000

class Sysmem:
    """One MAP_SYSMEM_FD allocation: a file and made-up DMA segments, listed at the mapping's start as server.c does."""
    def __init__(self, work, n, size):
        self.n, self.size = n, max((size + 0x3fff) & ~0x3fff, 0x4000)
        self.path = os.path.join(work, f"amd_sysmem_{n}.bin")
        self.fd = os.open(self.path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o600)
        os.ftruncate(self.fd, self.size)
        self.mm = mmap.mmap(self.fd, self.size)
        base = IOVA_BASE + n * IOVA_STRIDE
        self.mm[:32] = struct.pack("<QQ", base, self.size) + bytes(16)
    def close(self):
        self.mm.close(); os.close(self.fd); os.unlink(self.path)

class Gpu:
    def __init__(self):
        self.errors, self.counts = [], collections.Counter()
        self.reset()
    def reset(self):
        self.sysmem, self.maps, self.queues, self.regs, self.vram = [], [], {}, {}, {}
        self.kernels, self.image, self.lib_va, self.hdp_flushed, self.stopped, self.kargs_map = {}, b"", None, False, False, None
    def err(self, msg):
        if len(self.errors) < 50: print(f"fake AMD GPU: ERROR {msg}", flush=True)
        self.errors.append(msg)

    # ── GPU memory: every access must be inside a FAKE_MAP range ──
    def find(self, va, n):
        for m in self.maps:
            if m["va"] <= va and va + n <= m["va"] + m["size"]: return m
        return None
    def rw(self, va, n, data=None, what="access"):
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
        out = bytearray()
        for page in range(off >> 12, (off + n + 0xfff) >> 12):   # VRAM: sparse 4 KiB pages
            lo, hi = max(off, page << 12), min(off + n, (page + 1) << 12)
            if data is None: out += bytes(self.vram.get(page, bytes(0x1000))[lo - (page << 12):hi - (page << 12)])
            else:
                p = self.vram.setdefault(page, bytearray(0x1000))
                p[lo - (page << 12):hi - (page << 12)] = data[lo - off:hi - off]
        return bytes(out) if data is None else None
    def u32(self, va, what): return struct.unpack("<I", self.rw(va, 4, what=what))[0]

    # ── registers (BAR5) ──
    def rd(self, dw):
        if dw == REG["hdp_remap"]: return HDP_FLUSH * 4
        if dw == REG["ih_wptr"]: return (8 << 2) if self.stopped and FAULT else 0   # one entry once the GPU stopped
        return self.regs.get(dw, 0)
    def wr(self, dw, v):
        if dw == HDP_FLUSH:
            self.hdp_flushed = True
            self.counts["hdp flushes"] += 1
        self.regs[dw] = v

    # ── doorbells ──
    def doorbell(self, off, value):
        q = next((q for q in self.queues.values() if q["doorbell"] == off), None)
        if q is None: return self.err(f"a doorbell at BAR2+{off:#x} that no queue has")
        self.counts[f"{q['kind']} doorbells"] += 1
        wptr = struct.unpack("<Q", self.sysmem[q["wptr_map"]].mm[q["wptr_off"]:q["wptr_off"] + 8])[0]
        # the doorbell is posted, so the client may already have stored a later submit's write pointer: never an earlier one
        if wptr < value: self.err(f"{q['kind']} doorbell {value} but the write pointer says {wptr}: wptr must be stored before the doorbell")
        if not self.hdp_flushed: self.err(f"{q['kind']} doorbell without an HDP flush since the last one (signal_doorbell's order)")
        self.hdp_flushed = False
        if value < q["done"]: return self.err(f"{q['kind']} doorbell went backwards ({value} after {q['done']})")
        ring = self.sysmem[q["ring_map"]].mm
        unit = 4 if q["kind"] == "compute" else 1   # put_value: dwords (compute) or bytes (SDMA)
        nbytes = q["ring_size"]
        def dword(i):   # the ring's i-th dword since the start, wrapping
            o = q["ring_off"] + (i * 4) % nbytes
            return struct.unpack("<I", ring[o:o + 4])[0]
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
        rp = self.sysmem[q["rptr_map"]].mm
        rp[q["rptr_off"]:q["rptr_off"] + 8] = struct.pack("<Q", q["done"])   # the engine reports its read pointer

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
            for i, x in enumerate(v[1:]): self.regs[("sh", 0x2c00 + v[0] + i)] = x
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

    def dispatch(self, v):
        sh = lambda r: self.regs.get(("sh", r), 0)
        self.counts["launches"] += 1
        if FAULT and self.counts["launches"] >= 3:
            self.stopped = True   # the wave faulted: nothing more runs, and the IH ring has the SQ entry
            return
        prog = (sh(0x2e0c) | (sh(0x2e0d) << 32)) << 8
        kargs = sh(0x2e40) | (sh(0x2e41) << 32)
        scratch = (sh(0x2e10) | (sh(0x2e11) << 32)) << 8
        match = None
        for name, k in self.kernels.items():
            lib = prog - k["entry"] - k["kd_off"]
            if self.lib_va in (None, lib) and self.find(lib, len(self.image)) and self.rw(lib + k["kd_off"], 64, what="the kernel descriptor") == self.image[k["kd_off"]:k["kd_off"] + 64]:
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
        if self.find(kargs, k["kernarg_size"]) is None or self.find(kargs, 1)["sysmem"] != self.kargs_map:
            self.err(f"{match}: kernargs at {kargs:#x} are not inside the kernargs buffer")
        if k["kernarg_size"]:
            args = self.rw(kargs, k["kernarg_size"], what="the kernargs")
            for o in range(0, k["kernarg_size"] - 7, 8):   # 64-bit words in the VRAM or sysmem windows must be mapped pointers
                p = struct.unpack_from("<Q", args, o)[0]
                if (0x10_0000_0000 <= p < 0x20_0000_0000 or 0x7f_0000_0000 <= p < 0x80_0000_0000) and self.find(p, 1) is None:
                    self.err(f"{match}: kernarg word {o} is {p:#x}, in the GPU's windows but unmapped")
        if self.find(scratch, 1) is None or self.find(scratch, 1)["sysmem"] >= 0: self.err(f"{match}: scratch at {scratch:#x} is not in VRAM")

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
    while (hdr := recv_exact(conn, 33)) is not None:
        cmd, _, bar, a0, a1, a2 = REQ.unpack(hdr)
        gpu.counts[f"cmd {cmd}"] += 1
        if cmd == MMIO_WRITE:
            data = recv_exact(conn, a1)
            if data is None: break
            if bar not in BARS or a0 + a1 > BARS[bar][1]: gpu.err(f"MMIO_WRITE outside BAR{bar} ({a0:#x}+{a1:#x})"); continue
            if bar == 2:
                if a1 != 8 or a0 % 8: gpu.err(f"a {a1}-byte doorbell write at {a0:#x}"); continue
                gpu.doorbell(a0, struct.unpack("<Q", data)[0])
            elif bar == 5:
                if a1 % 4 or a0 % 4: gpu.err(f"a {a1}-byte register write at {a0:#x}"); continue
                for o in range(0, a1, 4): gpu.wr((a0 + o) // 4, struct.unpack_from("<I", data, o)[0])
            else: gpu.err(f"an MMIO write to BAR{bar}, which the C++ runtime never makes")
            continue
        if cmd == MMIO_READ:
            if bar not in BARS or a0 + a1 > BARS[bar][1]: conn.sendall(RESP.pack(1, 0, 0)); gpu.err(f"MMIO_READ outside BAR{bar}"); continue
            if bar == 5: data = b"".join(struct.pack("<I", gpu.rd((a0 + o) // 4)) for o in range(0, a1, 4))
            elif bar == 0 and IH_RING_PADDR <= a0 and a0 + a1 <= IH_RING_PADDR + IH_RING_SIZE:   # the IH ring: an SQ MEMVIOL
                entry = [10 | (239 << 8), 0, 0, 0, (2 << 21), (2 << 6), 0, 0] if gpu.stopped and FAULT else [0] * 8   # client GFX, SQ_INTERRUPT_ID
                data = b"".join(struct.pack("<I", entry[((a0 - IH_RING_PADDR) // 4 + i) % 8]) for i in range(a1 // 4))
            else: data = bytes(a1)
            conn.sendall(RESP.pack(0, a1, 0) + data); continue
        if cmd == MAP_BAR: conn.sendall(RESP.pack(0, *BARS[bar]) if bar in BARS else RESP.pack(1, 0, 0)); continue
        if cmd == CFG_READ:
            word = CFG.get(a0 & ~3, 0)
            conn.sendall(RESP.pack(0, (word >> (8 * (a0 & 3))) & ((1 << (8 * a1)) - 1), 0)); continue
        if cmd == MAP_SYSMEM_FD:
            s = Sysmem(work, len(gpu.sysmem), a0)
            gpu.sysmem.append(s)
            conn.sendmsg([RESP.pack(0, s.size, s.n)], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, struct.pack("i", s.fd))]); continue
        if cmd in (FAKE_MAP, FAKE_QUEUE, FAKE_HSACO):
            payload = recv_exact(conn, a0)
            if cmd == FAKE_MAP:
                m = json.loads(payload)
                gpu.maps.append(m)
                if m.get("kargs"): gpu.kargs_map = m["sysmem"]
            elif cmd == FAKE_QUEUE: q = json.loads(payload); gpu.queues[q["kind"]] = dict(q, done=q["put"])
            else:
                info = json.loads(payload[:a1])
                gpu.kernels, gpu.image = info["kernels"], payload[a1:]
            conn.sendall(RESP.pack(0, 0, 0)); continue
        gpu.err(f"command {cmd}, which BEAGLE never sends"); conn.sendall(RESP.pack(1, 0, 0))
    conn.close()
    for s in gpu.sysmem: s.close()
    print("fake TinyGPU.app (AMD device): client done: " + json.dumps(dict(sorted((k, v) for k, v in gpu.counts.items() if not k.startswith("kernel ")))), flush=True)
    kernels = {k[7:]: v for k, v in gpu.counts.items() if k.startswith("kernel ")}
    print(f"fake TinyGPU.app (AMD device): kernels launched: {json.dumps(dict(sorted(kernels.items())))}", flush=True)
    print("fake TinyGPU.app (AMD device): " + ("NO ERRORS" if not gpu.errors else f"{len(gpu.errors)} ERRORS, first: " + "; ".join(gpu.errors[:3])), flush=True)
    gpu.errors, gpu.counts = [], collections.Counter()
    gpu.reset()

FAULT = os.environ.get("FAKE_AMD_FAULT", "") == "1"

def main():
    sock_path, work = sys.argv[1], sys.argv[2]
    os.makedirs(work, exist_ok=True)
    if os.path.exists(sock_path): os.unlink(sock_path)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(sock_path); srv.listen(1)
    gpu = Gpu()
    print("fake TinyGPU.app (AMD device) listening", flush=True)
    while True: serve(srv.accept()[0], gpu, work)

if __name__ == "__main__":
    main()
