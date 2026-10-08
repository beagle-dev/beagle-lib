"""A fake TinyGPU.app serving a fake RX 7900 (1002:744c), for the AMD boot (TODO.md plan step A2) and the AMD C++ runtime
(A1h), offline: the plugin boots it and runs on it as on the eGPU. The card is fake_am_gpu.py's register-level model, on
which tinygrad's AMDev and BEAGLE's C++ port boot.

What it models:
  - the protocol of TinyGPU.app's server.c: CFG_READ and CFG_WRITE (the card's config space), MAP_BAR, RESIZE_BAR (BAR0
    stays 256 MiB), MMIO_READ and MMIO_WRITE on BAR0 (VRAM's window), BAR2 (doorbells) and BAR5 (registers), and
    MAP_SYSMEM_FD (files with a DMA segment list);
  - the registers and VRAM: fake_am_gpu.py (FAKE_AMD_STATE cold, warm or dirty; the state lasts across sessions, as the
    GPU's does, and a session's sysmem goes with it);
  - a GPU front end: a doorbell runs its queue from where it last stopped to the doorbell's value, in order, before the next
    request: gfx11 PM4 (WAIT_REG_MEM, ACQUIRE_MEM, SET_SH_REG, DISPATCH_DIRECT, EVENT_WRITE, RELEASE_MEM) and SDMA 6 (NOP,
    COPY_LINEAR, FENCE, POLL_REGMEM), every packet decoded and checked; the read pointer then reports the queue's progress.
    Kernels are not run (results are wrong by design). With FAKE_AMD_HSACO=<variant>[,<variant>...] (SP_4 ... DP_256: the
    variants the process's instances use, TODO.md plan step A5) each dispatch must name a kernel of the build's HSACO of one
    of them (kernels/tinygpu_hsaco/, what the plugin embeds), its image uploaded unchanged, with
    that kernel's rsrc registers and each of its pointer arguments (the HSACO metadata's global_buffer args) null or
    mapped; scratch must be in VRAM.
Every GPU access (a packet's address, a copy's source and destination, a kernel's pointers, its program and scratch) must
lie inside memory mapped for the GPU, through the GMC page tables: every system address inside a MAP_SYSMEM_FD allocation of
the session (fake_am_gpu's DART check, also at every TLB flush). A session that ends with a queue live is an error too
(TinyGPU.app unwires the sysmem it polls).
FAKE_AMD_FAULT=1: from the 3rd dispatch on, the GPU stops (signals nothing more) and posts an SQ MEMVIOL in the IH ring.
FAKE_AMD_HANG=1: the same stop with no fault posted, a GPU that hangs. FAKE_AMD_DROP_AT=<n>: TinyGPU.app goes away, the
session's connection closed at its n-th request.
FAKE_AMD_RECORD=<path>: each session's requests, headers and payloads, to <path>.<n> (n from 0).
    python3 fake_amd_device.py <socket path> <work dir>
It prints "fake TinyGPU.app (AMD device) listening", and after each session its counts and NO ERRORS or the errors."""
import os, sys, json, mmap, struct, socket, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fake_am_gpu as amg

REQ, RESP = struct.Struct("<BIIQQQ"), struct.Struct("<BQQ")
MAP_BAR, MAP_SYSMEM_FD, CFG_READ, CFG_WRITE, MMIO_READ, MMIO_WRITE, RESIZE_BAR = 1, 2, 3, 4, 6, 7, 11
BARS = amg.BARS
IOVA_BASE, IOVA_STRIDE = 0x80_0000_0000, 0x1_0000_0000

def msgpack(b, i=0):
    """The msgpack value at b[i:] and the offset after it (the types LLVM's AMDGPU metadata uses, and the rest of the format's
    fixed-width ones)."""
    t = b[i]; i += 1
    if t <= 0x7f or t >= 0xe0: return t - (t >= 0xe0) * 0x100, i
    if t in (0xc0, 0xc2, 0xc3): return (None, False, True)[(t > 0xc0) + (t > 0xc2)], i
    w = {0xcc: 1, 0xcd: 2, 0xce: 4, 0xcf: 8, 0xd0: 1, 0xd1: 2, 0xd2: 4, 0xd3: 8}.get(t)
    if w: return int.from_bytes(b[i:i + w], "big", signed=t >= 0xd0), i + w
    if t in (0xca, 0xcb): w = 4 << (t - 0xca); return struct.unpack(">fd"[t - 0xca], b[i:i + w])[0], i + w
    if 0xa0 <= t <= 0xbf or t in (0xd9, 0xda, 0xdb, 0xc4, 0xc5, 0xc6):   # str and bin
        w = {0xd9: 1, 0xda: 2, 0xdb: 4, 0xc4: 1, 0xc5: 2, 0xc6: 4}.get(t, 0)
        n = t & 0x1f if not w else int.from_bytes(b[i:i + w], "big"); i += w
        return (bytes(b[i:i + n]) if t in (0xc4, 0xc5, 0xc6) else bytes(b[i:i + n]).decode()), i + n
    if 0x80 <= t <= 0x9f or t in (0xdc, 0xdd, 0xde, 0xdf):   # map and array
        w = {0xdc: 2, 0xdd: 4, 0xde: 2, 0xdf: 4}.get(t, 0)
        n = t & 0xf if not w else int.from_bytes(b[i:i + w], "big"); i += w
        out = []
        for _ in range(n * (2 if t <= 0x8f or t >= 0xde else 1)):
            v, i = msgpack(b, i); out.append(v)
        return (dict(zip(out[::2], out[1::2])) if t <= 0x8f or t >= 0xde else out), i
    raise ValueError(f"msgpack type {t:#x}")

def load_hsaco(variant):
    """The build's HSACO of a variant, what the plugin embeds and uploads: the image as BeagleAMDProgram relocates it, and per
    kernel its descriptor offset, entry, rsrc registers, kernarg size (the oracle daemon's parse, the A1h fake daemon's
    table) and the kernarg offsets of its pointer arguments (the global_buffer args of its NT_AMDGPU_METADATA note)."""
    import amd_compile_helper as ach   # the oracle's (tgpaths.setup, through fake_am_gpu)
    from tinygrad.runtime.support.elf import elf_loader
    h = (amg.tgpaths.GPU_DIR / f"kernels/tinygpu_hsaco/{variant}_gfx1100.hsaco").read_bytes()
    _image, kernels = ach.parse_kernels(h)
    img, sections, relocs = elf_loader(h)
    img = bytearray(img)
    for off, sym, typ, add in relocs:   # BeagleAMDProgram's relocation loop
        assert typ == 5
        img[off:off + 8] = struct.pack("<q", sym - off + add)
    note, o, meta = bytes(next(s.content for s in sections if s.name == ".note")), 0, None
    while o < len(note):   # ELF notes: namesz, descsz and type, then the name and the desc, each padded to 4 bytes
        namesz, descsz, typ = struct.unpack_from("<III", note, o)
        name, o = note[o + 12:o + 12 + namesz], o + 12 + ((namesz + 3) & ~3)
        if typ == 32 and name.rstrip(b"\0") == b"AMDGPU": meta = msgpack(note, o)[0]   # NT_AMDGPU_METADATA
        o += (descsz + 3) & ~3
    ptrs = {k[".symbol"][:-3]: [a[".offset"] for a in k.get(".args", []) if a[".value_kind"] == "global_buffer"]
            for k in meta["amdhsa.kernels"]}
    table = {}
    for name, (kd, d) in kernels.items():
        lds = ((d.group_segment_fixed_size + 511) // 512) & 0x1FF
        table[name] = dict(kd_off=kd, entry=d.kernel_code_entry_byte_offset, rsrc1=d.compute_pgm_rsrc1 | (1 << 20),
                           rsrc2=d.compute_pgm_rsrc2 | (lds << 15), rsrc3=d.compute_pgm_rsrc3, kernarg_size=d.kernarg_size,
                           ptrs=ptrs[name])
    return table, bytes(img)

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
        self.variants = [dict(zip(("kernels", "image"), load_hsaco(v))) for v in os.environ.get("FAKE_AMD_HSACO", "").split(",") if v]
        self.reset()
    def reset(self):
        """A session's end: TinyGPU.app unwires its sysmem; the card keeps its registers, VRAM and queues."""
        self.sysmem, self.last_kargs, self.stopped = [], None, False
        for var in self.variants: var["lib"] = None   # where the session uploaded each variant's image
        self.am.sysmem_segs, self.am.hdp_flushed = [], False
    def err(self, msg): self.am.err(msg)

    # ── GPU memory: through the page tables the boot set ──
    def sys_at(self, iova, n):
        for s in self.sysmem:
            if s.base <= iova and iova + n <= s.base + s.size: return s, iova - s.base
        return None, None
    def mapped(self, va, n=1): return self.am.translate(va, min(n, 0x1000 - (va & 0xfff))) is not None
    def rw(self, va, n, data=None, what="access"):
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
    def doorbell(self, off, value):
        q = self.am.queues.get(off)
        if q is None: return self.err(f"a doorbell at BAR2+{off:#x} that no queue has")
        self.counts[f"{q['kind']} doorbells"] += 1
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
        if not am.r.get(a("regIH_RB_BASE")): return self.err("an SQ fault to post, but no IH ring programmed (a fake warm card: use FAKE_AMD_STATE=cold)")
        base = (am.pair("regIH_RB_BASE", "regIH_RB_BASE_HI") << 8) - am.mc_base()
        am.vram_write(base, struct.pack("<8I", 10 | (239 << 8), 0, 0, 0, (2 << 21), (2 << 6), 0, 0))
        am.r[a("regIH_RB_WPTR")] = 8 << 2

    def dispatch(self, v):
        sh = lambda r: self.am.r.get(("sh", r), 0)
        self.counts["launches"] += 1
        if FAULT and self.counts["launches"] >= 3: return self.fault()
        if HANG and self.counts["launches"] >= 3: self.stopped = True; return
        prog = (sh(0x2e0c) | (sh(0x2e0d) << 32)) << 8
        kargs = sh(0x2e40) | (sh(0x2e41) << 32)
        scratch = (sh(0x2e10) | (sh(0x2e11) << 32)) << 8
        if self.last_kargs is not None and kargs < self.last_kargs: self.counts["kernargs wraps"] += 1   # the slots start over
        self.last_kargs = kargs
        in_vram = lambda va: self.mapped(va) and not (self.am.translate(va, 1) or (True,))[0]
        if not self.variants:   # no FAKE_AMD_HSACO: only where things are
            if not in_vram(prog): self.err(f"a dispatch of {prog:#x}, which is not in mapped VRAM")
            if not self.mapped(kargs): self.err(f"kernargs at {kargs:#x} are not mapped")
            if not in_vram(scratch): self.err(f"scratch at {scratch:#x} is not in VRAM")
            return
        match = k = None
        for var in self.variants:
            for name, kk in var["kernels"].items():
                lib = prog - kk["entry"] - kk["kd_off"]
                if var["lib"] not in (None, lib) or not self.mapped(lib, len(var["image"])) or \
                   self.rw(lib + kk["kd_off"], 64, what="the kernel descriptor") != var["image"][kk["kd_off"]:kk["kd_off"] + 64]: continue
                if var["lib"] is None:   # this variant's upload, which must be its whole image (of several, the one it is)
                    if self.rw(lib, len(var["image"]), what="the program image") != var["image"]:
                        if len(self.variants) > 1: continue
                        self.err("the uploaded program image differs from the HSACO's")
                    var["lib"] = lib
                match, k = name, kk
                break
            if match: break
        if match is None: return self.err(f"a dispatch of {prog:#x}, which is no kernel of the uploaded image")
        self.counts[f"kernel {match}"] += 1
        for reg, want in ((0x2e12, k["rsrc1"]), (0x2e13, k["rsrc2"]), (0x2e28, k["rsrc3"])):
            if sh(reg) != want: self.err(f"{match}: register {reg:#x} is {sh(reg):#x}, not {want:#x}")
        if k["kernarg_size"]:
            args = self.rw(kargs, k["kernarg_size"], what="the kernargs")
            for o in k["ptrs"]:   # its pointer arguments (plan step A4: by the metadata, so an int is never taken for one)
                p = struct.unpack_from("<Q", args, o)[0]
                if p and not self.mapped(p): self.err(f"{match}: its pointer argument at kernarg offset {o} is {p:#x}, which is not mapped")
        if not in_vram(scratch): self.err(f"{match}: scratch at {scratch:#x} is not in VRAM")

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
    nreq = 0
    while (hdr := recv_exact(conn, 33)) is not None:
        nreq += 1
        if nreq == DROP_AT: break   # TinyGPU.app gone: this request is never served
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
        gpu.err(f"command {cmd}, which BEAGLE never sends"); conn.sendall(RESP.pack(1, 0, 0))
    conn.close()
    for s in gpu.sysmem: s.close()
    if am.queues:   # plan step A2k: TinyGPU.app unwires the session's sysmem now, while these queues may still read it
        gpu.err(f"the session ended with {len(am.queues)} queue(s) live ({', '.join(sorted(q['kind'] for q in am.queues.values()))}): "
                "TinyGPU.app unwires the sysmem they poll (on the Mac, a DART fault)")
    if getattr(gpu, "record_path", None):   # before "client done", which the tests wait for and then end the fake on
        with open(gpu.record_path, "wb") as f: f.write(b"".join(gpu.record))
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
HANG = os.environ.get("FAKE_AMD_HANG", "") == "1"
DROP_AT = int(os.environ.get("FAKE_AMD_DROP_AT", "0"))

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
        gpu.record, gpu.record_path = ([], f"{record}.{n}") if record else (None, None)
        serve(conn, gpu, work)
        n += 1

if __name__ == "__main__":
    main()
