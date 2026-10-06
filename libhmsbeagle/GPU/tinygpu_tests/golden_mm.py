"""Golden test for TinyGPUMemory.h and TinyGPUNVMemory.h (TODO.md plan step C6) against the code they port: tinygrad's
TLSFAllocator, PageTableTraverseContext, MemoryManager, NVPageTableEntry, NVMemoryManager and PCIIfaceBase.alloc/free, with
BEAGLE's palloc patch (nv_init_helper imported, as the daemon has it). Each case runs twice against the same fake TinyGPU.app
(256 MiB of BAR1 VRAM, BAR0 writes recorded, MAP_SYSMEM_FD files with made-up DMA segments): tinygrad's code in this process,
then golden_mm.cpp. Both must print the same results (addresses, mappings, exceptions with their text, the allocators' final
states) and send the same requests, byte for byte, reads included, leaving the same VRAM.
  - TLSF: 10^5 allocations and frees (failures included) on five allocators, and the final buckets.
  - handoff: tinygrad boots (a mm_trace-like sequence with P2's teardown images), the daemon's own _mm_export is taken at the
    fork point, and the plugin's allocations at level vram (the pool) and sysmem (its four buffers, then the pool) run from
    there: MMU v2 (8188 MiB, the RTX 4060) and v3 (16304 MiB, GB20x).
  - random: 300 operations of every kind, from tinygrad's constructor (a 256 MiB card, so every zeroed allocation is inside
    BAR1: C3's client refuses a write past BAR1 that server.c would drop) and from a fork point (1 GiB, nothing zeroed past
    it), MMU v2 and v3, three seeds each.
  - fence: a system-memory mapping at an address TinyGPU.app never handed out is refused before anything is sent.
  - DMA segment lists of 1, 3 and 32 segments.
Then perturbed copies of the port must fail. No GPU and no TinyGPU.app: a private socket and TMPDIR.
    python golden_mm.py"""
import os, sys, json, random, socket, struct, tempfile, threading, subprocess, hashlib, types
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tgpaths
tgpaths.setup()
sys.path.insert(0, str(tgpaths.HERE / "replay"))
import nv_init_helper  # noqa: F401  BEAGLE's palloc patch (patch 2), as in the daemon
import nv_dispatch_daemon as d   # _mm_export, _tlsf_save, _HANDOFF_BUFS: the daemon's own
import tgwire
from tinygrad.runtime.support.system import APLRemotePCIDevice, RemoteCmd, PCIIfaceBase
from tinygrad.runtime.support.nv.nvdev import NVDev, NVMemoryManager, NVPageTableEntry
from tinygrad.runtime.support.memory import TLSFAllocator
from tinygrad.runtime.autogen import nv

HERE, WORK = str(tgpaths.HERE), str(tgpaths.WORK)
REQ, RESP = "<BIIQQQ", "<BQQ"
MB = 1 << 20
BARS = {0: (0x1c_0000_0000, 16 * MB), 1: (0x1d_0000_0000, 256 * MB)}
WPR_RSVD = 0x1f7c00000   # a gspFwRsvdStart for the Ada cases (the RTX 4060's is about this)

def recv_exact(conn, n):
    b = bytearray()
    while len(b) < n:
        chunk = conn.recv(n - len(b))
        if not chunk: return None
        b += chunk
    return bytes(b)

class FakeTG:
    """TinyGPU.app as the memory manager sees it. Every request but MAP_BAR is recorded (tinygrad caches bar_info). A write
    past BAR1 is dropped and a read past it fails, as server.c does. The n-th MAP_SYSMEM_FD gets segs[n % len(segs)] DMA
    segments at fresh device addresses with gaps between them (below 2^40)."""
    def __init__(self, priv, snap=None, segs=(1,)):
        self.priv, self.segs, self.rec, self.keep = priv, segs, bytearray(), []
        self.hooks, self.vals = [], {}   # BAR0 4-byte writes go to each hook(address, value) (golden_rm.py's GSP doorbell); 4-byte
                                         # reads return vals[address] (a value, or a function of the address; 0 if absent)
        self.bar1, self.iova, self.nsys = (bytearray(snap[0]), snap[1], snap[2]) if snap else (bytearray(256 * MB), 0x40_0000_0000, 0)
    def snapshot(self): return bytes(self.bar1), self.iova, self.nsys
    def serve(self, conn):
        while (hdr := recv_exact(conn, 33)) is not None:
            cmd, _, bar, a0, a1, _ = struct.unpack(REQ, hdr)
            if cmd != RemoteCmd.MAP_BAR: self.rec += hdr
            if cmd == RemoteCmd.MMIO_WRITE:
                data = recv_exact(conn, a1); self.rec += data
                if bar == 1 and a0 + a1 <= len(self.bar1): self.bar1[a0:a0 + a1] = data
                if bar == 0 and a1 == 4:
                    for h in self.hooks: h(a0, struct.unpack("<I", data)[0])
            elif cmd == RemoteCmd.MAP_BAR: conn.sendall(struct.pack(RESP, 0, *BARS[bar]))
            elif cmd == RemoteCmd.MMIO_READ and bar == 1 and a0 + a1 <= len(self.bar1):
                conn.sendall(struct.pack(RESP, 0, a1, 0) + bytes(self.bar1[a0:a0 + a1]))
            elif cmd == RemoteCmd.MMIO_READ and bar == 0 and a1 == 4:
                v = self.vals.get(a0, 0)
                conn.sendall(struct.pack(RESP, 0, 4, 0) + struct.pack("<I", (v(a0) if callable(v) else v) & 0xffffffff))
            elif cmd == RemoteCmd.MAP_SYSMEM_FD: self.sysmem(conn, a0)
            else:
                msg = b"not served"
                conn.sendall(struct.pack(RESP, 1, len(msg), 0) + msg)
        conn.close()
    def sysmem(self, conn, size):
        mapped = max((size + 0xfff) & ~0xfff, 0x4000)
        pages, n = mapped // 0x1000, self.segs[self.nsys % len(self.segs)]
        n = min(n, pages)
        segl = []
        for i in range(n):
            cnt = pages // n + (1 if i < pages % n else 0)
            segl.append((self.iova, cnt * 0x1000))
            self.iova += cnt * 0x1000 + 0x40000   # a gap: no two segments merge
        f = tempfile.TemporaryFile(dir=self.priv)
        f.truncate(mapped)
        f.write(b"".join(struct.pack("<QQ", a, s) for a, s in segl) + bytes(16))
        f.flush()
        socket.send_fds(conn, [struct.pack(RESP, 0, mapped, self.nsys)], [f.fileno()])
        self.nsys += 1
        self.keep.append(f)

def listen(priv):
    path = f"{priv}/s.sock"
    if os.path.exists(path): os.unlink(path)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(path); srv.listen(1); srv.settimeout(120)
    return srv, path

def serve_once(srv, fake):
    def run():
        conn, _ = srv.accept(); conn.settimeout(300); fake.serve(conn)
    t = threading.Thread(target=run, daemon=True); t.start()
    return t

# ── tinygrad's side ───────────────────────────────────────────────────────────────────────────────────────────────────
def py_dev(sock_path, mmu, vram_mb):
    """NVDev._early_mmu_init (nvdev.py:123-147) on the fake, with the chip's include() sequence; the manager is tinygrad's
    NVMemoryManager with a fresh class VA allocator."""
    pci = object.__new__(APLRemotePCIDevice)
    pci.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    pci.sock.connect(sock_path)
    pci.pcibus, pci.dev_id, pci.sysmem_fds = "usb4", 0, {}
    orig = APLRemotePCIDevice.alloc_sysmem
    def alloc_sysmem(size, vaddr=0, contiguous=False):   # counted as the daemon's inherited-fd device keeps them
        view, paddrs = orig(pci, size, vaddr, contiguous)
        pci.sysmem_fds[view.addr] = -1
        return view, paddrs
    pci.alloc_sysmem = alloc_sysmem
    dev = NVDev.__new__(NVDev)
    dev.pci_dev, dev.devfmt, dev.smi_dev, dev.is_booting, dev.mmu_ver = pci, "usb4", False, True, mmu
    for name, arch in tgwire.INCLUDES["ada" if mmu == 2 else "gb20x"]: dev.include(name, arch)
    dev.pte_t, dev.pde_t, dev.dual_pde_t = [dev.__dict__[f"NV_MMU_VER{mmu}_{g}"] for g in ("PTE", "PDE", "DUAL_PDE")]
    dev.vram_size = vram_mb * MB
    dev.vram, dev.mmio = pci.map_bar(1), pci.map_bar(0, fmt='I')
    dev.large_bar = dev.vram.nbytes >= dev.vram_size
    dev.fmc_boot = mmu == 3
    dev.gsp = types.SimpleNamespace(wpr_meta=bytes(nv.GspFwWprMeta(gspFwRsvdStart=WPR_RSVD)))
    NVMemoryManager.va_allocator = TLSFAllocator((1 << 44), base=0x1000000000)
    bits, shifts = (56, [12, 21, 29, 38, 47, 56]) if mmu == 3 else (48, [12, 21, 29, 38, 47])
    dev.mm = NVMemoryManager(dev, dev.vram_size - (64 << 20), boot_size=(2 << 20), pt_t=NVPageTableEntry, va_bits=bits, va_shifts=shifts,
                             va_base=0, palloc_ranges=[(x, x) for x in [512 << 20, 2 << 20, 4 << 10]], reserve_ptable=not dev.large_bar)
    class Iface(PCIIfaceBase):
        def __init__(self): self.pci_dev, self.dev_impl, self.vram_bar, self.dev = pci, dev, 1, object()
    return dev, Iface()

def sync(pci):
    """A round trip outside the recording (a MAP_BAR, which tinygrad's cached bar_info never sends again): every request
    sent before it has been served."""
    pci._rpc(pci.sock, pci.dev_id, RemoteCmd.MAP_BAR, bar=1)

def fmt_map(m): return f"{m.va_addr} {m.size} {m.aspace.value} " + ",".join(f"{p}:{s}" for p, s in m.paddrs)

class PyRunner:
    """golden_mm.cpp's operations on tinygrad's objects: one result line per operation."""
    def __init__(self, dev, ifa): self.dev, self.ifa, self.maps, self.bufs, self.out = dev, ifa, [], [], []
    def run(self, line):
        op, *a = line.split()
        a = [int(x, 0) for x in a]
        mm = self.dev.mm
        try:
            if op == "valloc":
                self.maps.append(None); self.bufs.append(None)
                self.maps[-1] = m = mm.valloc(a[0], a[1], uncached=bool(a[2]), contiguous=bool(a[3]), zero=bool(a[4]))
                r = fmt_map(m)
            elif op == "vfree": mm.vfree(self.maps[a[0]]); r = "ok"
            elif op == "alloc":
                self.maps.append(None); self.bufs.append(None)
                self.bufs[-1] = b = self.ifa.alloc(a[0], host=bool(a[1]), uncached=bool(a[2]), cpu_access=bool(a[3]), contiguous=bool(a[4]),
                                                   force_devmem=bool(a[5]), zero=bool(a[6]))
                r = f"{b.va_addr} {b.size} {b.meta.hMemory} {fmt_map(b.meta.mapping)}"
            elif op == "free": self.ifa.free(self.bufs[a[0]]); r = "ok"
            elif op == "palloc": r = str(mm.palloc(a[0], a[1], zero=bool(a[2]), ptable=bool(a[3])))
            elif op == "pfree": mm.pfree(a[0], ptable=bool(a[1])); r = "ok"
            elif op == "alloc_vaddr": r = str(mm.alloc_vaddr(a[0], a[1]))
            elif op == "booted": self.dev.is_booting = False; r = "ok"
            else: raise ValueError(op)
        except (MemoryError, AssertionError, RuntimeError, KeyError, ValueError, IndexError) as e:
            r = f"error {type(e).__name__}: {e}"
        self.out.append(r)
        return r
    def finish(self):
        mm = self.dev.mm
        for name, a in (("boot", mm.boot_allocator), ("ptable", mm.ptable_allocator), ("pa", mm.pa_allocator), ("va", mm.va_allocator)):
            self.out.append(f"state {name} " + " ".join(map(str, d._tlsf_save(a))))
        pa = mm.pa_allocator
        self.out.append(f"vram_end {max((pa.base + s + b[0] for s, b in pa.blocks.items() if not b[3]), default=0)}")
        return self.out

def boot_phase(dev, ifa):
    """What tinygrad allocates from boot to the handoff, as mm_trace.py approximates it, with P2's two teardown images."""
    mm = dev.mm
    dev.is_booting = False
    for size in (0x5c00, 0x9f00, 0x5c00, 0x9f00): mm.palloc((size + 0xfff) & ~0xfff)   # NV_FLCN.init_sw's images and P2's
    res_va = mm.alloc_vaddr(512 << 20); mm.page_tables(res_va, 512 << 20)                # init_golden_image
    mm.valloc(4 << 10, contiguous=True)
    mm.valloc(0x1000, contiguous=True); mm.palloc(0x5000)
    for sz in [0x400000, 0x100000, 0x100000, 0x200000, 0x80000, 0x80000, 0x200000, 0x40000, 0x40000, 0x40000]: mm.valloc(sz, contiguous=True)
    ifa.alloc(0x300000, contiguous=True, cpu_access=True, force_devmem=True)             # gpfifo_area
    for _ in range(2):
        ifa.alloc(48 << 20, uncached=True); mm.valloc(0x1000, contiguous=True); mm.palloc(0x5000)
    for sz in [0x400000, 0x100000, 0x100000]: mm.valloc(sz, contiguous=True)
    ifa.alloc(0x200000, cpu_access=True)
    for _ in range(8): ifa.alloc(2 << 20, host=True)
    ifa.alloc(0x1000, host=True, uncached=True, cpu_access=True)
    ifa.alloc(16 << 20, cpu_access=True)

def handoff_ops(level, pool):
    bufs = [f"alloc {size} {int(s.get('host', False))} {int(s.get('uncached', False))} {int(s.get('cpu_access', False))} 0 0 0"
            for _, size, s in d._HANDOFF_BUFS]
    return (bufs if level == "sysmem" else []) + [f"alloc {pool} 0 0 0 0 0 0"]

# ── cases ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
class Case:
    def __init__(self, name, mmu, vram_mb, segs=(1,), fork=None, ops=None, gen=None, pre_fork=None, fence=None):
        self.name, self.mmu, self.vram_mb, self.segs, self.fork, self.ops, self.gen, self.pre_fork, self.fence = \
            name, mmu, vram_mb, segs, fork, ops, gen, pre_fork, fence

def random_ops(seed, zero_ok, n=300):
    """A generator of operations that sees each result (on tinygrad's side) before choosing the next."""
    rng = random.Random(seed)
    def gen(run):
        live_v, live_b, live_p, idx, nsys = [], [], [], 0, 0
        for _ in range(n):
            k = rng.random()
            if k < 0.25:
                size = rng.choice([0x1000, 0x3000, 0x10000, 0x18000, 0x200000, 0x250000, 0x800000, 0x1000000, rng.randrange(1, 64) * 0x1000])
                contiguous = rng.random() < 0.3 and (zero_ok or size > (64 << 10))
                r = run(f"valloc {size} {rng.choice([0x1000, 0x1000, 0x10000, 0x200000])} {int(rng.random() < 0.3)} {int(contiguous)} "
                        f"{int(zero_ok and rng.random() < 0.3)}")
                if not r.startswith("error"): live_v.append(idx)
                idx += 1
            elif k < 0.40 and live_v:
                run(f"vfree {live_v.pop(rng.randrange(len(live_v)))}")
            elif k < 0.65:
                host = rng.random() < 0.3 and nsys < 40
                cpu = not host and rng.random() < 0.3 and nsys < 40
                size = rng.choice([0x1000, 0x4000, 0x5000, 0x200000, 0x300000, 0x900000, rng.randrange(1, 300) * 0x1000])
                if not zero_ok and not host and not cpu: size = max(size, 0x11000)   # contiguous=cpu_access only on sysmem here
                r = run(f"alloc {size} {int(host)} {int(rng.random() < 0.3)} {int(cpu)} 0 {int(rng.random() < 0.1)} {int(zero_ok and rng.random() < 0.2)}")
                nsys += host or cpu
                if not r.startswith("error"): live_b.append(idx)
                idx += 1
            elif k < 0.75 and live_b:
                run(f"free {live_b.pop(rng.randrange(len(live_b)))}")
            elif k < 0.88:
                ptable = rng.random() < 0.3
                size = rng.choice([0x1000, 0x2000, 0x5000, 0x10000, 0x20000, 0x200000])
                r = run(f"palloc {size} {rng.choice([0x1000, 0x10000, 0x200000])} {int(zero_ok and rng.random() < 0.4)} {int(ptable)}")
                if not r.startswith("error"): live_p.append((int(r), ptable))
            elif k < 0.95 and live_p:
                p, ptable = live_p.pop(rng.randrange(len(live_p)))
                run(f"pfree {p} {int(ptable)}")
            else:
                run(f"alloc_vaddr {rng.choice([0x1000, 0x5000, 0x200000, 0x40000000])} {rng.choice([0x1000, 0x200000])}")
    return gen

def cases():
    out = []
    for mmu, vram in ((2, 8188), (3, 16304)):
        for level in ("vram", "sysmem"):
            pre = (lambda dev, ifa, lvl=level: [ifa.alloc(size, **spec) for _, size, spec in d._HANDOFF_BUFS] if lvl == "vram" else None)
            out.append(Case(f"handoff {level}, MMU v{mmu} ({vram} MiB)", mmu, vram, fork=True, pre_fork=pre,
                            ops=handoff_ops(level, vram * MB // 2)))
    for mmu in (2, 3):
        for seed in (1, 2, 3):
            out.append(Case(f"random from the constructor, MMU v{mmu}, seed {seed}", mmu, 256, segs=(1, 3, 32), ops=["booted"],
                            gen=random_ops(100 * mmu + seed, zero_ok=True)))
            out.append(Case(f"random from a fork point, MMU v{mmu}, seed {seed}", mmu, 1024, segs=(3, 1, 32), fork=True,
                            gen=random_ops(200 * mmu + seed, zero_ok=False)))
    out.append(Case("segments: 1, 3 and 32 per allocation", 2, 256, segs=(1, 3, 32), ops=["booted"] + [
        f"alloc {s} 1 0 0 0 0 0" for s in (0x4000, 0x20000, 0x200000, 0x4000, 0x21000, 0x280000)]))
    return out

FENCE_OPS = ["booted", "alloc 16384 1 0 0 0 0 0", "valloc 8192 4096 0 0 0"]

def run_case(c, exe, priv, quiet=False):
    """Returns (identical, summary)."""
    srv, path = listen(priv)
    fake = FakeTG(priv, segs=c.segs)
    t = serve_once(srv, fake)
    dev, ifa = py_dev(path, c.mmu, c.vram_mb)
    py = PyRunner(dev, ifa)
    if c.fork:
        boot_phase(dev, ifa)
        if c.pre_fork: c.pre_fork(dev, ifa)
        export = json.dumps(d._mm_export(types.SimpleNamespace(iface=types.SimpleNamespace(dev_impl=dev, pci_dev=dev.pci_dev))))
        sync(dev.pci_dev)   # tinygrad's writes are posted: the fake must have taken all of them before the snapshot
        snap, mark = fake.snapshot(), len(fake.rec)
    else:
        py.out.append(f"root {dev.mm.root_page_table.paddr}")
        mark = 0
    ops = []
    def run(line): ops.append(line); return py.run(line)
    for line in c.ops or []: run(line)
    if c.gen: c.gen(run)
    py_out = py.finish()
    dev.pci_dev.sock.close(); t.join(timeout=60)
    py_rec, py_bar1 = bytes(fake.rec[mark:]), hashlib.sha256(fake.bar1).hexdigest()

    fake2 = FakeTG(priv, snap=snap, segs=c.segs) if c.fork else FakeTG(priv, segs=c.segs)
    t2 = serve_once(srv, fake2)
    with open(f"{priv}/ops.txt", "w") as f: f.write("\n".join(ops) + "\n")
    if c.fork:
        with open(f"{priv}/export.json", "w") as f: f.write(export)
        args = [exe, "mm", f"{priv}/export.json", f"{priv}/ops.txt"]
    else: args = [exe, "scratch", str(c.mmu), str(c.vram_mb * MB), f"{priv}/ops.txt"]
    r = subprocess.run(args, capture_output=True, text=True, timeout=600, env=dict(os.environ, TMPDIR=priv, APL_REMOTE_SOCK=path, BEAGLE_TINYGPU_NO_LAUNCH="1"))
    t2.join(timeout=60); srv.close()
    cpp_out = r.stdout.splitlines() + ([f"exit {r.returncode}: {r.stderr.strip()}"] if r.returncode else [])
    crec, cbar1 = bytes(fake2.rec), hashlib.sha256(fake2.bar1).hexdigest()
    same = py_out == cpp_out and py_rec == crec and py_bar1 == cbar1
    frames = count_frames(py_rec)
    errs = sum(l.startswith("error") for l in py_out)
    summary = f"{len(ops)} ops ({errs} raised), {frames['MMIO_READ']} BAR reads, {frames['MMIO_WRITE']} writes, {frames['MAP_SYSMEM_FD']} sysmem"
    if not same and not quiet:
        diff = next((i for i, (a, b) in enumerate(zip(py_out, cpp_out)) if a != b), min(len(py_out), len(cpp_out)))
        summary += (f"\n   first differing result line {diff} of {len(py_out)}/{len(cpp_out)}: tinygrad {py_out[diff:diff + 1]} c++ {cpp_out[diff:diff + 1]}"
                    f"\n   streams: {len(py_rec)} vs {len(crec)} bytes, first difference at "
                    f"{next((i for i, (a, b) in enumerate(zip(py_rec, crec)) if a != b), min(len(py_rec), len(crec)))}; VRAM {'same' if py_bar1 == cbar1 else 'differs'}")
    return same, summary

def count_frames(rec):
    out, i, names = {"MMIO_READ": 0, "MMIO_WRITE": 0, "MAP_SYSMEM_FD": 0}, 0, {6: "MMIO_READ", 7: "MMIO_WRITE", 2: "MAP_SYSMEM_FD"}
    while i + 33 <= len(rec):
        cmd, _, _, _, a1, _ = struct.unpack_from(REQ, rec, i)
        if cmd in names: out[names[cmd]] += 1
        i += 33 + (a1 if cmd == 7 else 0)
    return out

def run_tlsf(exe, priv):
    rng, fails, total = random.Random(7), 0, 0
    for size, base, n in ((1 << 44, 0x1000000000, 30000), (8106 * MB, 18 * MB, 30000), (1 * MB, 0, 30000), (16 * MB, 2 * MB, 9990), (0, 2 * MB, 10)):
        a, live, ops, want = TLSFAllocator(size, base), [], [f"config {size} {base}"], []
        for _ in range(n):
            if live and rng.random() < 0.42:
                addr = live.pop(rng.randrange(len(live)))
                ops.append(f"f {addr}"); a.free(addr); want.append("ok")
                continue
            k = rng.random()
            sz = (1 << rng.randrange(0, 34) if k < 0.3 else rng.randrange(1, 1 << rng.randrange(1, 32)) if k < 0.6 else
                  rng.randrange(1, 65) if k < 0.8 else rng.randrange(1, 1 << 20) * 0x1000)
            al = rng.choice([1, 1, 16, 0x1000, 0x1000, 0x10000, 2 << 20, 512 << 20, max(1 << (sz.bit_length() - 1), 0x1000)])
            ops.append(f"a {sz} {al}")
            try:
                addr = a.alloc(sz, al); live.append(addr); want.append(str(addr))
            except MemoryError as e: want.append(f"error MemoryError: {e}")
        want.append("state " + " ".join(map(str, d._tlsf_save(a))))
        with open(f"{priv}/tlsf.txt", "w") as f: f.write("\n".join(ops) + "\n")
        r = subprocess.run([exe, "tlsf", f"{priv}/tlsf.txt"], capture_output=True, text=True, timeout=600)
        got = r.stdout.splitlines()
        ok = got == want
        fails += not ok
        total += n
        failed = sum(w.startswith("error") for w in want)
        print(f"{'IDENTICAL' if ok else 'MISMATCH '} TLSF size {size:#x} base {base:#x}: {n} operations, {failed} failed allocations, "
              f"{len(a.blocks)} blocks at the end")
        if not ok:
            i = next((i for i, (x, y) in enumerate(zip(want, got)) if x != y), min(len(want), len(got)))
            print(f"   first difference at line {i}: tinygrad {want[i:i + 1]} c++ {got[i:i + 1]} (op {ops[i + 1] if i + 1 < len(ops) else '-'})")
    return total, fails

def run_fence(exe, priv):
    """C++ only (tinygrad has no fence): the mapping is refused, and sends nothing: the stream equals the same operations
    without it."""
    streams, outs = [], []
    for ops in (FENCE_OPS, FENCE_OPS + ["fence 8192 0x7700000000"]):
        srv, path = listen(priv)
        fake = FakeTG(priv)
        t = serve_once(srv, fake)
        with open(f"{priv}/ops.txt", "w") as f: f.write("\n".join(ops) + "\n")
        r = subprocess.run([exe, "scratch", "2", str(256 * MB), f"{priv}/ops.txt"], capture_output=True, text=True, timeout=120,
                           env=dict(os.environ, TMPDIR=priv, APL_REMOTE_SOCK=path, BEAGLE_TINYGPU_NO_LAUNCH="1"))
        t.join(timeout=60); srv.close()
        streams.append(bytes(fake.rec)); outs.append(r.stdout.splitlines())
    refused = any(l.startswith("error RuntimeError: IOVA fence: 0x7700000000+0x1000 is in no DMA segment") for l in outs[1])
    return refused and streams[0] == streams[1]

# a perturbed copy of the port must be caught: (header, text, replacement, the case that shows it)
PERTURBED = [("TinyGPUMemory.h", "uint64_t start = storage[l1][l2][0], nsize", "uint64_t start = storage[l1][l2].back(), nsize", "TLSF"),
             ("TinyGPUMemory.h", "uint64_t new_start = tg_round_up(start, align);", "uint64_t new_start = tg_round_up(start + base, align) - base;",
              "handoff vram, MMU v2 (8188 MiB)"),
             ("TinyGPUMemory.h", "kPallocZeroLimit = 64 << 10", "kPallocZeroLimit = 96 << 10", "random from the constructor, MMU v2, seed 1"),
             ("TinyGPUNVMemory.h", "nv_regs::nvbits hi = q(2 * entry_id + 1);\n        return (hi << 64) | q(2 * entry_id);",
              "nv_regs::nvbits lo = q(2 * entry_id);\n        return ((nv_regs::nvbits)q(2 * entry_id + 1) << 64) | lo;", "handoff sysmem, MMU v2 (8188 MiB)"),
             ("TinyGPUNVMemory.h", 'if (is_page(entry_id)) return read_fields(entry_id)["valid"] != 0;',
              'if (is_page(entry_id)) return pte().decode(entry(entry_id))["valid"] != 0;', "random from the constructor, MMU v3, seed 1")]   # a vfree of huge pages

def main():
    exe = f"{WORK}/golden_mm"
    tgpaths.build_cpp(f"{HERE}/golden_mm.cpp", exe)
    priv = tempfile.mkdtemp(dir="/tmp", prefix="tgmm.")
    tempfile.tempdir = priv
    total, fails = run_tlsf(exe, priv)
    print(f"TLSF: {total} operations")
    all_cases = cases()
    for c in all_cases:
        same, summary = run_case(c, exe, priv)
        fails += not same
        print(f"{'IDENTICAL' if same else 'MISMATCH '} {c.name}: {summary}")
    ok = run_fence(exe, priv)
    fails += not ok
    print(f"{'PASS' if ok else 'FAIL'} fence: a system-memory mapping at a device address TinyGPU.app never handed out is refused, sending nothing")
    caught, gpu = 0, tgpaths.REPO / "libhmsbeagle" / "GPU"
    for hdr, old, new, case in PERTURBED:
        inc = f"{priv}/perturbed"; os.makedirs(f"{inc}/libhmsbeagle/GPU", exist_ok=True)
        for f in ("TinyGPUMemory.h", "TinyGPUNVMemory.h"):
            text = (gpu / f).read_text()
            if f == hdr: assert text.count(old) == 1, (hdr, old); text = text.replace(old, new)
            open(f"{inc}/libhmsbeagle/GPU/{f}", "w").write(text)
        pexe = f"{priv}/golden_mm_perturbed"
        tgpaths.build_cpp(f"{HERE}/golden_mm.cpp", pexe, "-iquote", inc)   # found before the repository's
        if case == "TLSF":
            with open(os.devnull, "w") as null:
                saved, sys.stdout = sys.stdout, null
                try: _, f = run_tlsf(pexe, priv)
                finally: sys.stdout = saved
            hit = f > 0
        else: hit = not run_case(next(c for c in all_cases if c.name == case), pexe, priv, quiet=True)[0]
        caught += hit
        print(f"perturbed {hdr} ({old.splitlines()[0][:60]} ...): {'REJECTED' if hit else 'NOT CAUGHT'} by '{case}'")
    fails += len(PERTURBED) - caught
    print(f"C6 memory manager vs tinygrad: {'all identical' if not fails else f'{fails} FAILED'}")
    sys.exit(1 if fails else 0)

if __name__ == "__main__":
    main()
