"""Fake TinyGPU.app for offline tests of BEAGLE's C++ NV dispatch: answers the vendor probe, records posted
GPFIFO/GPPut writes, and on each doorbell runs the pushbuffers like a GPU front end would (semaphore acquires
checked against the signal, QMD chains and their releases, copy-engine DMA and semaphores) over the file-backed
buffers fake_nv_daemon.py handed to the plugin. Kernels themselves are not emulated.
After a C++ runtime handoff (the handoff carries "pool_va") it also runs the compute queue's local-memory setup
and semaphore release, and checks every launched QMD with tinygrad's own QMD reader: local memory set up first,
program address inside the uploaded image, every valid constant buffer mapped, and one local-memory size.
With FAKE_NV_HANG=1 it plays a GPU that stopped making progress: it runs the work but never writes a semaphore release,
so the plugin's timeline wait times out (the hung path, plan step P2).
    <tinygrad venv>/python fake_tinygpu_server.py <socket path> <memory dir>"""
import os, sys, json, mmap, socket, struct, types
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tgpaths
tgpaths.setup()
import nv_dispatch_daemon as d  # tinygrad's ops_nv (the hcq1 pin)

sock_path, MEM = sys.argv[1], sys.argv[2]
REQ, RESP = "<BIIQQQ", "<BQQ"
HANG = os.environ.get("FAKE_NV_HANG", "0") not in ("", "0")
CFG_READ, MMIO_WRITE = 3, 7
errors, stats = [], {"batches": 0, "launches": 0, "copies": 0, "copy_bytes": 0, "releases": 0, "posted_writes": 0}
runtime_seen = {"local_mem": None, "slm": set(), "programs": set(), "cbufs_checked": 0}

def recv_exact(conn, n):
    b = bytearray()
    while len(b) < n:
        chunk = conn.recv(n - len(b))
        if not chunk: return None
        b += chunk
    return bytes(b)

class Fifo:
    def __init__(self, h, k):
        self.ring_bar, self.ring_off, self.gpput_bar, self.gpput_off = h[f"{k}_ring_bar"], h[f"{k}_ring_off"], h[f"{k}_gpput_bar"], h[f"{k}_gpput_off"]
        self.entries, self.token, self.get, self.gpput, self.ring, self.name = h[f"{k}_entries"], h[f"{k}_token"], h[f"{k}_put"], None, {}, k

class GPU:
    def __init__(self):
        self.h = json.load(open(f"{MEM}/handoff.json"))
        self.regions = []
        for name in ("cmdq", "kargs", "staging", "signal", "vram"):
            f = open(f"{MEM}/{name}.bin", "r+b")
            size = os.fstat(f.fileno()).st_size
            va = self.h[f"{name}_va"] if name != "vram" else json.load(open(f"{MEM}/vram.json"))["va"]
            self.regions.append((va, size, mmap.mmap(f.fileno(), size)))
        self.fifos = [Fifo(self.h, "c"), Fifo(self.h, "d")]
        self.last_release = 0
        self.runtime = "pool_va" in self.h
        if self.runtime:
            self.qmd_dev = types.SimpleNamespace(iface=types.SimpleNamespace(compute_class=self.h["compute_class"]))

    def mem(self, va, n):
        for base, size, mm in self.regions:
            if base <= va and va + n <= base + size: return mm, va - base
        raise RuntimeError(f"GPU access to unmapped VA {va:#x} (+{n})")

    def read(self, va, n): mm, off = self.mem(va, n); return mm[off:off + n]
    def write(self, va, data): mm, off = self.mem(va, len(data)); mm[off:off + len(data)] = data
    def u64(self, va): return struct.unpack("<Q", self.read(va, 8))[0]

    def bits(self, qmd, key):
        hi, lo = self.h[f"{key}_hi"], self.h[f"{key}_lo"]
        num = int.from_bytes(qmd[lo // 8:hi // 8 + 1], "little")
        return (num >> (lo % 8)) & ((1 << (hi - lo + 1)) - 1)

    def release(self, addr, value, what):
        if value != self.last_release + 1: errors.append(f"{what} released {value}, expected {self.last_release + 1}")
        self.last_release = value
        if not HANG: self.write(addr, struct.pack("<Q", value))
        stats["releases"] += 1

    def check_runtime_qmd(self, qmd_va, qmd):
        q = d.ops_nv.QMD(self.qmd_dev)
        q.mv[:] = qmd
        v5 = q.ver >= 4
        if runtime_seen["local_mem"] is None: errors.append(f"QMD at {qmd_va:#x} launched before local memory was set up")
        prog = ((q.read("program_address_upper_shifted4") << 32 | q.read("program_address_lower_shifted4")) << 4 if v5
                else q.read("program_address_upper") << 32 | q.read("program_address_lower"))
        try:
            if not any(self.read(prog, 16)): errors.append(f"QMD at {qmd_va:#x}: no code at program address {prog:#x}")
        except RuntimeError as e: errors.append(f"QMD at {qmd_va:#x}: program address: {e}")
        runtime_seen["programs"].add(prog)
        runtime_seen["slm"].add(q.read("shader_local_memory_high_size_shifted4") << 4 if v5 else q.read("shader_local_memory_high_size"))
        for i in range(8):
            if not q.read(f"constant_buffer_valid_{i}"): continue
            addr = ((q.read(f"constant_buffer_addr_upper_shifted6_{i}") << 32 | q.read(f"constant_buffer_addr_lower_shifted6_{i}")) << 6
                    if v5 else q.read(f"constant_buffer_addr_upper_{i}") << 32 | q.read(f"constant_buffer_addr_lower_{i}"))
            try: self.read(addr, q.read(f"constant_buffer_size_shifted4_{i}"))
            except RuntimeError as e: errors.append(f"QMD at {qmd_va:#x}: constant buffer {i}: {e}")
            runtime_seen["cbufs_checked"] += 1

    def run_qmds(self, qmd_va):
        h = self.h
        while True:
            qmd = self.read(qmd_va, h["qmd_bytes"])
            if self.runtime: self.check_runtime_qmd(qmd_va, qmd)
            grid = struct.unpack_from("<3I", qmd, h["q_grid"])
            if 0 in grid: errors.append(f"QMD at {qmd_va:#x} has grid {grid}")
            stats["launches"] += 1
            if self.bits(qmd, "q_rel_en"):
                lo, hi = struct.unpack_from("<II", qmd, h["q_rel_addr"])
                self.release(lo | ((hi & 0xff) << 32), struct.unpack_from("<Q", qmd, h["q_rel_payload"])[0], "QMD")  # VA < 2^40; the dword holds other fields too
            if not self.bits(qmd, "q_dep_enable"): return
            qmd_va = self.bits(qmd, "q_dep_ptr") << 8

    def run_pushbuffer(self, pb_va, n):
        h, words, i, dma = self.h, struct.unpack(f"<{n}I", self.read(pb_va, n * 4)), 0, {}
        stats["batches"] += 1
        while i < n:
            hdr = words[i]
            count, subc, mthd = (hdr >> 16) & 0x1fff, (hdr >> 13) & 7, (hdr & 0x1fff) << 2
            args, i = words[i + 1:i + 1 + count], i + 1 + count
            if (hdr >> 28) != 2: errors.append(f"method header {hdr:#x} is not incrementing")
            elif subc == 0 and mthd == h["m_sem_addr_lo"]:
                addr, value = args[0] | args[1] << 32, args[2] | args[3] << 32
                if args[4] == h["f_sem_release"]: self.release(addr, value, "compute semaphore")
                elif args[4] != h["f_sem_acquire"]: errors.append(f"unexpected SEM_EXECUTE {args[4]:#x}")
                elif self.u64(addr) < value: errors.append(f"acquire of {value} but semaphore is {self.u64(addr)}: work out of order")
            elif subc == 0 and mthd == h["m_non_stall_interrupt"]: pass
            elif subc == 1 and mthd == h["m_local_mem_a"]:
                dma["local_mem"] = args[0] << 32 | args[1]
                try: self.read(dma["local_mem"], 1)
                except RuntimeError as e: errors.append(f"local memory: {e}")
            elif subc == 1 and mthd == h["m_local_mem_nt_a"]:
                runtime_seen["local_mem"] = (dma.pop("local_mem", None), args[0] << 32 | args[1])
                if args[2] != 0xff: errors.append(f"SET_SHADER_LOCAL_MEMORY_NON_THROTTLED third word {args[2]:#x}")
            elif subc == 1 and mthd == h["m_invalidate"]: pass
            elif subc == 1 and mthd == h["m_pcas_a"]: dma["qmd"] = args[0] << 8
            elif subc == 1 and mthd == h["m_pcas2_b"]: self.run_qmds(dma.pop("qmd"))
            elif subc == 4 and mthd == h["m_dma_offset_in_upper"]: dma["src"], dma["dst"] = args[0] << 32 | args[1], args[2] << 32 | args[3]
            elif subc == 4 and mthd == h["m_dma_line_length_in"]: dma["n"] = args[0]
            elif subc == 4 and mthd == h["m_dma_sem_a"]: dma["sem"], dma["value"] = args[0] << 32 | args[1], args[2]
            elif subc == 4 and mthd == h["m_dma_launch"] and args[0] == h["f_dma_copy"]:
                self.write(dma["dst"], self.read(dma["src"], dma["n"]))
                stats["copies"] += 1; stats["copy_bytes"] += dma["n"]
            elif subc == 4 and mthd == h["m_dma_launch"] and args[0] == h["f_dma_sem"]: self.release(dma["sem"], dma["value"], "copy")
            else: errors.append(f"unknown method subc={subc} mthd={mthd:#x}")

    def mmio_write(self, bar, off, data):
        stats["posted_writes"] += 1
        for f in self.fifos:
            if bar == f.ring_bar and f.ring_off <= off < f.ring_off + f.entries * 8:
                f.ring[(off - f.ring_off) // 8] = struct.unpack("<Q", data)[0]; return
            if bar == f.gpput_bar and off == f.gpput_off:
                f.gpput = struct.unpack("<I", data)[0]; return
        if bar == self.h["db_bar"] and off == self.h["db_off"]:
            f = next((f for f in self.fifos if f.token == struct.unpack("<I", data)[0]), None)
            if f is None: errors.append(f"doorbell with unknown token {data.hex()}"); return
            while f.get % f.entries != f.gpput:
                entry = f.ring.pop(f.get % f.entries, None)
                if entry is None: errors.append(f"{f.name}: GPFIFO slot {f.get % f.entries} rung but never written"); return
                if not entry >> 41 & 1: errors.append(f"{f.name}: GPFIFO entry {entry:#x} lacks bit 41")
                self.run_pushbuffer(entry & ((1 << 40) - 1), (entry >> 42) & 0x1fffff)
                f.get += 1
            return
        errors.append(f"unexpected MMIO write bar={bar} off={off:#x} len={len(data)}")

if os.path.exists(sock_path): os.unlink(sock_path)
srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
srv.bind(sock_path); srv.listen(1)
print("fake TinyGPU.app listening", flush=True)
def serve(conn):  # one client at a time, like TinyGPU.app; GPU state is loaded at the first MMIO write
    gpu = None
    while (hdr := recv_exact(conn, 33)) is not None:
        cmd, dev, bar, a0, a1, a2 = struct.unpack(REQ, hdr)
        if os.path.exists(f"{MEM}/fini"): errors.append(f"command {cmd} after the daemon's GSP unload")   # plan step P3
        if cmd == CFG_READ:   # 10de:2882, an RTX 4060, or with FAKE_NV_CHIP=gb205 10de:2f04, an RTX 5070 (plan step B1)
            conn.sendall(struct.pack(RESP, 0, 0x2f0410de if os.environ.get("FAKE_NV_CHIP") == "gb205" else 0x288210de, 0))
        elif cmd == MMIO_WRITE:
            data = recv_exact(conn, a1)
            try:
                if gpu is None: gpu = GPU()
                gpu.mmio_write(bar, a0, data)
            except Exception as e: errors.append(f"{type(e).__name__}: {e}")
        else:
            msg = f"fake TinyGPU.app: command {cmd} not expected after the vendor probe".encode()
            errors.append(msg.decode()); conn.sendall(struct.pack(RESP, 1, len(msg), 0) + msg)
    conn.close()
    print("fake TinyGPU.app: client done:", json.dumps(stats), flush=True)
    if runtime_seen["programs"]:
        lm = runtime_seen["local_mem"]
        print(f"fake TinyGPU.app: C++ runtime: local memory {lm[0]:#x} bytes_per_tpc {lm[1]:#x}; "
              f"{len(runtime_seen['programs'])} distinct program addresses launched; QMD local-memory sizes "
              f"{sorted(hex(x) for x in runtime_seen['slm'])}; {runtime_seen['cbufs_checked']} constant buffers checked", flush=True)
        if len(runtime_seen["slm"]) != 1: errors.append(f"QMDs disagree on the local-memory size: {runtime_seen['slm']}")
    print("fake TinyGPU.app: " + ("NO ERRORS" if not errors else f"{len(errors)} ERRORS, first: " + "; ".join(errors[:5])), flush=True)

while True: serve(srv.accept()[0])  # until killed
